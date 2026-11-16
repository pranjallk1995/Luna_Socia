"""Luna's ears and voice, on the GPU (or the CPU with LUNASOCIA_VOICE_DEVICE=cpu).

Speech to text: faster-whisper. Text to speech: Kokoro (82M, ONNX). Both models are
downloaded to the data volume the first time they are used. On the GPU they take about
1.1 GB together and are loaded and warmed up when Socia starts: the first call in a process
compiles CUDA kernels for this card (about 30 s), which CUDA_CACHE_PATH keeps on the data
volume for the next start. If the GPU cannot be used, they run on the CPU instead.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import urllib.request
from pathlib import Path

from app.config import SociaConfig
from app.model_usage import ModelUsage

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
# Half the CPU: enough to transcribe a voice message in about real time, leaving the rest to the apps.
WHISPER_THREADS = max(4, (os.cpu_count() or 8) // 2)
logger = logging.getLogger("socia")
# Emoji and other pictographs, which a voice would read out by name.
PICTOGRAPHS = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")


# Devanagari, which only the Hindi phonemizer can read.
DEVANAGARI = re.compile("[\u0900-\u097F]")
SAMPLE = "Hey you. I was just thinking about you… how was your day?"


def language(text: str, voice: str) -> str:
    """Kokoro's language for this text in this voice: Hindi for Devanagari, else the voice's English."""
    if DEVANAGARI.search(text):
        return "hi"
    return "en-gb" if voice.startswith("b") else "en-us"


def speakable(text: str) -> str:
    """The words of a message, without emoji or the text of actions like *smiles*."""
    text = PICTOGRAPHS.sub("", text)
    text = re.sub(r"\*[^*]{1,60}\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.rename(target)


class Voice:
    def __init__(self, config: SociaConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.whisper = None
        self.kokoro = None
        # Where each one actually runs, once loaded: "GPU" or "CPU".
        self.whisper_device = self.kokoro_device = ""
        # For Luna Sentia's model monitor, which cannot see models that run outside Ollama.
        self.hearing = ModelUsage(f"Whisper {config.stt_model} (faster-whisper)",
                                  "Hears voice messages: speech to text",
                                  lambda: self._state(self.whisper, self.whisper_device))
        self.speaking = ModelUsage("Kokoro 82M (kokoro-onnx)", "Her voice: reads her replies aloud",
                                   lambda: self._state(self.kokoro, self.kokoro_device))

    def _state(self, model, device: str) -> tuple[bool, str]:
        if model is None:
            return False, f"loads at start ({self.config.voice_device.upper()}); stays loaded"
        return True, device

    @property
    def on_gpu(self) -> bool:
        return self.config.voice_device == "cuda"

    def _load_whisper(self) -> None:
        if self.whisper is not None:
            return
        from faster_whisper import WhisperModel

        root = str(self.config.models_dir / "whisper")
        if self.on_gpu:
            try:
                self.whisper = WhisperModel(self.config.stt_model, device="cuda", compute_type="int8_float16",
                                            download_root=root)
                self.whisper_device = "GPU (int8/float16)"
                return
            except Exception as error:  # noqa: BLE001 - she still hears on the CPU
                logger.warning("Whisper could not use the GPU, using the CPU: %s", error)
        self.whisper = WhisperModel(self.config.stt_model, device="cpu", compute_type="int8",
                                    cpu_threads=WHISPER_THREADS, download_root=root)
        self.whisper_device = "CPU (int8)"

    def _load_kokoro(self) -> None:
        if self.kokoro is not None:
            return
        from kokoro_onnx import Kokoro

        folder = self.config.models_dir / "kokoro"
        for name in KOKORO_FILES:
            if not (folder / name).exists():
                download(f"{KOKORO_RELEASE}/{name}", folder / name)
        model, voices = str(folder / KOKORO_FILES[0]), str(folder / KOKORO_FILES[1])
        if self.on_gpu:
            try:
                import onnxruntime

                options = onnxruntime.SessionOptions()
                options.log_severity_level = 3
                session = onnxruntime.InferenceSession(model, options, providers=[
                    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"])
                if session.get_providers()[0] == "CUDAExecutionProvider":
                    self.kokoro = Kokoro.from_session(session, voices)
                    self.kokoro_device = "GPU (ONNX, CUDA)"
                    return
                logger.warning("Kokoro could not use the GPU, using the CPU")
            except Exception as error:  # noqa: BLE001 - she still speaks on the CPU
                logger.warning("Kokoro could not use the GPU, using the CPU: %s", error)
        self.kokoro = Kokoro(model, voices)
        self.kokoro_device = "CPU (ONNX)"

    def preload(self) -> None:
        """Load both models and run each once, so the first voice message is not the slow one."""
        import numpy

        try:
            with self.lock:
                self._load_kokoro()
                samples, rate = self.kokoro.create("Mm-hm, I'm here.", voice=self.config.tts_voice, speed=1.0,
                                                   lang="en-us")
                self._load_whisper()
                speech = numpy.interp(numpy.arange(0, len(samples), rate / 16000), numpy.arange(len(samples)),
                                      samples).astype(numpy.float32)
                list(self.whisper.transcribe(speech, beam_size=1, language="en")[0])
            logger.info("Her voice is ready: Whisper on the %s, Kokoro on the %s", self.whisper_device,
                        self.kokoro_device)
        except Exception:  # noqa: BLE001 - loaded on first use instead
            logger.exception("Could not load her voice at start")

    def transcribe(self, audio: bytes) -> str:
        with self.lock, self.hearing.record():
            self._load_whisper()
            from faster_whisper import decode_audio

            samples = decode_audio(io.BytesIO(audio))
            segments, _ = self.whisper.transcribe(samples, beam_size=1, vad_filter=True,
                                                  language=self.language(samples))
            return " ".join(segment.text.strip() for segment in segments).strip()

    def language(self, samples) -> str | None:
        """The likeliest of the owner's languages (LUNASOCIA_STT_LANGUAGES), or None to let Whisper choose."""
        allowed = self.config.stt_languages
        if not allowed:
            return None
        if len(allowed) == 1:
            return allowed[0]
        _, _, scores = self.whisper.detect_language(samples, vad_filter=True)
        likely = dict(scores)
        return max(allowed, key=lambda code: likely.get(code, 0.0))

    def speak(self, text: str, voice: str | None = None) -> bytes:
        """A WAV file of her saying ``text`` in ``voice`` (a Kokoro voice id)."""
        import soundfile

        with self.lock, self.speaking.record():
            self._load_kokoro()
            words = speakable(text) or "Mm-hm."
            voice = voice or self.config.tts_voice
            samples, rate = self.kokoro.create(words, voice=voice, speed=1.0, lang=language(words, voice))
        buffer = io.BytesIO()
        soundfile.write(buffer, samples, rate, format="WAV")
        return buffer.getvalue()
