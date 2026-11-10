"""Luna's ears and voice, on the CPU so the GPU stays free for her chat and image models.

Speech to text: faster-whisper. Text to speech: Kokoro (82M, ONNX). Both models are
downloaded to the data volume the first time they are used.
"""

from __future__ import annotations

import io
import os
import re
import threading
import urllib.request
from pathlib import Path

from app.config import SociaConfig

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
# Half the CPU: enough to transcribe a voice message in about real time, leaving the rest to the apps.
WHISPER_THREADS = max(4, (os.cpu_count() or 8) // 2)
# Emoji and other pictographs, which a voice would read out by name.
PICTOGRAPHS = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")


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

    def transcribe(self, audio: bytes) -> str:
        with self.lock:
            if self.whisper is None:
                from faster_whisper import WhisperModel

                self.whisper = WhisperModel(self.config.stt_model, device="cpu", compute_type="int8",
                                            cpu_threads=WHISPER_THREADS,
                                            download_root=str(self.config.models_dir / "whisper"))
            segments, _ = self.whisper.transcribe(io.BytesIO(audio), beam_size=1, vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()

    def speak(self, text: str) -> bytes:
        """A WAV file of Luna saying ``text``."""
        import soundfile

        with self.lock:
            if self.kokoro is None:
                from kokoro_onnx import Kokoro

                folder = self.config.models_dir / "kokoro"
                for name in KOKORO_FILES:
                    if not (folder / name).exists():
                        download(f"{KOKORO_RELEASE}/{name}", folder / name)
                self.kokoro = Kokoro(str(folder / KOKORO_FILES[0]), str(folder / KOKORO_FILES[1]))
            samples, rate = self.kokoro.create(speakable(text) or "Mm-hm.", voice=self.config.tts_voice, speed=1.0, lang="en-us")
        buffer = io.BytesIO()
        soundfile.write(buffer, samples, rate, format="WAV")
        return buffer.getvalue()
