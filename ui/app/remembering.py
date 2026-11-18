"""Remembering by herself: after each text turn, a small classifier decides what is worth keeping.

Her chat model has a `remember` tool, but it seldom uses it, and sometimes says "I'll remember that" without
calling it. So, after each reply, two neurons (logistic regression, trained in notebooks/03_remember_training)
read an embedding of the turn:
- your message, with her line before it: a lasting fact about you? (her memory of you)
- her reply, with your message before it: something new and lasting about herself? (her bio)

Only on a yes does her chat model write the fact, as one short line from those words alone, or NONE when there is
nothing lasting after all (a mood, a reaction, advice). A fact naming someone, somewhere or a number the chat never
did is dropped: it was copied from the writer's examples. The embedding model
then compares it with what she has, and a fact she already knows is not saved again. The chat text itself is never
stored. A missed fact means you remind her; so the neurons speak only when fairly sure (0.5).

The embedding model is Qwen3-Embedding 0.6B as ONNX, run on the CPU by default (about 0.3 s a turn, 2.4 GB of RAM
and no GPU memory). The neurons come from the data volume (``remember/neurons.npz``, trained on your own chats too)
or else the ones shipped here (``remember_neurons.npz``, trained on written examples only). The neurons read the
message and the line before it embedded apart, side by side.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config import SociaConfig
from app.model_usage import ModelUsage, measure_cuda_context, measuring
from app.store import SociaStore
from app.voice import download

logger = logging.getLogger(__name__)

EMBED_REPO = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
EMBED_REVISION = "c25a394dd583836952667c12f008335071b3f43d"
# What the neurons read the message for: the same words as in training (notebooks/remember_classifier.py).
TASKS = {
    "you": "Decide whether the message states a lasting fact about the person who wrote it, worth remembering",
    "her": "Decide whether the message states a new, lasting fact about the woman who wrote it, about herself",
}
SHIPPED_NEURONS = Path(__file__).with_name("remember_neurons.npz")
# A written fact this close to a line she has (cosine of their embeddings) is the same fact.
SAME_FACT = 0.8
MAX_TOKENS = 512
PAD = "<|endoftext|>"
CONTEXT_START = "(start of the chat)"

# Her chat model writes the fact, or NONE. The examples' names are made up and never in a real chat: a fact
# naming someone or somewhere the chat did not is copied from here, and is dropped (see grounded).
ABOUT_THEM = """Rewrite the last chat message as one short, lasting fact about the person who wrote it, for a memory
list, or answer NONE.
- Start with "They" or "Their" (for a moment the two of you share: "You and they").
- Only what the Message itself says about them. The previous message only helps read a short answer; never take a
  fact, advice or idea from it.
- NONE when the message holds nothing lasting about them: small talk, a mood, a reaction, a question, what the
  message is about, a request, or something only true today.
- Keep names, places and dates exactly as written. No emojis. Write only the fact, or NONE.

Message: "ugh my boss Oskarine made us redo the whole deck"
Fact: Their boss is called Oskarine

Previous message: "Do you have siblings?"
Message: "two, both older"
Fact: They have two older siblings

Message: "remember that I can't stand loud music"
Fact: They can't stand loud music

Message: "lol you're so silly"
Fact: NONE

Message: "so tired today, long day"
Fact: NONE

Previous message: "Try warm milk before bed, it always works for me!"
Message: "hmm ok maybe"
Fact: NONE

Message: "let's talk properly from now on"
Fact: NONE

Message: "we finally watched Zalvador Nights together, our first movie night"
Fact: You and they had their first movie night watching Zalvador Nights

{before}Message: "{text}"
Fact:"""

ABOUT_HER = """Rewrite the last chat message as one short, lasting fact about the woman who wrote it, about herself,
for her bio, or answer NONE.
- Start with "She" or "Her". Only what this message says about her own life, past, likes or family.
- NONE for flirting, feelings in the moment, plans or ideas for the two of you, advice, questions, or anything
  about the other person.
- Keep names, places and dates exactly as written. No emojis. Write only the fact, or NONE.

Previous message: "favourite fruit?"
Message: "Jackfruit!! the sweet kind only"
Fact: Her favourite fruit is sweet jackfruit

Message: "I took swimming lessons every summer as a kid"
Fact: She took swimming lessons every summer as a child

Message: "my cousin Ilvana is basically my sister, we talk every day 💕"
Fact: Her cousin Ilvana is like a sister to her, and they talk every day

Message: "Haha you're too cute 😘 Let's plan a movie night soon! 🍿"
Fact: NONE

Message: "When cravings hit, drink some water first, it really helps!"
Fact: NONE

{before}Message: "{text}"
Fact:"""

# Words a fact may start with or use that are not names: they need not be in the chat.
NOT_NAMES = {"they", "their", "them", "she", "her", "you", "i", "none"}


def query(who: str, text: str) -> str:
    return f"Instruct: {TASKS[who]}\nQuery: {text}"


def fact_line(answer: str) -> str:
    """The writer's answer as one fact: its first line, without "Fact:", quotes or a list dash; "" for NONE."""
    line = answer.strip().splitlines()[0].strip() if answer.strip() else ""
    line = line.removeprefix("Fact:").strip().strip('"').removeprefix("- ").strip().rstrip(".")
    return "" if line.strip("*. ").upper() == "NONE" else line


def grounded(fact: str, *said: str) -> bool:
    """Whether every name, place and number in the fact was in what was said: "Their boss is Neha" from a chat
    that never named Neha was copied from an example, not learnt."""
    heard = " ".join(said).casefold()
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*|\d+", fact)
    for index, word in enumerate(words):
        if word.isdigit() or (word[0].isupper() and index > 0 and word.casefold() not in NOT_NAMES):
            if word.casefold().rstrip("'’s") not in heard:
                return False
    return True


def known_lines(companion: dict[str, Any], who: str) -> list[str]:
    """What she already has: her memory's lines (about you) or her bio's sentences (about her)."""
    if who == "you":
        return [line.removeprefix("- ").strip() for line in (companion.get("memory") or "").splitlines() if line.strip()]
    bio = str((companion.get("persona") or {}).get("about") or "")
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", bio) if part.strip()]


class Neuron:
    """One logistic-regression neuron: standardised features, a weight each, a bias, a sigmoid and a threshold."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, weight: np.ndarray, bias: float, threshold: float) -> None:
        self.mean, self.std, self.weight = mean.astype(np.float32), std.astype(np.float32), weight.astype(np.float32)
        self.bias, self.threshold = float(bias), float(threshold)

    def proba(self, features: np.ndarray) -> float:
        z = float(((features - self.mean) / self.std) @ self.weight + self.bias)
        return float(1 / (1 + np.exp(-z)))


def load_neurons(path: Path) -> dict[str, Neuron]:
    data = np.load(path)
    return {who: Neuron(data[f"{who}_mean"], data[f"{who}_std"], data[f"{who}_weight"], float(data[f"{who}_bias"]),
                        float(data[f"{who}_threshold"])) for who in ("you", "her")}


class Embedder:
    """Qwen3-Embedding 0.6B in ONNX Runtime: unit-length vectors from the last token, as it is meant to be read."""

    def __init__(self, config: SociaConfig) -> None:
        self.config = config
        self.device = config.remember_device
        self.lock = threading.Lock()
        self.session = self.tokenizer = None
        self.layers: list[str] = []
        self.where = ""
        self.usage = ModelUsage("Qwen3-Embedding 0.6B (ONNX)", "Decides what she remembers about you and herself",
                                lambda: (self.session is not None, self.where or "loads at start"))

    @property
    def folder(self) -> Path:
        return self.config.models_dir / "qwen3-embedding-0.6b"

    @property
    def model_file(self) -> str:
        return "model_fp16.onnx" if self.device == "cuda" else "model.onnx"

    def download(self) -> None:
        """The tokenizer and the model (2.4 GB for the CPU, 1.2 GB in fp16 for the GPU), once."""
        base = f"https://huggingface.co/{EMBED_REPO}/resolve/{EMBED_REVISION}"
        for name, remote in (("tokenizer.json", "tokenizer.json"), (self.model_file, f"onnx/{self.model_file}"),
                             (f"{self.model_file}_data", f"onnx/{self.model_file}_data")):
            if not (self.folder / name).exists():
                logger.info("Downloading %s for remembering", name)
                download(f"{base}/{remote}", self.folder / name)

    def load(self) -> None:
        if self.session is not None:
            return
        import onnxruntime
        from tokenizers import Tokenizer

        self.download()
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        # A few threads: it runs between her replies, beside her voice.
        options.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            measure_cuda_context()
            providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        self.tokenizer = Tokenizer.from_file(str(self.folder / "tokenizer.json"))
        self.session = onnxruntime.InferenceSession(str(self.folder / self.model_file), options, providers=providers)
        self.layers = sorted({i.name.split(".")[1] for i in self.session.get_inputs()
                              if i.name.startswith("past_key_values")}, key=int)
        self.where = "GPU (ONNX, CUDA)" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "CPU (ONNX)"

    def preload(self) -> None:
        """Download and load it at start, beside her voice, and run it once."""
        try:
            with measuring(self.usage):
                self.vectors(["hello"])
            logger.info("The remembering model is ready, on the %s", self.where)
        except Exception:  # noqa: BLE001 - loaded on first use instead
            logger.exception("Could not load the remembering model at start")

    def vectors(self, texts: list[str]) -> np.ndarray:
        with self.lock, self.usage.record():
            self.load()
            pad = self.tokenizer.token_to_id(PAD)
            encodings = [self.tokenizer.encode(text).ids[-MAX_TOKENS:] for text in texts]
            width = max(map(len, encodings))
            ids = np.full((len(texts), width), pad, dtype=np.int64)
            mask = np.zeros((len(texts), width), dtype=np.int64)
            for row, encoding in enumerate(encodings):  # padded on the left: the last token is each text's own
                ids[row, width - len(encoding):] = encoding
                mask[row, width - len(encoding):] = 1
            feeds = {"input_ids": ids, "attention_mask": mask, "position_ids": np.clip(np.cumsum(mask, 1) - 1, 0, None)}
            # Exported for generation: an empty cache for each layer.
            empty = np.zeros((len(texts), 8, 0, 128), dtype=np.float16 if self.device == "cuda" else np.float32)
            for layer in self.layers:
                feeds[f"past_key_values.{layer}.key"] = empty
                feeds[f"past_key_values.{layer}.value"] = empty
            hidden = self.session.run(["last_hidden_state"], feeds)[0][:, -1].astype(np.float32)
        return hidden / np.linalg.norm(hidden, axis=1, keepdims=True)


class Rememberer:
    """After a text turn: the neurons decide, her chat model writes the fact, and it is saved unless she has it."""

    def __init__(self, config: SociaConfig, store: SociaStore, complete, embedder: Embedder | None = None) -> None:
        self.config, self.store, self.complete = config, store, complete
        self.embedder = embedder or Embedder(config)
        own = config.data_dir / "remember" / "neurons.npz"
        self.neurons_from = own if own.exists() else SHIPPED_NEURONS
        self.neurons = load_neurons(self.neurons_from)
        logger.info("Remembering with the neurons in %s", self.neurons_from)

    def preload(self) -> None:
        self.embedder.preload()

    async def write(self, who: str, before: str, text: str) -> str:
        quoted = f'Previous message: "{before[:200]}"\n' if before else ""
        prompt = (ABOUT_THEM if who == "you" else ABOUT_HER).format(before=quoted, text=text[:800])
        return fact_line(await self.complete(prompt))

    async def after_turn(self, companion_id: int, before: str, message: str, reply: str) -> list[dict[str, Any]]:
        """What this turn taught her: each fact saved, about them ("them") or herself ("me"), with how sure the
        neuron was. ``before`` is her line before your message."""
        sides = [("you", "them", before, message), ("her", "me", message, reply)]
        sides = [side for side in sides if side[3].strip()]
        if not sides:
            return []
        texts = [query(who, text) for who, _, _, text in sides] + [prior or CONTEXT_START for _, _, prior, _ in sides]
        vectors = await asyncio.to_thread(self.embedder.vectors, texts)
        saved = []
        for index, (who, about, prior, text) in enumerate(sides):
            features = np.concatenate([vectors[index], vectors[len(sides) + index]])
            sure = self.neurons[who].proba(features)
            if sure < self.neurons[who].threshold:
                continue
            fact = await self.write(who, prior, text)
            if not fact:
                continue
            # About you, from your words (her line before only helps read a short answer); about her, from hers.
            if not grounded(fact, text, prior if who == "you" else ""):
                logger.info("Not in the chat, dropped: %s", fact)
                continue
            companion = await asyncio.to_thread(self.store.companion, companion_id)
            if companion is None:
                return saved
            lines = known_lines(companion, who)
            if lines:
                compare = await asyncio.to_thread(self.embedder.vectors, [fact, *lines])
                closest = float((compare[1:] @ compare[0]).max())
                if closest >= SAME_FACT:
                    logger.info("Already known (%.2f): %s", closest, fact)
                    continue
            if who == "you":
                added = await asyncio.to_thread(self.store.remember, companion_id, fact, SociaConfig.MEMORY_CHARS)
            else:
                added = bool(await asyncio.to_thread(self.store.add_to_bio, companion_id, [fact], SociaConfig.ABOUT_CHARS))
            if added:
                logger.info("Remembered about %s (%.2f): %s", about, sure, fact)
                saved.append({"about": about, "fact": fact, "sure": round(sure, 2)})
        return saved
