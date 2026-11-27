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

The embedding model is Qwen3-Embedding 0.6B in Ollama (``qwen3-embedding:0.6b``, the one Lexia's search uses: 1 GB
of the GPU while loaded, a few hundredths of a second a turn). The neurons come from the data volume (``remember/neurons.npz``, trained on your own chats too)
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

import httpx
import numpy as np

from app.config import SociaConfig
from app.ollama import APP_HEADER
from app.store import SociaStore
from app.voice import download

logger = logging.getLogger(__name__)

# The embedding model's tokenizer, for counting the tokens of her memory she reads (the model is in Ollama).
TOKENIZER_URL = ("https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/"
                 "c25a394dd583836952667c12f008335071b3f43d/tokenizer.json")
# What the neurons read the message for: the same words as in training (notebooks/remember_classifier.py).
TASKS = {
    "you": "Decide whether the message states a lasting fact about the person who wrote it, worth remembering",
    "her": "Decide whether the message states a new, lasting fact about the woman who wrote it, about herself",
}
SHIPPED_NEURONS = Path(__file__).with_name("remember_neurons.npz")
# A written fact this close to a line she has (cosine of their embeddings) is the same fact. Closeness alone is
# blunt ("a cat named Mini" and "a dog named Mini" are 0.85 apart; "works as a nurse" and "her job is nursing"
# only 0.68), so a near copy is the same at once, and a fact merely close to some lines is shown to her chat model
# beside them, which says whether it is the same fact. Without the chat model, SAME_FACT decides.
SAME_FACT = 0.8
SURELY_SAME = 0.95
MAYBE_SAME = 0.65
JUDGED_LINES = 3
SAME_PROMPT = """Do these two lines state the same fact, about the same person or thing? Different details (a cat
and a dog, pizza and biryani, two different names) make them different facts; the same fact in other words is the
same. Answer YES or NO only.
A: {new}
B: {known}"""
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
    """Qwen3-Embedding 0.6B in Ollama: unit-length vectors from the last token, as it is meant to be read. Its
    tokenizer, here, counts tokens."""

    def __init__(self, config: SociaConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.config = config
        self.model = config.embed_model
        self.lock = threading.Lock()
        self.tokenizer = None
        self.client = httpx.Client(base_url=config.ollama_url, transport=transport, headers=APP_HEADER,
                                   timeout=httpx.Timeout(120.0, connect=10.0))

    @property
    def folder(self) -> Path:
        return self.config.models_dir / "qwen3-embedding-0.6b"

    def load(self) -> None:
        """The tokenizer (11 MB), downloaded once. The ONNX copy of the model Socia ran before is deleted."""
        with self.lock:
            if self.tokenizer is not None:
                return
            from tokenizers import Tokenizer

            for old in self.folder.glob("model*.onnx*"):
                logger.info("Deleting %s: the embedding model is Ollama's now", old.name)
                old.unlink(missing_ok=True)
            path = self.folder / "tokenizer.json"
            if not path.exists():
                download(TOKENIZER_URL, path)
            self.tokenizer = Tokenizer.from_file(str(path))

    def preload(self) -> None:
        try:
            self.load()
        except Exception:  # noqa: BLE001 - loaded on first use instead
            logger.exception("Could not load the embedding model's tokenizer at start")

    def count(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text).ids)

    def vectors(self, texts: list[str]) -> np.ndarray:
        response = self.client.post("/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        vectors = np.array(response.json()["embeddings"], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


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

    async def known_vectors(self, companion: dict[str, Any], who: str) -> np.ndarray:
        """What she has, embedded: her memory's lines reuse the vectors recall keeps for them, and the rest
        (new lines, her bio's sentences) are embedded now."""
        lines = known_lines(companion, who)
        if not lines:
            return np.zeros((0, 0), dtype=np.float32)
        stored: dict[str, np.ndarray] = {}
        if who == "you" and hasattr(self.store, "memory_lines"):
            for row in await asyncio.to_thread(self.store.memory_lines, companion["id"]):
                if row.get("embedded_by") == self.embedder.model and row.get("vector"):
                    stored[row["line"]] = np.frombuffer(row["vector"], dtype=np.float32)
        missing = [line for line in lines if line not in stored]
        if missing:
            for line, vector in zip(missing, await asyncio.to_thread(self.embedder.vectors, missing)):
                stored[line] = vector
        return np.stack([stored[line] for line in lines])

    async def new_facts(self, companion: dict[str, Any], who: str, facts: list[str]) -> list[str]:
        """The facts she does not have yet: each compared by meaning (cosine of embeddings) with her memory
        (``who`` "you") or her bio ("her"), and with the facts before it in ``facts``. Every way she remembers
        goes through here. If the embedding model cannot be reached, the facts are kept rather than lost."""
        facts = [fact for fact in facts if fact.strip()]
        if not facts:
            return []
        try:
            known = await self.known_vectors(companion, who)
            candidates = await asyncio.to_thread(self.embedder.vectors, [fact.removeprefix("- ").strip() for fact in facts])
        except Exception:  # noqa: BLE001 - a duplicate is better than a lost fact
            logger.exception("Could not compare new facts with what she has; keeping them")
            return facts
        pool = list(known) if known.size else []
        texts = known_lines(companion, who) if known.size else []
        kept = []
        for fact, vector in zip(facts, candidates):
            same = await self.same_as_known(fact, vector, pool, texts)
            if same:
                logger.info("Already known (%s): %s", same, fact)
                continue
            kept.append(fact)
            pool.append(vector)
            texts.append(fact.removeprefix("- ").strip())
        return kept

    async def same_as_known(self, fact: str, vector: np.ndarray, pool: list[np.ndarray], texts: list[str]) -> str:
        """Why ``fact`` is one she has (a near copy, or the line her chat model said it repeats), or "" if new."""
        if not pool:
            return ""
        scores = np.stack(pool) @ vector
        order = np.argsort(-scores)
        best = float(scores[order[0]])
        if best >= SURELY_SAME:
            return f"{best:.2f}"
        close = [int(index) for index in order[:JUDGED_LINES] if scores[index] >= MAYBE_SAME]
        if not close:
            return ""
        if self.complete is None:
            return f"{best:.2f}" if best >= SAME_FACT else ""
        try:
            for index in close:
                answer = await self.complete(SAME_PROMPT.format(new=fact.removeprefix("- ").strip(), known=texts[index]))
                if answer.strip().upper().startswith("YES"):
                    return f"{float(scores[index]):.2f}, same as {texts[index]!r}"
        except Exception:  # noqa: BLE001 - closeness alone decides then
            logger.exception("Could not ask whether a fact is one she has")
            return f"{best:.2f}" if best >= SAME_FACT else ""
        return ""

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
            if not await self.new_facts(companion, who, [fact]):
                continue
            if who == "you":
                added = await asyncio.to_thread(self.store.remember, companion_id, fact, SociaConfig.MEMORY_CHARS)
            else:
                added = bool(await asyncio.to_thread(self.store.add_to_bio, companion_id, [fact], SociaConfig.ABOUT_CHARS))
            if added:
                logger.info("Remembered about %s (%.2f): %s", about, sure, fact)
                saved.append({"about": about, "fact": fact, "sure": round(sure, 2)})
        return saved
