"""Part B of notebook 03: deciding *when* to remember with a small classifier, not by training her chat model.

1. A small embedding model (Qwen3-Embedding-0.6B, about 1.2 GB, run here with transformers) turns a message, with
   the line before it, into 1,024 numbers. (Ollama has the same model, but its engine hung loading it on 17 Nov.)
2. Two tiny classifiers (logistic regression: one "neuron" each) read those numbers:
   - on your messages: is there a lasting fact about you? (them / none)
   - on her replies: did she say something new and lasting about herself? (me / none)
3. Only when one says yes, her chat model writes the fact in one line, seeing what she already knows, and may
   still answer NONE (a fact she already has, or her bio again).

Trained on labelled lines Claude wrote (results/remember/generated.jsonl) and tested on your labelled chats
(results/remember/labels.csv), which it never trains on.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import SociaConfig
from app.ollama import Ollama
from app.store import SociaStore

GENERATED = Path(__file__).parent / "results" / "remember" / "generated.jsonl"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TASKS = {
    "you": "Decide whether the message states a lasting fact about the person who wrote it, worth remembering",
    "her": "Decide whether the message states a new, lasting fact about the woman who wrote it, about herself",
}
POSITIVE = {"you": "them", "her": "me"}


# ------------------------------------------------------------------------------------------------- data

def load_generated(path: Path = GENERATED) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def real_rows(store: SociaStore, labels: list[dict[str, Any]], companion_id: int, none_per_chat: int = 10,
              seed: int = 7) -> list[dict[str, Any]]:
    """Your labelled messages (and hers), each with the line before it from the other side; plus some of your
    messages that were not even candidates, as "none"."""
    by_id = {int(row["message_id"]): row for row in labels}
    rng = random.Random(seed)
    out = []
    for chat in store.chats(companion_id):
        rows = store.recent(chat["id"], 100000)
        unlisted = []
        for index, row in enumerate(rows):
            text = " ".join((row["content"] or "").split())
            if not text or row.get("image_id"):
                continue
            before = next((" ".join((r["content"] or "").split()) for r in reversed(rows[:index])
                           if r["role"] != row["role"]), "") if index else ""
            who = "you" if row["role"] == "user" else "her"
            label = by_id.get(row["id"])
            item = {"who": who, "before": before[:300], "text": text[:400], "ref": f"message {row['id']}"}
            if label and label["label"] in ("them", "me", "none"):
                out.append({**item, "label": label["label"], "fact": label["fact"]})
            elif not label and who == "you":
                unlisted.append({**item, "label": "none", "fact": ""})
        rng.shuffle(unlisted)
        out += unlisted[:none_per_chat]
    return out


def query(row: dict[str, Any]) -> str:
    """What the embedding model reads: the task, then the line before (if any) and the message."""
    before = f"Previous message: {row['before']}\n" if row.get("before") else ""
    return f"Instruct: {TASKS[row['who']]}\nQuery: {before}Message: {row['text']}"


# ----------------------------------------------------------------------------------------------- embed

_embedder: dict[str, Any] = {}


def embedder():
    """The embedding model and its tokenizer, loaded once (on the GPU if there is room, else the CPU)."""
    if not _embedder:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 2**30 else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side="left")
        model = AutoModel.from_pretrained(EMBED_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32)
        _embedder.update(tokenizer=tokenizer, model=model.to(device).eval(), device=device)
    return _embedder["tokenizer"], _embedder["model"], _embedder["device"]


def embed(config: SociaConfig | None, rows: list[dict[str, Any]], batch: int = 32, plain: bool = False):
    """Each row as a unit-length vector: the embedding model's last token, as Qwen3-Embedding is meant to be read.
    ``plain``: the text alone, without the task (for comparing facts)."""
    import torch

    tokenizer, model, device = embedder()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            texts = [r["text"] if plain else query(r) for r in rows[start:start + batch]]
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            hidden = model(**tokens).last_hidden_state[:, -1]
            vectors.append(torch.nn.functional.normalize(hidden.float(), dim=1).cpu())
    return torch.cat(vectors)


# ------------------------------------------------------------------------------------------- classifier

class Neuron:
    """Logistic regression: one weight per number of the embedding, a bias, and a sigmoid. Trained with the
    classes balanced, so a rare "yes" counts as much as the common "no"."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.mean = self.std = self.weight = self.bias = None

    def fit(self, x, y, steps: int = 400, lr: float = 0.02, decay: float = 1e-2, seed: int = 1) -> "Neuron":
        import torch

        torch.manual_seed(seed)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mean) / self.std
        weight = torch.zeros(x.shape[1], requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        positives = y.sum().clamp(min=1)
        pos_weight = (len(y) - positives) / positives
        optimizer = torch.optim.Adam([weight, bias], lr=lr, weight_decay=decay)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ weight + bias, y.float(),
                                                                        pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
        self.weight, self.bias = weight.detach(), bias.detach()
        return self

    def proba(self, x):
        import torch

        return torch.sigmoid(((x - self.mean) / self.std) @ self.weight + self.bias)

    def predict(self, x):
        return self.proba(x) >= self.threshold

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "weight": self.weight, "bias": self.bias, "threshold": self.threshold}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Neuron":
        neuron = cls(state["threshold"])
        neuron.mean, neuron.std, neuron.weight, neuron.bias = state["mean"], state["std"], state["weight"], state["bias"]
        return neuron


def labels_of(rows: list[dict[str, Any]], who: str):
    import torch

    return torch.tensor([row["label"] == POSITIVE[who] for row in rows])


def counts(truth, guess) -> dict[str, Any]:
    tp = int((truth & guess).sum()); fp = int((~truth & guess).sum())
    fn = int((truth & ~guess).sum()); tn = int((~truth & ~guess).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"facts caught": f"{tp}/{tp + fn}", "false alarms": f"{fp}/{fp + tn}", "precision": round(precision, 2),
            "recall": round(recall, 2), "right": f"{tp + tn}/{len(truth)}"}


def cross_validate(x, y, folds: int = 5, seed: int = 3, threshold: float = 0.5):
    """Each line scored by a classifier that never saw it: the generated set split five ways."""
    import torch

    order = list(range(len(y)))
    random.Random(seed).shuffle(order)
    proba = torch.zeros(len(y))
    for fold in range(folds):
        held = order[fold::folds]
        kept = [i for i in order if i not in set(held)]
        neuron = Neuron(threshold).fit(x[kept], y[kept])
        proba[held] = neuron.proba(x[held])
    return proba


def thresholds(truth, proba, values=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> list[dict[str, Any]]:
    return [{"threshold": value, **counts(truth, proba >= value)} for value in values]


def mistakes(rows: list[dict[str, Any]], truth, proba, threshold: float, limit: int = 15) -> list[dict[str, Any]]:
    wrong = [i for i in range(len(rows)) if bool(truth[i]) != bool(proba[i] >= threshold)]
    wrong.sort(key=lambda i: -abs(float(proba[i]) - threshold))
    return [{"p": round(float(proba[i]), 2), "should be": rows[i]["label"], "before": rows[i].get("before", "")[:40],
             "text": rows[i]["text"][:90]} for i in wrong[:limit]]


# ------------------------------------------------------------------------------------------ fact writer

# The neuron has already decided there is a fact: the writer only words it, from the message alone (shown her
# memory or bio, the 8B answered NONE to plain facts and copied lines from them instead). Whether she already
# knows it is checked after, by the embedding model (``already_known``).
ABOUT_THEM = """Rewrite the last chat message as one short fact about the person who wrote it, for a memory list.
Start with "They" or "Their" (for a moment the two of you share: "You and they"). Keep names, places and dates
exactly as written. No emojis. Write only the fact.

Message: "ugh my boss Neha made us redo the whole deck"
Fact: Their boss is called Neha

Previous message: "Do you have siblings?"
Message: "two, both older"
Fact: They have two older siblings

Message: "remember that I can't stand loud music"
Fact: They can't stand loud music

Message: "we finally watched Dil Chahta Hai together tonight, our first movie night"
Fact: You and they had your first movie night watching Dil Chahta Hai

{before}Message: "{text}"
Fact:"""

ABOUT_HER = """Rewrite the last chat message as one short fact about the woman who wrote it, about herself, for her
bio. Start with "She" or "Her". Use only what this message says. Keep names, places and dates exactly as written.
No emojis. Write only the fact.

Previous message: "favourite fruit?"
Message: "Mangoes!! alphonso only 🥭"
Fact: Her favourite fruit is the Alphonso mango

Message: "I took swimming lessons every summer as a kid"
Fact: She took swimming lessons every summer as a child

Message: "my cousin Ria is basically my sister, we talk every day 💕"
Fact: Her cousin Ria is like a sister to her, and they talk every day

{before}Message: "{text}"
Fact:"""


async def write_fact(ollama: Ollama, row: dict[str, Any], her: dict[str, Any] | None = None, name: str = "") -> str:
    """The fact in one line, worded by her chat model from the message (and the line before it) alone."""
    before = f'Previous message: "{row["before"][:200]}"\n' if row.get("before") else ""
    prompt = (ABOUT_THEM if row["who"] == "you" else ABOUT_HER).format(before=before, text=row["text"])
    answer = (await ollama.complete(prompt)).strip()
    line = answer.splitlines()[0].strip() if answer else ""
    return line.removeprefix("Fact:").strip().strip('"').removeprefix("- ").strip()


def embed_texts(texts: list[str]):
    """Plain sentences (no task), as Qwen3-Embedding embeds documents: for comparing facts with each other."""
    rows = [{"who": "you", "text": text} for text in texts]
    return embed(None, rows, plain=True)


def known_lines(her: dict[str, Any], who: str) -> list[str]:
    """What she already has: her memory's lines (about them) or her bio's sentences (about her)."""
    import re

    if who == "you":
        return [line.removeprefix("- ").strip() for line in (her.get("memory") or "").splitlines() if line.strip()]
    bio = str((her.get("persona") or {}).get("about") or "")
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", bio) if part.strip()]


def already_known(fact: str, lines: list[str], threshold: float = 0.8) -> tuple[bool, float, str]:
    """Whether a line she has says the same: the closest line and how close (cosine similarity, 1 = the same)."""
    if not lines:
        return False, 0.0, ""
    vectors = embed_texts([fact, *lines])
    similarity = vectors[1:] @ vectors[0]
    best = int(similarity.argmax())
    return float(similarity[best]) >= threshold, round(float(similarity[best]), 3), lines[best]


# ------------------------------------------------------------------------------------------- pipeline

class Rememberer:
    """The whole idea: both classifiers, then the fact writer, for one turn (your message and her reply)."""

    def __init__(self, config: SociaConfig, you: Neuron, her_neuron: Neuron) -> None:
        self.config, self.neurons = config, {"you": you, "her": her_neuron}

    def decide(self, row: dict[str, Any]) -> tuple[bool, float]:
        p = float(self.neurons[row["who"]].proba(embed(self.config, [row]))[0])
        return p >= self.neurons[row["who"]].threshold, p

    async def turn(self, ollama: Ollama, her: dict[str, Any], name: str, before: str, message: str,
                   reply: str, dedupe: float = 0.8) -> dict[str, Any]:
        """What gets remembered after one turn: about them from your message, about her from her reply. Each side:
        the neuron's probability, the fact written, and the line she already had that said the same (if any)."""
        started = time.perf_counter()
        out: dict[str, Any] = {"them": "", "me": ""}
        for key, row in (("them", {"who": "you", "before": before, "text": message}),
                         ("me", {"who": "her", "before": message, "text": reply})):
            out[f"wrote_{key}"], out[f"known_{key}"], out[f"p_{key}"] = "", "", 0.0
            if not row["text"].strip():
                continue  # no words (she only made a tool call): nothing to read
            yes, out[f"p_{key}"] = self.decide(row)
            if yes:
                fact = await write_fact(ollama, row)
                known, similarity, line = already_known(fact, known_lines(her, row["who"]), dedupe)
                out[f"wrote_{key}"] = fact
                if known:
                    out[f"known_{key}"] = f"{similarity}: {line}"
                else:
                    out[key] = fact
        out["seconds"] = round(time.perf_counter() - started, 2)
        out["got"] = "them" if out["them"] else "me" if out["me"] else "none"
        return out

    def save(self, path: Path) -> None:
        import torch

        torch.save({"embed_model": EMBED_MODEL, "tasks": TASKS,
                    "you": self.neurons["you"].state(), "her": self.neurons["her"].state()}, path)
