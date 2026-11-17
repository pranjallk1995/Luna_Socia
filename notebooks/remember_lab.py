"""Helpers for notebook 03: teaching her chat model when to call `remember`, by itself.

The notebook shows each step; the code that is long or fiddly lives here:
- labels: which of your messages (and hers) hold a lasting fact, in a CSV you edit;
- synthetic examples: made-up messages with a known right answer, her replies written by her own model;
- examples: each one exactly as Socia sends a turn (her prompt, her tools, the chat so far), and the reply
  she should give, with the remember call she should make;
- training: QLoRA on the Hugging Face copy of her chat model, the loss only on her reply;
- export: the adapter as GGUF, added to Ollama on top of her chat model;
- evaluation: the same checks before and after.

Nothing here writes to her database.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import tarfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.chat import model_messages, without_written_pictures, written_reactions, written_remembers
from app.config import SociaConfig
from app.ollama import Ollama
from app.persona import TOOLS, system_prompt
from app.store import SociaStore

RESULTS = Path(__file__).parent / "results" / "remember"
TRAINING = Path("/training")
LABELS = RESULTS / "labels.csv"
SYNTHETIC = RESULTS / "synthetic.jsonl"
LABEL_COLUMNS = ["message_id", "chat_id", "who", "text", "hint", "label", "fact"]
KINDS = ("them", "me", "none")

# Your messages that may hold a lasting fact about you: shown to you to label. The rest count as small talk.
THEIR_HINT = re.compile(
    r"\b(my|mine|i'?m|i am|i'?ve|i have|i work|i live|i love|i like|i hate|i don'?t like|i can'?t stand|i prefer|"
    r"remember|don'?t forget|favou?rite|birthday|anniversary|sister|brother|mom|mum|dad|mother|father|friend|"
    r"cat|cats|dog|pet|job|office|boss|college|school|moving|trip|flight|tomorrow|next week|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\b", re.IGNORECASE)
# Her messages that may say something lasting about herself (her favourites, her past, her family).
HER_HINT = re.compile(
    r"\b(my favou?rite|i grew up|i was born|when i was (a )?(kid|little|young)|i used to|my (mom|mum|dad|sister|"
    r"brother|best friend|family|grandma|nani|dadi)|i'?ve always|i studied|i went to|my first|i'?m from|"
    r"i can'?t stand|i hate|i'?m allergic|my dream)\b", re.IGNORECASE)
# Short messages that are only reactions, never facts.
TOO_SHORT = 12


# --------------------------------------------------------------------------------------------- labels

def candidates(store: SociaStore, companion_id: int) -> list[dict[str, Any]]:
    """Every message in her chats that may hold a lasting fact: yours about you, hers about her."""
    found = []
    for chat in store.chats(companion_id):
        for row in store.recent(chat["id"], 100000):
            text = " ".join((row["content"] or "").split())
            if len(text) < TOO_SHORT:
                continue
            who = "you" if row["role"] == "user" else "her"
            hint = THEIR_HINT if who == "you" else HER_HINT
            match = hint.search(text)
            if match:
                found.append({"message_id": row["id"], "chat_id": chat["id"], "who": who, "text": text[:400],
                              "hint": match.group(0).lower(), "label": "", "fact": ""})
    return sorted(found, key=lambda row: row["message_id"])


def write_labels(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Add new candidates to the labels file, keeping every label already in it; (kept, added)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    old = read_labels() if LABELS.exists() else []
    known = {int(row["message_id"]) for row in old}
    new = [row for row in rows if int(row["message_id"]) not in known]
    merged = sorted([*old, *new], key=lambda row: int(row["message_id"]))
    with LABELS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in LABEL_COLUMNS} for row in merged)
    return len(old), len(new)


def read_labels() -> list[dict[str, Any]]:
    with LABELS.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["label"] = (row.get("label") or "").strip().lower()
        row["fact"] = " ".join((row.get("fact") or "").split()).removeprefix("- ")
    return rows


def label_problems(rows: list[dict[str, Any]]) -> list[str]:
    """What is wrong in the labels: an unknown label, a fact without a label or a label without a fact."""
    problems = []
    for row in rows:
        where = f"message {row['message_id']}"
        if row["label"] and row["label"] not in KINDS:
            problems.append(f"{where}: label must be them, me, none or blank, not {row['label']!r}")
        if row["label"] in ("them", "me") and not row["fact"]:
            problems.append(f"{where}: a {row['label']} label needs the fact")
        if row["label"] == "me" and row["who"] != "her":
            problems.append(f"{where}: 'me' is for her messages about herself")
        if row["label"] == "them" and row["who"] != "you":
            problems.append(f"{where}: 'them' is for your messages about you")
    return problems


# ------------------------------------------------------------------------------------------- examples

@dataclass
class Example:
    """One turn: what Socia sends her model, and the reply she should give."""

    source: str                     # "chat" (your real chats) or "synthetic"
    kind: str                       # them, me or none
    messages: list[dict[str, Any]]  # system prompt, then the chat so far, ending with your message
    reply: str                      # her words
    facts: list[str] = field(default_factory=list)  # what she should remember
    ref: str = ""                   # where it came from

    @property
    def target(self) -> dict[str, Any]:
        calls = [{"type": "function", "function": {"name": "remember", "arguments": {
            "fact": fact, **({"about": "me"} if self.kind == "me" else {})}}} for fact in self.facts]
        return {"role": "assistant", "content": self.reply, **({"tool_calls": calls} if calls else {})}

    def to_json(self) -> dict[str, Any]:
        return {"source": self.source, "kind": self.kind, "messages": self.messages, "reply": self.reply,
                "facts": self.facts, "ref": self.ref}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Example":
        return cls(**data)


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2} - {"they", "their", "them", "she",
                                                                                   "her", "the", "and", "with"}


def without_fact(text: str, facts: list[str], sentences: bool = False) -> str:
    """Her memory (or bio) without lines that already say one of these facts, so the example teaches
    her to remember something new, as it was when you told her."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text) if sentences else text.splitlines()
    kept = []
    for part in parts:
        line = words(part)
        if line and any(len(line & words(fact)) >= max(2, len(words(fact)) // 2) for fact in facts):
            continue
        kept.append(part)
    return ("\n" if not sentences else " ").join(kept).strip()


def her_reply(text: str) -> str:
    """Her words as the model should write them: no tool calls she typed out, no pictures in brackets."""
    _, text = written_remembers(text)
    _, text = written_reactions(text)
    return without_written_pictures(text).strip()


class Builder:
    """Turns chats and labels into examples, each exactly as Socia would have sent that turn."""

    def __init__(self, config: SociaConfig, store: SociaStore, her: dict[str, Any], name: str, looks: str) -> None:
        self.config, self.store, self.her, self.name, self.looks = config, store, her, name, looks
        self.owner_about = store.owner_profile().get("about") or ""

    def prompt(self, kind: str, facts: list[str], now: datetime) -> str:
        her = dict(self.her)
        if kind == "them" and facts:
            her["memory"] = without_fact(her.get("memory") or "", facts)
        if kind == "me" and facts:
            persona = dict(her.get("persona") or {})
            persona["about"] = without_fact(str(persona.get("about") or ""), facts, sentences=True)
            her["persona"] = persona
        return system_prompt(her, self.name, now, self.looks, "", self.owner_about)

    def turn(self, rows: list[dict[str, Any]], kind: str, facts: list[str], now: datetime) -> list[dict[str, Any]]:
        history = model_messages(rows[-SociaConfig.HISTORY_MESSAGES:])
        return [{"role": "system", "content": self.prompt(kind, facts, now)}, *history]

    def from_labels(self, labels: list[dict[str, Any]], none_per_chat: int = 12, seed: int = 7) -> list[Example]:
        """Examples from your chats: each labelled message, and some unlabelled ones of yours as small talk."""
        by_id = {int(row["message_id"]): row for row in labels}
        rng = random.Random(seed)
        examples = []
        for chat in self.store.chats(self.her["id"]):
            rows = self.store.recent(chat["id"], 100000)
            small_talk = []
            for index, row in enumerate(rows):
                label = by_id.get(row["id"])
                if row.get("image_id"):
                    continue  # photo turns go to the vision model, not this one
                if row["role"] == "user":
                    reply = next((r for r in rows[index + 1:index + 2] if r["role"] == "luna"), None)
                    if reply is None or reply.get("image_id") or not her_reply(reply["content"] or ""):
                        continue
                    when = row["created_at"]
                    if label and label["label"] == "them":
                        examples.append(Example("chat", "them", self.turn(rows[:index + 1], "them", [label["fact"]], when),
                                                her_reply(reply["content"]), [label["fact"]], f"message {row['id']}"))
                    elif (label and label["label"] == "none") or (not label and len(row["content"] or "") >= TOO_SHORT):
                        small_talk.append(Example("chat", "none", self.turn(rows[:index + 1], "none", [], when),
                                                  her_reply(reply["content"]), [], f"message {row['id']}"))
                elif label and label["label"] == "me" and index > 0 and rows[index - 1]["role"] == "user":
                    examples.append(Example("chat", "me", self.turn(rows[:index], "me", [label["fact"]], row["created_at"]),
                                            her_reply(row["content"] or ""), [label["fact"]], f"message {row['id']}"))
            rng.shuffle(small_talk)
            examples += small_talk[:none_per_chat]
        return examples


# ---------------------------------------------------------------------------------------- synthetic

PETS = [("cat", "Biscuit"), ("dog", "Bruno"), ("cat", "Mochi"), ("dog", "Simba"), ("parrot", "Mithu"),
        ("cat", "Pepper"), ("dog", "Coco"), ("rabbit", "Bun")]
PEOPLE = [("sister", "Asha"), ("brother", "Rohan"), ("best friend", "Kabir"), ("mom", "Meera"), ("cousin", "Tanya"),
          ("dad", "Vikram"), ("roommate", "Arjun"), ("friend", "Neha")]
CITIES = ["Pune", "Bangalore", "Delhi", "Goa", "Jaipur", "Hyderabad", "Kolkata", "Chennai", "Shimla", "Kochi"]
JOBS = [("designer", "a small studio"), ("data analyst", "a bank"), ("teacher", "a school"),
        ("software engineer", "a startup"), ("doctor", "a hospital"), ("photographer", "a magazine")]
FOODS = ["biryani", "sushi", "dosa", "pasta", "momos", "chole bhature", "ramen", "pav bhaji", "tacos", "paneer tikka"]
DISLIKES = ["olives", "mushrooms", "karela", "coriander", "pineapple on pizza", "raw onions", "cold coffee"]
HOBBIES = ["playing guitar", "cycling", "chess", "painting", "running", "photography", "cooking", "gaming"]
DATES = ["12 March", "3 June", "21 August", "9 October", "28 December", "15 January", "5 May", "30 September"]
SHOWS = ["The Office", "Dark", "Friends", "Breaking Bad", "Kota Factory", "Panchayat"]
OPENERS = ["", "", "", "btw ", "oh and ", "fun fact: ", "ok so ", "guess what, ", "hey, ", "you know what? "]
ENDINGS = ["", "", "", " 😊", " haha", " 🙈", "!", " lol", " ❤️"]


def synthetic_turns(seed: int = 11, them: int = 90, none: int = 70, me: int = 40) -> list[dict[str, Any]]:
    """Made-up messages from you with a known right answer: a lasting fact about you, small talk, or a question
    that makes her say something about herself (her fact is read from her reply later)."""
    rng = random.Random(seed)

    def pick(values):
        return rng.choice(values)

    def casual(text: str) -> str:
        text = pick(OPENERS) + text + pick(ENDINGS)
        return text.lower() if rng.random() < 0.4 else text[0].upper() + text[1:]

    makers = [
        lambda: (lambda a, n: (f"my {a} is called {n}", f"They have a {a} named {n}"))(*pick(PETS)),
        lambda: (lambda a, n: (f"I got a {a} today! named her {n}", f"They have a {a} named {n}"))(*pick(PETS)),
        lambda: (lambda r, n, c: (f"my {r} {n} lives in {c}", f"Their {r} {n} lives in {c}"))(*pick(PEOPLE), pick(CITIES)),
        lambda: (lambda r, n: (f"ugh my {r} {n} keeps stealing my hoodies", f"Their {r} is called {n}"))(*pick(PEOPLE)),
        lambda: (lambda j, w: (f"I work as a {j} at {w}", f"They work as a {j} at {w}"))(*pick(JOBS)),
        lambda: (lambda c: (f"I live in {c} now, moved last year", f"They live in {c}"))(pick(CITIES)),
        lambda: (lambda c: (f"I grew up in {c}", f"They grew up in {c}"))(pick(CITIES)),
        lambda: (lambda f: (f"{f} is my favourite food", f"Their favourite food is {f}"))(pick(FOODS)),
        lambda: (lambda f: (f"I could eat {f} every single day", f"They love {f}"))(pick(FOODS)),
        lambda: (lambda d: (f"I can't stand {d}", f"They dislike {d}"))(pick(DISLIKES)),
        lambda: (lambda d: (f"remember that I hate {d}", f"They hate {d}"))(pick(DISLIKES)),
        lambda: (lambda h: (f"I've been {h} every weekend lately", f"They enjoy {h}"))(pick(HOBBIES)),
        lambda: (lambda d: (f"my birthday is on {d}", f"Their birthday is on {d}"))(pick(DATES)),
        lambda: (lambda c, d: (f"I'm flying to {c} on {d}", f"They are travelling to {c} on {d}"))(pick(CITIES), pick(DATES)),
        lambda: (lambda r, n, d: (f"my {r} {n} is visiting on {d}", f"Their {r} {n} is visiting on {d}"))(*pick(PEOPLE), pick(DATES)),
        lambda: (lambda s: (f"{s} is my comfort show, I've watched it like 5 times", f"Their comfort show is {s}"))(pick(SHOWS)),
        lambda: (lambda: ("I'm allergic to peanuts", "They are allergic to peanuts"))(),
        lambda: (lambda: ("I'm vegetarian, never eaten meat", "They are vegetarian"))(),
        lambda: (lambda d: (f"let's call it our anniversary, {d} is the day we met", f"Your anniversary with them is {d}"))(pick(DATES)),
        lambda: (lambda s: (f"tonight was our first movie night, {s}! I want to remember it", f"Your first movie night together was watching {s}"))(pick(SHOWS)),
    ]
    small_talk = ["lol ok", "good morning", "good night baby", "how was your day?", "what are you doing?",
                  "I had maggi tonight, too tired to cook", "so sleepy today", "haha you're cute", "it's raining here",
                  "my head hurts a bit today", "I'm watching a movie right now", "traffic was crazy today",
                  "ok I'm back", "miss you", "what should I eat for dinner?", "tell me a joke", "hmm",
                  "that's funny", "I'm bored", "work was long today", "just woke up", "brb, shower",
                  "you look nice today", "what's up?", "I'm having coffee", "ugh mondays"]
    her_questions = ["what's your favourite colour?", "where did you grow up?", "what's your favourite song these days?",
                     "do you have any siblings?", "what did you want to be as a kid?", "what's your favourite flower?",
                     "what's your favourite book?", "which city would you love to live in?", "are you a morning person?",
                     "what's your favourite season?", "what's your go-to karaoke song?", "do you have a pet?",
                     "what's your favourite dessert besides lava cake?", "where did you study?", "what scares you?",
                     "what's your favourite perfume?", "what's your dream holiday?", "what's your favourite movie genre?"]
    turns = []
    for _ in range(them):
        message, fact = rng.choice(makers)()
        turns.append({"kind": "them", "message": casual(message), "facts": [fact]})
    for _ in range(none):
        turns.append({"kind": "none", "message": casual(rng.choice(small_talk)), "facts": []})
    for _ in range(me):
        turns.append({"kind": "me", "message": casual(rng.choice(her_questions)), "facts": []})
    return turns


async def her_words(ollama: Ollama, messages: list[dict[str, Any]]) -> str:
    """Her reply in words only (no tools), from her own chat model."""
    said = ""
    async for piece in ollama.chat(messages):
        said += piece.get("text", "")
    return her_reply(said)


ABOUT_HER = """Here is a message {her} wrote in a chat:

"{reply}"

Her bio already says:
{bio}

Does her message say one specific, lasting fact about herself that her bio does not say yet (a favourite, her
past, her family, a habit)? If yes, write it as one short sentence starting with "Her" or "She", such as
"Her favourite flower is the lotus". If not, or if it only repeats her bio, write NONE. Write nothing else."""


async def her_fact(ollama: Ollama, her_name: str, bio: str, reply: str) -> str:
    """The lasting fact about herself in one reply of hers, or "" — the label for a "me" example, checked by you."""
    answer = (await ollama.complete(ABOUT_HER.format(her=her_name, reply=reply, bio=bio or "(empty)"))).strip()
    answer = answer.strip().strip('"').splitlines()[0].strip() if answer.strip() else ""
    return "" if not answer or answer.upper().startswith("NONE") else answer


async def make_synthetic(builder: Builder, ollama: Ollama, turns: list[dict[str, Any]], context_rows: list[list[dict]],
                         seed: int = 13, on_each=None) -> list[Example]:
    """Each made-up message after a few real messages from your chats (or none), with her own reply."""
    rng = random.Random(seed)
    bio = str((builder.her.get("persona") or {}).get("about") or "")
    examples = []
    for number, turn in enumerate(turns):
        before = rng.choice(context_rows) if context_rows and rng.random() < 0.6 else []
        before = before[-rng.randint(2, 8):] if before else []
        before = [row for row in before if not row.get("image_id")]
        now = datetime(2026, 11, rng.randint(1, 28), rng.randint(8, 23), rng.randint(0, 59))
        rows = [*before, {"id": -1, "role": "user", "content": turn["message"], "image_id": None, "photo_note": None,
                          "reaction": None, "image_kind": None, "image_prompt": None}]
        messages = builder.turn(rows, turn["kind"], turn["facts"], now)
        reply = await her_words(ollama, messages)
        facts = list(turn["facts"])
        kind = turn["kind"]
        if kind == "me":
            fact = await her_fact(ollama, builder.her["name"], bio, reply)
            if fact:
                facts = [fact]
                messages = builder.turn(rows, "me", facts, now)
            else:
                kind = "none"  # she said nothing new about herself: no call is right
        example = Example("synthetic", kind, messages, reply, facts, f"synthetic {number}")
        examples.append(example)
        if on_each:
            on_each(example)
    return examples


def save_examples(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example.to_json(), ensure_ascii=False, default=str) + "\n")


def load_examples(path: Path) -> list[Example]:
    return [Example.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split(examples: list[Example], test_share: float = 0.2, seed: int = 5) -> tuple[list[Example], list[Example]]:
    """Train and test sets with the same mix of sources and kinds; the test set is never trained on."""
    rng = random.Random(seed)
    train, test = [], []
    groups: dict[tuple[str, str], list[Example]] = {}
    for example in examples:
        groups.setdefault((example.source, example.kind), []).append(example)
    for group in groups.values():
        rng.shuffle(group)
        cut = max(1, round(len(group) * test_share)) if len(group) > 2 else 0
        test += group[:cut]
        train += group[cut:]
    rng.shuffle(train)
    return train, test


def mix(examples: list[Example]) -> dict[str, int]:
    return dict(sorted(Counter(f"{e.source}/{e.kind}" for e in examples).items()))


# ------------------------------------------------------------------------------------------- tokens

def chat_messages(example: Example) -> list[dict[str, Any]]:
    """The turn for the tokenizer's chat template: Socia's messages, tool calls as Qwen3 expects them."""
    out = []
    for message in [*example.messages, example.target]:
        message = {key: value for key, value in message.items() if key in ("role", "content", "tool_calls")}
        if message.get("tool_calls"):
            message["tool_calls"] = [{"type": "function", "function": {
                "name": call["function"]["name"],
                "arguments": call["function"].get("arguments") or {}}} for call in message["tool_calls"]]
        out.append(message)
    return out


def encode(tokenizer, example: Example, max_tokens: int) -> dict[str, Any] | None:
    """Token ids for the whole turn, and where her reply starts: the loss is only on her reply.
    None if it does not fit in ``max_tokens``."""
    messages = chat_messages(example)
    prompt = tokenizer.apply_chat_template(messages[:-1], tools=TOOLS, add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
    full = tokenizer.apply_chat_template(messages, tools=TOOLS, enable_thinking=False, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("Her reply does not follow the prompt as rendered: check the chat template.")
    start = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    # The template ends with a newline after <|im_end|>; the model stops at <|im_end|>.
    while ids and tokenizer.decode(ids[-1:]).strip() == "":
        ids = ids[:-1]
    if len(ids) > max_tokens:
        return None
    return {"ids": ids, "start": start, "kind": example.kind}


# ------------------------------------------------------------------------------------------- training

def train(model, batches: list[dict[str, Any]], tests: list[dict[str, Any]], epochs: int = 2, lr: float = 1e-4,
          accumulate: int = 8, call_weight: float = 3.0, on_step=None, seed: int = 3) -> dict[str, list]:
    """A plain training loop: the loss only on her reply, with the remember call's tokens weighted ``call_weight``
    times her words (the call is what she gets wrong; her words she already writes well).

    Only her reply's hidden states go through the output layer, so the 151k-word logits never take the GPU for
    the whole prompt."""
    import bitsandbytes as bnb
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    rng = random.Random(seed)
    base = model.get_base_model()
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(batches) * epochs / accumulate)
    warmup = max(1, steps // 10)

    def rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    call_start = None

    def loss_of(item: dict[str, Any]) -> torch.Tensor:
        nonlocal call_start
        ids = torch.tensor([item["ids"]], device=device)
        hidden = base.model(input_ids=ids, use_cache=False).last_hidden_state[0, item["start"] - 1:-1]
        logits = base.lm_head(hidden).float()
        targets = ids[0, item["start"]:]
        losses = F.cross_entropy(logits, targets, reduction="none")
        weights = torch.ones_like(losses)
        if item.get("call_from") is not None:
            weights[item["call_from"] - item["start"]:] = call_weight
        return (losses * weights).sum() / weights.sum()

    history = {"step": [], "loss": [], "test_step": [], "test_loss": []}
    model.train()
    step, running, seen = 0, 0.0, 0
    started = time.time()
    for epoch in range(epochs):
        order = list(range(len(batches)))
        rng.shuffle(order)
        for number, index in enumerate(order, start=1):
            loss = loss_of(batches[index]) / accumulate
            loss.backward()
            running += loss.item()
            seen += 1
            if seen % accumulate == 0 or number == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                history["step"].append(step)
                history["loss"].append(running * accumulate / max(1, seen % accumulate or accumulate))
                running = 0.0
                if on_step:
                    on_step(step, steps, history["loss"][-1], time.time() - started)
        model.eval()
        with torch.no_grad():
            test_loss = sum(loss_of(item).item() for item in tests) / max(1, len(tests))
        model.train()
        history["test_step"].append(step)
        history["test_loss"].append(test_loss)
        if on_step:
            on_step(step, steps, None, time.time() - started, test_loss=test_loss, epoch=epoch + 1)
    model.eval()
    return history


def mark_calls(tokenizer, item: dict[str, Any]) -> dict[str, Any]:
    """Where the first <tool_call> starts in her reply, so those tokens can weigh more."""
    tool_call = tokenizer.convert_tokens_to_ids("<tool_call>")
    reply = item["ids"][item["start"]:]
    item["call_from"] = item["start"] + reply.index(tool_call) if tool_call in reply else None
    return item


# --------------------------------------------------------------------------------------------- export

LLAMA_CPP_TAG = "b6000"


def llama_cpp() -> Path:
    """llama.cpp's Python converters (no build needed), downloaded once into /training."""
    folder = TRAINING / f"llama.cpp-{LLAMA_CPP_TAG}"
    if not (folder / "convert_lora_to_gguf.py").exists():
        url = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_TAG}.tar.gz"
        data = httpx.get(url, follow_redirects=True, timeout=300).raise_for_status().content
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(TRAINING, filter="data")
    return folder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_to_ollama(config: SociaConfig, adapter: Path, name: str) -> dict[str, Any]:
    """Her chat model with the adapter on top, as a new Ollama model: the adapter is uploaded as a blob, then
    created FROM her chat model (its template and settings) with ADAPTER. Her chat model is not changed."""
    digest = f"sha256:{sha256(adapter)}"
    with httpx.Client(base_url=config.ollama_url, timeout=600) as client:
        if client.head(f"/api/blobs/{digest}").status_code != 200:
            with adapter.open("rb") as file:
                client.post(f"/api/blobs/{digest}", content=file).raise_for_status()
        reply = client.post("/api/create", json={"model": name, "from": config.chat_model,
                                                 "adapters": {adapter.name: digest}, "stream": False})
        reply.raise_for_status()
        return reply.json()


# ------------------------------------------------------------------------------------------ evaluation

def calls_of(found: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Her remember calls as (them or me, fact)."""
    out = []
    for call in found:
        function = call.get("function") or {}
        if function.get("name") != "remember":
            continue
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
        about = "me" if str(arguments.get("about") or "").lower() == "me" else "them"
        out.append((about, str(arguments.get("fact") or "")))
    return out


async def answer(config: SociaConfig, model: str, messages: list[dict[str, Any]]) -> tuple[str, list[tuple[str, str]]]:
    """One reply from a model exactly as Socia asks (her tools too), with the remember calls, typed out or made."""
    ollama = Ollama(replace(config, chat_model=model))
    said, calls = "", []
    async for piece in ollama.chat(messages, TOOLS):
        said += piece.get("text", "")
        calls += piece.get("tool_calls", [])
    typed, said = written_remembers(said)
    return " ".join(said.split()), calls_of([*calls, *typed])


async def evaluate(config: SociaConfig, model: str, cases: list[Example], on_each=None) -> list[dict[str, Any]]:
    """Each case's expected kind against what the model did: right if it made the call it should (them or me),
    or none when none was expected."""
    results = []
    for case in cases:
        said, calls = await answer(config, model, case.messages)
        got = calls[0][0] if calls else "none"
        result = {"ref": case.ref, "source": case.source, "expected": case.kind, "got": got, "right": got == case.kind,
                  "message": case.messages[-1]["content"][:80], "fact": case.facts[0] if case.facts else "",
                  "remembered": "; ".join(fact for _, fact in calls), "reply": said[:120]}
        results.append(result)
        if on_each:
            on_each(result)
    return results


def scores(results: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for kind in KINDS:
        rows = [r for r in results if r["expected"] == kind]
        if rows:
            out[kind] = f"{sum(r['right'] for r in rows)}/{len(rows)}"
    false_calls = sum(r["expected"] == "none" and r["got"] != "none" for r in results)
    out["false calls"] = str(false_calls)
    return out


def fixed_cases(builder: Builder) -> list[Example]:
    """model_evaluations/remember.py's twelve messages, each in a fresh chat: the same check as before any
    training, so the numbers compare."""
    from remember import MESSAGES  # model_evaluations/remember.py

    now = datetime(2026, 11, 17, 21, 0)
    cases = []
    for number, (kind, message) in enumerate(MESSAGES):
        rows = [{"id": -1, "role": "user", "content": message, "image_id": None, "photo_note": None, "reaction": None,
                 "image_kind": None, "image_prompt": None}]
        cases.append(Example("fixed", kind, builder.turn(rows, "none", [], now), "", [], f"fixed {number}"))
    return cases
