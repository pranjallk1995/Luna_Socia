"""Twenty questions with her, to see whether she keeps a long chat together.

Two games, each played to the end against the model Socia runs, with her own system prompt and tools:

- guesser: the owner thinks of something and answers her yes/no questions; she must use every
  answer so far to close in. Scored: solved, questions used, questions asked twice.
- keeper: she thinks of something and the owner asks; she must answer every question true to
  the secret she chose. Scored: answers a judge finds wrong once she says what it was.

The owner's side is the same model at temperature 0, told only how to play. Settings are
compared as variants, such as her repeat penalty, or private notes she can write a secret into.

    python games.py <variant> <out.jsonl> [--games N]
    variants: current (repeat penalty 1.15), no-penalty (1.0), notes (1.0 and a note tool), think (notes, reasoning on),
              product and product-no-penalty (Socia's own note tool, notes and prompt, at 1.15 and 1.0)
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime

import httpx

from app.config import SociaConfig
from app.chat import with_secret_nudge, without_written_pictures, written_notes
from app.persona import DEFAULT_PERSONA, TOOLS, system_prompt

# Made up, everyday things, varied enough to need real narrowing.
SECRETS = ["an elephant", "a pizza", "a guitar", "the moon", "a cricket bat", "a penguin"]
MAX_QUESTIONS = 20
OWNER = "Sam"

# Socia's tools without its note tool, for the variants from before Socia had one.
OLD_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] != "note"]
VARIANTS = {
    "current": {"repeat_penalty": 1.15, "notes": False},
    "no-penalty": {"repeat_penalty": 1.0, "notes": False},
    "notes": {"repeat_penalty": 1.0, "notes": True},
    # The ceiling: Qwen3 reasoning before each reply (slower, and its thoughts are not kept).
    "think": {"repeat_penalty": 1.0, "notes": True, "think": True},
    # Socia's own note tool, notes and prompt, at its repeat penalty and without one.
    "product": {"repeat_penalty": 1.15, "product": True},
    "product-no-penalty": {"repeat_penalty": 1.0, "product": True},
}

NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "note",
        "description": ("Write down something private to remember in this chat: a secret you picked for a game, "
                        "the score, a plan. They never see it; your notes are shown to you on every message."),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to remember, in a short line."}}, "required": ["text"]},
    },
}
NOTES = "\n\nYour private notes for this chat (only you see them; keep to them):\n{notes}"

ANSWERER = """You are playing twenty questions. The secret is: {secret}.
Answer the other player's last message about the secret with exactly one of: Yes. / No. / Sometimes.
If they guess the secret itself (or an obvious synonym), answer exactly: CORRECT.
Their message:
{message}"""
ASKER = """You are playing twenty questions: the other player has thought of something and you guess it.
The game so far (your questions and their answers):
{transcript}
Write your next message only: one short yes/no question, or a guess like "Is it a ...?". No other words.
If they already said your guess is right, write only: DONE"""
JUDGE = """Twenty questions. The secret is: {secret}
Question: {question}
Answer given: {answer}
Is the answer given true of {secret}? (A playful answer counts if its yes or no is right.) Answer only TRUE or FALSE."""
SAME = """The secret written down was "{a}". Later the player said: "{b}"
Did the player name the same secret? Answer only YES or NO."""


class Player:
    def __init__(self, config: SociaConfig, variant: dict) -> None:
        self.config = config
        self.variant = variant
        self.client = httpx.Client(base_url=config.ollama_url, timeout=900)

    def plain(self, prompt: str) -> str:
        """The owner's side or the judge: the same model, temperature 0."""
        body = {"model": self.config.chat_model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                "think": False, "options": {"num_ctx": self.config.context_tokens, "temperature": 0}}
        return (self.client.post("/api/chat", json=body).json().get("message") or {}).get("content", "").strip()

    def her(self, history: list[dict], notes: list[str]) -> str:
        """Her reply as Socia would get it, with her notes and the note tool in the notes variant."""
        companion = {"name": "Luna", "persona": dict(DEFAULT_PERSONA)}
        product = self.variant.get("product")
        prompt = system_prompt(companion, OWNER, datetime(2026, 11, 13, 20, 30), notes="\n".join(notes) if product else "")
        if notes and not product:
            prompt += NOTES.format(notes="\n".join(f"- {n}" for n in notes))
        tools = TOOLS if product else [*OLD_TOOLS, NOTE_TOOL] if self.variant.get("notes") else OLD_TOOLS
        messages = [{"role": "system", "content": prompt},
                    *(with_secret_nudge(history, "\n".join(notes)) if product else history)]
        options = {"num_ctx": self.config.context_tokens, "temperature": self.config.temperature,
                   "repeat_penalty": self.variant["repeat_penalty"], "repeat_last_n": 256}
        body = {"model": self.config.chat_model, "messages": messages, "tools": tools, "stream": False,
                "think": bool(self.variant.get("think")), "options": options}
        message = self.client.post("/api/chat", json=body).json().get("message") or {}
        text = (message.get("content") or "").strip()
        calls = [c for c in message.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "note"]
        if product:
            # Socia takes a note she wrote out as a call, and out of her text.
            calls += written_notes(text)
            text = without_written_pictures(text)
        for call in calls:
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            if str(arguments.get("text") or "").strip():
                notes.append(str(arguments["text"]).strip())
        if calls and not text:
            # She only wrote a note: let her say something, as Socia does after a tool.
            follow = [*messages, {"role": "assistant", "content": "", "tool_calls": calls},
                      {"role": "tool", "content": "Noted. Only you can see it."}]
            body = {**body, "messages": follow}
            body.pop("tools")
            text = ((self.client.post("/api/chat", json=body).json().get("message") or {}).get("content") or "").strip()
        return text


def asked_twice(questions: list[str]) -> int:
    seen, twice = set(), 0
    for question in questions:
        key = re.sub(r"[^a-z ]", "", question.casefold()).strip()
        twice += key in seen
        seen.add(key)
    return twice


def guesser(player: Player, secret: str) -> dict:
    """The owner thinks of the secret; she asks."""
    history = [{"role": "user", "content": "Let's play 20 questions! I'm thinking of something. "
                                           "You ask me yes/no questions and guess what it is. Go!"}]
    notes: list[str] = []
    questions = []
    for _ in range(MAX_QUESTIONS):
        reply = player.her(history, notes)
        history.append({"role": "assistant", "content": reply})
        questions.append(reply)
        answer = player.plain(ANSWERER.format(secret=secret, message=reply))
        if answer.upper().startswith("CORRECT"):
            history.append({"role": "user", "content": "Yesss! You got it! 🎉"})
            return {"solved": True, "questions": len(questions), "asked_twice": asked_twice(questions),
                    "history": history, "notes": notes}
        history.append({"role": "user", "content": answer})
    return {"solved": False, "questions": len(questions), "asked_twice": asked_twice(questions),
            "history": history, "notes": notes}


def keeper(player: Player, rounds: int = 12) -> dict:
    """She thinks of something; the owner asks, then asks what it was."""
    history = [{"role": "user", "content": "Let's play 20 questions! You think of something (an animal, object or "
                                           "place) and keep it secret, I'll ask yes/no questions. Ready?"}]
    notes: list[str] = []
    history.append({"role": "assistant", "content": player.her(history, notes)})
    # The secret counts only if she wrote it down before the first question.
    committed = notes[0] if notes else None
    pairs = []
    for _ in range(rounds):
        transcript = "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs) or "(no questions yet)"
        question = (player.plain(ASKER.format(transcript=transcript)).splitlines() or ["DONE"])[0][:200]
        if question.strip().upper().startswith("DONE"):
            break
        history.append({"role": "user", "content": question})
        answer = player.her(history, notes)
        history.append({"role": "assistant", "content": answer})
        pairs.append((question, answer))
    history.append({"role": "user", "content": "Ok I give up 🙈 What was it? Tell me exactly what you picked."})
    reveal_text = player.her(history, notes)
    history.append({"role": "assistant", "content": reveal_text})
    result = {"secret": committed, "answers": len(pairs), "committed": committed is not None, "wrong": None,
              "reveal_matches": None, "history": history, "notes": notes}
    if committed is None:
        # Nothing was kept, so there is nothing to keep to: her answers can only be made up as she goes.
        return result
    secret = re.sub(r"^\s*secret\s*:\s*", "", committed, flags=re.IGNORECASE)
    result["wrong"] = [n for n, (q, a) in enumerate(pairs, 1)
                       if player.plain(JUDGE.format(secret=secret, question=q, answer=a[:200])).upper().startswith("FALSE")]
    result["reveal_matches"] = player.plain(SAME.format(a=committed, b=reveal_text[:200])).upper().startswith("YES")
    return result


def main() -> None:
    variant_name, out = sys.argv[1], sys.argv[2]
    games = int(sys.argv[sys.argv.index("--games") + 1]) if "--games" in sys.argv else 3
    player = Player(SociaConfig.from_env(), VARIANTS[variant_name])
    with open(out, "w") as file:
        for number in range(games):
            for game, play in (("guesser", lambda: guesser(player, SECRETS[number % len(SECRETS)])),
                               ("keeper", lambda: keeper(player))):
                started = time.monotonic()
                result = {"variant": variant_name, "game": game, "number": number, **play(),
                          "seconds": round(time.monotonic() - started)}
                if game == "guesser":
                    result["secret"] = SECRETS[number % len(SECRETS)]
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                summary = {k: v for k, v in result.items() if k not in ("history",)}
                print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
