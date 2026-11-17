"""Does Luna remember what she should, and only that? Her real prompt and chat model, nothing saved.

Each message goes, in a fresh chat, to her chat model exactly as in Socia (her prompt and tools), a few times.
A remember call is counted as about them, about herself ("about": "me"), or none. Expected:
- about them: lasting facts the owner tells (pets and their names, family, work, a plan, "remember that...");
- about her: questions that make her say something lasting about herself (her favourite colour);
- none: small talk and things only true today.

With --reader, the same messages are one chat instead, with her replies, and the background reader (what reads
her chats once quiet) lists what it would remember about them and add to her bio.

With --chats, the reader goes through every chat with her (the owner's words), lists what it would remember about
them, and, with --save, puts that in her memory (which the owner can edit or clear on her edit page).

Run it beside the running Socia (in the notebooks' container):
    python model_evaluations/remember.py [--repeat 3] [--reader] [--chats [--save]]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime

from app.config import SociaConfig
from app.memory import MemoryReader, facts_about_her, facts_about_them
from app.ollama import Ollama
from app.persona import TOOLS, system_prompt
from app.store import SociaStore

MESSAGES = [
    ("them", "my cats are Mini, Maxi and Melon 🐱"),
    ("them", "Melon is the tabby with the green collar, Maxi has the pink bell"),
    ("them", "I work at a startup in Bangalore, building apps"),
    ("them", "my sister Asha is visiting in May!"),
    ("them", "remember that I hate olives"),
    ("me", "what's your favourite colour?"),
    ("me", "where did you grow up?"),
    ("me", "what's your favourite song these days?"),
    ("none", "lol ok"),
    ("none", "I had maggi tonight, too tired to cook"),
    ("none", "good night baby"),
    ("none", "how was your day?"),
]


def about(calls: list[dict]) -> list[str]:
    found = []
    for call in calls:
        function = call.get("function") or {}
        if function.get("name") != "remember":
            continue
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        found.append(f"{'me' if str(arguments.get('about')).lower() == 'me' else 'them'}: {arguments.get('fact')}")
    return found


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--companion", type=int, default=1)
    parser.add_argument("--name", default="Pranjall")
    parser.add_argument("--reader", action="store_true")
    parser.add_argument("--chats", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    config = SociaConfig.from_env()
    store = SociaStore(config.database_url)
    her = store.companion(args.companion)
    ollama = Ollama(config)
    prompt = system_prompt(her, args.name, datetime(2026, 11, 17, 21, 0))
    if args.chats:
        await read_all_chats(store, Ollama(config), her, args.save)
        return
    if args.reader:
        await read_chat(ollama, her, prompt)
        return
    score = Counter()
    for expected, message in MESSAGES:
        for _ in range(args.repeat):
            said, calls = "", []
            async for piece in ollama.chat([{"role": "system", "content": prompt},
                                            {"role": "user", "content": message}], TOOLS):
                said += piece.get("text", "")
                calls += piece.get("tool_calls", [])
            facts = about(calls)
            got = facts[0].split(":")[0] if facts else "none"
            score[(expected, got == expected)] += 1
            print(f"{expected:>4} {'ok ' if got == expected else 'XX '} {message[:45]:<45} | {'; '.join(facts) or '-'}"
                  f" | {' '.join(said.split())[:70]}", flush=True)
    for expected in ("them", "me", "none"):
        right, total = score[(expected, True)], score[(expected, True)] + score[(expected, False)]
        print(f"{expected}: {right}/{total} right")


async def read_chat(ollama: Ollama, her: dict, prompt: str) -> None:
    """The messages as one chat with her replies, then what the background reader takes from it."""
    chat, messages = [{"role": "system", "content": prompt}], []
    for _, message in MESSAGES:
        chat.append({"role": "user", "content": message})
        said = ""
        async for piece in ollama.chat(chat):
            said += piece.get("text", "")
        chat.append({"role": "assistant", "content": said})
        messages += [{"role": "user", "text": message}, {"role": "luna", "text": said}]
        print(f"> {message}\n  {' '.join(said.split())[:110]}", flush=True)
    # The same steps as the background reader.
    about_them = await facts_about_them(ollama.complete, her["name"], messages)
    bio = str((her.get("persona") or {}).get("about") or "").strip()
    about_her = await facts_about_her(ollama.complete, her["name"], bio, messages)
    print("\nRemembered about them:", *about_them, sep="\n  ")
    print("Added to her bio:", *about_her, sep="\n  ")


async def read_all_chats(store: SociaStore, ollama: Ollama, her: dict, save: bool) -> None:
    """Every chat with her, the owner's words, through the reader; her memory from it with --save."""
    facts: list[str] = []
    for chat in store.chats(her["id"]):
        rows = store.recent(chat["id"], 100000)
        messages = [{"role": row["role"], "text": row["content"] or ""} for row in rows]
        found = await facts_about_them(ollama.complete, her["name"], messages)
        print(f"chat {chat['id']} ({sum(r['role'] == 'user' for r in rows)} of their messages): {len(found)} facts", flush=True)
        for fact in found:
            if fact.casefold() not in (f.casefold() for f in facts):
                facts.append(fact)
                print("  ", fact, flush=True)
    reader = MemoryReader(store, ollama, asyncio.Lock(), read_chats=False)
    memory = await reader.condense("\n".join(facts))
    print(f"\nHer memory ({len(memory)} characters):\n{memory}")
    if save:
        store.update_companion(her["id"], {"memory": memory})
        print("Saved.")


if __name__ == "__main__":
    asyncio.run(main())
