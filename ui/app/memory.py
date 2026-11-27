"""What a girlfriend remembers about the owner, from chats exported from ChatGPT or Claude.

An export (the .zip either service emails, or the conversations.json inside it) is parsed
into conversations and kept in the database. In the background, the chat model reads the
owner's own words in each one for lasting facts (people, plans, likes, work), then folds
them into her memory, a short list that goes into every reply. Chats only wait while she
is answering, so the GPU is never asked for two things at once by Socia.

The conversations stay in the database as the training data for her fine-tuned model.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from typing import Any

from app.config import SociaConfig

logger = logging.getLogger("socia.memory")


class ExportError(ValueError):
    """An upload that is not a ChatGPT or Claude export."""


READ_PROMPT = """Below are messages a person wrote to an AI assistant, in one conversation titled "{title}".
List the lasting facts they reveal about the person themselves: their name, people in their life, where they
live and work, what they study or do, their plans, likes and dislikes, habits, health, and what matters to them.
Only write what they actually say: never guess or infer from what they asked for. A fact about someone else
stays that person's ("their sister Asha loves Coldplay"). Leave out the task itself, budgets and details of
that one task, anything about the assistant, and anything only true that day.
Write each fact as one short line starting with "- ", about "they". If there are none, write only NONE.

Messages:
{messages}"""

CHAT_READ_PROMPT = """Below are messages a person wrote to {her}, their girlfriend, in one chat.
List the lasting facts they reveal about themselves and their life: their name, people in their life, their pets
(each pet's name, and what tells it apart in photos: its colour, markings, collar or tag), where they live and work,
what they do, their plans and dates, likes and dislikes, habits, health, and what matters to them.
Only write what they actually say or show: never guess. Write names exactly as they write them. Leave out:
- anything about {her} herself, flirting and compliments, and games and their answers;
- what they ate, did or felt today or tonight, greetings and goodnights, and questions they asked;
- their own name, unless they tell it.
Write each fact as one short line starting with "- ", about "they". If there are none, write only NONE.

Messages:
{messages}"""

HER_READ_PROMPT = """{her} is a girlfriend in a chat. Her bio so far:
{bio}

Below are messages {her} wrote in one chat. List the lasting facts she states about herself that her bio does not
already say: her favourites (colour, food, film, song, place), her past, family, friends, work, home, hobbies,
habits, and what she likes or dislikes. Only what she states as true about herself:
- not anything already in her bio, even in other words;
- not what the person she talks to said about themselves, even when she agrees (they hate olives: that is theirs);
- not flirting, not what she is doing, feeling or wearing right now, not plans for today or tomorrow, not a game's
  secret, and nothing that goes against her bio.
Write each fact as one short line starting with "- ", about "she" ("- Her favourite colour is lavender"). If there
are none, write only NONE.

Messages:
{messages}"""

VERIFY_PROMPT = """Here is what a person wrote to their girlfriend:
{messages}

Do the person's own words plainly say this about the person: "{fact}"?
Answer NO if it is a guess, if it is about someone else (their girlfriend included), if it comes from a joke or a
game, or if it is only true that day. Answer only YES or NO."""

MERGE_PROMPT = """Below are notes about a person, gathered from many conversations, older ones first.
Rewrite them as one list of the most important lasting facts about them, at most {lines} lines,
each a short line starting with "- ", about "they". Merge duplicates, keep names and specifics,
prefer newer notes when they disagree, and drop trivia. Write only the list.

Notes:
{notes}"""


def timestamp(value: Any) -> str | None:
    """An export's time (seconds since 1970, or ISO text) as ISO text."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def chatgpt_messages(conversation: dict[str, Any]) -> list[dict[str, str]]:
    """The conversation's main thread: from its current message back to the start."""
    mapping = conversation.get("mapping") or {}
    node_id = conversation.get("current_node")
    if node_id not in mapping:
        # Without a current node, the last message written stands in for it.
        dated = [(node.get("message") or {}).get("create_time") or 0 for node in mapping.values()]
        node_id = list(mapping)[dated.index(max(dated))] if dated else None
    chain, seen = [], set()
    while node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        chain.append(node.get("message"))
        node_id = node.get("parent")
    messages = []
    for message in reversed(chain):
        if not message:
            continue
        role = (message.get("author") or {}).get("role")
        content = message.get("content") or {}
        if role not in ("user", "assistant") or content.get("content_type") not in ("text", "multimodal_text"):
            continue
        text = "\n".join(part for part in content.get("parts") or [] if isinstance(part, str)).strip()
        if text:
            messages.append({"role": role, "text": text})
    return messages


def claude_messages(conversation: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    for message in conversation.get("chat_messages") or []:
        role = {"human": "user", "assistant": "assistant"}.get(message.get("sender"))
        text = str(message.get("text") or "").strip()
        if not text:
            text = "\n".join(str(part.get("text") or "") for part in message.get("content") or []
                             if isinstance(part, dict) and part.get("type") == "text").strip()
        if role and text:
            messages.append({"role": role, "text": text})
    return messages


def conversations_json(data: bytes) -> bytes:
    """conversations.json, from an export's .zip or as it is."""
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [info for info in archive.infolist() if info.filename.rsplit("/", 1)[-1] == "conversations.json"]
        if not names:
            raise ExportError("That .zip has no conversations.json. Upload the export ChatGPT or Claude emailed you.")
        if names[0].file_size > SociaConfig.MAX_EXPORT_JSON_BYTES:
            raise ExportError("That export is too large to read.")
        return archive.read(names[0])


def parse_export(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    """Which service an export is from ("chatgpt" or "claude"), and its conversations.

    Each conversation: {"id", "title", "started_at", "messages": [{"role", "text"}]};
    conversations without a message from the owner are left out.
    """
    try:
        items = json.loads(conversations_json(data))
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise ExportError("That file is not a ChatGPT or Claude export.") from error
    if not isinstance(items, list):
        raise ExportError("That file is not a ChatGPT or Claude export.")
    items = [item for item in items if isinstance(item, dict)]
    if any("mapping" in item for item in items):
        source = "chatgpt"
    elif any("chat_messages" in item for item in items):
        source = "claude"
    elif not items:
        raise ExportError("That export has no conversations.")
    else:
        raise ExportError("That file is not a ChatGPT or Claude export.")
    conversations = []
    for number, item in enumerate(items):
        if source == "chatgpt":
            messages = chatgpt_messages(item)
            external = item.get("conversation_id") or item.get("id")
            started = item.get("create_time")
            title = item.get("title")
        else:
            messages = claude_messages(item)
            external = item.get("uuid")
            started = item.get("created_at")
            title = item.get("name")
        if not any(message["role"] == "user" for message in messages):
            continue
        conversations.append({"id": str(external or f"#{number}"), "title": str(title or "").strip()[:200],
                              "started_at": timestamp(started), "messages": messages})
    return source, conversations


def their_words(messages: list[dict[str, str]], limit: int = SociaConfig.MEMORY_READ_CHARS, role: str = "user") -> str:
    """What the owner (or, with role "luna", she) wrote in one conversation, each message trimmed, up to ``limit``
    characters."""
    lines, used = [], 0
    for message in messages:
        if message["role"] != role:
            continue
        text = " ".join(message["text"].split())[:800]
        if used + len(text) > limit:
            break
        lines.append(f"- {text}")
        used += len(text) + 3
    return "\n".join(lines)


# Words that say nothing about which fact it is ("she loves", "her favourite").
PLAIN_WORDS = {"she", "her", "hers", "they", "their", "them", "the", "and", "with", "for", "from", "has", "have", "had",
               "was", "were", "are", "is", "likes", "like", "loves", "love", "enjoys", "enjoy", "favourite", "favorite",
               "really", "very", "also", "always", "most", "best", "lot", "lots", "some", "this", "that", "these"}


# Lines about the owner that hold only the moment: what they did tonight, how tired they are, a goodnight.
PASSING = re.compile(r"\b(?:today|tonight|right now|this (?:morning|evening|week)|tired|good ?night|good ?morning|"
                     r"says?\b|asks?\b|asked\b|their name is they)", re.IGNORECASE)


# The owner's messages read at once: the model lists facts reliably from a short stretch, not a whole chat.
CHAT_STRETCH = 30


# Judgements of their character the model writes as facts from a girlfriend chat ("They have a habit of being
# persistent", "They are looking for a romantic connection"): not things they said.
JUDGEMENT = re.compile(r"\b(?:habit of|values?\b|valuing|looking for|preference for|interest in|tend to|tends to|"
                       r"seems?|enjoys? being|like being|likes being|focused on)", re.IGNORECASE)


async def facts_about_them(ask, her: str, messages: list[dict[str, str]]) -> list[str]:
    """The lasting facts about the owner in what they wrote to her in a chat: their own words only (what their photos
    showed made "facts" such as "They have a lamp"), CHAT_STRETCH messages at a time, each fact then checked against
    their words with a yes or no."""
    theirs = [{"role": "user", "text": re.sub(r"\s*\[with a photo: [^\]]*\]", "", m["text"]).strip()}
              for m in messages if m["role"] == "user"]
    theirs = [m for m in theirs if m["text"]]
    facts: list[str] = []
    for start in range(0, len(theirs), CHAT_STRETCH):
        words = their_words(theirs[start:start + CHAT_STRETCH])
        if not words:
            continue
        for fact in fact_lines(await ask(CHAT_READ_PROMPT.format(her=her, messages=words))):
            if PASSING.search(fact) or JUDGEMENT.search(fact) or fact.casefold() in (f.casefold() for f in facts):
                continue
            answer = await ask(VERIFY_PROMPT.format(messages=words, fact=fact.removeprefix("- ")))
            if answer.strip().upper().startswith("YES"):
                facts.append(fact)
    return facts


# What may go into her bio: specific, lasting facts. The model also lists vague ones ("She likes to miss people")
# however it is asked, so only these kinds are kept.
LASTING = re.compile(
    r"\b(?:favou?rites?|grew up|born|(?:is|comes) from|lives? in|sisters?|brothers?|mother|mom|mum|father|dad|parents|"
    r"cousins?|best friend|friend named|dog|cat|pets?|works? (?:as|at|in)|job|stud(?:y|ies|ied)|college|school|"
    r"allergic|hates|can'?t stand|named|called)\b", re.IGNORECASE)


async def facts_about_her(ask, her: str, bio: str, messages: list[dict[str, str]]) -> list[str]:
    """What she said about herself in a chat that her bio does not say yet: specific and lasting only, and never
    a second favourite of a kind she has one of."""
    words = their_words(messages, role="luna")
    if not words:
        return []
    kept = []
    for fact in fact_lines(await ask(HER_READ_PROMPT.format(her=her, bio=bio or "(empty)", messages=words))):
        favourite = re.search(r"favou?rite ([a-z]+(?: [a-z]+)?) (?:is|are)\b", fact, re.IGNORECASE)
        if favourite and re.search(rf"favou?rite {re.escape(favourite.group(1))}\b", f"{bio}\n" + "\n".join(kept),
                                   re.IGNORECASE):
            continue
        if LASTING.search(fact) and not known(fact, bio):
            kept.append(fact)
    return kept


def known(fact: str, text: str) -> bool:
    """Whether ``text`` already says ``fact``: nearly all of its telling words are in it ("She loves baking" when
    the bio says "She loves baking and her favourite is choco lava cake")."""
    words = {w for w in re.findall(r"[a-z]{3,}", fact.casefold()) if w not in PLAIN_WORDS}
    have = set(re.findall(r"[a-z]{3,}", text.casefold()))
    return bool(words) and len(words & have) >= 0.75 * len(words)


def fact_lines(text: str) -> list[str]:
    """The "- " lines of the model's answer; NONE or anything else gives none."""
    facts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ", "• ")) and len(line) > 3:
            facts.append("- " + line[2:].strip())
    return facts


class MemoryReader:
    """Reads imported chats in the background, one at a time, and keeps each girlfriend's memory."""

    # Her chats are read once quiet for this long, so a conversation is read whole, not mid-way.
    CHAT_QUIET_SECONDS = 600
    # Messages of one chat read at once; the rest are read next time.
    CHAT_MESSAGES = 200
    # Notes merged at once fit the model's context with room for the answer.
    MERGE_BATCH_CHARS = 14000
    MEMORY_LINES = 45

    def __init__(self, store, ollama, turn: asyncio.Lock, read_chats: bool = True, rememberer=None) -> None:
        self.store = store
        # Compares new facts with what she has by meaning (app/remembering.py); None keeps the word checks only.
        self.rememberer = rememberer
        self.read_chats = read_chats
        self.ollama = ollama
        self.turn = turn
        self.wake = asyncio.Event()
        self.task: asyncio.Task | None = None
        # What each companion's reader is doing, for the page: {"error": ...}.
        self.errors: dict[int, str] = {}
        self.busy: int | None = None

    def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self.run())
        self.wake.set()

    async def quiet(self) -> None:
        """Wait while she is answering: a reply always comes first."""
        while self.turn.locked():
            await asyncio.sleep(1.0)

    async def ask(self, prompt: str) -> str:
        await self.quiet()
        return await self.ollama.complete(prompt)

    async def run(self) -> None:
        while True:
            self.wake.clear()
            try:
                worked = await self.step()
            except Exception:  # noqa: BLE001 - shown on the page by step; tried again later
                logger.exception("Reading memories failed")
                worked = False
                await asyncio.sleep(30)
            if not worked:
                try:
                    await asyncio.wait_for(self.wake.wait(), 300)
                except TimeoutError:
                    pass

    async def step(self) -> bool:
        """Read one imported chat, or merge what was read into memory; False when nothing is left."""
        try:
            return await self.work()
        except Exception as error:
            if self.busy is not None:
                self.errors[self.busy] = str(error)[:300] or type(error).__name__
            raise
        finally:
            self.busy = None

    async def work(self) -> bool:
        source = await asyncio.to_thread(self.store.unread_source)
        if source:
            self.busy = source["companion_id"]
            if source["source"] == "socia":
                her = await asyncio.to_thread(self.store.companion, source["companion_id"])
                facts = await facts_about_them(self.ask, (her or {}).get("name") or "her", source["messages"])
                await self.learn_about_her(source)
            else:
                words = their_words(source["messages"])
                facts = fact_lines(await self.ask(READ_PROMPT.format(title=source["title"] or "untitled",
                                                                     messages=words))) if words else []
            await asyncio.to_thread(self.store.set_source_facts, source["id"], "\n".join(facts))
            self.errors.pop(source["companion_id"], None)
            # Merged every so often, so her memory grows while a big export is read.
            pending = await asyncio.to_thread(self.store.unmerged_facts, source["companion_id"])
            if sum(len(row["facts"]) for row in pending) >= self.MERGE_BATCH_CHARS:
                await self.merge(source["companion_id"])
            return True
        # Her chats, once quiet: the owner's new messages become a source like an imported chat.
        for chat in (await asyncio.to_thread(self.store.quiet_chats, self.CHAT_QUIET_SECONDS) if self.read_chats else []):
            if await asyncio.to_thread(self.store.chat_source, chat, self.CHAT_MESSAGES):
                return True
        for companion in await asyncio.to_thread(self.store.companions):
            if await asyncio.to_thread(self.store.unmerged_facts, companion["id"]):
                self.busy = companion["id"]
                await self.merge(companion["id"])
                return True
        return False

    async def learn_about_her(self, source: dict[str, Any]) -> list[str]:
        """What she said about herself in a chat (her favourite colour, her past), added to her bio so she says
        the same next time."""
        her = await asyncio.to_thread(self.store.companion, source["companion_id"])
        if her is None:
            return []
        bio = str((her.get("persona") or {}).get("about") or "").strip()
        facts = await facts_about_her(self.ask, her["name"], bio, source["messages"])
        if self.rememberer is not None:
            facts = await self.rememberer.new_facts(her, "her", facts)
        added = await asyncio.to_thread(self.store.add_to_bio, her["id"], facts, SociaConfig.ABOUT_CHARS)
        if added:
            logger.info("Added %d facts she told about herself to her bio", len(added))
        return added

    async def merge(self, companion_id: int) -> None:
        """Fold newly read facts into her memory, keeping it under SociaConfig.MEMORY_CHARS."""
        companion = await asyncio.to_thread(self.store.companion, companion_id)
        rows = await asyncio.to_thread(self.store.unmerged_facts, companion_id)
        if companion is None or not rows:
            return
        lines, seen = [], set()
        for line in (companion["memory"] or "").splitlines():
            if line.strip() and line.strip().casefold() not in seen:
                seen.add(line.strip().casefold())
                lines.append(line.strip())
        read = []
        for line in "\n".join(row["facts"] for row in rows).splitlines():
            if line.strip() and line.strip().casefold() not in seen:
                seen.add(line.strip().casefold())
                read.append(line.strip())
        if self.rememberer is not None:
            # A fact she has in other words is not merged in again (her chat model would only fold it back).
            read = await self.rememberer.new_facts(companion, "you", read)
        if not read:
            # Nothing she did not know: her memory stays as it is, and the chats count as merged.
            await asyncio.to_thread(self.store.merge_memory, companion_id, companion["memory"] or "", [row["id"] for row in rows])
            return
        lines += read
        notes = "\n".join(lines)
        memory = await self.condense(notes)
        await asyncio.to_thread(self.store.merge_memory, companion_id, memory, [row["id"] for row in rows])

    async def condense(self, notes: str) -> str:
        """Notes short enough for every reply, merged in parts when there are many."""
        lines = [line for line in notes.splitlines() if line.strip()]
        while sum(len(line) + 1 for line in lines) > SociaConfig.MEMORY_CHARS:
            batches, batch, size = [], [], 0
            for line in lines:
                if batch and size + len(line) > self.MERGE_BATCH_CHARS:
                    batches.append(batch)
                    batch, size = [], 0
                batch.append(line)
                size += len(line) + 1
            batches.append(batch)
            merged = []
            for part in batches:
                answer = fact_lines(await self.ask(MERGE_PROMPT.format(lines=self.MEMORY_LINES, notes="\n".join(part))))
                merged.extend(answer or part[:self.MEMORY_LINES])
            if len(merged) >= len(lines):
                # The model kept everything: keep the newest notes instead of looping.
                lines = merged[-self.MEMORY_LINES:]
                break
            lines = merged
        return "\n".join(lines)[:SociaConfig.MEMORY_CHARS]

    def progress(self, companion: dict[str, Any]) -> dict[str, Any]:
        sources, read = int(companion.get("sources") or 0), int(companion.get("read") or 0)
        return {"sources": sources, "read": read, "reading": read < sources or self.busy == companion["id"],
                "error": self.errors.get(companion["id"])}
