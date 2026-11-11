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


def their_words(messages: list[dict[str, str]], limit: int = SociaConfig.MEMORY_READ_CHARS) -> str:
    """What the owner wrote in one conversation, each message trimmed, up to ``limit`` characters."""
    lines, used = [], 0
    for message in messages:
        if message["role"] != "user":
            continue
        text = " ".join(message["text"].split())[:800]
        if used + len(text) > limit:
            break
        lines.append(f"- {text}")
        used += len(text) + 3
    return "\n".join(lines)


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

    # Notes merged at once fit the model's context with room for the answer.
    MERGE_BATCH_CHARS = 14000
    MEMORY_LINES = 45

    def __init__(self, store, ollama, turn: asyncio.Lock) -> None:
        self.store = store
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
        for companion in await asyncio.to_thread(self.store.companions):
            if await asyncio.to_thread(self.store.unmerged_facts, companion["id"]):
                self.busy = companion["id"]
                await self.merge(companion["id"])
                return True
        return False

    async def merge(self, companion_id: int) -> None:
        """Fold newly read facts into her memory, keeping it under SociaConfig.MEMORY_CHARS."""
        companion = await asyncio.to_thread(self.store.companion, companion_id)
        rows = await asyncio.to_thread(self.store.unmerged_facts, companion_id)
        if companion is None or not rows:
            return
        notes = "\n".join(line for line in [companion["memory"], *(row["facts"] for row in rows)] if line.strip())
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
