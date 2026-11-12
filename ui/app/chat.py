"""One turn of a conversation: what she sees, what she says, and the pictures she sends."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps

from app.config import SociaConfig
from app.images import ComfyUI
from app.ollama import LOOKS_PROMPT, Ollama
from app.persona import TOOLS, clean_persona, image_prompt, system_prompt
from app.store import SociaStore

logger = logging.getLogger("socia")
# Photos are kept at most this size on their longer side; the image model uses about 1024.
PHOTO_MAX_SIDE = 2048
# A picture she wrote out in brackets ("[You sent a picture: ...]") instead of calling
# create_image: it is taken out of her text and taken as the call she meant.
WRITTEN_PICTURE = re.compile(
    r"\[\s*(?:you\s+)?(?:sends?|sent)?\s*(?:a\s+)?(?:picture|photo|selfie|image|pic)\s*:\s*([^\]]*)\]?",
    re.IGNORECASE)


def serialize(row: dict[str, Any]) -> dict[str, Any]:
    """A message as the page shows it."""
    return {
        "id": row["id"],
        "chat_id": row.get("chat_id"),
        "role": row["role"],
        "content": row["content"],
        "image_url": f"/api/images/{row['image_id']}" if row.get("image_id") else None,
        "image_kind": row.get("image_kind"),
        "created_at": row["created_at"].isoformat(),
    }


def serialize_companion(row: dict[str, Any]) -> dict[str, Any]:
    """A girlfriend as the page shows her; her memory's progress is added by the web layer."""
    last = row.get("last_active")
    return {
        "id": row["id"],
        "name": row["name"],
        "persona": clean_persona(row.get("persona") or {}),
        "voice": row["voice"],
        "memory": row.get("memory") or "",
        "chats": int(row.get("chats") or 0),
        "last_active": last.isoformat() if last else None,
        "created_at": row["created_at"].isoformat(),
    }


def serialize_chat(row: dict[str, Any]) -> dict[str, Any]:
    """A chat as the page lists it."""
    last = (row.get("last_content") or "").strip() or ("📷 Picture" if row.get("last_image_id") else "")
    if last and row.get("last_role") == "user":
        last = f"You: {last}"
    return {
        "id": row["id"],
        "companion_id": row.get("companion_id"),
        "title": row["title"] or "New chat",
        "last": last,
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def chat_title(text: str, photo: bool) -> str:
    """A chat's name, from its first message."""
    words = " ".join(text.split())
    if not words:
        return "Photo" if photo else ""
    limit = SociaConfig.CHAT_TITLE_CHARS
    return words if len(words) <= limit else words[:limit - 1].rstrip() + "…"


NAME_PROMPT = """Here is the start of a private chat between {name} and {her}, {name}'s girlfriend.

{transcript}

Give this chat a short name, two to five words, the way someone would label it in their chat list: what it is
about or its mood, such as "Rainy Sunday plans" or "Missing you tonight". Answer with the name only: no quotes,
no full stop, no emoji, no explanation."""
NAME_WORDS = 6
# How often chats that could not be named are tried again.
NAME_RECHECK_SECONDS = 600


def clean_name(reply: str) -> str:
    """The model's name for a chat, as one short line; '' if it gave nothing usable."""
    line = next((line for line in reply.splitlines() if line.strip()), "")
    line = line.strip().strip("\"'`*#").removeprefix("Name:").strip().strip("\"'`*").rstrip(".!")
    words = line.split()[:NAME_WORDS]
    name = " ".join(words)
    return chat_title(name, False) if name else ""


def model_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as the chat model reads it: photos and pictures become notes in brackets.

    A reply she already sent once is left out the second time: a model that sees itself
    repeat a line (a refusal, say) tends to send it again and again.
    """
    messages, said = [], set()
    for row in rows:
        content = row["content"]
        if row["role"] == "user":
            if row.get("photo_note"):
                content = f"{content}\n[Photo: {row['photo_note']}]".strip()
            messages.append({"role": "user", "content": content})
        else:
            content = without_written_pictures(content)
            picture = row.get("image_prompt") if row.get("image_kind") == "generated" else None
            key = " ".join(f"{content} {picture or ''}".split()).casefold()
            if key in said:
                continue
            said.add(key)
            if picture:
                # Her pictures appear as the create_image calls they were, so she learns to
                # call the tool again rather than to write a picture out in words.
                call = {"function": {"name": "create_image", "arguments": {
                    "description": picture, "include_luna": True, "use_their_photo": False}}}
                messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                messages.append({"role": "tool", "content": "The picture was sent."})
            else:
                messages.append({"role": "assistant", "content": content})
    return messages


def without_written_pictures(text: str) -> str:
    """Her text without any picture she wrote out in brackets."""
    return "\n".join(line.rstrip() for line in WRITTEN_PICTURE.sub("", text).splitlines()).strip()


def written_picture(text: str) -> dict[str, Any] | None:
    """A create_image call for a picture she wrote out in brackets, if she did."""
    match = WRITTEN_PICTURE.search(text)
    if not match or not match.group(1).strip():
        return None
    return {"function": {"name": "create_image", "arguments": {
        "description": match.group(1).strip(), "include_luna": True, "use_their_photo": False}}}


def normalized_photo(data: bytes) -> bytes:
    """A photo upright, at most PHOTO_MAX_SIDE, as JPEG."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    image.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


class Conversation:
    def __init__(self, config: SociaConfig, store: SociaStore, ollama: Ollama, comfy: ComfyUI) -> None:
        self.config = config
        self.store = store
        self.ollama = ollama
        self.comfy = comfy
        # One reply at a time, as in a real conversation.
        self.turn = asyncio.Lock()

    def save_file(self, data: bytes, suffix: str) -> str:
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{suffix}"
        (self.config.images_dir / name).write_bytes(data)
        return name

    async def send(self, companion: dict[str, Any], chat_id: int | None, text: str, photo: bytes | None, name: str,
                   timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        """Store what was sent, then stream her answer as events for the page.

        Without a chat, a new one with her is started and named after this message.
        """
        async with self.turn:
            if chat_id is None:
                chat = await asyncio.to_thread(self.store.create_chat, companion["id"])
                chat_id = chat["id"]
            await asyncio.to_thread(self.store.title_chat, chat_id, chat_title(text, bool(photo)))
            chat = await asyncio.to_thread(self.store.chat, chat_id)
            yield {"type": "chat", "chat": serialize_chat(chat)}
            image_id = note = None
            if photo:
                photo = await asyncio.to_thread(normalized_photo, photo)
                image_id = await asyncio.to_thread(self.store.add_image, chat_id, "photo", self.save_file(photo, ".jpg"))
            sent = await asyncio.to_thread(self.store.add_message, chat_id, "user", text, image_id)
            yield {"type": "user", "message": serialize(sent)}
            if photo:
                yield {"type": "status", "text": "Looking at your photo"}
                try:
                    note = await self.ollama.describe_photo(photo)
                except Exception as error:  # noqa: BLE001 - she answers without seeing it
                    logger.warning("Could not describe the photo: %s", error)
                    note = "(the photo did not load for you)"
                await asyncio.to_thread(self.store.set_photo_note, sent["id"], note)
            async for event in self.answer(companion, chat_id, name, timezone):
                yield event
            if chat["named"] == "":
                # Her first answer is in: name the chat after how it began.
                named = await self.name_chat(chat_id, companion["name"], name)
                if named:
                    yield {"type": "chat", "chat": serialize_chat(named)}

    async def name_chat(self, chat_id: int, her: str, name: str = "You") -> dict[str, Any] | None:
        """A short name for a chat from its first messages, by the chat model; None if it gave none."""
        rows = await asyncio.to_thread(self.store.opening, chat_id)
        lines = []
        for row in rows:
            content = " ".join((row["content"] or "").split())[:300] or ("(a photo)" if row["image_id"] else "")
            if content:
                lines.append(f"{her if row['role'] == 'luna' else name}: {content}")
        if not lines:
            return None
        try:
            reply = await self.ollama.complete(NAME_PROMPT.format(name=name, her=her, transcript="\n".join(lines)))
        except Exception as error:  # noqa: BLE001 - the first message stays its name
            logger.warning("Could not name chat %s: %s", chat_id, error)
            return None
        title = clean_name(reply)
        return await asyncio.to_thread(self.store.name_chat, chat_id, title) if title else None

    async def name_old_chats(self) -> None:
        """Keep naming chats that have no name yet: those from before chats were named, and any the model missed."""
        while True:
            try:
                await self.name_waiting_chats()
            except Exception:  # noqa: BLE001 - tried again later
                logger.exception("Naming old chats failed")
            await asyncio.sleep(NAME_RECHECK_SECONDS)

    async def name_waiting_chats(self) -> None:
        """Name each chat without a name, one at a time, between her replies.

        A chat whose name is not its first message was renamed by the owner, so it keeps it.
        """
        for chat in await asyncio.to_thread(self.store.unnamed_chats):
            first = next(iter(await asyncio.to_thread(self.store.opening, chat["id"], 1)), None)
            stand_in = chat_title(first["content"] or "", bool(first["image_id"])) if first else ""
            if chat["title"] and chat["title"] != stand_in:
                await asyncio.to_thread(self.store.keep_name, chat["id"])
                continue
            companion = await asyncio.to_thread(self.store.companion, chat["companion_id"])
            async with self.turn:
                await self.name_chat(chat["id"], companion["name"] if companion else "Luna")

    def now(self, timezone: str | None) -> datetime:
        try:
            return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now()
        except (KeyError, ValueError):
            return datetime.now()

    async def looks(self, companion_id: int) -> str:
        """How she looks, read once from her profile picture and again when it changes."""
        profile = self.config.profile_path(companion_id)
        notes = profile.with_suffix(".txt")
        if not profile.exists():
            return ""
        if notes.exists() and notes.stat().st_mtime >= profile.stat().st_mtime:
            return notes.read_text()
        try:
            looks = await self.ollama.describe_photo(profile.read_bytes(), LOOKS_PROMPT)
        except Exception as error:  # noqa: BLE001 - she can talk without it
            logger.warning("Could not describe the profile picture: %s", error)
            return ""
        notes.write_text(looks)
        return looks

    async def answer(self, companion: dict[str, Any], chat_id: int, name: str,
                     timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        rows = await asyncio.to_thread(self.store.recent, chat_id, SociaConfig.HISTORY_MESSAGES)
        prompt = system_prompt(companion, name, self.now(timezone), await self.looks(companion["id"]))
        messages = [{"role": "system", "content": prompt}, *model_messages(rows)]
        text, calls = "", []
        try:
            async for piece in self.ollama.chat(messages, TOOLS):
                if "text" in piece:
                    text += piece["text"]
                    yield {"type": "text", "text": piece["text"]}
                calls += piece.get("tool_calls", [])
        except Exception as error:  # noqa: BLE001 - shown in the chat
            logger.exception("Chat model failed")
            yield {"type": "error", "text": f"{companion['name']} could not answer: {error}"}
            return

        image_id = None
        call = next((c for c in calls if (c.get("function") or {}).get("name") == "create_image"), None)
        if call is None:
            call = written_picture(text)
        text = without_written_pictures(text)
        if call:
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            yield {"type": "image_started"}
            try:
                image_id = await self.picture(companion["id"], chat_id, arguments)
                yield {"type": "image", "url": f"/api/images/{image_id}"}
            except Exception as error:  # noqa: BLE001 - shown in the chat
                logger.exception("Image failed")
                yield {"type": "error", "text": f"The picture did not come out: {error}"}
            if not text.strip():
                # She only sent the picture; let her say something with it.
                follow = [*messages, {"role": "assistant", "content": "", "tool_calls": [call]},
                          {"role": "tool", "content": "The picture was sent." if image_id else "The camera failed."}]
                try:
                    async for piece in self.ollama.chat(follow):
                        if "text" in piece:
                            text += piece["text"]
                            yield {"type": "text", "text": piece["text"]}
                except Exception:  # noqa: BLE001 - the picture alone is enough
                    logger.exception("Caption failed")

        saved = await asyncio.to_thread(self.store.add_message, chat_id, "luna", text.strip(), image_id)
        yield {"type": "done", "message": serialize(saved)}

    async def picture(self, companion_id: int, chat_id: int, arguments: dict[str, Any]) -> int:
        description = str(arguments.get("description") or "").strip() or "a cosy selfie"
        profile = self.config.profile_path(companion_id)
        # Her profile picture first, then her other pictures; none without a profile picture.
        lunas = [profile, *self.config.references(companion_id)] if arguments.get("include_luna") and profile.exists() else []
        photo = await asyncio.to_thread(self.store.last_photo, chat_id) if arguments.get("use_their_photo") else None
        prompt, order = image_prompt(description, len(lunas), photo is not None)
        pictures = {"luna": iter(lunas), "photo": iter([self.config.images_dir / photo["file"]] if photo else [])}
        references = [next(pictures[name]).read_bytes() for name in order]
        # Ollama's models out of the GPU first; ComfyUI frees its own when done.
        await self.ollama.unload_all()
        image = await self.comfy.generate(prompt, references)
        file = await asyncio.to_thread(self.save_file, image, ".png")
        return await asyncio.to_thread(self.store.add_image, chat_id, "generated", file, description)
