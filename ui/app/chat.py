"""One turn of the conversation: what Luna sees, what she says, and the pictures she sends."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps

from app.config import SociaConfig
from app.images import ComfyUI
from app.ollama import LOOKS_PROMPT, Ollama
from app.persona import TOOLS, image_prompt, system_prompt
from app.store import SociaStore

logger = logging.getLogger("socia")
# Photos are kept at most this size on their longer side; the image model uses about 1024.
PHOTO_MAX_SIDE = 2048


def serialize(row: dict[str, Any]) -> dict[str, Any]:
    """A message as the page shows it."""
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "image_url": f"/api/images/{row['image_id']}" if row.get("image_id") else None,
        "image_kind": row.get("image_kind"),
        "created_at": row["created_at"].isoformat(),
    }


def model_messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as the chat model reads it: photos and pictures become notes in brackets."""
    messages = []
    for row in rows:
        content = row["content"]
        if row["role"] == "user":
            if row.get("photo_note"):
                content = f"{content}\n[Photo: {row['photo_note']}]".strip()
            messages.append({"role": "user", "content": content})
        else:
            if row.get("image_kind") == "generated" and row.get("image_prompt"):
                content = f"{content}\n[You sent a picture: {row['image_prompt']}]".strip()
            messages.append({"role": "assistant", "content": content})
    return messages


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

    async def send(self, text: str, photo: bytes | None, name: str, timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        """Store what was sent, then stream Luna's answer as events for the page."""
        async with self.turn:
            image_id = note = None
            if photo:
                photo = await asyncio.to_thread(normalized_photo, photo)
                image_id = await asyncio.to_thread(self.store.add_image, "photo", self.save_file(photo, ".jpg"))
            sent = await asyncio.to_thread(self.store.add_message, "user", text, image_id)
            yield {"type": "user", "message": serialize(sent)}
            if photo:
                yield {"type": "status", "text": "Looking at your photo"}
                try:
                    note = await self.ollama.describe_photo(photo)
                except Exception as error:  # noqa: BLE001 - she answers without seeing it
                    logger.warning("Could not describe the photo: %s", error)
                    note = "(the photo did not load for you)"
                await asyncio.to_thread(self.store.set_photo_note, sent["id"], note)
            async for event in self.answer(name, timezone):
                yield event

    def now(self, timezone: str | None) -> datetime:
        try:
            return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now()
        except (KeyError, ValueError):
            return datetime.now()

    async def looks(self) -> str:
        """How Luna looks, read once from her profile picture and again when it changes."""
        profile, notes = self.config.profile_path, self.config.profile_path.with_suffix(".txt")
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

    async def answer(self, name: str, timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        rows = await asyncio.to_thread(self.store.recent, SociaConfig.HISTORY_MESSAGES)
        prompt = system_prompt(name, self.now(timezone), await self.looks())
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
            yield {"type": "error", "text": f"Luna could not answer: {error}"}
            return

        image_id = None
        call = next((c for c in calls if (c.get("function") or {}).get("name") == "create_image"), None)
        if call:
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            yield {"type": "image_started"}
            try:
                image_id = await self.picture(arguments)
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

        saved = await asyncio.to_thread(self.store.add_message, "luna", text.strip(), image_id)
        yield {"type": "done", "message": serialize(saved)}

    async def picture(self, arguments: dict[str, Any]) -> int:
        description = str(arguments.get("description") or "").strip() or "a cosy selfie"
        profile = self.config.profile_path
        with_luna = bool(arguments.get("include_luna")) and profile.exists()
        photo = await asyncio.to_thread(self.store.last_photo) if arguments.get("use_their_photo") else None
        prompt, order = image_prompt(description, with_luna, photo is not None)
        files = {"luna": profile, "photo": self.config.images_dir / photo["file"] if photo else None}
        references = [files[name].read_bytes() for name in order]
        # Ollama's models out of the GPU first; ComfyUI frees its own when done.
        await self.ollama.unload_all()
        image = await self.comfy.generate(prompt, references)
        file = await asyncio.to_thread(self.save_file, image, ".png")
        return await asyncio.to_thread(self.store.add_image, "generated", file, description)
