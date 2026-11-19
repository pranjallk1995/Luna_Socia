"""One turn of a conversation: what she sees, what she says, and the pictures she sends."""

from __future__ import annotations

import asyncio
import base64
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

from app.apps import APPS, DIRECT_SEPARATOR, LunaApps, app_tools_spec, apps_prompt, is_app_call
from app.config import SociaConfig
from app.faces import Faces, faces_line
from app.images import ComfyUI
from app.ollama import LOOKS_PROMPT, Ollama
from app.persona import REACTIONS, TOOLS, clean_persona, image_prompt, system_prompt
from app.store import SociaStore

logger = logging.getLogger("socia")
# Photos are kept at most this size on their longer side; the image model uses about 1024.
PHOTO_MAX_SIDE = 2048
# A picture she wrote out in brackets ("[You sent a picture: ...]") instead of calling
# create_image: it is taken out of her text and taken as the call she meant.
WRITTEN_PICTURE = re.compile(
    r"\[\s*(?:you\s+)?(?:sends?|sent)?\s*(?:a\s+)?(?:picture|photo|selfie|image|pic)\s*:\s*([^\]]*)\]?",
    re.IGNORECASE)
# A note she wrote out (note("the moon")) instead of calling the tool: taken out and kept as one.
# A reaction she wrote out ("react: ❤️" on a line of its own) instead of calling the tool.
WRITTEN_REACTION = re.compile(r"^[ \t]*react(?:ion)?\s*[:(]\s*[\"']?([^\s\"')]+)[\"')]*[ \t]*$\n?", re.IGNORECASE | re.MULTILINE)
# A remember call she typed out ('remember {"about":"them","fact":"..."}') instead of making it.
WRITTEN_REMEMBER = re.compile(r"\bremember\s*(\{[^{}]*\})", re.IGNORECASE)
WRITTEN_NOTE = re.compile(r"\bnote\(\s*(?:text\s*=\s*)?[\"'“]?(.*?)[\"'”]?\s*\)", re.IGNORECASE)


# Words that mean she is to pick something and keep it secret (a guessing game, a riddle, a surprise).
SECRET_HINT = re.compile(
    r"\b(?:20|twenty|21)\s*questions\b|\bguess(?:ing)?\b|\b(?:think|thinking) of (?:something|someone|a|an)\b|"
    r"\b(?:pick|choose|imagine) (?:something|someone|a|an)\b|\bsecret\b|\briddle\b|\bsurprise\b|\bi spy\b|"
    r"\bwould you rather\b|\btruth or dare\b|\bgame\b", re.IGNORECASE)
# ...unless the owner is the one who picked it, and she is to guess.
OWNER_PICKED = re.compile(
    r"\bi(?:'m| am)\s+thinking\b|\bi(?:'ve| have)?\s+(?:picked|chosen|thought of|got)\b|\bi\s+(?:picked|chose)\b|"
    r"\byou\s+(?:guess|ask)\b|\bguess what i\b", re.IGNORECASE)
# A message about a photo or picture just sent: the vision model then looks at it again to answer.
ABOUT_PICTURE = re.compile(
    r"\b(?:photos?|pics?|pictures?|images?|selfies?|circled?|marked|arrow|differen(?:t|ce)|compare|look(?:ing)? (?:at|closely|again)|"
    r"zoom|in (?:it|this|that)|this one|that one|the one above|who(?:'s| is| are) (?:this|that|there|in)|"
    r"which (?:one|is better))\b", re.IGNORECASE)
# The owner's words asking her for a picture while she looks at photos: a selfie or photo of her, herself put
# into their photo, or their photo changed. Without them, a photo gets words only: the vision model, given the
# camera, took a picture of its own for a plain "LOOK!!!" with a photo.
ASKS_FOR_PICTURE = re.compile(
    r"\b(?:selfies?|pics?|pictures?|photos?|nudes?|snaps?)\s+(?:of\s+)?(?:you|yourself|us|both of us)\b|"
    r"\b(?:send|take|click|snap|make|draw|create|share)\b[^.?!]{0,40}\b(?:selfies?|pics?|pictures?|photos?|nudes?|images?)\b|"
    r"\bshow (?:me )?(?:you|yourself)\b|\byou and me\b|\bus together\b|\bwith me in\b|"
    r"\b(?:put|add|place|insert|photoshop)\b[^.?!]{0,30}\b(?:you|yourself)\b|"
    r"\b(?:join|step|get|be|jump|come)\b[^.?!]{0,20}\b(?:in|into)\b[^.?!]{0,20}\b(?:this|it|photo|pic|picture|frame|here)\b|"
    r"\b(?:improve|edit|retouch|enhance|fix|change|brighten|redo|recreate)\b[^.?!]{0,30}\b(?:this|it|photo|pic|picture|"
    r"image|lighting|light|background|colou?rs?)\b", re.IGNORECASE)
# How far back the photo a message asks about may be.
PICTURE_ASKED_WITHIN = 4
SEEING = """

Right now you see the pictures themselves: they are attached to {name}'s last message, in the order it lists. Look at
them closely and answer from what you see: who is in them, faces, clothes, anything drawn or written on them, and how
they differ. Under each picture, "Faces:" says whose faces are in it, from face recognition: it is right, so go by it,
not by how a face looks to you. You are in a picture only when its "Faces:" says "you"; a woman it does not call you
is not you, however much she looks like you, and with no face visible you cannot tell who it is. If you said earlier
that one of these was you and its "Faces:" disagrees, you were wrong: say so. Earlier photos you know only from the
descriptions in brackets."""
# Rounds of app use one reply may take: listing an app's tools, reading one, using it, and a retry.
MAX_APP_ROUNDS = 5
SECRET_NUDGE = ("[If you are the one to pick a secret here (the thing to guess, an answer), pick it now and call "
                "note with the secret itself, such as note(\"Secret: a penguin\"), before you reply.]")


def with_secret_nudge(messages: list[dict[str, Any]], notes: str) -> list[dict[str, Any]]:
    """Her messages with a reminder to write a secret down, when the owner's last one starts a game
    and she has no notes yet; a reminder in the system prompt alone she tends to pass over."""
    if notes.strip() or not messages or messages[-1]["role"] != "user":
        return messages
    said = messages[-1]["content"]
    if not SECRET_HINT.search(said) or OWNER_PICKED.search(said):
        return messages
    return [*messages[:-1], {**messages[-1], "content": f"{messages[-1]['content']}\n{SECRET_NUDGE}"}]


def serialize(row: dict[str, Any]) -> dict[str, Any]:
    """A message as the page shows it."""
    return {
        "id": row["id"],
        "chat_id": row.get("chat_id"),
        "role": row["role"],
        "content": row["content"],
        "image_url": f"/api/images/{row['image_id']}" if row.get("image_id") else None,
        "image_kind": row.get("image_kind"),
        "reaction": row.get("reaction"),
        "apps": row.get("apps") or [],
        "created_at": row["created_at"].isoformat(),
        "edited": row.get("edited_at") is not None,
        # What she saved with this reply, each with its place in the list (to forget it by); not the forgotten.
        "remembered": [{"index": index, "about": item.get("about"), "fact": item.get("fact")}
                       for index, item in enumerate(row.get("remembered") or []) if not item.get("forgotten")],
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


# Before she answers, which app (if any) the owner's message needs, and which few of its tools: a small
# chat model rarely starts on the apps by itself, so the code takes that first step for her.
ROUTE_PROMPT = """{name}'s apps:
{apps}

The end of {name}'s chat with {her}, their girlfriend:
{transcript}

Does {name}'s last message ask {her} to use, check or look in one of these apps right now (play music, find or look
at photos, lights, meals, money, documents, the server)? Answer with the app's key only (one of: {keys}), or NONE."""
PICK_PROMPT = """{name} asked {her}: "{message}"

{app}'s tools:
{lines}

Which tools does {her} need for this, at most three, most useful first? Answer with their exact names, comma-separated,
and nothing else."""
CHOSEN_APP = ("\n\nFor {name}'s last message, use {app}: your functions {functions} do it. Call the right one now, "
              "in this reply, then tell {name} what it returned, in your own words. Never say you opened, checked or "
              "saw something in an app without calling it.")
# A tool call she wrote into her text instead of making it, as small models sometimes do:
# {"name": ..., "arguments": {...}}, with or without <tool_call> tags around it.
WRITTEN_CALL = re.compile(r"(?:</?tool_call>\s*)*(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:.*\})\s*(?:</?tool_call>)*",
                          re.DOTALL)
CALL_MARKER = re.compile(r"</?tool_call>|\{\s*\"name\"\s*:|\bremember\s*\{", re.IGNORECASE)


def written_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """The tool calls she wrote out as text, and her text without them (and without stray tags)."""
    calls = []
    for match in WRITTEN_CALL.finditer(text):
        try:
            call = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            calls.append({"function": {"name": call["name"], "arguments": call.get("arguments") or {}}})
    cleaned = re.sub(r"</?tool_call>", "", WRITTEN_CALL.sub("", text)) if calls or CALL_MARKER.search(text) else text
    return calls, cleaned.strip()


def shown_part(said: str, shown: int) -> str:
    """What of her reply so far can go to the page: everything before a tool call she started writing out,
    holding back a tail that could be the start of one."""
    marker = CALL_MARKER.search(said)
    if marker:
        end = marker.start()
    else:
        # A "<" or "{" near the end may be a tool call still arriving: wait for the next piece.
        tail = max(len(said) - len("</tool_call>"), 0)
        start = max(said.rfind("<", tail), said.rfind("{", tail))
        end = start if start >= shown else len(said)
    return said[shown:end] if end > shown else ""


NAME_PROMPT = """Here is the start of a private chat between {name} and {her}, {name}'s girlfriend.

{transcript}

Give this chat a short name, two to five words, the way someone would label it in their chat list: what it is
about or its mood, such as "Rainy Sunday plans" or "Missing you tonight". Answer with the name only: no quotes,
no full stop, no emoji, no explanation."""
NAME_WORDS = 6
# Her own words saying a picture of her is sent or on its way. Only her reply is read: the owner talking
# about photos (or sending one) is no promise, and checking on those words sent pictures nobody asked for.
PICTURE_PROMISE = re.compile(
    r"\b(?:here (?:it is|you go|i am)|(?:just )?sent (?:you|it)|sending (?:you|it|this|one)|resend|"
    r"let me (?:show|send|take|snap|get)|i'?ll (?:send|show|take|snap)|"
    r"(?:a|this|my|one|another) (?:pic|pics|picture|photo|selfie|nude|nudes)(?! of (?:her|him|them|it|your))|there!)",
    re.IGNORECASE)
PROMISE_PROMPT = """The end of a private chat between {name} and {her}, {name}'s girlfriend:

{transcript}

Does {her}'s last message send a picture of herself, or say she is sending or about to send one now? If not,
answer only NONE. If it does, describe that picture in detail in English, as {her} would for her camera: the place,
the light, what she wears (or that she is naked, if that is what was asked for), her pose and expression, and the
camera (selfie, mirror shot, candid). Answer only the description, starting with "I"."""
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
    repeat a line (a refusal, say) tends to send it again and again. The owner's reactions to
    her messages come before their next message, as a note in brackets.
    """
    messages, said, reacted = [], set(), []
    for row in rows:
        content = row["content"]
        if row["role"] == "user":
            if row.get("photo_note"):
                content = f"{content}\n[Photo: {row['photo_note']}]".strip()
            content = "\n".join([*reacted, content]).strip()
            reacted = []
            messages.append({"role": "user", "content": content})
        else:
            content = without_written_pictures(content)
            if row.get("reaction"):
                said_what = " ".join(content.split())[:80] or "your picture"
                reacted.append(f'[Reacted {row["reaction"]} to your message: "{said_what}"]')
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
    """Her text without any picture she wrote out in brackets, or note she wrote out."""
    text = WRITTEN_NOTE.sub("", WRITTEN_PICTURE.sub("", text))
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def written_reactions(text: str) -> tuple[list[dict[str, Any]], str]:
    """react calls for the reactions she wrote out instead of calling the tool, and her text without them."""
    calls = [{"function": {"name": "react", "arguments": {"emoji": match.group(1)}}}
             for match in WRITTEN_REACTION.finditer(text)]
    return calls, WRITTEN_REACTION.sub("", text).strip() if calls else text


def written_remembers(text: str) -> tuple[list[dict[str, Any]], str]:
    """remember calls she typed out instead of making them, and her text without them."""
    calls = []
    for match in WRITTEN_REMEMBER.finditer(text):
        try:
            arguments = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(arguments, dict) and arguments.get("fact"):
            calls.append({"function": {"name": "remember", "arguments": arguments}})
    return calls, WRITTEN_REMEMBER.sub("", text).strip() if calls else text


def written_notes(text: str) -> list[dict[str, Any]]:
    """note calls for the notes she wrote out instead of calling the tool."""
    return [{"function": {"name": "note", "arguments": {"text": match.group(1).strip()}}}
            for match in WRITTEN_NOTE.finditer(text) if match.group(1).strip()]


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


def sees_pictures(rows: list[dict[str, Any]]) -> bool:
    """Whether she is to see the pictures themselves for this reply: the owner's last message has a photo,
    or asks about a photo or picture of the last few messages."""
    if not rows or rows[-1]["role"] != "user":
        return False
    if rows[-1].get("image_id"):
        return True
    recent = rows[-PICTURE_ASKED_WITHIN - 1:-1]
    return any(row.get("image_file") for row in recent) and bool(ABOUT_PICTURE.search(rows[-1]["content"] or ""))


def seen_size(data: bytes, side: int) -> bytes:
    """A picture at most ``side`` on its longer side, as JPEG: what the vision model reads."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def ago(count: int) -> str:
    return "just now" if count == 0 else "1 message ago" if count == 1 else f"{count} messages ago"


class Conversation:
    def __init__(self, config: SociaConfig, store: SociaStore, ollama: Ollama, comfy: ComfyUI,
                 apps: LunaApps | None = None, faces: Faces | None = None, rememberer: Any = None,
                 recall: Any = None) -> None:
        self.config = config
        # Decides after each text turn what is worth remembering (app/remembering.py); None leaves it to her tool.
        self.rememberer = rememberer
        self.recall = recall
        self.upkeeps: set[asyncio.Task] = set()
        # Face recognition, paired with the vision model: whose faces are in the pictures it sees.
        self.faces = faces
        self.store = store
        self.ollama = ollama
        self.comfy = comfy
        # The owner's Luna apps, once connected.
        self.apps = apps
        # One reply at a time, as in a real conversation.
        self.turn = asyncio.Lock()
        # Whether the last reply looked at pictures (with the vision model), and work left running after a reply.
        self.saw = False
        self.background: set[asyncio.Task] = set()

    def save_file(self, data: bytes, suffix: str) -> str:
        self.config.images_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{suffix}"
        (self.config.images_dir / name).write_bytes(data)
        return name

    async def send(self, companion: dict[str, Any], chat_id: int | None, text: str,
                   photo: bytes | list[bytes] | None, name: str, timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        """Store what was sent, then stream her answer as events for the page.

        Several photos (up to PHOTOS_SEEN, as many as she sees at once) are a message each, the text with
        the last, and she answers them together. Without a chat, a new one with her is started and named
        after this message.
        """
        photos = [p for p in (photo if isinstance(photo, list) else [photo]) if p][:SociaConfig.PHOTOS_SEEN]
        async with self.turn:
            if chat_id is None:
                chat = await asyncio.to_thread(self.store.create_chat, companion["id"])
                chat_id = chat["id"]
            await asyncio.to_thread(self.store.title_chat, chat_id, chat_title(text, bool(photos)))
            chat = await asyncio.to_thread(self.store.chat, chat_id)
            yield {"type": "chat", "chat": serialize_chat(chat)}
            sent_photos = []
            for index, data in enumerate(photos):
                data = await asyncio.to_thread(normalized_photo, data)
                image_id = await asyncio.to_thread(self.store.add_image, chat_id, "photo", self.save_file(data, ".jpg"))
                last = index == len(photos) - 1
                sent = await asyncio.to_thread(self.store.add_message, chat_id, "user", text if last else "", image_id)
                sent_photos.append((sent, data))
                yield {"type": "user", "message": serialize(sent)}
            if not photos:
                sent = await asyncio.to_thread(self.store.add_message, chat_id, "user", text, None)
                yield {"type": "user", "message": serialize(sent)}
            if photos:
                yield {"type": "status", "text": "Looking at your photo" + ("s" if len(photos) > 1 else "")}
            async for event in self.answer(companion, chat_id, name, timezone):
                yield event
            for sent, data in sent_photos:
                # She saw the photo itself; a description of it stays for her later replies, which only read text.
                # The vision model is still loaded, so this takes a second.
                try:
                    note = await self.ollama.describe_photo(data)
                except Exception as error:  # noqa: BLE001 - later replies go without it
                    logger.warning("Could not describe the photo: %s", error)
                    note = "(the photo did not load for you)"
                people = await asyncio.to_thread(self.faces_in, data, companion, name or "love")
                if people:
                    note = f"{note} {people}"
                await asyncio.to_thread(self.store.set_photo_note, sent["id"], note)
            if self.saw:
                # The vision model took the chat model's place on the GPU: swap back while the owner reads.
                task = asyncio.create_task(self.back_to_chat())
                self.background.add(task)
                task.add_done_callback(self.background.discard)
            if chat["named"] == "":
                # Her first answer is in: name the chat after how it began.
                named = await self.name_chat(chat_id, companion["name"], name)
                if named:
                    yield {"type": "chat", "chat": serialize_chat(named)}

    async def choose_app_tools(self, rows: list[dict[str, Any]], running: list[Any], her: str,
                               name: str) -> tuple[Any, list[dict[str, Any]]] | None:
        """The app the owner's last message needs and its few tools as her functions, or None."""
        if not rows or rows[-1]["role"] != "user" or not (rows[-1]["content"] or "").strip():
            return None
        transcript = "\n".join(f"{her if row['role'] == 'luna' else name}: {' '.join((row['content'] or '').split())[:300]}"
                               for row in rows[-4:] if row["content"])
        keys = [app.key for app in running]
        try:
            answer = await self.ollama.complete(ROUTE_PROMPT.format(
                name=name, her=her, transcript=transcript, keys=", ".join(keys),
                apps="\n".join(f"- {app.key}: {app.purpose}" for app in running)))
            key = next((k for k in keys if re.search(rf"\b{k}\b", answer.lower())), None)
            if key is None:
                return None
            lines = await self.apps.tool_lines(key)
            picked = await self.ollama.complete(PICK_PROMPT.format(
                name=name, her=her, message=" ".join(rows[-1]["content"].split())[:300], app=APPS[key].name, lines=lines))
            names = [n.strip(" `'\"-.") for n in re.split(r"[,\n]", picked) if n.strip(" `'\"-.")][:3]
            specs = await self.apps.direct_specs(key, names)
        except Exception as error:  # noqa: BLE001 - she answers with the three app tools alone
            logger.warning("Could not choose an app for the message: %s", error)
            return None
        return (APPS[key], specs) if specs else None

    async def back_to_chat(self) -> None:
        """Her chat model back on the GPU after a reply that looked at pictures, before her next reply."""
        async with self.turn:
            if not self.saw:
                return
            try:
                await self.ollama.back_to_chat()
                self.saw = False
            except Exception as error:  # noqa: BLE001 - her next reply loads it instead
                logger.warning("Could not bring her chat model back: %s", error)
            finally:
                # The face models came with the vision model, and leave with it.
                if self.faces is not None:
                    await asyncio.to_thread(self.faces.unload)

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

    def faces_in(self, picture: bytes, companion: dict[str, Any], name: str) -> str:
        """Whose faces are in a picture, by face recognition: her (her profile picture and other pictures)
        and the owner (their photos), or someone else; '' without face recognition."""
        if self.faces is None:
            return ""
        her = companion["name"]
        profile = self.config.profile_path(companion["id"])
        people = {her: [profile, *self.config.references(companion["id"])] if profile.exists() else [],
                  name: self.config.owner_faces()}
        try:
            return faces_line(self.faces.who(picture, people), her, name)
        except Exception as error:  # noqa: BLE001 - the vision model looks without it
            logger.warning("Could not read the faces: %s", error)
            return ""

    def with_pictures(self, messages: list[dict[str, Any]], rows: list[dict[str, Any]], companion: dict[str, Any],
                      name: str) -> list[dict[str, Any]]:
        """Her messages with the newest pictures of the chat attached to the owner's last message, and a list
        there of what each one is, with whose faces face recognition finds in it.

        Her profile picture is not attached: shown as picture 1, she took it for one of the owner's photos
        and said "the first one is me" of every set he sent. Face recognition tells her whose faces she sees."""
        pictures, labels = [], []
        # The owner's last messages together (several photos sent at once) were all sent just now.
        together = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["role"] != "user"), -1) + 1
        shown = [(0 if index >= together else len(rows) - 1 - index, row)
                 for index, row in enumerate(rows) if row.get("image_file")]
        for back, row in shown[-SociaConfig.PHOTOS_SEEN:]:
            path = self.config.images_dir / row["image_file"]
            if not path.exists():
                continue
            data = path.read_bytes()
            pictures.append(seen_size(data, SociaConfig.PHOTO_SEEN_SIDE))
            if row.get("image_kind") == "generated":
                label = f"the picture you sent of yourself {ago(back)}"
            else:
                said = " ".join((row["content"] or "").split())[:80]
                label = f"the photo {name} sent {ago(back)}" + (f' with "{said}"' if said else "")
            people = self.faces_in(data, companion, name)
            labels.append(f"{label}. {people}" if people else label)
        if not pictures:
            return messages
        listed = "; ".join(f"{number}) {label}" for number, label in enumerate(labels, start=1))
        last = messages[-1]
        return [*messages[:-1], {**last, "content": f"{last['content']}\n[Pictures you see now: {listed}]".strip(),
                                 "images": [base64.b64encode(picture).decode() for picture in pictures]}]

    async def answer(self, companion: dict[str, Any], chat_id: int, name: str,
                     timezone: str | None) -> AsyncIterator[dict[str, Any]]:
        rows = await asyncio.to_thread(self.store.recent, chat_id, SociaConfig.HISTORY_MESSAGES)
        chat = await asyncio.to_thread(self.store.chat, chat_id)
        notes = (chat or {}).get("notes") or ""
        if sees_pictures(rows):
            # The vision model does not keep a secret as her chat model does: it told the owner the one she
            # picked for a guessing game while answering a photo. Her notes wait for her next text reply.
            notes = ""
        owner = await asyncio.to_thread(self.store.owner_profile)
        if self.recall is not None and companion.get("memory"):
            # Her core memories, and the details your message is about (app/recall.py).
            said = (rows[-1]["content"] or "") if rows and rows[-1]["role"] == "user" else ""
            try:
                companion = {**companion, "memory": await self.recall.memory(companion, " ".join(said.split()))}
            except Exception:  # noqa: BLE001 - she reads all of it instead
                logger.exception("Could not recall by topic")
        prompt = system_prompt(companion, name, self.now(timezone), await self.looks(companion["id"]), notes,
                               owner.get("about") or "")
        tools = TOOLS
        seeing = self.saw = sees_pictures(rows)
        may_draw = True
        if seeing:
            # A photo, or a question about one: the vision model looks at the pictures and answers as her,
            # with the last few messages that fit beside them. It gets the camera only when asked for a picture.
            rows = rows[-SociaConfig.PHOTO_HISTORY_MESSAGES:]
            prompt += SEEING.format(name=name or "love")
            may_draw = bool(ASKS_FOR_PICTURE.search(rows[-1]["content"] or ""))
            if not may_draw:
                tools = [tool for tool in TOOLS if tool["function"]["name"] != "create_image"]
        elif self.apps:
            try:
                running = await self.apps.running()
            except Exception:  # noqa: BLE001 - she answers without the apps
                logger.exception("Could not reach the Luna apps")
                running = []
            if running:
                prompt += apps_prompt(running, name or "love")
                tools = [*TOOLS, *app_tools_spec([app.key for app in running])]
                chosen = await self.choose_app_tools(rows, running, companion["name"], name or "love")
                if chosen:
                    app, specs = chosen
                    tools = [*tools, *specs]
                    functions = ", ".join(spec["function"]["name"] for spec in specs)
                    prompt += CHOSEN_APP.format(name=name or "love", app=app.name, functions=functions)
        messages = [{"role": "system", "content": prompt}, *with_secret_nudge(model_messages(rows), notes)]
        if seeing:
            # The vision model swaps in while the face models load and read the pictures beside it.
            swapping = asyncio.create_task(self.ollama.to_vision())
            messages = await asyncio.to_thread(self.with_pictures, messages, rows, companion, name or "love")
            try:
                await swapping
            except Exception as error:  # noqa: BLE001 - the reply loads it instead
                logger.warning("Could not start loading the vision model: %s", error)
        text, calls, used_apps = "", [], []
        try:
            for _ in range(MAX_APP_ROUNDS + 1):
                said, asked, shown = "", [], 0
                async for piece in self.ollama.chat(messages, tools, seeing=seeing):
                    if "text" in piece:
                        said += piece["text"]
                        # A tool call she writes out as text never reaches the page.
                        out = shown_part(said, shown)
                        if out:
                            shown += len(out)
                            yield {"type": "text", "text": out}
                    asked += piece.get("tool_calls", [])
                # The held-back tail, unless a written tool call starts in it.
                rest = said[shown:]
                if rest and not CALL_MARKER.search(rest):
                    yield {"type": "text", "text": rest}
                if not asked and CALL_MARKER.search(said):
                    asked, said = written_calls(said)
                text += said
                app_calls = [c for c in asked if is_app_call((c.get("function") or {}).get("name") or "")]
                calls += [c for c in asked if c not in app_calls]
                if not app_calls or not self.apps:
                    break
                # She used an app: run it, show her the answer, and let her go on.
                results = []
                for call in app_calls:
                    arguments = call["function"].get("arguments") or {}
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments or "{}")
                    function = call["function"]["name"]
                    key, direct, tool_name = function.partition(DIRECT_SEPARATOR)
                    if not direct:
                        key, tool_name = str(arguments.get("app", "")).lower().removeprefix("luna ").strip(), str(arguments.get("tool") or "")
                    app = APPS.get(key)
                    yield {"type": "status", "text": f"Using {app.name if app else 'an app'}"}
                    results.append({"role": "tool", "content": await self.apps.handle(function, arguments)})
                    used = {"app": app.key, "tool": tool_name} if app else None
                    if used and (direct or function == "use_app") and used not in used_apps:
                        # Shown under her reply, so the owner sees what she did.
                        used_apps.append(used)
                messages = [*messages, {"role": "assistant", "content": said, "tool_calls": app_calls}, *results]
        except Exception as error:  # noqa: BLE001 - shown in the chat
            logger.exception("Chat model failed")
            yield {"type": "error", "text": f"{companion['name']} could not answer: {error}"}
            return

        image_id = None
        reacted, text = written_reactions(text)
        typed, text = written_remembers(text)
        calls = [*calls, *reacted, *typed]
        noted = await self.write_notes(chat_id, [*calls, *written_notes(text)])
        remembered = await self.remember(companion["id"], calls)
        if remembered:
            yield {"type": "remembered", "facts": [item["fact"] for item in remembered]}
        reaction = await self.react(calls, rows)
        if reaction:
            yield {"type": "reaction", **reaction}
        call = next((c for c in calls if (c.get("function") or {}).get("name") == "create_image"), None)
        if call is None:
            call = written_picture(text)
        text = without_written_pictures(text)
        dropped = None
        if not may_draw and call is not None:
            # A picture nobody asked for, in a call she typed out: dropped, her words stay.
            logger.info("Dropped a picture she was not asked for while looking at a photo")
            dropped, call = call, None
        if call is None and rows and not rows[-1].get("image_id") and PICTURE_PROMISE.search(text):
            # She said a picture is coming but did not take it: ask what she meant to send, and take that.
            call = await self.promised_picture(companion["name"], name, rows, text)
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
        if not text.strip() and (call or reaction or noted or dropped or remembered):
            # She only sent a picture, a reaction or a note; let her say something with it.
            used, results = [], []
            for note in noted:
                used.append({"function": {"name": "note", "arguments": {"text": note}}})
                results.append({"role": "tool", "content": "Noted. Only you can see it."})
            for item in remembered:
                used.append({"function": {"name": "remember", "arguments": {"fact": item["fact"]}}})
                results.append({"role": "tool", "content": "Remembered for every chat."})
            if reaction:
                used.append({"function": {"name": "react", "arguments": {"emoji": reaction["reaction"]}}})
                results.append({"role": "tool", "content": "Your reaction is on their message."})
            if call:
                used.append(call)
                results.append({"role": "tool", "content": "The picture was sent." if image_id else "The camera failed."})
            if dropped:
                used.append(dropped)
                results.append({"role": "tool", "content": "Not sent: they did not ask for a picture. Answer their "
                                                           "photo in words."})
            follow = [*messages, {"role": "assistant", "content": "", "tool_calls": used}, *results]
            try:
                async for piece in self.ollama.chat(follow, seeing=seeing):
                    if "text" in piece:
                        text += piece["text"]
                        yield {"type": "text", "text": piece["text"]}
            except Exception:  # noqa: BLE001 - the picture or reaction alone is enough
                logger.exception("Caption failed")

        if self.rememberer is not None and not seeing and rows and rows[-1]["role"] == "user":
            # What this turn taught her, besides what her remember tool saved: shown with it under her reply.
            yours = rows[-1]
            before = next((r["content"] or "" for r in reversed(rows[:-1]) if r["role"] != "user"), "")
            try:
                learnt = await self.rememberer.after_turn(companion["id"], " ".join(before.split()),
                                                          " ".join((yours["content"] or "").split()),
                                                          " ".join(text.split()))
            except Exception:  # noqa: BLE001 - her reply stands without it
                logger.exception("Remembering after the turn failed")
                learnt = []
            known = {item["fact"] for item in remembered}
            new = [{"about": item["about"], "fact": item["fact"]} for item in learnt if item["fact"] not in known]
            if new:
                remembered = [*remembered, *new]
                yield {"type": "remembered", "facts": [item["fact"] for item in remembered]}

        saved = await asyncio.to_thread(self.store.add_message, chat_id, "luna", text.strip(), image_id, None, used_apps,
                                        remembered or None)
        if self.recall is not None:
            # Sorting new lines and forgetting the weakest, after she has answered.
            task = asyncio.create_task(self.recall.upkeep(companion["id"]))
            self.upkeeps.add(task)
            task.add_done_callback(self.upkept)
        yield {"type": "done", "message": serialize(saved)}

    def upkept(self, task: asyncio.Task) -> None:
        self.upkeeps.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Her memory's upkeep failed", exc_info=task.exception())

    async def remember(self, companion_id: int, calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        """The facts she chose to remember, added to her memory for every chat; the ones that are new, each with
        whom it is about ("them": her memory of you, "me": her bio)."""
        remembered = []
        for call in calls:
            if (call.get("function") or {}).get("name") != "remember":
                continue
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    continue
            fact = " ".join(str(arguments.get("fact") or arguments.get("text") or "").split())[:SociaConfig.NOTE_CHARS]
            if not fact:
                continue
            if str(arguments.get("about") or "").lower() == "me":
                # About herself: into her bio (which is about "she"), so she says the same next time.
                fact = re.sub(r"^(?:I'm|I am)\b", "She is", fact)
                fact = re.sub(r"^My\b", "Her", re.sub(r"^I\b", "She", fact))
                if await asyncio.to_thread(self.store.add_to_bio, companion_id, [fact], SociaConfig.ABOUT_CHARS):
                    remembered.append({"about": "me", "fact": fact})
            elif await asyncio.to_thread(self.store.remember, companion_id, fact, SociaConfig.MEMORY_CHARS):
                remembered.append({"about": "them", "fact": fact})
        return remembered

    async def write_notes(self, chat_id: int, calls: list[dict[str, Any]]) -> list[str]:
        """What she wrote down with note, added to the chat's private notes."""
        noted = []
        for call in calls:
            if (call.get("function") or {}).get("name") != "note":
                continue
            arguments = call["function"].get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments or "{}")
                except json.JSONDecodeError:
                    continue
            note = " ".join(str(arguments.get("text") or "").split())[:SociaConfig.NOTE_CHARS]
            if note:
                await asyncio.to_thread(self.store.add_note, chat_id, note, SociaConfig.NOTES_CHARS)
                noted.append(note)
        return noted

    async def react(self, calls: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Her reaction on the owner's last message, if she called react with one she has; else None."""
        call = next((c for c in calls if (c.get("function") or {}).get("name") == "react"), None)
        if call is None or not rows or rows[-1]["role"] != "user":
            return None
        arguments = call["function"].get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return None
        emoji = str(arguments.get("emoji") or "").strip()
        # A model may drop the emoji's variation selector (❤ for ❤️); either is the same reaction.
        emoji = next((r for r in REACTIONS if emoji and emoji.replace("\ufe0f", "") == r.replace("\ufe0f", "")), None)
        if emoji is None:
            return None
        await asyncio.to_thread(self.store.set_reaction, rows[-1]["id"], emoji)
        return {"message_id": rows[-1]["id"], "reaction": emoji}

    async def promised_picture(self, her: str, name: str, rows: list[dict[str, Any]],
                               text: str) -> dict[str, Any] | None:
        """A create_image call for the picture her reply promises, or None if it promises none."""
        lines = [f"{her if row['role'] == 'luna' else name}: {' '.join((row['content'] or '').split())[:300]}"
                 for row in rows[-6:] if row["content"]]
        lines.append(f"{her}: {' '.join(text.split())}")
        try:
            answer = await self.ollama.complete(PROMISE_PROMPT.format(her=her, name=name, transcript="\n".join(lines)))
        except Exception as error:  # noqa: BLE001 - her text alone goes out
            logger.warning("Could not check for a promised picture: %s", error)
            return None
        answer = answer.strip().strip('"')
        if len(answer) < 20 or answer.upper().startswith("NONE"):
            return None
        return {"function": {"name": "create_image", "arguments": {
            "description": answer, "include_luna": True, "use_their_photo": False}}}

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
