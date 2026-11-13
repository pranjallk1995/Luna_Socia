"""Luna Socia's page and API, for the owner only, behind Luna's single sign-on.

The page lists the owner's girlfriends (companions) and creates new ones; everything else
(chats, pictures, memory, voice) belongs to one of them.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import html
import json
import logging
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.avatar import Avatar
from app.chat import Conversation, normalized_photo, serialize, serialize_chat, serialize_companion
from app.config import SociaConfig
from app.images import ComfyUI
from app import model_usage
from app.memory import ExportError, MemoryReader, parse_export
from app.persona import CHOICES, DEFAULT_PERSONA, REACTIONS, STYLES, VOICE_IDS, VOICES, clean_persona, example
from app.store import SociaStore
from app.voice import SAMPLE, Voice

logger = logging.getLogger("socia")
# While she takes a picture nothing else is sent; a comment this often keeps tunnels and phones connected.
KEEPALIVE_SECONDS = 15

STATIC = Path(__file__).parent / "static"
REFERENCE_NAME = re.compile(r"\d+-[0-9a-f]{8}\.jpg")


def asset_version() -> str:
    """A short hash of the page's own files, so a new release gets new asset URLs."""
    digest = hashlib.sha256()
    for name in ("app.js", "styles.css"):
        digest.update((STATIC / name).read_bytes())
    return digest.hexdigest()[:12]


class Assets(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def message_page(title: str, message: str, status: int) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title>"
        "<body style='font-family:ui-monospace,monospace;background:#111;color:#f5f5f5;display:grid;place-items:center;"
        f"min-height:90vh;margin:0 16px'><main style='max-width:40rem'><h1>{html.escape(title)}</h1><p>{message}</p></main>",
        status_code=status,
    )


class Sessions:
    """Who a luna_sso cookie belongs to, asked of Sentia's auth service and remembered briefly."""

    def __init__(self, config: SociaConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = f"{config.auth_internal_url}/session"
        self.client = httpx.AsyncClient(transport=transport, timeout=5.0)
        self.known: dict[str, tuple[dict[str, Any] | None, float]] = {}

    async def user(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        cached = self.known.get(token)
        if cached and time.monotonic() - cached[1] < SociaConfig.SESSION_RECHECK_SECONDS:
            return cached[0]
        try:
            response = await self.client.get(self.url, headers={SociaConfig.SESSION_HEADER: token})
            user = response.json() if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return cached[0] if cached else None
        if len(self.known) > 256:
            self.known.clear()
        self.known[token] = (user, time.monotonic())
        return user


def download_name(image: dict[str, Any], suffix: str) -> str:
    """A downloaded picture's file name: "luna-2026-11-11-1615.png", or "photo-..." for the owner's photos."""
    sender = (image.get("companion_name") or "picture") if image.get("kind") == "generated" else "photo"
    sender = re.sub(r"[^a-z0-9]+", "-", sender.casefold()).strip("-") or "picture"
    created = image.get("created_at")
    stamp = created.astimezone().strftime("%Y-%m-%d-%H%M") if created else str(image.get("id", ""))
    return f"{sender}-{stamp}{suffix}"


def first_name(user: dict[str, Any]) -> str:
    return (str(user.get("name") or "").split() or [""])[0]


def create_app(config: SociaConfig, store: SociaStore, conversation: Conversation, comfy: ComfyUI, voice: Voice,
               auth_transport: httpx.AsyncBaseTransport | None = None, reader: MemoryReader | None = None) -> Starlette:
    sessions = Sessions(config, auth_transport)
    samples: dict[str, bytes] = {}

    async def owner(request: Request) -> dict[str, Any] | Response:
        """The owner's session, or the page or error to answer with."""
        user = await sessions.user(request.cookies.get(SociaConfig.COOKIE_NAME))
        wants_page = not request.url.path.startswith("/api/")
        if user is None:
            if wants_page:
                here = f"{config.public_url}{request.url.path}"
                return RedirectResponse(f"{config.auth_public_url}/login?next={quote(here, safe='')}", 303)
            return JSONResponse({"error": "not signed in"}, 401)
        if user.get("user_id") != SociaConfig.OWNER_ID:
            if wants_page:
                return message_page("Owner only", "Luna Socia is the owner's private conversation.", 403)
            return JSONResponse({"error": "owner only"}, 403)
        return user

    async def her(request: Request) -> dict[str, Any] | Response:
        """The girlfriend named in the path, for the owner, or the error to answer with."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        row = await asyncio.to_thread(store.companion, request.path_params["companion_id"])
        return row if row else JSONResponse({"error": "no such girlfriend"}, 404)

    def companion_json(row: dict[str, Any]) -> dict[str, Any]:
        found = serialize_companion(row)
        found["has_profile"] = config.profile_path(row["id"]).exists()
        found["memory_progress"] = reader.progress(row) if reader else {
            "sources": int(row.get("sources") or 0), "read": int(row.get("read") or 0), "reading": False, "error": None}
        return found

    def her_fields(body: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any] | str:
        """What a create or edit changes, checked; or what is wrong with it."""
        fields: dict[str, Any] = {}
        if "name" in body or current is None:
            name = " ".join(str(body.get("name") or "").split())[:SociaConfig.NAME_CHARS]
            if not name:
                return "Give her a name."
            fields["name"] = name
        if "persona" in body or current is None:
            if not isinstance(body.get("persona", {}), dict):
                return "persona must be an object."
            fields["persona"] = clean_persona(body.get("persona") or {}, current and current.get("persona"))
        if "voice" in body:
            if body["voice"] not in VOICE_IDS:
                return "Choose one of her voices."
            fields["voice"] = body["voice"]
        if "memory" in body:
            fields["memory"] = str(body.get("memory") or "").strip()[:SociaConfig.MEMORY_CHARS]
        return fields

    page = (STATIC / "index.html").read_text().replace("__ASSETS__", asset_version()).replace("__BRAND__", SociaConfig.BRAND)

    async def index(request: Request) -> Response:
        user = await owner(request)
        return user if isinstance(user, Response) else HTMLResponse(page, headers={"Cache-Control": "no-store"})

    async def options(request: Request) -> Response:
        """The choices a girlfriend is made from, for the page's create and edit form."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        styles = [{"id": key, "label": style["label"], "example": example({**DEFAULT_PERSONA, "style": key})}
                  for key, style in STYLES.items()]
        return JSONResponse({"styles": styles, "lengths": list(CHOICES["length"]), "emoji": list(CHOICES["emoji"]),
                             "languages": list(CHOICES["language"]), "voices": VOICES, "reactions": REACTIONS,
                             "default_persona": DEFAULT_PERSONA,
                             "default_voice": config.tts_voice if config.tts_voice in VOICE_IDS else VOICES[0]["id"],
                             "limits": {"name": SociaConfig.NAME_CHARS, "about": SociaConfig.ABOUT_CHARS,
                                        "memory": SociaConfig.MEMORY_CHARS, "references": SociaConfig.MAX_REFERENCES,
                                        "export_bytes": SociaConfig.MAX_EXPORT_BYTES}})

    async def companions(request: Request) -> Response:
        """Every girlfriend (GET), or a new one (POST {name, persona, voice})."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        if request.method == "POST":
            try:
                body = await request.json()
            except ValueError:
                return JSONResponse({"error": "Send her as JSON."}, 400)
            fields = her_fields(body if isinstance(body, dict) else {}, None)
            if isinstance(fields, str):
                return JSONResponse({"error": fields}, 400)
            voice_id = fields.get("voice") or config.tts_voice
            row = await asyncio.to_thread(store.create_companion, fields["name"], fields["persona"],
                                          voice_id if voice_id in VOICE_IDS else VOICES[0]["id"])
            return JSONResponse({"companion": companion_json(row)}, 201)
        rows = await asyncio.to_thread(store.companions)
        return JSONResponse({"companions": [companion_json(row) for row in rows]})

    async def companion(request: Request) -> Response:
        row = await her(request)
        if isinstance(row, Response):
            return row
        if request.method == "DELETE":
            if conversation.turn.locked():
                return JSONResponse({"error": "She is still answering."}, 409)
            files = await asyncio.to_thread(store.delete_companion, row["id"])
            for file in files or []:
                (config.images_dir / file).unlink(missing_ok=True)
            shutil.rmtree(config.companion_dir(row["id"]), ignore_errors=True)
            return JSONResponse({"deleted": True})
        if request.method == "PATCH":
            try:
                body = await request.json()
            except ValueError:
                return JSONResponse({"error": "Send the changes as JSON."}, 400)
            fields = her_fields(body if isinstance(body, dict) else {}, row)
            if isinstance(fields, str):
                return JSONResponse({"error": fields}, 400)
            row = await asyncio.to_thread(store.update_companion, row["id"], fields)
        return JSONResponse({"companion": companion_json(row)})

    async def chats(request: Request) -> Response:
        """Her chats, newest activity first. Chats start with their first message (see send)."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        rows = await asyncio.to_thread(store.chats, row["id"])
        return JSONResponse({"chats": [serialize_chat(chat) for chat in rows]})

    async def chat(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        chat_id = request.path_params["chat_id"]
        if request.method == "DELETE":
            if conversation.turn.locked():
                return JSONResponse({"error": "She is still answering."}, 409)
            files = await asyncio.to_thread(store.delete_chat, chat_id)
            if files is None:
                return JSONResponse({"error": "no such chat"}, 404)
            for file in files:
                (config.images_dir / file).unlink(missing_ok=True)
            return JSONResponse({"deleted": True})
        if request.method == "PATCH":
            body = await request.json()
            title = " ".join(str(body.get("title") or "").split())[:SociaConfig.CHAT_TITLE_CHARS]
            if not title:
                return JSONResponse({"error": "Give the chat a name."}, 400)
            row = await asyncio.to_thread(store.rename_chat, chat_id, title)
            if row is None:
                return JSONResponse({"error": "no such chat"}, 404)
        else:
            row = await asyncio.to_thread(store.chat, chat_id)
            if row is None:
                return JSONResponse({"error": "no such chat"}, 404)
        return JSONResponse({"chat": serialize_chat(row)})

    async def messages(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        chat_id = request.path_params["chat_id"]
        try:
            before = int(request.query_params["before"]) if request.query_params.get("before") else None
        except ValueError:
            return JSONResponse({"error": "before must be a message id"}, 400)
        if await asyncio.to_thread(store.chat, chat_id) is None:
            return JSONResponse({"error": "no such chat"}, 404)
        rows = await asyncio.to_thread(store.recent, chat_id, SociaConfig.MESSAGES_PAGE, before)
        return JSONResponse({"messages": [serialize(row) for row in rows], "page": SociaConfig.MESSAGES_PAGE})

    async def delete_message(request: Request) -> Response:
        """Forget one message, with its photo or picture; she no longer sees it either."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        if conversation.turn.locked():
            return JSONResponse({"error": "She is still answering."}, 409)
        files = await asyncio.to_thread(store.delete_message, request.path_params["message_id"])
        if files is None:
            return JSONResponse({"error": "no such message"}, 404)
        for file in files:
            (config.images_dir / file).unlink(missing_ok=True)
        return JSONResponse({"deleted": True})

    async def react(request: Request) -> Response:
        """The owner's reaction on one of her messages: an emoji to put it on, or null to take it off."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        body = await request.json()
        reaction = body.get("reaction") if isinstance(body, dict) else None
        if reaction is not None and reaction not in REACTIONS:
            return JSONResponse({"error": f"React with one of {' '.join(REACTIONS)}."}, 400)
        message = await asyncio.to_thread(store.message, request.path_params["message_id"])
        if message is None:
            return JSONResponse({"error": "no such message"}, 404)
        if message["role"] != "luna":
            return JSONResponse({"error": "Only her messages take your reactions."}, 400)
        row = await asyncio.to_thread(store.set_reaction, message["id"], reaction)
        return JSONResponse({"message": serialize(row)})

    # Answers still running after their page went away; kept so they are not garbage-collected.
    answering: set[asyncio.Task[None]] = set()

    async def send(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        form = await request.form(max_part_size=SociaConfig.MAX_UPLOAD_BYTES)
        text = str(form.get("text") or "").strip()
        upload = form.get("photo")
        photo = await upload.read() if upload is not None and hasattr(upload, "read") else None
        if not text and not photo:
            return JSONResponse({"error": "Say something or send a photo."}, 400)
        try:
            companion_id = int(form.get("companion_id") or "")
            chat_id = int(form["chat_id"]) if form.get("chat_id") else None
        except ValueError:
            return JSONResponse({"error": "companion_id and chat_id must be ids"}, 400)
        row = await asyncio.to_thread(store.companion, companion_id)
        if row is None:
            return JSONResponse({"error": "That girlfriend was deleted."}, 404)
        if chat_id is not None:
            existing = await asyncio.to_thread(store.chat, chat_id)
            if existing is None or existing["companion_id"] != companion_id:
                return JSONResponse({"error": "That chat was deleted."}, 404)
        if conversation.turn.locked():
            return JSONResponse({"error": "She is still answering the last message."}, 409)

        # Her answer runs on its own, so a page that drops off (a phone losing signal mid-picture)
        # does not cancel it: the picture and her reply are saved and appear when the chat reloads.
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def reply() -> None:
            try:
                async for event in conversation.send(row, chat_id, text, photo or None, first_name(user),
                                                     str(form.get("timezone") or "")):
                    queue.put_nowait(event)
            except Exception as error:  # noqa: BLE001 - shown in the chat
                logger.exception("Her answer failed")
                queue.put_nowait({"type": "error", "text": f"{row['name']} could not answer: {error}"})
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(reply())
        answering.add(task)
        task.add_done_callback(answering.discard)

        async def events():
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield ": still here\n\n"
                    continue
                if event is None:
                    return
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    async def image(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        row = await asyncio.to_thread(store.image, int(request.path_params["image_id"]))
        path = config.images_dir / row["file"] if row else None
        if not path or not path.exists():
            return JSONResponse({"error": "no such image"}, 404)
        if request.query_params.get("download"):
            # Named for who sent it and when: "luna-2026-11-11-1615.png", or "photo-..." for yours.
            return FileResponse(path, filename=download_name(row, path.suffix),
                                headers={"Cache-Control": "private, no-cache"})
        return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    async def profile(request: Request) -> Response:
        row = await her(request)
        if isinstance(row, Response):
            return row
        path = config.profile_path(row["id"])
        if request.method == "POST":
            form = await request.form(max_part_size=SociaConfig.MAX_UPLOAD_BYTES)
            upload = form.get("photo")
            if upload is None or not hasattr(upload, "read"):
                return JSONResponse({"error": "Choose a picture."}, 400)
            data = await asyncio.to_thread(normalized_photo, await upload.read())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            # A crop chosen for the old picture would cut the new one anywhere.
            Avatar(path).reset()
            # A new face needs its own confirmation that it is AI-generated before nude pictures.
            persona = clean_persona({"nude_pictures": "off"}, row.get("persona"))
            await asyncio.to_thread(store.update_companion, row["id"], {"persona": persona})
            return JSONResponse({"saved": True})
        if not path.exists():
            return JSONResponse({"error": "She has no profile picture yet."}, 404)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    async def avatar(request: Request) -> Response:
        """Her round avatar, cut from the profile picture where the owner chose."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        path = config.profile_path(row["id"])
        if not path.exists():
            return JSONResponse({"error": "She has no profile picture yet."}, 404)
        image = await asyncio.to_thread(Avatar(path).render)
        return Response(image, media_type="image/png", headers={"Cache-Control": "no-cache"})

    async def avatar_crop(request: Request) -> Response:
        """Where the avatar is cut: read, chosen (PUT), or back to the default (DELETE)."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        path = config.profile_path(row["id"])
        if not path.exists():
            return JSONResponse({"error": "She has no profile picture yet."}, 404)
        avatar = Avatar(path)
        if request.method == "PUT":
            try:
                body = await request.json()
                return JSONResponse(await asyncio.to_thread(avatar.save, body))
            except (ValueError, TypeError, KeyError):
                return JSONResponse({"error": "A crop needs numbers x, y and zoom."}, 400)
        if request.method == "DELETE":
            await asyncio.to_thread(avatar.reset)
        return JSONResponse(await asyncio.to_thread(avatar.crop))

    async def references(request: Request) -> Response:
        """Her other pictures: listed, added (several at once), or one served."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        companion_id = row["id"]
        folder = config.references_dir(companion_id)
        if request.method == "POST":
            form = await request.form(max_part_size=SociaConfig.MAX_UPLOAD_BYTES)
            uploads = [u for u in form.getlist("photos") if hasattr(u, "read")]
            if not uploads:
                return JSONResponse({"error": "Choose a picture."}, 400)
            if len(config.references(companion_id)) + len(uploads) > SociaConfig.MAX_REFERENCES:
                return JSONResponse({"error": f"She can have {SociaConfig.MAX_REFERENCES} other pictures at most."}, 400)
            folder.mkdir(parents=True, exist_ok=True)
            for upload in uploads:
                data = await asyncio.to_thread(normalized_photo, await upload.read())
                # Named by the time they were added, so they keep their order.
                (folder / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.jpg").write_bytes(data)
        base = f"/api/companions/{companion_id}/references"
        return JSONResponse({"references": [{"name": path.name, "url": f"{base}/{path.name}"}
                                            for path in config.references(companion_id)],
                             "max": SociaConfig.MAX_REFERENCES})

    async def reference(request: Request) -> Response:
        row = await her(request)
        if isinstance(row, Response):
            return row
        name = request.path_params["name"]
        path = config.references_dir(row["id"]) / name
        if not REFERENCE_NAME.fullmatch(name) or not path.exists():
            return JSONResponse({"error": "no such picture"}, 404)
        if request.method == "DELETE":
            path.unlink()
            return JSONResponse({"deleted": True})
        return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    async def memory(request: Request) -> Response:
        """What she remembers (GET), or the owner's edit of it (PUT {memory})."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        if request.method == "PUT":
            body = await request.json()
            row = await asyncio.to_thread(store.update_companion, row["id"], her_fields({"memory": body.get("memory")}, row))
        found = companion_json(row)
        return JSONResponse({"memory": found["memory"], "progress": found["memory_progress"]})

    async def memory_imports(request: Request) -> Response:
        """Chats exported from ChatGPT or Claude, added to her memory (POST), or forgotten (DELETE)."""
        row = await her(request)
        if isinstance(row, Response):
            return row
        if request.method == "DELETE":
            forgotten = await asyncio.to_thread(store.forget_sources, row["id"])
            return JSONResponse({"forgotten": forgotten})
        form = await request.form(max_files=1)
        upload = form.get("export")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "Choose your ChatGPT or Claude export."}, 400)
        if (upload.size or 0) > SociaConfig.MAX_EXPORT_BYTES:
            return JSONResponse({"error": "That export is too large."}, 400)
        try:
            source, found = await asyncio.to_thread(parse_export, await upload.read())
        except ExportError as error:
            return JSONResponse({"error": str(error)}, 400)
        finally:
            # An export can be hundreds of MB, spooled to disk: freed now, not when collected.
            await form.close()
        added = await asyncio.to_thread(store.add_memory_sources, row["id"], source, found)
        if reader:
            reader.start()
        row = await asyncio.to_thread(store.companion, row["id"])
        return JSONResponse({"source": source, "conversations": len(found), "added": added,
                             "progress": companion_json(row)["memory_progress"]})

    async def voice_sample(request: Request) -> Response:
        """A few words in one of the voices, for choosing hers."""
        user = await owner(request)
        if isinstance(user, Response):
            return user
        voice_id = request.path_params["voice"]
        if voice_id not in VOICE_IDS:
            return JSONResponse({"error": "no such voice"}, 404)
        if voice_id not in samples:
            try:
                samples[voice_id] = await asyncio.to_thread(voice.speak, SAMPLE, voice_id)
            except Exception as error:  # noqa: BLE001 - shown on the page
                return JSONResponse({"error": f"The voice is not available: {error}"}, 502)
        return Response(samples[voice_id], media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})

    async def transcribe(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        audio = await request.body()
        if not audio or len(audio) > SociaConfig.MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Send a short voice clip."}, 400)
        try:
            text = await asyncio.to_thread(voice.transcribe, audio)
        except Exception as error:  # noqa: BLE001 - shown on the page
            return JSONResponse({"error": f"Could not hear that: {error}"}, 502)
        return JSONResponse({"text": text})

    async def speech(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        row = await asyncio.to_thread(store.message, int(request.path_params["message_id"]))
        if not row or row["role"] != "luna" or not row["content"].strip():
            return JSONResponse({"error": "nothing to say"}, 404)
        try:
            audio = await asyncio.to_thread(voice.speak, row["content"], row.get("voice"))
        except Exception as error:  # noqa: BLE001 - shown on the page
            return JSONResponse({"error": f"Her voice is not available: {error}"}, 502)
        return Response(audio, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})

    async def status(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        return JSONResponse({"chat_model": config.chat_model, "images": await comfy.available(),
                             "brand": SociaConfig.BRAND})

    async def models(request: Request) -> Response:
        """For Luna Sentia's model monitor: the voice models Socia runs itself, outside Ollama.

        The monitor sends LUNASOCIA_API_TOKEN; the owner's session works too.
        """
        scheme, _, credential = request.headers.get("authorization", "").partition(" ")
        trusted = (config.api_token and scheme.lower() == "bearer"
                   and hmac.compare_digest(credential.strip().encode(), config.api_token.encode()))
        if not trusted:
            user = await owner(request)
            if isinstance(user, Response):
                return user
        return JSONResponse(model_usage.report("socia"))

    async def logout(request: Request) -> Response:
        return RedirectResponse(f"{config.auth_public_url}/logout?next={quote(config.public_url + '/', safe='')}", 303)

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "version": SociaConfig.VERSION})

    @asynccontextmanager
    async def lifespan(app: Starlette):
        # Imported chats left unread when Socia last stopped are read on.
        if reader:
            reader.start()
        # Chats started before chats were named get a name, between her replies.
        naming = asyncio.create_task(conversation.name_old_chats())
        yield
        naming.cancel()

    her_path = "/api/companions/{companion_id:int}"
    return Starlette(lifespan=lifespan, routes=[
        Route("/", index),
        Route("/api/options", options),
        Route("/api/companions", companions, methods=["GET", "POST"]),
        Route(her_path, companion, methods=["GET", "PATCH", "DELETE"]),
        Route(f"{her_path}/chats", chats),
        Route(f"{her_path}/profile", profile, methods=["GET", "POST"]),
        Route(f"{her_path}/avatar", avatar),
        Route(f"{her_path}/avatar/crop", avatar_crop, methods=["GET", "PUT", "DELETE"]),
        Route(f"{her_path}/references", references, methods=["GET", "POST"]),
        Route(f"{her_path}/references/{{name}}", reference, methods=["GET", "DELETE"]),
        Route(f"{her_path}/memory", memory, methods=["GET", "PUT"]),
        Route(f"{her_path}/memory/imports", memory_imports, methods=["POST", "DELETE"]),
        Route("/api/voices/{voice}/sample", voice_sample),
        Route("/api/chats/{chat_id:int}", chat, methods=["GET", "PATCH", "DELETE"]),
        Route("/api/chats/{chat_id:int}/messages", messages),
        Route("/api/messages/{message_id:int}", delete_message, methods=["DELETE"]),
        Route("/api/messages/{message_id:int}/reaction", react, methods=["PUT"]),
        Route("/api/send", send, methods=["POST"]),
        Route("/api/images/{image_id:int}", image),
        Route("/api/transcribe", transcribe, methods=["POST"]),
        Route("/api/speech/{message_id:int}", speech),
        Route("/api/status", status),
        Route("/api/models", models),
        Route("/logout", logout),
        Route("/health", health),
        Mount("/static", Assets(directory=STATIC), name="static"),
    ])
