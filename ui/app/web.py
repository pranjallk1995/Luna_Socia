"""Luna Socia's page and API, for the owner only, behind Luna's single sign-on."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.chat import Conversation, normalized_photo, serialize
from app.config import SociaConfig
from app.images import ComfyUI
from app.store import SociaStore
from app.voice import Voice

STATIC = Path(__file__).parent / "static"


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


def first_name(user: dict[str, Any]) -> str:
    return (str(user.get("name") or "").split() or [""])[0]


def create_app(config: SociaConfig, store: SociaStore, conversation: Conversation, comfy: ComfyUI, voice: Voice,
               auth_transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    sessions = Sessions(config, auth_transport)

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

    page = (STATIC / "index.html").read_text().replace("__ASSETS__", asset_version()).replace("__BRAND__", SociaConfig.BRAND)

    async def index(request: Request) -> Response:
        user = await owner(request)
        return user if isinstance(user, Response) else HTMLResponse(page, headers={"Cache-Control": "no-store"})

    async def messages(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        if request.method == "DELETE":
            for file in await asyncio.to_thread(store.clear):
                (config.images_dir / file).unlink(missing_ok=True)
            return JSONResponse({"cleared": True})
        try:
            before = int(request.query_params["before"]) if request.query_params.get("before") else None
        except ValueError:
            return JSONResponse({"error": "before must be a message id"}, 400)
        rows = await asyncio.to_thread(store.recent, 50, before)
        return JSONResponse({"messages": [serialize(row) for row in rows]})

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
        if conversation.turn.locked():
            return JSONResponse({"error": "Luna is still answering the last message."}, 409)

        async def events():
            async for event in conversation.send(text, photo or None, first_name(user), str(form.get("timezone") or "")):
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
        return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    async def profile(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        path = config.profile_path
        if request.method == "POST":
            form = await request.form(max_part_size=SociaConfig.MAX_UPLOAD_BYTES)
            upload = form.get("photo")
            if upload is None or not hasattr(upload, "read"):
                return JSONResponse({"error": "Choose a picture."}, 400)
            data = await asyncio.to_thread(normalized_photo, await upload.read())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return JSONResponse({"saved": True})
        if not path.exists():
            return JSONResponse({"error": "Luna has no profile picture yet."}, 404)
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

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
            audio = await asyncio.to_thread(voice.speak, row["content"])
        except Exception as error:  # noqa: BLE001 - shown on the page
            return JSONResponse({"error": f"Luna's voice is not available: {error}"}, 502)
        return Response(audio, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})

    async def status(request: Request) -> Response:
        user = await owner(request)
        if isinstance(user, Response):
            return user
        return JSONResponse({"chat_model": config.chat_model, "images": await comfy.available(),
                             "profile": config.profile_path.exists(), "brand": SociaConfig.BRAND})

    async def logout(request: Request) -> Response:
        return RedirectResponse(f"{config.auth_public_url}/logout?next={quote(config.public_url + '/', safe='')}", 303)

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "version": SociaConfig.VERSION})

    return Starlette(routes=[
        Route("/", index),
        Route("/api/messages", messages, methods=["GET", "DELETE"]),
        Route("/api/send", send, methods=["POST"]),
        Route("/api/images/{image_id:int}", image),
        Route("/api/profile", profile, methods=["GET", "POST"]),
        Route("/api/transcribe", transcribe, methods=["POST"]),
        Route("/api/speech/{message_id:int}", speech),
        Route("/api/status", status),
        Route("/logout", logout),
        Route("/health", health),
        Mount("/static", Assets(directory=STATIC), name="static"),
    ])
