"""Luna's phone: the owner's Luna apps, reached through their MCP servers.

Socia signs in to them the way Claude does: it registers itself with Luna Sentia's
sign-on as an OAuth client, the owner approves it once, and it keeps the access and
refresh tokens that come back. That token names no app, so every Luna MCP server
accepts it, as the owner. The owner can disconnect at any time.

The chat model's context is small, so Luna gets three tools rather than every app's:
one to list an app's tools, one to see a tool's arguments, and one to use it. A small chat
model rarely starts that on its own, so before she answers, the app and the few tools a
message needs are picked for her and handed to her as plain functions (``app__tool``).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import SociaConfig
from app.store import SociaStore

logger = logging.getLogger("socia.apps")


@dataclass(frozen=True)
class LunaApp:
    key: str
    name: str
    url: str
    purpose: str


# Every Luna MCP server, at its address on luna_sentia_network, and what it is for.
LUNA_APPS = (
    LunaApp("fiora", "Luna Fiora", "http://luna-fiora-mcp:8000/mcp", "their money: income, spending, budgets"),
    LunaApp("lexia", "Luna Lexia", "http://luna-lexia-mcp:8000/mcp", "their documents: payslips, statements, tax proofs"),
    LunaApp("lumia", "Luna Lumia", "http://luna-lumia-mcp:8000/mcp", "the lights at home"),
    LunaApp("vibia", "Luna Vibia", "http://luna-vibia-mcp:8000/mcp", "their music, which can play on the speakers at home"),
    LunaApp("visia", "Luna Visia", "http://luna-visia-mcp:8000/mcp", "their photo gallery"),
    LunaApp("vita", "Luna Vita", "http://luna-vita-mcp:8000/mcp", "their meals, nutrition and health"),
    LunaApp("sentia", "Luna Sentia", "http://luna-sentia-monitor:8001/mcp", "the home server's GPU and models"),
)
APPS = {app.key: app for app in LUNA_APPS}

# The longest a tool's answer may be in her context, and how long tool lists are kept.
RESULT_CHARS = 2400
TOOLS_SECONDS = 300.0
CALL_SECONDS = 60.0
# A new access token is fetched this long before the old one expires.
REFRESH_EARLY_SECONDS = 120
# How long the owner has to approve Socia on Sentia's page.
PENDING_SECONDS = 600


class NotConnected(RuntimeError):
    pass


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def first_sentence(text: str | None) -> str:
    line = " ".join((text or "").split())
    end = line.find(". ")
    return line if end < 0 else line[:end + 1]


def result_text(result: Any) -> str:
    """A tool's answer as text for her: its structured content, or its text parts, shortened."""
    if getattr(result, "structuredContent", None):
        text = json.dumps(result.structuredContent, ensure_ascii=False, default=str)
    else:
        parts = [getattr(part, "text", None) or f"[{getattr(part, 'type', 'content')}]" for part in result.content or []]
        text = "\n".join(parts)
    if getattr(result, "isError", False):
        text = f"The app answered with an error: {text}"
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS - 1] + "…"


class LunaApps:
    """The owner's connection to the Luna apps, and calls to their tools."""

    def __init__(self, config: SociaConfig, store: SociaStore, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.store = store
        self.auth = httpx.AsyncClient(base_url=config.auth_internal_url, transport=transport, timeout=10.0)
        self.transport = transport
        self.pending: dict[str, tuple[str, float]] = {}
        self.refreshing = asyncio.Lock()
        self.tool_lists: dict[str, tuple[float, list[Any]]] = {}

    @property
    def redirect_uri(self) -> str:
        return f"{self.config.public_url}/apps/callback"

    # Connecting

    async def client(self) -> dict[str, Any]:
        """Socia's OAuth client at Sentia's sign-on, registered the first time it is needed."""
        row = await asyncio.to_thread(self.store.apps_connection)
        if row and row["client_id"] and row["redirect_uri"] == self.redirect_uri:
            return row
        reply = await self.auth.post("/oauth/register", json={
            "client_name": "Luna Socia", "redirect_uris": [self.redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        })
        reply.raise_for_status()
        body = reply.json()
        return await asyncio.to_thread(self.store.set_apps_client, body["client_id"], body.get("client_secret"),
                                       self.redirect_uri)

    async def start(self) -> str:
        """Where to send the owner to approve Socia: Sentia's sign-on, asking for every Luna app."""
        client = await self.client()
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(24)
        now = time.monotonic()
        self.pending = {key: value for key, value in self.pending.items() if now - value[1] < PENDING_SECONDS}
        self.pending[state] = (verifier, now)
        query = urlencode({"response_type": "code", "client_id": client["client_id"], "redirect_uri": self.redirect_uri,
                           "code_challenge": challenge, "code_challenge_method": "S256", "state": state})
        return f"{self.config.auth_public_url}/oauth/authorize?{query}"

    async def finish(self, code: str, state: str) -> None:
        pending = self.pending.pop(state, None)
        if pending is None or time.monotonic() - pending[1] > PENDING_SECONDS:
            raise ValueError("This approval expired. Connect again.")
        client = await self.client()
        await self.token({"grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_uri,
                          "code_verifier": pending[0]}, client)

    async def token(self, fields: dict[str, str], client: dict[str, Any]) -> str:
        reply = await self.auth.post("/oauth/token", data={
            **fields, "client_id": client["client_id"], "client_secret": client["client_secret"] or ""})
        if reply.status_code != 200:
            raise NotConnected(f"Luna Sentia's sign-on refused Socia ({reply.status_code}); connect the apps again.")
        body = reply.json()
        await asyncio.to_thread(self.store.set_apps_tokens, body["access_token"], body.get("refresh_token"),
                                time.time() + float(body.get("expires_in") or 3600))
        return body["access_token"]

    async def disconnect(self) -> None:
        await asyncio.to_thread(self.store.set_apps_tokens, None, None, None)
        self.tool_lists.clear()

    async def connected(self) -> bool:
        row = await asyncio.to_thread(self.store.apps_connection)
        return bool(row and row["refresh_token"])

    async def access_token(self) -> str:
        """A current access token, refreshed shortly before it expires; refresh tokens work once."""
        async with self.refreshing:
            row = await asyncio.to_thread(self.store.apps_connection)
            if not row or not row["refresh_token"]:
                raise NotConnected("The Luna apps are not connected.")
            if row["access_token"] and (row["access_expires"] or 0) - time.time() > REFRESH_EARLY_SECONDS:
                return row["access_token"]
            try:
                return await self.token({"grant_type": "refresh_token", "refresh_token": row["refresh_token"]}, row)
            except NotConnected:
                await self.disconnect()
                raise

    # Using

    async def session_call(self, app: LunaApp, action):
        """Open one MCP session to an app as the owner and run ``action(session)`` in it."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        token = await self.access_token()
        http = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, transport=self.transport,
                                 timeout=httpx.Timeout(CALL_SECONDS, connect=5.0))
        async with http, streamable_http_client(app.url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await action(session)

    async def tools(self, key: str) -> list[Any]:
        app = APPS[key]
        cached = self.tool_lists.get(key)
        if cached and time.monotonic() - cached[0] < TOOLS_SECONDS:
            return cached[1]

        async def listed(session):
            return (await session.list_tools()).tools

        tools = await self.session_call(app, listed)
        self.tool_lists[key] = (time.monotonic(), tools)
        return tools

    async def running(self) -> list[LunaApp]:
        """The apps that answer now, for her prompt and the page."""
        if not await self.connected():
            return []

        async def answers(app: LunaApp) -> bool:
            try:
                await asyncio.wait_for(self.tools(app.key), 15)
                return True
            except NotConnected:
                raise
            except Exception:  # noqa: BLE001 - an app that is not running is simply left out
                return False

        found = await asyncio.gather(*(answers(app) for app in LUNA_APPS))
        return [app for app, ok in zip(LUNA_APPS, found) if ok]

    async def status(self) -> dict[str, Any]:
        """For the page: whether Socia is connected, and every app with whether it answers now."""
        connected = await self.connected()
        try:
            running = {app.key for app in await self.running()} if connected else set()
        except NotConnected:
            connected, running = False, set()
        return {"connected": connected, "apps": [
            {"key": app.key, "name": app.name, "purpose": app.purpose, "running": app.key in running}
            for app in LUNA_APPS]}

    async def tool_lines(self, key: str) -> str:
        """An app's tools, one line each, for choosing among them."""
        return "\n".join(f"- {tool.name}: {first_sentence(tool.description)}" for tool in await self.tools(key))

    async def direct_specs(self, key: str, names: list[str]) -> list[dict[str, Any]]:
        """Some of an app's tools as her own functions, named ``app__tool``, with their real arguments."""
        tools = {tool.name: tool for tool in await self.tools(key)}
        return [{"type": "function", "function": {
            "name": f"{key}{DIRECT_SEPARATOR}{name}",
            "description": f"{APPS[key].name}: {' '.join((tools[name].description or '').split())[:600]}",
            "parameters": tools[name].inputSchema or {"type": "object", "properties": {}}}}
            for name in names if name in tools]

    async def handle(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one of her app tools (or an ``app__tool`` function) and return what she reads back."""
        if DIRECT_SEPARATOR in name:
            key, tool_name = name.split(DIRECT_SEPARATOR, 1)
            return await self.handle("use_app", {"app": key, "tool": tool_name, "arguments": arguments})
        key = str(arguments.get("app") or "").lower().removeprefix("luna ").strip()
        if key not in APPS:
            return f"There is no Luna app called {arguments.get('app')!r}. The apps are: {', '.join(APPS)}."
        app = APPS[key]
        try:
            tools = await self.tools(key)
            if name == "app_tools":
                lines = [f"- {tool.name}: {first_sentence(tool.description)}" for tool in tools]
                return f"{app.name} can:\n" + "\n".join(lines)
            tool = next((tool for tool in tools if tool.name == arguments.get("tool")), None)
            if tool is None:
                return f"{app.name} has no tool {arguments.get('tool')!r}; call app_tools to see its tools."
            if name == "app_tool_help":
                schema = json.dumps(tool.inputSchema, ensure_ascii=False)
                return f"{tool.name}: {' '.join((tool.description or '').split())}\nArguments (JSON schema): {schema}"
            given = arguments.get("arguments") or {}
            if isinstance(given, str):
                given = json.loads(given or "{}")

            async def called(session):
                return await session.call_tool(tool.name, given, read_timeout_seconds=timedelta(seconds=CALL_SECONDS))

            return result_text(await self.session_call(app, called))
        except NotConnected as error:
            return str(error)
        except Exception as error:  # noqa: BLE001 - she says it did not work
            logger.warning("%s %s failed: %s", app.name, arguments.get("tool") or name, error)
            return f"{app.name} did not answer: {type(error).__name__}. It may not be running."


APP_TOOL_NAMES = {"app_tools", "app_tool_help", "use_app"}
# An app tool handed to her directly is named app__tool, e.g. visia__search_images.
DIRECT_SEPARATOR = "__"


def is_app_call(name: str) -> bool:
    """Whether a function she called is one of the apps': a meta tool or an ``app__tool``."""
    key, separator, _ = name.partition(DIRECT_SEPARATOR)
    return name in APP_TOOL_NAMES or (bool(separator) and key in APPS)


def app_tools_spec(keys: list[str]) -> list[dict[str, Any]]:
    """Her three tools for the apps that are connected and running."""
    app = {"type": "string", "enum": keys, "description": "The Luna app."}
    return [
        {"type": "function", "function": {
            "name": "app_tools",
            "description": "List what one of their Luna apps can do: its tools, one line each.",
            "parameters": {"type": "object", "properties": {"app": app}, "required": ["app"]}}},
        {"type": "function", "function": {
            "name": "app_tool_help",
            "description": "See one app tool's full description and the arguments it takes, before using it.",
            "parameters": {"type": "object", "properties": {"app": app, "tool": {"type": "string"}},
                           "required": ["app", "tool"]}}},
        {"type": "function", "function": {
            "name": "use_app",
            "description": "Use a tool of one of their Luna apps, for them, and read its answer.",
            "parameters": {"type": "object", "properties": {
                "app": app, "tool": {"type": "string", "description": "The tool's exact name."},
                "arguments": {"type": "object", "description": "The tool's arguments."}},
                "required": ["app", "tool", "arguments"]}}},
    ]


def apps_prompt(apps: list[LunaApp], name: str) -> str:
    """What she knows about the apps on her phone."""
    if not apps:
        return ""
    lines = "\n".join(f"- {app.key}: {app.purpose}" for app in apps)
    return (f"\n\nYour phone has {name}'s Luna apps, and you can use them for {name}:\n{lines}\n"
            f"When {name} asks for something one of them does (play a song, dim the lights, find a photo, what they "
            "ate, what they spent), do it: app_tools shows what an app can do, app_tool_help a tool's arguments, and "
            "use_app does it. Then tell them what happened in your own words, briefly, as yourself. Never make up "
            "what an app said, and never use an app they did not ask for.")
