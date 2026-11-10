"""Luna's chat and vision models, through Sentia's monitor on its way to Ollama."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import SociaConfig

# Socia names itself to the monitor, which shows each call under its app.
APP_HEADER = {"X-Luna-App": "socia"}
PHOTO_PROMPT = ("Describe this photo for someone who cannot see it, in 2-4 sentences: who and what is in it, "
                "the place, the mood, and any text. Plain description, no preamble.")
LOOKS_PROMPT = ("Describe only the woman's lasting appearance in one or two sentences, for recognizing her: face, "
                "hair, skin, eyes, build, age range. Not her clothes, pose or the scene. No preamble.")


class Ollama:
    def __init__(self, config: SociaConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.client = httpx.AsyncClient(base_url=config.ollama_url, transport=transport, headers=APP_HEADER,
                                        timeout=httpx.Timeout(300.0, connect=10.0))

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Stream Luna's reply: {"text": ...} pieces, then {"tool_calls": [...]} if she uses a tool."""
        body = {
            "model": self.config.chat_model,
            "messages": messages,
            "stream": True,
            # Qwen3 thinks before answering unless told not to; a text reply needs no reasoning.
            "think": False,
            "options": {"num_ctx": self.config.context_tokens, "temperature": self.config.temperature},
        }
        if tools:
            body["tools"] = tools
        async with self.client.stream("POST", "/api/chat", json=body) as response:
            if response.status_code != 200:
                raise httpx.HTTPStatusError(f"Ollama answered {response.status_code}: {(await response.aread())[:300]!r}",
                                            request=response.request, response=response)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                message = chunk.get("message") or {}
                if message.get("content"):
                    yield {"text": message["content"]}
                if message.get("tool_calls"):
                    yield {"tool_calls": message["tool_calls"]}

    async def describe_photo(self, image: bytes, prompt: str = PHOTO_PROMPT) -> str:
        """What is in a photo, from the vision model, for Luna to react to."""
        reply = await self.client.post("/api/chat", json={
            "model": self.config.vision_model,
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]}],
            "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.2},
        })
        reply.raise_for_status()
        return (reply.json().get("message") or {}).get("content", "").strip()

    async def unload_all(self) -> None:
        """Free the GPU for an image: every model Ollama holds, whichever app loaded it."""
        reply = await self.client.get("/api/ps")
        reply.raise_for_status()
        for model in reply.json().get("models") or []:
            await self.client.post("/api/generate", json={"model": model["name"], "keep_alive": 0})
