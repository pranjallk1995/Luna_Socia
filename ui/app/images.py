"""Pictures from Luna Sentia's image service (Qwen-Image 2.1 Turbo with diffusers), with Luna's face from her
profile picture.

The service takes the GPU to itself for each picture: it unloads Ollama's models, reads the prompt and up to
16 reference pictures (which its text encoder both sees and splices into the picture's latents; the picture
takes the first one's shape), draws in 8 steps, and leaves the GPU empty for the monitor to load her models
back. Socia sends the prompt and references, then asks until the picture is done.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import httpx

from app.config import SociaConfig

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0


class ImageError(RuntimeError):
    pass


class ImageService:
    def __init__(self, config: SociaConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.client = httpx.AsyncClient(base_url=config.imagegen_url, transport=transport,
                                        timeout=httpx.Timeout(60.0, connect=5.0))

    async def available(self) -> bool:
        try:
            reply = await self.client.get("/health")
            return reply.status_code == 200 and bool(reply.json().get("ready"))
        except (httpx.HTTPError, ValueError):
            return False

    async def generate(self, prompt: str, references: list[bytes]) -> bytes:
        reply = await self.client.post("/pictures", json={
            "prompt": prompt, "resolution": self.config.image_resolution,
            "references": [base64.b64encode(image).decode() for image in references]})
        if reply.status_code != 202:
            raise ImageError(f"The image service refused the picture: {error_of(reply)}")
        picture = reply.json()["id"]
        done = False
        try:
            image = await asyncio.wait_for(self.result(picture), SociaConfig.IMAGE_TIMEOUT_SECONDS)
            done = True
            return image
        finally:
            # A picture given up on (too slow, or the reply cancelled) is stopped, so it does not hold the GPU.
            if not done:
                await self.cancel(picture)

    async def cancel(self, picture: str) -> None:
        try:
            await self.client.delete(f"/pictures/{picture}")
        except httpx.HTTPError:
            logger.warning("Could not stop the picture %s", picture)

    async def result(self, picture: str) -> bytes:
        while True:
            await asyncio.sleep(POLL_SECONDS)
            reply = await self.client.get(f"/pictures/{picture}")
            if reply.status_code == 404:
                raise ImageError("The image service lost the picture (was it restarted?).")
            state = reply.json()
            if state.get("status") == "done":
                image = await self.client.get(f"/pictures/{picture}/image")
                image.raise_for_status()
                return image.content
            if state.get("status") in ("failed", "cancelled"):
                raise ImageError(f"The picture failed: {state.get('error') or state['status']}")


def error_of(reply: httpx.Response) -> str:
    try:
        return str(reply.json().get("error") or reply.text)[:500]
    except ValueError:
        return reply.text[:500]
