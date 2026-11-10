"""Pictures from Qwen-Image 2.1 in ComfyUI, with Luna's face from her profile picture.

The workflow follows ComfyUI's own "Qwen Image 2.1 Turbo" template (8 steps, cfg 1,
the turbo sigmas), without its prompt enhancer. Up to 16 reference images go to the
text encoder, which both sees them and splices them into the latent sequence; the
output takes the first reference's shape.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any

import httpx

from app.config import SociaConfig

# Qwen-Image 2.1 Turbo's sampling schedule, from its model_index.json via ComfyUI's template.
TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
POLL_SECONDS = 1.0


def workflow(config: SociaConfig, prompt: str, references: list[str], seed: int) -> dict[str, Any]:
    """ComfyUI's API graph; ``references`` are file names already uploaded to its input folder."""
    graph: dict[str, Any] = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": config.image_model, "weight_dtype": "default"}},
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "auto", "dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": config.image_text_encoder, "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": config.image_vae}},
        "encode": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt, "negative_prompt": "",
            "resolution": config.image_resolution}},
        "guider": {"class_type": "CFGGuider",
                   "inputs": {"model": ["cache", 0], "positive": ["encode", 0], "negative": ["encode", 1], "cfg": 1.0}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "ManualSigmas", "inputs": {"sigmas": TURBO_SIGMAS}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
            "latent_image": ["encode", 2]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "socia"}},
    }
    for number, name in enumerate(references, start=1):
        graph[f"ref{number}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["encode"]["inputs"][f"images.image_{number}"] = [f"ref{number}", 0]
    return graph


class ImageError(RuntimeError):
    pass


class ComfyUI:
    def __init__(self, config: SociaConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.client = httpx.AsyncClient(base_url=config.comfyui_url, transport=transport,
                                        timeout=httpx.Timeout(60.0, connect=5.0))
        # One image at a time: the GPU holds one image model, and Ollama waits meanwhile.
        self.lock = asyncio.Lock()

    async def available(self) -> bool:
        try:
            return (await self.client.get("/system_stats")).status_code == 200
        except httpx.HTTPError:
            return False

    async def upload(self, image: bytes) -> str:
        name = f"socia-{uuid.uuid4().hex}.png"
        reply = await self.client.post("/upload/image", files={"image": (name, image, "image/png")},
                                       data={"overwrite": "true"})
        reply.raise_for_status()
        return reply.json()["name"]

    async def generate(self, prompt: str, references: list[bytes]) -> bytes:
        async with self.lock:
            names = [await self.upload(image) for image in references]
            graph = workflow(self.config, prompt, names, random.randrange(2**48))
            client_id = uuid.uuid4().hex
            reply = await self.client.post("/prompt", json={"prompt": graph, "client_id": client_id})
            if reply.status_code != 200:
                raise ImageError(f"ComfyUI refused the workflow: {reply.text[:500]}")
            prompt_id = reply.json()["prompt_id"]
            try:
                return await asyncio.wait_for(self.result(prompt_id), SociaConfig.IMAGE_TIMEOUT_SECONDS)
            finally:
                # Hand the GPU back to Ollama.
                await self.client.post("/free", json={"unload_models": True, "free_memory": True})

    async def result(self, prompt_id: str) -> bytes:
        while True:
            await asyncio.sleep(POLL_SECONDS)
            history = (await self.client.get(f"/history/{prompt_id}")).json().get(prompt_id)
            if not history:
                continue
            status = history.get("status") or {}
            if status.get("status_str") == "error":
                messages = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                detail = messages[0][1].get("exception_message", "") if messages else ""
                raise ImageError(f"ComfyUI failed: {detail.strip()[:500]}")
            for output in (history.get("outputs") or {}).values():
                for image in output.get("images") or []:
                    view = await self.client.get("/view", params={
                        "filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")})
                    view.raise_for_status()
                    return view.content
            if status.get("completed"):
                raise ImageError("ComfyUI finished without an image.")
