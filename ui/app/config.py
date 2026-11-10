"""Settings for Luna Socia, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class SociaConfig:
    VERSION: ClassVar[str] = os.getenv("LUNA_VERSION", "").strip()
    BRAND: ClassVar[str] = f"Luna Socia · v{VERSION}" if VERSION else "Luna Socia"
    PORT: ClassVar[int] = 8501
    # Header the auth service reads a luna_sso token from, and the owner's fixed account id.
    SESSION_HEADER: ClassVar[str] = "X-Luna-Session"
    OWNER_ID: ClassVar[str] = "owner"
    COOKIE_NAME: ClassVar[str] = "luna_sso"
    SESSION_RECHECK_SECONDS: ClassVar[float] = 60.0
    # Messages of the conversation Luna sees on each reply.
    HISTORY_MESSAGES: ClassVar[int] = 40
    # Largest photo or voice clip accepted from the page.
    MAX_UPLOAD_BYTES: ClassVar[int] = 25 * 1024 * 1024
    # The longest one image may take, including loading the model from disk.
    IMAGE_TIMEOUT_SECONDS: ClassVar[float] = 600.0

    database_url: str
    auth_internal_url: str
    auth_public_url: str
    public_url: str
    ollama_url: str = "http://luna-sentia-ollama:11434"
    comfyui_url: str = "http://luna-sentia-comfyui:8188"
    chat_model: str = "qwen3:14b"
    vision_model: str = "qwen3-vl:8b-instruct"
    # 8k holds the last 40 short messages and leaves room on the GPU; it must equal
    # SENTIA_RESIDENT_CONTEXT, or Ollama reloads the model for every reply.
    context_tokens: int = 8192
    temperature: float = 0.8
    # Qwen-Image 2.1's pixel budget: about resolution x resolution, keeping the aspect ratio.
    image_resolution: int = 1024
    image_model: str = "qwen_image_2.1_turbo_bf16.safetensors"
    image_text_encoder: str = "qwen3vl_8b_int8_convrot.safetensors"
    image_vae: str = "qwen_image_2.1_vae_bf16.safetensors"
    stt_model: str = "large-v3-turbo"
    tts_voice: str = "af_heart"
    data_dir: Path = Path("/data")

    @classmethod
    def from_env(cls) -> SociaConfig:
        def required(name: str) -> str:
            value = os.getenv(name, "").strip()
            if not value:
                raise RuntimeError(f"{name} is required.")
            return value

        def setting(name: str, default: str) -> str:
            return os.getenv(name, "").strip() or default

        return cls(
            database_url=required("DATABASE_URL"),
            auth_internal_url=setting("LUNA_SSO_INTERNAL_URL", "http://luna-sentia-auth:8000").rstrip("/"),
            auth_public_url=required("LUNA_SSO_URL").rstrip("/"),
            public_url=required("UI_PUBLIC_URL").rstrip("/"),
            ollama_url=setting("LUNASOCIA_OLLAMA_URL", cls.ollama_url).rstrip("/"),
            comfyui_url=setting("LUNASOCIA_COMFYUI_URL", cls.comfyui_url).rstrip("/"),
            chat_model=setting("LUNASOCIA_CHAT_MODEL", cls.chat_model),
            vision_model=setting("LUNASOCIA_VISION_MODEL", cls.vision_model),
            context_tokens=int(setting("LUNASOCIA_CONTEXT_TOKENS", str(cls.context_tokens))),
            temperature=float(setting("LUNASOCIA_TEMPERATURE", str(cls.temperature))),
            image_resolution=int(setting("LUNASOCIA_IMAGE_RESOLUTION", str(cls.image_resolution))),
            image_model=setting("LUNASOCIA_IMAGE_MODEL", cls.image_model),
            image_text_encoder=setting("LUNASOCIA_IMAGE_TEXT_ENCODER", cls.image_text_encoder),
            image_vae=setting("LUNASOCIA_IMAGE_VAE", cls.image_vae),
            stt_model=setting("LUNASOCIA_STT_MODEL", cls.stt_model),
            tts_voice=setting("LUNASOCIA_TTS_VOICE", cls.tts_voice),
            data_dir=Path(setting("LUNASOCIA_DATA_DIR", "/data")),
        )

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def profile_path(self) -> Path:
        """Luna's profile picture: her face in every image she appears in."""
        return self.data_dir / "luna" / "profile.png"

    @property
    def models_dir(self) -> Path:
        """Voice models, downloaded on first use."""
        return self.data_dir / "models"
