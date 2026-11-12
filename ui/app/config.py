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
    # Messages the page loads at a time, and the longest chat title, taken from its first message.
    MESSAGES_PAGE: ClassVar[int] = 50
    CHAT_TITLE_CHARS: ClassVar[int] = 60
    # Largest photo or voice clip accepted from the page.
    MAX_UPLOAD_BYTES: ClassVar[int] = 25 * 1024 * 1024
    # More pictures of Luna besides her profile picture, sent with it as references for her face.
    # Each one makes a picture slower and needs more GPU memory.
    MAX_REFERENCES: ClassVar[int] = 4
    # The longest one image may take, including loading the model from disk.
    IMAGE_TIMEOUT_SECONDS: ClassVar[float] = 600.0
    # The longest name a girlfriend can have, and what the owner writes about her.
    NAME_CHARS: ClassVar[int] = 40
    ABOUT_CHARS: ClassVar[int] = 2000
    # Chats exported from ChatGPT or Claude (a .zip or its conversations.json), and the
    # largest conversations.json read from one.
    MAX_EXPORT_BYTES: ClassVar[int] = 512 * 1024 * 1024
    MAX_EXPORT_JSON_BYTES: ClassVar[int] = 400 * 1024 * 1024
    # What she remembers about the owner goes into every reply, so it stays short:
    # about 1,200 tokens of the 8k context.
    MEMORY_CHARS: ClassVar[int] = 4800
    # The owner's own words from one imported chat that the model reads for facts.
    MEMORY_READ_CHARS: ClassVar[int] = 6000

    database_url: str
    auth_internal_url: str
    auth_public_url: str
    public_url: str
    ollama_url: str = "http://luna-sentia-ollama:11434"
    comfyui_url: str = "http://luna-sentia-comfyui:8188"
    chat_model: str = "huihui_ai/qwen3-abliterated:14b"
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
    # The languages voice messages are in: Whisper picks the likeliest of these instead of any of its
    # 99, which on a short or noisy clip can be a language nobody spoke.
    stt_languages: tuple[str, ...] = ("en", "hi")
    tts_voice: str = "af_heart"
    data_dir: Path = Path("/data")
    # Shared with Luna Sentia (LUNASOCIA_API_TOKEN in both .env files), whose model monitor
    # asks GET /api/models about the voice models; blank lets only the owner's session ask.
    api_token: str = ""

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
            stt_languages=tuple(code.strip().lower() for code in setting("LUNASOCIA_STT_LANGUAGES", "en,hi").split(",")
                                if code.strip()),
            tts_voice=setting("LUNASOCIA_TTS_VOICE", cls.tts_voice),
            data_dir=Path(setting("LUNASOCIA_DATA_DIR", "/data")),
            api_token=os.getenv("LUNASOCIA_API_TOKEN", "").strip(),
        )

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    def companion_dir(self, companion_id: int) -> Path:
        """One girlfriend's own files: her profile picture, avatar crop and other pictures."""
        return self.data_dir / "companions" / str(companion_id)

    def profile_path(self, companion_id: int) -> Path:
        """Her profile picture: her face in every image she appears in."""
        return self.companion_dir(companion_id) / "profile.png"

    def references_dir(self, companion_id: int) -> Path:
        """More pictures of her, so her face stays the same at other angles and expressions."""
        return self.companion_dir(companion_id) / "references"

    def references(self, companion_id: int) -> list[Path]:
        """Her other pictures, in the order they were added."""
        folder = self.references_dir(companion_id)
        return sorted(folder.glob("*.jpg"))[:self.MAX_REFERENCES] if folder.is_dir() else []

    @property
    def legacy_dir(self) -> Path:
        """Where Luna's files were before there could be more than one girlfriend."""
        return self.data_dir / "luna"

    @property
    def models_dir(self) -> Path:
        """Voice models, downloaded on first use."""
        return self.data_dir / "models"
