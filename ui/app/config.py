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
    # When the vision model answers about photos: the messages it reads, the photos and pictures it
    # sees (newest first), and their longer side; it reads about one token per 32 x 32 pixels.
    PHOTO_HISTORY_MESSAGES: ClassVar[int] = 16
    # Face prints at least this similar are the same person: her pictures scored 0.60-0.88 against her
    # others, women drawn to look like her 0.30-0.52 (model_evaluations/self_recognition.py).
    FACE_MATCH: ClassVar[float] = 0.55
    # Faces named for one picture, largest first; and photos of the owner's face kept.
    FACES_NAMED: ClassVar[int] = 4
    MAX_OWNER_PHOTOS: ClassVar[int] = 5
    PHOTOS_SEEN: ClassVar[int] = 3
    PHOTO_SEEN_SIDE: ClassVar[int] = 1024
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
    # Her private notes for one chat: a line is at most NOTE_CHARS, and the oldest lines go past NOTES_CHARS.
    NOTE_CHARS: ClassVar[int] = 300
    NOTES_CHARS: ClassVar[int] = 1500
    # The owner's own words from one imported chat that the model reads for facts.
    MEMORY_READ_CHARS: ClassVar[int] = 6000

    database_url: str
    auth_internal_url: str
    auth_public_url: str
    public_url: str
    ollama_url: str = "http://luna-sentia-ollama:11434"
    comfyui_url: str = "http://luna-sentia-comfyui:8188"
    chat_model: str = "huihui_ai/qwen3-abliterated:8b"
    vision_model: str = "huihui_ai/qwen3-vl-abliterated:8b-instruct"
    # 8k holds the last 40 short messages and leaves room on the GPU; it must equal
    # SENTIA_RESIDENT_CONTEXT, or Ollama reloads the model for every reply.
    context_tokens: int = 8192
    # The vision model's context. It sees the photos and answers as her, with her prompt and the
    # last PHOTO_HISTORY_MESSAGES messages: 8k holds that and three photos.
    vision_context_tokens: int = 8192
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
    # Where her voice runs: "cuda" (the GPU, loaded at start and kept) or "cpu".
    voice_device: str = "cuda"
    # Read her chats, once quiet, for lasting facts (LUNASOCIA_READ_CHATS=1). Off until it is reliable on real
    # chats: on 17 November it filled her memory with what photos showed and gave her the owner's cats.
    read_chats: bool = False
    # After each text turn, a small classifier decides whether your message or her reply holds a lasting fact, and
    # her chat model writes it into her memory or bio (app/remembering.py). LUNASOCIA_REMEMBER_AUTO=0 leaves it to
    # her remember tool alone. Its embedding model runs on the "cpu" (no GPU memory) or "cuda" (1.2 GB).
    remember_auto: bool = True
    remember_device: str = "cpu"
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
            vision_context_tokens=int(setting("LUNASOCIA_VISION_CONTEXT_TOKENS", str(cls.vision_context_tokens))),
            temperature=float(setting("LUNASOCIA_TEMPERATURE", str(cls.temperature))),
            image_resolution=int(setting("LUNASOCIA_IMAGE_RESOLUTION", str(cls.image_resolution))),
            image_model=setting("LUNASOCIA_IMAGE_MODEL", cls.image_model),
            image_text_encoder=setting("LUNASOCIA_IMAGE_TEXT_ENCODER", cls.image_text_encoder),
            image_vae=setting("LUNASOCIA_IMAGE_VAE", cls.image_vae),
            stt_model=setting("LUNASOCIA_STT_MODEL", cls.stt_model),
            stt_languages=tuple(code.strip().lower() for code in setting("LUNASOCIA_STT_LANGUAGES", "en,hi").split(",")
                                if code.strip()),
            tts_voice=setting("LUNASOCIA_TTS_VOICE", cls.tts_voice),
            voice_device=setting("LUNASOCIA_VOICE_DEVICE", cls.voice_device).lower(),
            read_chats=setting("LUNASOCIA_READ_CHATS", "") == "1",
            remember_auto=setting("LUNASOCIA_REMEMBER_AUTO", "1") != "0",
            remember_device=setting("LUNASOCIA_REMEMBER_DEVICE", cls.remember_device).lower(),
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

    @property
    def owner_photos_dir(self) -> Path:
        """Photos of the owner's face, so she knows them in the photos they send."""
        return self.data_dir / "owner" / "photos"

    def owner_photos(self) -> list[Path]:
        folder = self.owner_photos_dir
        return sorted(folder.glob("*.jpg"))[:self.MAX_OWNER_PHOTOS] if folder.is_dir() else []

    @property
    def owner_picture_path(self) -> Path:
        """The owner's own profile picture, shown at the top right."""
        return self.data_dir / "owner" / "profile.png"

    def owner_faces(self) -> list[Path]:
        """Every picture of the owner face recognition knows them by: their profile picture and photos."""
        picture = self.owner_picture_path
        return [*([picture] if picture.exists() else []), *self.owner_photos()]

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
