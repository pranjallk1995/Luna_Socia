"""Run Luna Socia: the owner's chats with their girlfriends, by text, voice and pictures."""

from __future__ import annotations

import logging
import time

import uvicorn

from app.chat import Conversation
from app.config import SociaConfig
from app.images import ComfyUI
from app.memory import MemoryReader
from app.ollama import Ollama
from app.persona import DEFAULT_PERSONA
from app.store import SociaStore
from app.voice import Voice
from app.web import create_app

logger = logging.getLogger("socia")


def adopt_legacy_files(config: SociaConfig, store: SociaStore) -> None:
    """Luna's files from before there could be several girlfriends move into her own folder."""
    legacy = config.legacy_dir
    if not legacy.is_dir():
        return
    luna = store.oldest_companion() or store.create_companion("Luna", dict(DEFAULT_PERSONA), config.tts_voice)
    target = config.companion_dir(luna["id"])
    if target.exists():
        logger.warning("Both %s and %s exist; leaving %s as it is", legacy, target, legacy)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(target)
    logger.info("Moved %s to %s", legacy, target)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = SociaConfig.from_env()
    store = SociaStore(config.database_url)
    for attempt in range(30):
        try:
            store.initialize()
            break
        except Exception as error:  # noqa: BLE001 - the database may still be starting
            logger.warning("Database not ready (%s), retrying", error)
            time.sleep(2)
    else:
        raise RuntimeError("Socia's database could not be reached.")
    adopt_legacy_files(config, store)
    comfy = ComfyUI(config)
    ollama = Ollama(config)
    conversation = Conversation(config, store, ollama, comfy)
    reader = MemoryReader(store, ollama, conversation.turn)
    app = create_app(config, store, conversation, comfy, Voice(config), reader=reader)
    uvicorn.run(app, host="0.0.0.0", port=SociaConfig.PORT, proxy_headers=True, forwarded_allow_ips="*",
                log_level="info")


if __name__ == "__main__":
    main()
