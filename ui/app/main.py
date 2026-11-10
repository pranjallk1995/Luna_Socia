"""Run Luna Socia: the owner's chat with Luna, by text, voice and pictures."""

from __future__ import annotations

import logging
import time

import uvicorn

from app.chat import Conversation
from app.config import SociaConfig
from app.images import ComfyUI
from app.ollama import Ollama
from app.store import SociaStore
from app.voice import Voice
from app.web import create_app

logger = logging.getLogger("socia")


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
    comfy = ComfyUI(config)
    app = create_app(config, store, Conversation(config, store, Ollama(config), comfy), comfy, Voice(config))
    uvicorn.run(app, host="0.0.0.0", port=SociaConfig.PORT, proxy_headers=True, forwarded_allow_ips="*",
                log_level="info")


if __name__ == "__main__":
    main()
