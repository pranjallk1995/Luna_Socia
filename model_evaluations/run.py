"""Answer every scenario in scenarios.json with one chat model, as Socia would; write JSONL.

usage: python run.py <label: old|new> <model> <out.jsonl> [--repeat 3]
Runs inside the ui container with the app's code on PYTHONPATH (/workspace/ui). Each
reply goes through Socia's own Ollama client, system prompt, tools and settings
(temperature, context); only the chat model differs, and nothing is stored or sent to
ComfyUI. Replies are sampled, so each scenario runs --repeat times.
"""

import argparse
import asyncio
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import SociaConfig
from app.ollama import Ollama
from app.persona import TOOLS, system_prompt

parser = argparse.ArgumentParser()
parser.add_argument("label", choices=["old", "new"])
parser.add_argument("model")
parser.add_argument("out", type=Path)
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
args = parser.parse_args()

config = dataclasses.replace(SociaConfig.from_env(), chat_model=args.model)
# A fixed name and time, so both models see the same prompt.
PROMPT = system_prompt("Sam", datetime(2026, 11, 5, 20, 30))


async def reply(ollama: Ollama, messages: list[dict]) -> dict:
    text, calls = "", []
    async for piece in ollama.chat([{"role": "system", "content": PROMPT}, *messages], TOOLS):
        text += piece.get("text", "")
        calls += piece.get("tool_calls", [])
    return {"text": text, "tool_calls": calls}


async def main() -> None:
    scenarios = json.loads(args.scenarios.read_text())
    ollama = Ollama(config)
    with args.out.open("w") as out:
        for scenario in scenarios:
            for attempt in range(args.repeat):
                started = time.monotonic()
                row = {"id": scenario["id"], "attempt": attempt, "label": args.label, "model": args.model}
                try:
                    row.update(await reply(ollama, scenario["messages"]))
                except Exception as error:  # noqa: BLE001 - a failure is a result
                    row["crash"] = f"{type(error).__name__}: {error}"[:500]
                row["seconds"] = round(time.monotonic() - started, 1)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{args.label}] {scenario['id']} #{attempt} {row['seconds']}s", flush=True)


asyncio.run(main())
