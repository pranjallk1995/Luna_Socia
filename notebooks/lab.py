"""Helpers for the notebooks: what Socia's code sends to her models, what comes back, and pictures inline.

Nothing here writes to her database.
"""

from __future__ import annotations

import copy
import io
import json
import time
from pathlib import Path
from typing import Any

import httpx
from IPython.display import HTML, Markdown, display
from PIL import Image

from app.config import SociaConfig
from app.store import SociaStore


class Recorder(httpx.AsyncHTTPTransport):
    """An httpx transport that keeps every call Socia's code makes, and its answer.

    Hand it to Socia's own clients (``Ollama(config, transport=recorder)``) and their requests go out
    unchanged; ``recorder.calls`` then shows exactly what was sent and what came back.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        start = time.monotonic()
        response = await super().handle_async_request(request)
        body = await response.aread()
        try:
            sent = json.loads(request.content or b"null")
        except ValueError:
            sent = f"<{len(request.content)} bytes>"
        self.calls.append({"method": request.method, "path": request.url.path, "request": sent,
                           "status": response.status_code, "response": body.decode(errors="replace"),
                           "seconds": round(time.monotonic() - start, 2)})
        headers = [(k, v) for k, v in response.headers.items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)


def short(value: Any, limit: int = 2000) -> Any:
    """A request or answer, readable: pictures as their size, very long text cut."""
    value = copy.deepcopy(value)

    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: ([f"<picture, {len(i) * 3 // 4 // 1024} KB>" for i in v]
                        if k == "images" and isinstance(v, list) and all(isinstance(i, str) for i in v)
                        else walk(v)) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(i) for i in item]
        if isinstance(item, str) and len(item) > limit:
            return item[:limit] + f"… <{len(item) - limit} more characters>"
        return item

    return walk(value)


def show_json(value: Any) -> None:
    display(Markdown(f"```json\n{json.dumps(short(value), indent=1, ensure_ascii=False)}\n```"))


def answers(call: dict[str, Any]) -> list[dict[str, Any]]:
    """An Ollama answer as its JSON pieces: one for a plain answer, one per line for a streamed one."""
    return [json.loads(line) for line in call["response"].splitlines() if line.strip()]


def stats(call: dict[str, Any]) -> dict[str, Any]:
    """How many tokens went in and came out of a model call, and how long it took."""
    last = answers(call)[-1]
    seconds = lambda key: round((last.get(key) or 0) / 1e9, 2)  # noqa: E731 - Ollama counts nanoseconds
    out = {"model": last.get("model"), "tokens in": last.get("prompt_eval_count"),
           "tokens out": last.get("eval_count"), "loading s": seconds("load_duration"),
           "reading s": seconds("prompt_eval_duration"), "writing s": seconds("eval_duration"),
           "total s": seconds("total_duration")}
    if last.get("eval_count") and last.get("eval_duration"):
        out["tokens/s"] = round(last["eval_count"] / (last["eval_duration"] / 1e9))
    return out


def thumb(picture: bytes | str | Path, side: int = 360, label: str = "") -> None:
    """A picture inline, small."""
    data = Path(picture).read_bytes() if isinstance(picture, (str, Path)) else picture
    display(HTML(row_html([(data, label)], side)))


def thumbs(pictures: list[tuple[bytes | str | Path, str]], side: int = 260) -> None:
    """Several pictures side by side, each with a label."""
    display(HTML(row_html([(Path(p).read_bytes() if isinstance(p, (str, Path)) else p, label)
                           for p, label in pictures], side)))


def row_html(pictures: list[tuple[bytes, str]], side: int) -> str:
    import base64

    cells = []
    for data, label in pictures:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        size = image.size
        image.thumbnail((side, side))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=85)
        source = base64.b64encode(buffer.getvalue()).decode()
        cells.append(f'<figure style="margin:0 8px 8px 0;display:inline-block;vertical-align:top">'
                     f'<img src="data:image/jpeg;base64,{source}"><figcaption style="font-size:12px">'
                     f'{label} <span style="opacity:.6">{size[0]}×{size[1]}</span></figcaption></figure>')
    return "".join(cells)


def transcript(messages: list[dict[str, Any]], width: int = 600) -> None:
    """Messages as the chat model reads them, one per block."""
    lines = []
    for message in messages:
        content = message.get("content") or ""
        if len(content) > width:
            content = content[:width] + f"… <{len(content) - width} more>"
        calls = message.get("tool_calls")
        extra = f"\n   ↳ calls {json.dumps(short(calls), ensure_ascii=False)}" if calls else ""
        pictures = f"  [+{len(message['images'])} picture(s)]" if message.get("images") else ""
        lines.append(f"**{message['role']}**{pictures}: {content}{extra}")
    display(Markdown("\n\n".join(line.replace("\n", "  \n") for line in lines)))


async def models_on_gpu(config: SociaConfig) -> None:
    """What Ollama holds right now, and at which context: a model asked for at another context reloads."""
    async with httpx.AsyncClient(base_url=config.ollama_url, timeout=10) as client:
        models = (await client.get("/api/ps")).json().get("models") or []
    if not models:
        print("Ollama holds no model.")
    for model in models:
        print(f"{model['name']:<48} {model.get('size_vram', 0) / 2**30:5.1f} GB on the GPU, "
              f"context {model.get('context_length', '?')}")


def photo_messages(store: SociaStore, limit: int = 15) -> list[dict[str, Any]]:
    """Your latest messages with a photo, newest first."""
    with store.connect() as db:
        return db.execute(
            "SELECT m.id, m.chat_id, m.content, m.photo_note, m.created_at, i.file FROM messages m "
            "JOIN images i ON i.id = m.image_id WHERE i.kind = 'photo' ORDER BY m.id DESC LIMIT %s",
            (limit,)).fetchall()


def her_pictures(store: SociaStore, limit: int = 15) -> list[dict[str, Any]]:
    """Her latest pictures, newest first, with the description she gave her camera."""
    with store.connect() as db:
        return db.execute(
            "SELECT i.id, i.chat_id, i.file, i.prompt, i.created_at, m.id AS message_id, m.content FROM images i "
            "LEFT JOIN messages m ON m.image_id = i.id WHERE i.kind = 'generated' ORDER BY i.id DESC LIMIT %s",
            (limit,)).fetchall()


def message_after(store: SociaStore, message: dict[str, Any]) -> dict[str, Any] | None:
    """What she actually answered to a message back then."""
    with store.connect() as db:
        return db.execute("SELECT m.*, i.prompt AS image_prompt FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                          "WHERE m.chat_id = %s AND m.id > %s ORDER BY m.id LIMIT 1",
                          (message["chat_id"], message["id"])).fetchone()


def table(rows: list[dict[str, Any]], columns: list[str], width: int = 70) -> None:
    """Rows as a small table."""
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join("| " + " | ".join(" ".join(str(row.get(c) or "").split())[:width].replace("|", "/")
                                     for c in columns) + " |\n" for row in rows)
    display(Markdown(head + body))


# The GPU, read the way Sentia's monitor reads it (NVML). Under WSL2 the card's memory counts
# everything on it (Ollama, Sentia's image service, the other apps, Windows); Ollama's own share is in /api/ps.

def gpu() -> dict[str, Any] | None:
    """The card right now: memory used and total (GB), how busy it is (%), temperature and power."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        busy = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        return {"used_gb": round(memory.used / 2**30, 2), "total_gb": round(memory.total / 2**30, 2),
                "busy_pct": busy, "temp_c": temperature, "power_w": round(power)}
    except Exception as error:  # noqa: BLE001 - no GPU in this container
        print(f"No GPU counters here ({error}); start the notebooks with ./start.sh.")
        return None


async def ollama_models(config: SociaConfig) -> dict[str, dict[str, Any]]:
    async with httpx.AsyncClient(base_url=config.ollama_url, timeout=10) as client:
        models = (await client.get("/api/ps")).json().get("models") or []
    return {m["name"]: {"gb": round(m.get("size_vram", 0) / 2**30, 2), "context": m.get("context_length")}
            for m in models}


class watch:
    """Watches the GPU while a cell's work runs, then says what it took.

        async with watch(config, "describe the photo"):
            note = await ollama.describe_photo(photo)

    Prints the card's memory before, at its peak and after, how busy it was, and which Ollama
    models were loaded, pushed out or kept: a model asked for at another context reloads.
    """

    def __init__(self, config: SociaConfig, what: str = "", every: float = 0.1) -> None:
        self.config, self.what, self.every = config, what, every
        self.samples: list[dict[str, Any]] = []

    async def __aenter__(self) -> "watch":
        import asyncio
        import threading

        self.models_before = await ollama_models(self.config)
        self.before = gpu()
        self.stop = threading.Event()

        def sample() -> None:
            while not self.stop.is_set():
                reading = gpu()
                if reading:
                    self.samples.append({**reading, "at": time.monotonic()})
                self.stop.wait(self.every)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.started = time.monotonic()
        self.thread.start()
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.seconds = time.monotonic() - self.started
        self.stop.set()
        self.thread.join()
        self.after = gpu()
        self.models_after = await ollama_models(self.config)
        self.report()

    def report(self) -> None:
        lines = [f"**{self.what or 'This cell'}**: {self.seconds:.1f} s"]
        if self.before and self.samples:
            peak = max(s["used_gb"] for s in self.samples)
            busy = [s["busy_pct"] for s in self.samples]
            lines.append(f"GPU memory: {self.before['used_gb']} GB before → **{peak} GB at the peak** → "
                         f"{self.after['used_gb']} GB after, of {self.before['total_gb']} GB")
            lines.append(f"GPU busy: {sum(busy) / len(busy):.0f}% on average, {max(busy)}% at the most; "
                         f"up to {max(s['power_w'] for s in self.samples)} W, {max(s['temp_c'] for s in self.samples)} °C")
        for name in sorted(set(self.models_before) | set(self.models_after)):
            before, after = self.models_before.get(name), self.models_after.get(name)
            if before and not after:
                change = f"pushed out (had {before['gb']} GB)"
            elif after and not before:
                change = f"**loaded**: {after['gb']} GB at context {after['context']}"
            elif before["context"] != after["context"] or before["gb"] != after["gb"]:
                change = f"**reloaded** at context {before['context']} → {after['context']} ({after['gb']} GB)"
            else:
                change = f"stayed: {after['gb']} GB at context {after['context']}"
            lines.append(f"- `{name}` {change}")
        display(Markdown("  \n".join(lines)))

    def chart(self) -> None:
        """Memory and busy-ness over the cell's run, as a small chart."""
        if not self.samples:
            print("No samples.")
            return
        import matplotlib.pyplot as plt

        start = self.samples[0]["at"]
        times = [s["at"] - start for s in self.samples]
        figure, memory = plt.subplots(figsize=(8, 2.6))
        memory.plot(times, [s["used_gb"] for s in self.samples], color="#3b6fd8", label="memory (GB)")
        memory.set_ylim(0, self.samples[0]["total_gb"])
        memory.set_ylabel("GB")
        memory.set_xlabel("seconds")
        busy = memory.twinx()
        busy.plot(times, [s["busy_pct"] for s in self.samples], color="#d8743b", alpha=0.7, label="busy (%)")
        busy.set_ylim(0, 100)
        busy.set_ylabel("busy %")
        figure.legend(loc="upper left")
        figure.tight_layout()
        plt.show()


async def load_model(config: SociaConfig, model: str, context: int) -> None:
    """Load a model onto the GPU at a context, as Socia's first call to it would, and say what it took."""
    async with httpx.AsyncClient(base_url=config.ollama_url, timeout=300) as client, watch(config, f"Loading {model}"):
        reply = await client.post("/api/generate", json={"model": model, "options": {"num_ctx": context}})
        reply.raise_for_status()
