# Luna Socia

![Status: early prototype](https://img.shields.io/badge/Status-early_prototype-D73A49)

> **Work in progress:** Luna Socia is an early prototype. Luna's voice still comes from a
> prompt alone; the fine-tuned model, links to the other Luna apps, and long-term memory
> are next.

> **Requires Luna Sentia, running and healthy:** Socia's chat, vision and image models
> run in [Luna Sentia](https://github.com/pranjallk1995/Luna_Sentia), and its data is in
> Sentia's PostgreSQL. Start Sentia first, as described under [Start](#start).

The owner's private chat with Luna: text, voice messages, photos, and pictures of Luna.
Socia is one service (`ui`): the chat page and its API, behind Luna's single sign-on, for
the owner only. Its models run in Luna Sentia, which every Luna app shares.

| Part | Model | Where it runs | Before changing it |
| --- | --- | --- | --- |
| Chat | `LUNASOCIA_CHAT_MODEL` (default `qwen3:14b`, about 9.6 GB with 8k context) | Sentia's Ollama, through its monitor, on the GPU | Compare it with [`model_evaluations/`](model_evaluations/README.md) and check it fits with Sentia's `model_checks/check_model.py --context 8192` |
| Photos sent to Luna | `LUNASOCIA_VISION_MODEL` (default `qwen3-vl:8b-instruct`) describes them for her | Sentia's Ollama, on the GPU | Sentia's `model_checks/check_model.py`, then a few photos by hand |
| Pictures of Luna | Qwen-Image 2.1 Turbo, 8 steps; her profile picture is the face reference | Sentia's ComfyUI, on the GPU | A few pictures by hand |
| Voice messages | faster-whisper `LUNASOCIA_STT_MODEL` (default `large-v3-turbo`, int8) | Here, on the CPU | A few voice messages by hand |
| Her voice | Kokoro 82M, voice `LUNASOCIA_TTS_VOICE` (default `af_heart`) | Here, on the CPU | Listen to a few replies |

## Luna ecosystem

![Luna ecosystem architecture](docs/luna-ecosystem.svg)

Socia is one service, with no MCP server, on Luna Sentia's `luna_sentia_network`. It signs
in through Sentia's `auth`, keeps its messages in Sentia's PostgreSQL (`lunasocia`), sends
chat and photo calls through Sentia's model monitor to Ollama, and asks Sentia's ComfyUI for
pictures. It is not on Luna Unia's home page yet.

## How a turn works

1. What you send is stored in the `lunasocia` database. A photo is first described by the
   vision model; Luna reads that description in brackets, which is how she "sees" it.
2. Luna's reply streams from the chat model, with her persona (`ui/app/persona.py`) and the
   last 40 messages. Her own looks, described once from her profile picture, are part of it.
3. When a picture fits (you ask for a selfie, or for her to step into your photo), she calls
   her `create_image` tool. Socia unloads every Ollama model, sends ComfyUI the Qwen-Image
   2.1 workflow (`ui/app/images.py`) with her profile picture and, if asked, your last photo
   as references, then has ComfyUI free its model again.

Measured on an RTX 5070 (12 GB): a reply starts in about 0.3 s once the chat model is
loaded (about 25 s to load it); a picture takes about 40 s; her voice about 0.4 s per
reply; a 4-second voice message about 4 s to transcribe.

## Start

Luna Sentia must be running, with `qwen3:14b` in its `SENTIA_MODELS`, its ComfyUI and
image-model download, and `LUNASOCIA_DB_PASSWORD` set (Sentia creates the `lunasocia`
role and database).

```bash
cp .env.example .env    # then fill it in
../Luna_Sentia/luna.sh start socia
```

Open it at `UI_PUBLIC_URL`: route that address to `127.0.0.1:18507` in the tunnel, under
`SENTIA_AUTH_COOKIE_DOMAIN`, so the sign-in cookie reaches it. Set Luna's profile picture
from the menu on her avatar: it becomes her face in every picture she takes.

## Data

The `luna_socia_data` volume holds her profile picture (`/data/luna`), the pictures and
photos of the conversation (`/data/images`), and the voice models, downloaded the first
time they are used (`/data/models`). "Start the conversation over" in the menu deletes the
messages and their pictures, and keeps her profile picture.

## Changing a model

Before changing `LUNASOCIA_CHAT_MODEL`, compare the new model with the current one:
[`model_evaluations/`](model_evaluations/README.md) answers a fixed set of made-up
conversations with both, through Socia's own prompt and `create_image` tool, and
scores her texting style, her language, whether she calls `create_image` when (and
only when) a picture is asked for, and time. The chat model stays resident, so check
that it fits with Luna Sentia's `model_checks/check_model.py <model> --context 8192`;
the full list of checks is in Sentia's
[Changing a model](https://github.com/pranjallk1995/Luna_Sentia#changing-a-model).

## Tests

```bash
docker compose build ui
docker run --rm --network none --entrypoint python -v "$PWD/ui/tests:/workspace/ui/tests:ro" \
  ghcr.io/pranjallk1995/luna-socia-ui:v0.1.1 -m unittest discover -s tests
```
