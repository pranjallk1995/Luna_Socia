# Luna Socia

![Status: early prototype](https://img.shields.io/badge/Status-early_prototype-D73A49)

> **Work in progress:** Luna Socia is an early prototype. Each girlfriend's voice still
> comes from her persona prompt; her own fine-tuned model (trained on the chats imported
> into her memory), links to the other Luna apps, and long-term memory of Socia's own chats
> are next.

> **Requires Luna Sentia, running and healthy:** Socia's chat, vision and image models
> run in [Luna Sentia](https://github.com/pranjallk1995/Luna_Sentia), and its data is in
> Sentia's PostgreSQL. Start Sentia first, as described under [Start](#start).

The owner's private chats with their AI girlfriends: text, voice messages, photos, and
pictures of her. The page opens on the girlfriends to choose from, with a card to create a
new one. Socia is one service (`ui`): the page and its API, behind Luna's single sign-on,
for the owner only. Its models run in Luna Sentia, which every Luna app shares.

| Part | Model | Where it runs | Before changing it |
| --- | --- | --- | --- |
| Chat | `LUNASOCIA_CHAT_MODEL` (default `huihui_ai/qwen3-abliterated:14b`, an uncensored Qwen3 14B, about 9.4 GB with 8k context) | Sentia's Ollama, through its monitor, on the GPU | Compare it with [`model_evaluations/`](model_evaluations/README.md) and check it fits with Sentia's `model_checks/check_model.py --context 8192` |
| Photos sent to Luna | `LUNASOCIA_VISION_MODEL` (default `qwen3-vl:8b-instruct`) describes them for her | Sentia's Ollama, on the GPU | Sentia's `model_checks/check_model.py`, then a few photos by hand |
| Pictures of Luna | Qwen-Image 2.1 Turbo, 8 steps; her profile picture and up to 4 other pictures of her are the face references | Sentia's ComfyUI, on the GPU | A few pictures by hand |
| Voice messages | faster-whisper `LUNASOCIA_STT_MODEL` (default `large-v3-turbo`, int8) | Here, on the CPU | A few voice messages by hand |
| Her voice | Kokoro 82M, in the voice chosen for her (16 female English and Hindi voices; `LUNASOCIA_TTS_VOICE`, default `af_heart`, for new girlfriends) | Here, on the CPU | Listen to a few replies |
| Her memory | The chat model reads imported ChatGPT or Claude chats for facts about you, in the background | Sentia's Ollama, on the GPU, only while no reply is being written | Import a small export and read what she remembers |

## Luna ecosystem

![Luna ecosystem architecture](docs/luna-ecosystem.svg)

Socia is one service, with no MCP server, on Luna Sentia's `luna_sentia_network`. It signs
in through Sentia's `auth`, keeps its messages in Sentia's PostgreSQL (`lunasocia`), sends
chat and photo calls through Sentia's model monitor to Ollama, and asks Sentia's ComfyUI for
pictures. It is not on Luna Unia's home page yet.

## Girlfriends

Create one from the card on the landing page, and change her later with "Edit her":

- **Who she is:** her name, and anything you write about her (age, work, what she loves,
  how you met), which goes into her persona.
- **Her picture:** her profile picture is her face in every picture she takes. Crop her
  avatar and add up to 4 more pictures of her from her chat's settings.
- **How she texts:** one of five styles (playful, sweet, sassy, shy, calm), reply length,
  how many emoji, and her language (yours, English, Hinglish, or Hindi in Devanagari).
- **Her voice:** one of Kokoro's English and Hindi voices, each with a sample to play.
  Messages in Devanagari are spoken with the Hindi phonemizer.
- **Her memory:** upload the export ChatGPT or Claude emails you (the .zip, or the
  `conversations.json` inside it). Socia keeps every conversation in `memory_sources`, then
  the chat model reads your own messages in each one (not the assistant's) for lasting
  facts, about 2 s a conversation, and merges them into a short list that goes into
  every reply (up to 4,800 characters). Reading pauses while she answers you; the form shows
  its progress, and you can edit what she remembers. Importing the same export again skips
  the conversations she already has.

Her persona and memory are prompt-only for now: a separate model per girlfriend would
not fit on the GPU beside the resident chat model. The imported conversations stay in the
database as the training data for her fine-tuned model later.

Luna, from before there could be several, becomes the first girlfriend when Socia starts:
her chats move to her, and her files to `/data/companions/<id>`.

## How a turn works

1. What you send is stored in the `lunasocia` database, in its chat: your first message in a
   new chat starts it and names it. A photo is first described by the
   vision model; Luna reads that description in brackets, which is how she "sees" it.
2. Her reply streams from the chat model, with her persona (`ui/app/persona.py`), what she
   remembers about you (`ui/app/memory.py`), and the last 40 messages of this chat. Her own
   looks, described once from her profile picture, are part of it.
3. When a picture fits (you ask for a selfie, or for her to step into your photo from this chat), she calls
   her `create_image` tool. Socia unloads every Ollama model, sends ComfyUI the Qwen-Image
   2.1 workflow (`ui/app/images.py`) with her profile picture, her other pictures and, if asked, your last photo
   as references, then has ComfyUI free its model again.

Measured on an RTX 5070 (12 GB): a reply starts in about 0.3 s once the chat model is
loaded (about 25 s to load it); a picture takes about 40 s; her voice about 0.4 s per
reply; a 4-second voice message about 4 s to transcribe.

## Start

Clone Socia beside Luna Sentia and start it; no `.env` is needed. Sentia's defaults pull
`huihui_ai/qwen3-abliterated:14b` and the image model, and create the `lunasocia` role and database with the
password `lunasocia-dev`, which is Socia's default too.

```bash
../Luna_Sentia/luna.sh start socia
```

Open it at http://localhost:18503 and sign in as the owner. With a tunnel, copy
`.env.example` to `.env`, set `UI_PUBLIC_URL`, and route that address to `127.0.0.1:18503`,
under `SENTIA_AUTH_COOKIE_DOMAIN`, so the sign-in cookie reaches it. Choose a girlfriend, or create
one: her profile picture becomes her face in every picture she takes.

## Data

The `luna_socia_data` volume holds each girlfriend's profile picture and avatar crop
(`/data/companions/<id>`) and her other pictures (`/data/companions/<id>/references`, added
and removed in her settings), the pictures and photos of the chats (`/data/images`), and the
voice models, downloaded the first time they are used (`/data/models`).

The database keeps each girlfriend (`companions`: persona, voice, memory), her chats (`chats`,
with their `messages` and `images`), and the conversations imported into her memory
(`memory_sources`). Every chat is kept, so her long-term memory can draw on all of them. On a
desktop, her chats are listed beside the conversation, and on a wide screen her profile is on
the other side; on a phone, open the list from the menu button at the top left. Start a new
chat with the pencil, and rename or delete a chat from the list. Deleting a chat removes its
messages and pictures; deleting a girlfriend (from "Edit her") removes her chats, pictures
and memory.

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
  ghcr.io/pranjallk1995/luna-socia-ui:v0.2.0 -m unittest discover -s tests
```
