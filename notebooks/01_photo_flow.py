# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 01 · What happens when you send her a photo
#
# Every step Socia's code takes, from your upload to her reply, one cell at a time. The cells call
# her real code (`ui/app` on this branch), read her real chats and use the real models. **Nothing is
# written to her database and no picture is made.** The vision model does take the GPU from her
# chat model while it looks, as in the chat, and the last step brings the chat model back.
#
# The flow, in `chat.py` (`Conversation.send`, then `Conversation.answer`):
#
# | Step | Code | What happens | Model |
# |---|---|---|---|
# | 1 | `normalized_photo` | your upload is turned upright, shrunk to 2048 px at most, saved as JPEG | |
# | 2 | `sees_pictures` | does she look at pictures for this reply? Yes if your message has a photo, or asks about one sent in the last 4 messages | |
# | 3 | `system_prompt` + `SEEING` | her persona, the time, her looks, what she remembers, her notes, and "right now you see the pictures themselves" | |
# | 4 | `with_pictures` | her profile picture and the chat's last 3 photos or pictures go with your message, with a list of what each one is | |
# | 5 | `Ollama.chat(seeing=True)` | the **vision model** reads the last 16 messages and the pictures, and writes her reply. It takes the chat model's place on the GPU | vision 8B |
# | 6 | after the reply | tool calls she typed out, a picture she asked for, a picture she only promised | |
# | 7 | `Ollama.describe_photo` | while it is still loaded, the vision model writes 2–4 sentences about the photo (`PHOTO_PROMPT`), kept on your message | vision 8B |
# | 8 | `model_messages` | her **later** replies come from the chat model, which reads text only: the photo is `[Photo: <those sentences>]` | |
# | 9 | `Ollama.back_to_chat` | the vision model leaves the GPU and her chat model comes back, while you read | chat 8B |

# %%
import io
import time

from PIL import Image

from app import chat as socia_chat
from app.config import SociaConfig
from app.ollama import PHOTO_PROMPT, Ollama
from app.persona import TOOLS, system_prompt
from app.store import SociaStore
from lab import (Recorder, gpu, load_model, message_after, models_on_gpu, photo_messages, show_json, stats, table,
                 thumb, thumbs, transcript, watch)

config = SociaConfig.from_env()
store = SociaStore(config.database_url)
# Socia's own Ollama client, with every request it makes kept in `recorder.calls`.
recorder = Recorder()
ollama = Ollama(config, transport=recorder)
conversation = socia_chat.Conversation(config, store, ollama, None)

print("chat model:  ", config.chat_model, "at", config.context_tokens, "tokens (stays on the GPU)")
print("vision model:", config.vision_model, "at", config.vision_context_tokens, "tokens (only for photos)")
print("the card:    ", gpu())
await models_on_gpu(config)

# %% [markdown]
# ## Load her chat model (optional)
#
# Sentia keeps her chat model on the GPU (`SENTIA_RESIDENT_MODEL`) and her voice stays loaded in
# Socia (about 1.75 GB). If the cell above shows the chat model missing, load it here and see what it
# takes. A model asked for at a **different context** than it is loaded at is reloaded.

# %%
await load_model(config, config.chat_model, config.context_tokens)

# %% [markdown]
# ## Pick a photo
#
# Your latest photos to her are listed below. Leave `MESSAGE_ID = None` for the newest, or set it to
# one of the ids.

# %%
MESSAGE_ID = None
# The name she calls you: Socia uses the first name of the account you signed in with.
NAME = "love"

photos = photo_messages(store)
table(photos, ["id", "chat_id", "content", "photo_note"])
message = next(p for p in photos if MESSAGE_ID in (None, p["id"]))
chat = store.chat(message["chat_id"])
companion = store.companion(chat["companion_id"])
stored = (config.images_dir / message["file"]).read_bytes()
thumb(stored, label=f"message {message['id']}: “{message['content']}”")

# %% [markdown]
# ## Step 1 · The upload is made ready
#
# Your original upload is not kept: what is stored is already this step's output, so here it changes
# little.

# %%
photo = socia_chat.normalized_photo(stored)
for label, data in [("stored", stored), ("ready", photo)]:
    print(f"{label:>6}: {Image.open(io.BytesIO(data)).size} {len(data) // 1024} KB")

# %% [markdown]
# ## Step 2 · Does she look at the pictures this time?
#
# The chat up to your photo (the last 40 messages, `HISTORY_MESSAGES`). Your message has a photo, so
# yes. A text message also makes her look again when it asks about a photo or picture of the last 4
# messages (`ABOUT_PICTURE`: "which one is better?", "what did I circle?", "who is in it?").

# %%
rows = store.recent(message["chat_id"], SociaConfig.HISTORY_MESSAGES, before=message["id"] + 1)
print("She looks at the pictures:", socia_chat.sees_pictures(rows))
for text in ["which one is better?", "what did I mark in the red circle?", "ok going to the gym"]:
    asked = [*rows, {"role": "user", "content": text, "image_file": None}]
    print(f"  then “{text}” →", "looks again" if socia_chat.sees_pictures(asked) else "chat model, text only")

# %% [markdown]
# ## Step 3 · Her prompt
#
# Her usual system prompt, rebuilt as it was at that moment, plus `SEEING` at the end. Only the last
# 16 messages go with the pictures (`PHOTO_HISTORY_MESSAGES`): they all have to fit in the vision
# model's 8k context.

# %%
looks = await conversation.looks(companion["id"])
notes = chat.get("notes") or ""
seen_rows = rows[-SociaConfig.PHOTO_HISTORY_MESSAGES:]
prompt = system_prompt(companion, NAME, message["created_at"], looks, notes) + socia_chat.SEEING.format(name=NAME)
print(prompt)

# %% [markdown]
# ## Step 4 · The pictures she sees
#
# Her profile picture (so she knows her own face), then the chat's last 3 photos or pictures of hers,
# each at most 1024 px. They go with your message, which lists what each one is. The vision model reads
# about one token per 32 × 32 pixels.

# %%
messages = [{"role": "system", "content": prompt}, *socia_chat.with_secret_nudge(socia_chat.model_messages(seen_rows), notes)]
messages = conversation.with_pictures(messages, seen_rows, companion["id"], NAME)
import base64

attached = [base64.b64decode(p) for p in messages[-1].get("images", [])]
thumbs([(data, f"picture {n}") for n, data in enumerate(attached, start=1)], 220)
for n, data in enumerate(attached, start=1):
    width, height = Image.open(io.BytesIO(data)).size
    print(f"picture {n}: {width}×{height}, about {(width // 32) * (height // 32)} tokens")
transcript(messages[-6:])

# %% [markdown]
# ## Step 5 · The vision model replies as her
#
# Watch the GPU lines: her chat model is **pushed out** and the vision model **loaded** (about 20 s the
# first time; the rest of the reply takes a few seconds). `tokens in` is how much of its 8k context
# this reply uses.

# %%
said, calls = "", []
async with watch(config, "Her reply, seeing the photos") as replied:
    async for piece in ollama.chat(messages, TOOLS, seeing=True):
        said += piece.get("text", "")
        calls += piece.get("tool_calls", [])
print(said or "(no words)")
if calls:
    show_json(calls)
print(stats(recorder.calls[-1]), "of", config.vision_context_tokens, "tokens")

# %%
replied.chart()

# %% [markdown]
# ## Step 6 · After the reply
#
# What Socia's code does with her words before you see them. If she only reacted or called a tool,
# she is asked once more for a few words to go with it.

# %%
asked, text = (socia_chat.written_calls(said) if not calls and socia_chat.CALL_MARKER.search(said)
               else (calls, said))
picture = next((c for c in asked if (c.get("function") or {}).get("name") == "create_image"), None)
picture = picture or socia_chat.written_picture(text)
promise = socia_chat.PICTURE_PROMISE.search(socia_chat.without_written_pictures(text))
print("Tools she called:", [(c.get("function") or {}).get("name") for c in asked])
print("Tool calls she typed out as text:", bool(asked) and not calls)
print("Picture she asked her camera for:", (picture or {}).get("function", {}).get("arguments"))
if not picture and promise:
    print(f"Promise words “{promise.group(0)}”: ignored, your message had a photo")
print("\nWhat you see:", socia_chat.without_written_pictures(text))

# %% [markdown]
# And what she actually answered back then (before 0.8.0 she only read a description of the photo):

# %%
then = message_after(store, message)
print(then["content"] if then else "(no answer)")
if then and then.get("image_prompt"):
    print("\nwith a picture of:", then["image_prompt"])

# %% [markdown]
# ## Step 7 · The photo's description, for later
#
# While the vision model is still loaded, it describes the photo. Her later replies come from the chat
# model, which reads text only, so this is all it will know about the photo. Compare with the note
# stored back then.

# %%
print(PHOTO_PROMPT, "\n")
async with watch(config, "Describing the photo"):
    note = await ollama.describe_photo(photo)
print("now: ", note)
print("then:", message["photo_note"])
stats(recorder.calls[-1])

# %% [markdown]
# ### Try it: ask the vision model anything about this photo
#
# Try: "Is there a red circle or arrow drawn on it? What does it mark?", "How many people are there,
# and where is each?", "Describe the woman's face in detail".

# %%
QUESTION = "Is anything drawn or marked on this photo (a circle, an arrow, text)? What does it point at?"
async with watch(config, "Asking the vision model"):
    print(await ollama.describe_photo(photo, QUESTION))
stats(recorder.calls[-1])

# %% [markdown]
# ## Step 8 · How her later replies see the photo
#
# Your next text message goes to the chat model. The photo is now `[Photo: …]` at the end of your
# message, with the description from step 7.

# %%
later = [*rows[:-1], {**rows[-1], "photo_note": note}]
transcript(socia_chat.model_messages(later)[-3:])

# %% [markdown]
# ## Step 9 · Her chat model comes back
#
# Socia does this in the background right after a reply that looked at pictures, so your next message
# does not wait for the reload.

# %%
async with watch(config, "Back to the chat model"):
    await ollama.back_to_chat()
