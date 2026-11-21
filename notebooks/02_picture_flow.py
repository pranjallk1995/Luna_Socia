# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 02 · What happens when she takes a picture
#
# From her `create_image` call to the picture you get, one cell at a time, with her real code,
# pictures and references. Nothing is written to her database. **Making a picture is off** until
# you set `MAKE = True`: it takes the GPU from her chat models for about 2 minutes, so she
# answers slowly right after.
#
# | Step | Code | What happens |
# |---|---|---|
# | 1 | her reply | she calls `create_image(description, include_luna, use_their_photo)`, or Socia's promise check writes the call for her (`PROMISE_PROMPT`) |
# | 2 | `Conversation.picture` | which pictures go with it: her profile picture and up to 4 more of her when she is in it, and your last photo when she edits it |
# | 3 | `image_prompt` | the instruction the image model gets, naming the pictures `<image1>`, `<image2>`… |
# | 4 | `ImageService.generate` | the request to Sentia's image service: the prompt, the references, the resolution |
# | 5 | the service's read pass | every reference is resized to about 1024×1024, read by the text encoder **and** turned into VAE latents that sit next to the picture being made |
# | 6 | `images.generate` | the service empties Ollama off the GPU, reads, draws, and gives the GPU back |

# %%
import dataclasses
import math
import time
from pathlib import Path

from app.config import SociaConfig
from app.images import ImageService
from app.ollama import Ollama
from app.persona import image_prompt
from app.store import SociaStore
from lab import Recorder, gpu, her_pictures, models_on_gpu, show_json, table, thumb, thumbs, watch

config = SociaConfig.from_env()
store = SociaStore(config.database_url)
recorder = Recorder()
ollama = Ollama(config, transport=recorder)
images = ImageService(config)
print("image service:", config.imagegen_url, "| ready:", await images.available())
print("references resized to about", config.image_resolution, "px square")
print("the card:", gpu())
await models_on_gpu(config)

# %% [markdown]
# ## Step 1 · Pick one of her pictures
#
# Her latest pictures, with the description she gave her camera. Leave `IMAGE_ID = None` for the
# newest. Socia does not keep whether she put herself in it (`include_luna`) or edited your photo
# (`use_their_photo`); set them below to replay it either way, or change `DESCRIPTION` to try your own.

# %%
IMAGE_ID = None
pictures = her_pictures(store)
table(pictures, ["id", "chat_id", "prompt"], 110)
picture = next(p for p in pictures if IMAGE_ID in (None, p["id"]))
thumb(config.images_dir / picture["file"], 420, label=f"picture {picture['id']}")
DESCRIPTION = picture["prompt"]
INCLUDE_LUNA = True
USE_THEIR_PHOTO = False
print(DESCRIPTION)

# %% [markdown]
# ## Step 2 · Which pictures go with it
#
# The same choice `Conversation.picture` makes. Her profile picture is always `<image1>` when she
# is in it, and the first picture also sets the new picture's shape.

# %%
chat_id = picture["chat_id"]
companion_id = store.chat(chat_id)["companion_id"]
profile = config.profile_path(companion_id)
lunas = [profile, *config.references(companion_id)] if INCLUDE_LUNA and profile.exists() else []
photo = store.last_photo(chat_id) if USE_THEIR_PHOTO else None
prompt, order = image_prompt(DESCRIPTION, len(lunas), photo is not None)
sources = {"luna": iter(lunas), "photo": iter([config.images_dir / photo["file"]] if photo else [])}
references = [next(sources[kind]) for kind in order]
thumbs([(path, f"<image{n}> {kind}") for n, (path, kind) in enumerate(zip(references, order), start=1)], 200)

# %% [markdown]
# ## Step 3 · The instruction the image model reads
#
# Note what it asks to keep: her **face and identity**, "but not their clothes, pose or background".
# Nothing asks for her body, and nothing says how many people to draw.

# %%
print(prompt)

# %% [markdown]
# ## Step 4 · The request
#
# What Socia sends Sentia's image service (`POST /pictures`), with each reference's base64 cut short.

# %%
show_json({"prompt": prompt, "resolution": config.image_resolution,
           "references": [f"<{Path(p).name}: {Path(p).stat().st_size // 1024} KB, base64>" for p in references]})

# %% [markdown]
# ## Step 5 · How much each reference weighs
#
# The service's read pass resizes each reference to about `resolution × resolution`
# pixels and turns it into VAE latents: one token per 16×16 pixels. They sit in the same sequence as
# the picture being made, which attends to all of them. Five references are five times the
# picture's own size: each one pulls on the result with its face, **and its clothes and pose**.

# %%
from PIL import Image

resolution = config.image_resolution
rows = []
for n, path in enumerate(references, start=1):
    width, height = Image.open(path).size
    ratio = width / height
    w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
    h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    rows.append({"image": f"<image{n}>", "file": Path(path).name, "size": f"{width}×{height}",
                 "resized": f"{w}×{h}", "latent tokens": (w // 16) * (h // 16)})
table(rows, ["image", "file", "size", "resized", "latent tokens"])
first = rows[0] if rows else None
print("The new picture:", first["resized"] if first else f"{resolution}×{resolution}",
      "=", first["latent tokens"] if first else (resolution // 16) ** 2, "tokens;",
      "the references together:", sum(r["latent tokens"] for r in rows), "tokens")

# %% [markdown]
# ## Step 6 · Make it (off until `MAKE = True`)
#
# Same as Socia: the image service empties Ollama off the GPU, reads, draws, then gives the GPU back.
# `TIMES` makes it more than once, to see how much it changes between seeds. The pictures are saved
# in `results/` (never in git).

# %%
MAKE = False
TIMES = 2
made = []
if MAKE:
    out = Path("results") / time.strftime("%Y-%m-%d_%H%M%S_picture")
    out.mkdir(parents=True, exist_ok=True)
    data = [Path(p).read_bytes() for p in references]
    for n in range(TIMES):
        async with watch(config, f"Picture {n + 1} of {TIMES}") as making:
            image = await images.generate(prompt, data)
        (out / f"{n + 1}.png").write_bytes(image)
        made.append((image, f"try {n + 1}"))
    (out / "prompt.txt").write_text(prompt)
    thumbs(made, 300)
    making.chart()

# %% [markdown]
# ## Try it: change the instruction or the references
#
# Edit and run, then run step 6 again. A few to try for the action items:
#
# - **Same body** (item 3): `prompt = prompt.replace("Keep her face and identity exactly as in them",
#   "Keep her face, hair, body shape and proportions exactly as in them")`
# - **Fewer references** (item 7): `references = references[:1]`, then rebuild the prompt with
#   `image_prompt(DESCRIPTION, 1, False)`
# - **One person only**: add "She is alone in the picture." to the description
# - **A new pose**: add "in a new pose, not the one in the references"

# %%
# prompt, order = image_prompt(DESCRIPTION + " She is alone in the picture.", len(lunas), photo is not None)
# references = references[:1]
print(prompt)
