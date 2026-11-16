# Notebooks

Socia's code run one step at a time, with every intermediate result shown: what the vision model
wrote about a photo, the exact request to each model, the tokens it used, and the GPU while it ran.
Later notebooks will hold the training and evaluation work. What to improve next is kept in
`ACTION_ITEMS.md` here, on this machine only: it has evaluation results and what happened in your chats.

| Notebook | What it shows |
|---|---|
| [01_photo_flow](01_photo_flow.py) | you send her a photo: the pictures the vision model sees, her prompt, her reply, the photo's note, the swap back to her chat model |
| [02_picture_flow](02_picture_flow.py) | she takes a picture: her references, the image model's instruction, the ComfyUI graph, the latents |

## Start

Luna Sentia and Luna Socia must be running.

```bash
./notebooks/start.sh        # prints the address to open
./notebooks/start.sh stop
```

Jupyter runs in a container beside Socia:

- **Her code:** Socia's own image, with this branch's `ui/app` mounted read-only. A change you
  make to her code here is in the next cell after a kernel restart.
- **Her settings and data:** read from the running Socia. Her pictures are read-only.
- **The GPU:** its counters only. `watch(...)` prints what each step took.

**In the browser:** open the address, then right-click a `.py` notebook and choose
**Open With → Notebook**.

**In VS Code:** open the `.py` file, choose **Select Kernel → Existing Jupyter Server**, and paste
the address. Then run cells with **Run Cell**.

## What goes into git

- **The notebooks:** kept as `.py` files ([jupytext](https://jupytext.readthedocs.io)), so a diff
  shows only code.
- **Never:** the `.ipynb` copies with their outputs (your chats and her pictures), `results/`,
  `ACTION_ITEMS.md`, and the settings `start.sh` writes. `.gitignore` keeps all of these out.

## Safe to run

- **Nothing is written to her database.**
- **Making a picture** (`MAKE = True` in 02) takes the GPU from her chat models for about 40 s.
- **Loading a model** at another context reloads it, as happens in the chat.
- **The Luna apps step** in 01 is off by default: it shares Socia's sign-in to your apps.
