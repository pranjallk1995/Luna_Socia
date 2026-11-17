# Notebooks

Socia's code run one step at a time, with every intermediate result shown: what the vision model
wrote about a photo, the exact request to each model, the tokens it used, and the GPU while it ran.
Later notebooks will hold the training and evaluation work. What to improve next is kept in
`ACTION_ITEMS.md` here, on this machine only: it has evaluation results and what happened in your chats.

| Notebook | What it shows |
|---|---|
| [01_photo_flow](01_photo_flow.py) | you send her a photo: the faces found in it and whose they are, the pictures the vision model sees, her prompt, her reply, the photo's note, the swap back to her chat model |
| [02_picture_flow](02_picture_flow.py) | she takes a picture: her references, the image model's instruction, the ComfyUI graph, the latents |
| [03_remember_training](03_remember_training.py) | teaching her chat model to call `remember` by herself: baseline, your labels, made-up examples in her own words, QLoRA training, the adapter into Ollama, before and after |

## Start, in VS Code

Luna Sentia and Luna Socia must be running. VS Code opens this repo inside a container beside
Socia and runs the cells there. Nothing listens on a port, so there is no Jupyter server, address
or token, and no browser.

1. **Once:** install the **Dev Containers** extension (`ms-vscode-remote.remote-containers`) in
   VS Code on Windows. The container installs Python, Jupyter and Jupytext into itself.
2. Open this folder in VS Code (through WSL as usual), then run **Dev Containers: Reopen in
   Container** from the command palette (or the prompt VS Code shows). The first time builds the
   image, about a minute. `notebooks/start.sh prepare` runs first, by itself.
3. Open a notebook:
   - **As a notebook:** right-click `01_photo_flow.py`, choose **Open With… → Jupyter Notebook**,
     pick the kernel **Python 3.12 (/usr/local/bin/python)**, and run cells with Shift+Enter.
     Outputs show under each cell and are never saved into the `.py`.
   - **Or as code:** open the `.py` and use **Run Cell** above each `# %%`. Outputs show in the
     Interactive Window beside it.
4. **Back to WSL:** **Dev Containers: Reopen Folder in WSL**. The container stops when VS Code
   leaves it.

Inside the container:

- **Her code:** Socia's own image, with this repo at `/workspace`. A change you make to
  `ui/app` is in the next cell after **Restart Kernel**, and in git as usual.
- **Her settings and data:** read from the running Socia (`notebooks/.socia.env`, and her
  `/data` read-only).
- **The GPU:** the cells use it as Socia does. `watch(...)` prints what each step took.
- **You:** files the notebooks write (in `results/`) belong to you, as anything else in WSL.

**Without VS Code** (a model evaluation from the terminal):

```bash
./notebooks/start.sh run    # the same container, in the background
docker exec -w /workspace luna_socia-notebooks python model_evaluations/self_recognition.py <folder>
./notebooks/start.sh stop
```

## What goes into git

- **The notebooks:** kept as `.py` files ([jupytext](https://jupytext.readthedocs.io)), so a diff
  shows only code.
- **Never:** outputs (your chats and her pictures), any `.ipynb` copy, `results/`,
  `ACTION_ITEMS.md`, and the settings `start.sh` writes. `.gitignore` keeps all of these out.

## Training (03)

Notebook 03 runs on its own kernel, **Luna training (torch)**. That's an environment in the same image
with torch, PEFT and bitsandbytes, which also sees Socia's libraries. The base model's Hugging Face
weights (16 GB), each run's adapter and llama.cpp's converters go into the Docker volume
`luna_socia_training`, mounted at `/training`, so they survive a rebuilt container. After pulling
this, run **Dev Containers: Rebuild Container** once, so the container gets the new kernel and the
volume. Training needs her chat model off the GPU; the notebook says when.

## Safe to run

- **Nothing is written to her database.** Notebook 03 adds a new model to Ollama when you get to step 6. Her chat model stays as it is.
- **Making a picture** (`MAKE = True` in 02) takes the GPU from her chat models for about 40 s.
- **Loading a model** at another context reloads it, as happens in the chat.
- **The Luna apps step** in 01 is off by default: it shares Socia's sign-in to your apps.
