# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Luna training (torch)
#     language: python
#     name: luna-train
# ---

# %% [markdown]
# # 03 · Teaching her to remember, by herself
#
# **The problem.** Her `remember` tool works, but she rarely uses it
# (`model_evaluations/remember.py` measures how rarely). She even answers "I'll remember that!" and
# never makes the call.
#
# **The goal.** She should call `remember` by herself, and only when it matters:
# - **about you** (`about: them`): people and pets and their names, where you live and work, likes
#   and dislikes, plans and dates, the moments you two share;
# - **about herself** (`about: me`): something new and lasting she tells you, such as a favourite or
#   her past, so she says the same thing next time;
# - **nothing** for small talk and things only true today.
#
# **How.** Use LoRA fine-tuning (QLoRA) on her own chat model, `huihui_ai/qwen3-abliterated:8b`.
# Each training example is one turn exactly as Socia sends it: her real prompt, her tools and the
# chat so far. The answer it learns is her reply plus the right `remember` call. The loss counts only
# her reply, and the call's tokens count three times as much as her words. The adapter, about 90 MB,
# goes into Ollama **on top of** her chat model as a new model. Her current model stays as it is
# until you switch.
#
# | Step | What | Your review |
# |---|---|---|
# | 1 | Baseline: the current model on the fixed eval | ✋ the numbers |
# | 2 | Candidates from your chats → `results/remember/labels.csv` | ✋ **you label them** |
# | 3 | Made-up examples, written by her own model | ✋ read through them |
# | 4 | Build the examples, split into train and test | ✋ the mix, one example in full |
# | 5 | Train (the GPU is needed: her chat model leaves it) | ✋ the loss curve |
# | 6 | Into Ollama as `luna-remember:8b-v1` | |
# | 7 | Before and after: fixed eval, test set, her voice | ✋ **switch or not** |
#
# **Safe to run:** nothing is written to her database. Steps 1, 3 and 7 use her chat model as a chat
# does. Step 5 needs her chat model off the GPU for about 20–40 minutes. Everything this notebook
# writes stays on this machine: labels and examples go in `notebooks/results/remember/`, and the
# base model and adapters in the `luna_socia_training` volume at `/training`. None of it goes into git.
#
# **Kernel:** pick **Luna training (torch)**. It has Socia's libraries plus torch, PEFT and
# bitsandbytes.

# %%
import json
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, "/workspace/model_evaluations")

from app import chat as socia_chat
from app.config import SociaConfig
from app.ollama import Ollama
from app.persona import TOOLS
from app.store import SociaStore
from lab import gpu, models_on_gpu, table, transcript

import remember_lab as rl

config = SociaConfig.from_env()
store = SociaStore(config.database_url)
ollama = Ollama(config)
conversation = socia_chat.Conversation(config, store, ollama, None)

COMPANION = 1          # Luna
NAME = store.owner_profile().get("name") or "love"  # what she calls you in her prompt
her = store.companion(COMPANION)
looks = await conversation.looks(COMPANION)
builder = rl.Builder(config, store, her, NAME, looks)

print("her chat model:", config.chat_model)
print("results:       ", rl.RESULTS)
print("the card:      ", gpu())
print(f"\nWhat she remembers about you now ({len(her['memory'] or '')} characters):\n{her['memory']}")
print(f"\nHer bio:\n{(her.get('persona') or {}).get('about')}")

# %% [markdown]
# ## 1 · Baseline
#
# The twelve messages of `model_evaluations/remember.py` (five facts about you, three questions about
# her, four small talk), each sent in a fresh chat to her current model, three times. Step 7 runs
# the same check on the trained model.

# %%
REPEAT = 3
fixed = rl.fixed_cases(builder) * REPEAT
baseline_fixed = await rl.evaluate(config, config.chat_model, fixed,
                                   on_each=lambda r: print(f"{r['expected']:>4} {'ok' if r['right'] else 'XX'}  "
                                                           f"{r['message'][:40]:<40} | {r['remembered'] or '-'}"))
print("\nbaseline:", rl.scores(baseline_fixed))
rl.RESULTS.mkdir(parents=True, exist_ok=True)
(rl.RESULTS / "baseline_fixed.json").write_text(json.dumps(baseline_fixed, indent=1, ensure_ascii=False))

# %% [markdown]
# ✋ **Stop:** these are the numbers to beat.
#
# ## 2 · Label your chats
#
# Each message in her chats that may hold a lasting fact is listed: yours where they mention your
# life, hers where she says something about herself. They go into `results/remember/labels.csv`.
# Running this again adds new messages and keeps your labels.
#
# **Open the CSV** (in VS Code, or Excel through `\\wsl$`) and fill in two columns:
#
# | `label` | when | `fact` |
# |---|---|---|
# | `them` | your message has a lasting fact about you | one short line, "They love pizza" |
# | `me` | her message says something lasting about herself | "Her favourite flower is the lotus" |
# | `none` | nothing worth keeping | leave blank |
# | *blank* | not sure yet | the message is left out |
#
# Tips:
# - Put a fact on the message where it first becomes clear: "I love pizza!" rather than a later
#   "remember that".
# - A fact only true today ("I'm tired") is `none`.
# - Your messages that aren't listed count as small talk (`none`). A few from each chat are used.

# %%
found = rl.candidates(store, COMPANION)
kept, added = rl.write_labels(found)
print(f"{len(found)} candidates: {kept} already in the file, {added} added → {rl.LABELS}")
print(Counter(row["who"] for row in found))
table(found[:25], ["message_id", "chat_id", "who", "hint", "text"])

# %% [markdown]
# ✋ **Stop: label the CSV**, then run the next cell to check it.

# %%
labels = rl.read_labels()
problems = rl.label_problems(labels)
print("\n".join(problems) or "No problems.")
print(Counter(row["label"] or "(blank)" for row in labels))
table([row for row in labels if row["label"] in ("them", "me")], ["message_id", "who", "label", "fact", "text"])

# %% [markdown]
# ## 3 · Made-up examples
#
# Your chats alone are too few, so made-up messages from you fill the gaps. Each comes with the right
# answer: a fact about you (pets, family, work, food, dates, a shared moment), small talk, or a
# question about her. Each is put after a few real messages from your chats, and **her own model
# writes the reply**, so her voice stays hers.
#
# For questions about her, the answer is read from her reply: the new fact she said about herself.
# If she only repeated her bio, the right answer is no call. Read through the facts before step 4.
# Wrong ones can be deleted from `results/remember/synthetic.jsonl`, one line per example.
#
# This uses her chat model for about 3–5 minutes, at about 1–2 s per example.

# %%
turns = rl.synthetic_turns(them=90, none=70, me=40)
context_rows = [store.recent(chat["id"], 100000) for chat in store.chats(COMPANION)]
synthetic = await rl.make_synthetic(builder, ollama, turns, context_rows,
                                    on_each=lambda e: print(f"{e.kind:>4} | {e.messages[-1]['content'][:45]:<45} | "
                                                            f"{'; '.join(e.facts) or '-':<40} | {e.reply[:60]}"))
rl.save_examples(rl.SYNTHETIC, synthetic)
print("\n", rl.mix(synthetic), "→", rl.SYNTHETIC)

# %% [markdown]
# ✋ **Stop:** read the `me` facts above especially. Delete any line you don't want from
# `synthetic.jsonl`, then go on.
#
# ## 4 · The examples
#
# Your labelled chats and the made-up examples, together, split 80/20 into train and test. The test
# set is never trained on. Step 7 checks it before and after.

# %%
labels = rl.read_labels()
assert not rl.label_problems(labels), "Fix the labels first (step 2)."
from_chats = builder.from_labels(labels, none_per_chat=12)
synthetic = rl.load_examples(rl.SYNTHETIC)
examples = [*from_chats, *synthetic]
train_set, test_set = rl.split(examples)
rl.save_examples(rl.RESULTS / "train.jsonl", train_set)
rl.save_examples(rl.RESULTS / "test.jsonl", test_set)
print("all:  ", rl.mix(examples))
print("train:", rl.mix(train_set))
print("test: ", rl.mix(test_set))

# %% [markdown]
# One example as her model will see it, rendered by Qwen3's own chat template: her prompt, her
# tools, the chat, and the reply with its `remember` call. The part trained on starts at
# `<|im_start|>assistant` at the end.

# %%
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# The Hugging Face weights of her Ollama model (the same abliterated Qwen3 8B, before Ollama's Q4_K_M).
BASE_REPO = "huihui-ai/Qwen3-8B-abliterated"
BASE_DIR = rl.TRAINING / "base" / BASE_REPO.replace("/", "--")
snapshot_download(BASE_REPO, local_dir=BASE_DIR, cache_dir=rl.TRAINING / "hf" / "hub",
                  allow_patterns=["*.json", "*.txt", "tokenizer*"])
tokenizer = AutoTokenizer.from_pretrained(BASE_DIR)

sample = next(e for e in train_set if e.kind == "them")
text = tokenizer.apply_chat_template(rl.chat_messages(sample), tools=TOOLS, enable_thinking=False, tokenize=False)
print(text[:1500], "\n…\n", text[-1200:])

MAX_TOKENS = 4096
encoded = [rl.encode(tokenizer, e, MAX_TOKENS) for e in train_set]
tests_encoded = [rl.encode(tokenizer, e, MAX_TOKENS) for e in test_set]
too_long = sum(item is None for item in encoded + tests_encoded)
batches = [rl.mark_calls(tokenizer, item) for item in encoded if item]
tests = [rl.mark_calls(tokenizer, item) for item in tests_encoded if item]
lengths = sorted(len(item["ids"]) for item in batches)
print(f"\n{len(batches)} train, {len(tests)} test; {too_long} left out as longer than {MAX_TOKENS} tokens")
print(f"tokens per example: median {lengths[len(lengths) // 2]}, longest {lengths[-1]}; "
      f"reply tokens trained on: {sum(len(i['ids']) - i['start'] for i in batches)}")

# %% [markdown]
# ✋ **Stop:** check the mix (about as many `none` as `them`, a fair number of `me`) and the
# example above.
#
# ## 5 · Train
#
# **The GPU.** Training needs about 9 GB. Her chat model (about 6 GB) has to leave the GPU. Sentia's
# monitor loads it back only once there's room, and there won't be while training runs. Her voice
# (about 1.5 GB) can stay. If the cell below shows more than about 2.5 GB used, stop Socia for the
# training (`docker stop luna_socia-ui-1` in WSL, then `docker start luna_socia-ui-1` after).
# ComfyUI must not be making a picture.
#
# **The base model.** The first run downloads the Hugging Face weights: 16 GB, into `/training`, once.

# %%
import httpx

with httpx.Client(base_url=config.ollama_url, timeout=60) as client:
    for model in client.get("/api/ps").json().get("models", []):
        client.post("/api/generate", json={"model": model["name"], "keep_alive": 0})
        print("unloaded", model["name"])
await models_on_gpu(config)
print(gpu())

# %%
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

snapshot_download(BASE_REPO, local_dir=BASE_DIR, cache_dir=rl.TRAINING / "hf" / "hub",
                  allow_patterns=["*.json", "*.txt", "tokenizer*", "*.safetensors"])
model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16))
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, LORA)
model.print_trainable_parameters()
print(gpu())

# %% [markdown]
# The run: 2 passes over the train set, 8 examples per update, a learning rate of 1e-4 that warms up
# and then decays. The loss is printed every update, and the test loss after each pass. If the test
# loss rises while the train loss keeps falling, it's learning your examples by heart. In that case,
# use 1 pass, or add more examples.

# %%
EPOCHS, LR, ACCUMULATE, CALL_WEIGHT = 2, 1e-4, 8, 3.0
RUN = f"v1-{datetime.now():%Y%m%d-%H%M}"


def progress(step, steps, loss, seconds, test_loss=None, epoch=None):
    if test_loss is not None:
        print(f"  pass {epoch}: test loss {test_loss:.3f}")
    else:
        card = gpu()
        print(f"step {step}/{steps}  loss {loss:.3f}  {seconds / 60:.1f} min  GPU {card['used_gb'] if card else '?'} GB")


history = rl.train(model, batches, tests, epochs=EPOCHS, lr=LR, accumulate=ACCUMULATE, call_weight=CALL_WEIGHT,
                   on_step=progress)
ADAPTER_DIR = rl.TRAINING / "runs" / RUN / "adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
(ADAPTER_DIR.parent / "history.json").write_text(json.dumps(history))
print("saved", ADAPTER_DIR)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))
plt.plot(history["step"], history["loss"], label="train")
plt.plot(history["test_step"], history["test_loss"], "o-", label="test")
plt.xlabel("update")
plt.ylabel("loss (her reply)")
plt.legend()
plt.show()

# %% [markdown]
# A quick look before Ollama: three test examples answered by the trained model here. You should see
# a `<tool_call>` with `remember` where one is expected, and none for small talk.

# %%
for example in [next(e for e in test_set if e.kind == kind) for kind in ("them", "me", "none")
                if any(e.kind == kind for e in test_set)]:
    prompt = tokenizer.apply_chat_template(rl.chat_messages(example)[:-1], tools=TOOLS, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors="pt").to(0)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=120, do_sample=False)
    print(f"[{example.kind}] {example.messages[-1]['content'][:70]}\n  expected: {example.facts or 'no call'}\n  "
          f"got: {tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=False)[:300]}\n")

# %% [markdown]
# ✋ **Stop:** check the curve and the three answers.
#
# ## 6 · Into Ollama
#
# The adapter is converted to GGUF with llama.cpp's converter (Python only, downloaded once), then
# uploaded to Ollama and created as a new model **from her chat model**, with its template and
# settings, plus this adapter. First the trained model is freed, so her chat model fits again.

# %%
import gc
import subprocess

del model
gc.collect()
torch.cuda.empty_cache()

converter = rl.llama_cpp()
ADAPTER_GGUF = ADAPTER_DIR.parent / "luna-remember.gguf"
subprocess.run([sys.executable, str(converter / "convert_lora_to_gguf.py"), "--base", str(BASE_DIR),
                "--outtype", "f16", "--outfile", str(ADAPTER_GGUF), str(ADAPTER_DIR)], check=True)
print(f"{ADAPTER_GGUF}: {ADAPTER_GGUF.stat().st_size / 1e6:.0f} MB")

NEW_MODEL = f"luna-remember:8b-{RUN}"
print(rl.add_to_ollama(config, ADAPTER_GGUF, NEW_MODEL))

# %% [markdown]
# ## 7 · Before and after
#
# The fixed eval from step 1, then the test set (never trained on), on both models. Then her voice:
# small talk side by side, to check she still sounds like herself. Each model loads in turn, as in a
# chat.

# %%
after_fixed = await rl.evaluate(config, NEW_MODEL, fixed)
print("fixed eval  before:", rl.scores(baseline_fixed))
print("            after: ", rl.scores(after_fixed))

before_test = await rl.evaluate(config, config.chat_model, test_set)
after_test = await rl.evaluate(config, NEW_MODEL, test_set)
print("test set    before:", rl.scores(before_test))
print("            after: ", rl.scores(after_test))
(rl.RESULTS / f"eval-{RUN}.json").write_text(json.dumps(
    {"model": NEW_MODEL, "fixed": [baseline_fixed, after_fixed], "test": [before_test, after_test]},
    indent=1, ensure_ascii=False))

# %%
table([{"expected": b["expected"], "message": b["message"], "before": b["remembered"] or "-",
        "after": a["remembered"] or "-"} for b, a in zip(before_test, after_test)],
      ["expected", "message", "before", "after"])

# %%
VOICE = ["how was your day?", "I'm so tired today", "what are you doing right now?", "tell me something sweet",
         "good night baby"]
rows = []
for message in VOICE:
    case = rl.fixed_cases(builder)[0]
    case.messages[-1] = {"role": "user", "content": message}
    before, _ = await rl.answer(config, config.chat_model, case.messages)
    after, calls = await rl.answer(config, NEW_MODEL, case.messages)
    rows.append({"message": message, "before": before, "after": after + (f"  [remember: {calls}]" if calls else "")})
table(rows, ["message", "before", "after"], width=60)

# %% [markdown]
# ✋ **Your call: switch or not.**
#
# She's better if the `them` and `me` rows go up, false calls stay near 0, and her voice above still
# sounds like her. To switch, put the new model in both settings and restart (nothing here does that):
#
# - `Luna_Socia/.env`: `LUNASOCIA_CHAT_MODEL=<NEW_MODEL>`
# - `Luna_Sentia/.env`: `SENTIA_RESIDENT_MODEL=<NEW_MODEL>` (so the monitor keeps it on the GPU)
#
# To go back, put `huihui_ai/qwen3-abliterated:8b` back in both. Each run is kept in
# `/training/runs/`, and its model stays in Ollama until you delete it (`ollama rm`).
#
# **Next round:** label the chats you have by then (step 2 keeps your earlier labels), add any facts
# she missed, and train `v2` the same way.

# %%
print("new model:", NEW_MODEL)

# %% [markdown]
# # Part B · A small classifier decides *when* (our brainstorm)
#
# Instead of retraining her chat model, keep it as it is and put a small decider beside it:
#
# ```
# your message (+ her line before it) ─► embedding model ─► neuron "fact about you?" ─► yes ─► her model writes "They …"
# her reply   (+ your message)        ─► embedding model ─► neuron "new fact about her?" ─► yes ─► her model writes "She …"
# ```
#
# - **The embedding model** is `Qwen3-Embedding-0.6B`, run here with transformers: a 1.2 GB download
#   into `/training`, on the GPU beside her chat model. It turns a line into 1,024 numbers. Ollama has
#   it too, but Ollama's engine hung loading it on 17 Nov.
# - **The neurons** are logistic regression: one weight per number, plus a sigmoid. They train in a
#   second on the CPU.
# - **The fact writer** is her chat model with one narrow job: write the fact in one line. It sees
#   what she already knows and may still answer NONE.
#
# **Training data:**
# - lines Claude wrote (`results/remember/generated.jsonl`): facts and non-facts for both of you,
#   in your texting style and hers, including the hard ones: corrections, testing the app, knowledge
#   questions, lists, photo talk, and what she is doing right now;
# - lines from your own chats that Claude labelled (`results/remember/from_chats.jsonl`). Those
#   chats are left out of the test.
#
# A fact she already has (her bio again) is still labelled a fact: the neuron can't know her bio,
# and the duplicate check (B2) catches it. This way the same neurons work for every girlfriend.
#
# **Test data:** your labelled chats from step 2, never trained on. Until you label them, B1 checks
# neurons trained on Claude's lines alone against the chats Claude labelled.
#
# Needs: the setup cell, step 1 (the baseline) and, for the real test, step 2's labels. Steps 3–7
# can be skipped.

# %%
import remember_classifier as rc
import torch

# Each written line gets a line before it, as every message in a real chat has (see rc.with_context).
written = rc.with_context(rc.load_generated())
from_chats = rc.load_generated(rc.FROM_CHATS)
TRAIN_CHATS = {row["chat"] for row in from_chats}
generated = [*written, *from_chats]  # everything trained on
labels = rl.read_labels() if rl.LABELS.exists() else []
real = (rc.real_rows(store, labels, COMPANION, skip_chats=TRAIN_CHATS)
        if any(row["label"] for row in labels) else [])
print("Claude's lines:   ", Counter((r["who"], r["label"]) for r in written))
print("from your chats:  ", Counter((r["who"], r["label"]) for r in from_chats), "chats", sorted(TRAIN_CHATS))
print("your chats:", Counter((r["who"], r["label"]) for r in real) or "not labelled yet (step 2)")
table(generated[::12], ["who", "label", "before", "text", "fact"])

# %%
import time as _time

started = _time.perf_counter()
rc.embedder()  # loads it (and downloads it the first time, 1.2 GB)
print(f"embedding model ready in {_time.perf_counter() - started:.1f} s, on the {rc.embedder()[2]}")
started = _time.perf_counter()
x_generated = rc.embed(config, generated)
x_real = rc.embed(config, real) if real else None
print(f"{len(generated) + len(real)} lines embedded in {_time.perf_counter() - started:.1f} s → {x_generated.shape[1]} numbers each (message + line before)")
await models_on_gpu(config)

# %% [markdown]
# ## B1 · How well does a neuron decide?
#
# First on the generated set, where each line is scored by a neuron trained on the other four fifths
# (cross-validation). Then on your real chats, by a neuron trained on all the generated lines. The
# real-chat numbers are the ones that count.

# %%
neurons, by_who = {}, {}
for who in ("you", "her"):
    rows = [r for r in generated if r["who"] == who]
    index = [i for i, r in enumerate(generated) if r["who"] == who]
    x, y = x_generated[index], rc.labels_of(rows, who)
    proba = rc.cross_validate(x, y)
    print(f"\n{who}: cross-validated on the generated set ({len(rows)} lines)")
    table(rc.thresholds(y, proba), ["threshold", "facts caught", "false alarms", "precision", "recall"])
    neurons[who] = rc.Neuron().fit(x, y)
    by_who[who] = (rows, y, proba)

# %%
if real:
    for who in ("you", "her"):
        index = [i for i, r in enumerate(real) if r["who"] == who]
        if not index:
            continue
        rows = [real[i] for i in index]
        y, proba = rc.labels_of(rows, who), neurons[who].proba(x_real[index])
        print(f"\n{who}: YOUR CHATS ({len(rows)} lines), neuron trained on the generated set only")
        table(rc.thresholds(y, proba), ["threshold", "facts caught", "false alarms", "precision", "recall"])
        print("Its biggest mistakes:")
        table(rc.mistakes(rows, y, proba, 0.5), ["p", "should be", "before", "text"])
else:
    print("Label your chats (step 2) to test on them. The training set's mistakes, for now:")
    for who in ("you", "her"):
        rows, y, proba = by_who[who]
        table(rc.mistakes(rows, y, proba, 0.5, limit=8), ["p", "should be", "before", "text"])

# %% [markdown]
# **Your style, before you label anything:** neurons trained on Claude's lines **alone**, tested
# on the lines from your chats that Claude labelled. They never saw your chats, so this shows how
# well lines written by Claude carry over to how you two actually text.

# %%
if from_chats:
    x_written, x_from = x_generated[:len(written)], x_generated[len(written):]
    for who in ("you", "her"):
        train_i = [i for i, r in enumerate(written) if r["who"] == who]
        test_i = [i for i, r in enumerate(from_chats) if r["who"] == who]
        neuron = rc.Neuron().fit(x_written[train_i], rc.labels_of([written[i] for i in train_i], who))
        rows = [from_chats[i] for i in test_i]
        y, proba = rc.labels_of(rows, who), neuron.proba(x_from[test_i])
        print(f"\n{who}: trained on Claude's lines, tested on YOUR CHATS {sorted(TRAIN_CHATS)} ({len(rows)} lines)")
        table(rc.thresholds(y, proba), ["threshold", "facts caught", "false alarms", "precision", "recall"])
        table(rc.mistakes(rows, y, proba, 0.5, limit=10), ["p", "should be", "before", "text"])

# %% [markdown]
# ✋ **Look:** pick a threshold per neuron. A missed fact means you remind her, which is fine (she
# can still save it herself when you ask). A false alarm puts something wrong in her memory or bio.
# So choose a threshold where false alarms are rare, even if she catches fewer facts: 0.5 or
# higher.

# %%
THRESHOLD = {"you": 0.5, "her": 0.5}
for who, value in THRESHOLD.items():
    neurons[who].threshold = value

# %% [markdown]
# ## B2 · The fact writer
#
# Once a neuron says yes, her chat model only **words** the fact, from the message and the line
# before it, guided by a few examples. It isn't asked whether the message holds a fact; the neuron
# already decided that. Shown her memory or bio, it said NONE to plain facts, or copied lines from
# them instead.
#
# **Already known?** The embedding model compares the written fact with each line of her memory
# (about you) or each sentence of her bio (about her). At a similarity of 0.8 or more, it's the
# same fact and isn't saved again. The margin is narrow: a reworded fact scored 0.80 and two
# different favourites 0.72, so check the pairs below.

# %%
import random

checks = [r for r in (real or generated) if r["label"] in ("them", "me")]
checks = random.Random(1).sample(checks, min(15, len(checks)))
written = []
for row in checks:
    fact = await rc.write_fact(ollama, row)
    known, similarity, line = rc.already_known(fact, rc.known_lines(her, row["who"]))
    written.append({"who": row["who"], "text": row["text"][:50], "expected": row["fact"], "written": fact,
                    "closest she has": f"{similarity} {line[:40]}", "known": "yes" if known else ""})
table(written, ["who", "text", "expected", "written", "closest she has", "known"], width=50)

# The duplicate check on its own: facts she has, reworded, against new ones.
pairs = [("They have a pet named Maxi", "Their cat is called Maxi"),
         ("They like chocolate red velvet lava cake", "They love red velvet lava cake"),
         ("They have a pet named Maxi", "Their sister Asha is visiting in May"),
         ("Her favourite snack is chocolate", "Her favourite colour is lavender"),
         ("She grew up in Mumbai", "She is from Mumbai, India")]
vectors = rc.embed_texts([text for pair in pairs for text in pair])
table([{"has": a, "new": b, "similarity": round(float(vectors[2 * i] @ vectors[2 * i + 1]), 3)}
       for i, (a, b) in enumerate(pairs)], ["has", "new", "similarity"])

# %% [markdown]
# ## B3 · Before and after, on the fixed eval
#
# The same twelve messages × 3 as step 1. For each, her current model replies as in a chat. The
# **your-message neuron** reads your message, and the **her-reply neuron** reads her reply. A fact
# about you counts as `them`, one about her as `me`, as the baseline counted her own `remember`
# calls.

# %%
rememberer = rc.Rememberer(config, neurons["you"], neurons["her"])
after_b = []
for case in rl.fixed_cases(builder) * REPEAT:
    message = case.messages[-1]["content"]
    reply, _ = await rl.answer(config, config.chat_model, case.messages)
    got = await rememberer.turn(ollama, her, NAME, "", message, reply)
    after_b.append({"expected": case.kind, "got": got["got"], "right": got["got"] == case.kind, "message": message[:40],
                    "remembered": got["them"] or got["me"], "p_them": round(got["p_them"], 2),
                    "p_me": round(got["p_me"], 2), "seconds": got["seconds"], "reply": reply[:60]})
    known = got["known_them"] or got["known_me"]
    print(f"{case.kind:>4} {'ok' if got['got'] == case.kind else 'XX'}  {message[:36]:<36} "
          f"p_them {got['p_them']:.2f} p_me {got['p_me']:.2f} | {got['them'] or got['me'] or '-'}"
          f"{f'  (already known, {known[:50]})' if known else ''}  ({got['seconds']} s)\n      her reply: {reply[:90]}")
print("\nbaseline (her remember tool):", rl.scores(baseline_fixed))
print("classifier + fact writer:    ", rl.scores(after_b))

# %% [markdown]
# ✋ **Verdict.** The brainstorm works if:
# - the `them` and `me` rows beat the baseline;
# - false calls stay near 0;
# - on your real chats (B1), it catches most facts with few false alarms.
#
# The extra time per turn is the `seconds` above: the embeddings, plus the fact writer only when a
# neuron says yes.
#
# If it does, it goes into Socia like this (not done here):
# 1. After each reply, in the background, as the chat reader once ran: embed both lines and ask the
#    two neurons.
# 2. On a yes, the fact writer runs, then `store.remember` (about you) or `store.add_to_bio`
#    (about her), shown as "Remembered:".
# 3. Her `remember` tool can stay for when you ask her outright.
#
# The cell below saves the neurons, trained on the generated lines plus your labelled chats, for
# that.

# %%
final = {}
for who in ("you", "her"):
    rows = [r for r in [*generated, *real] if r["who"] == who]
    x = torch.cat([x_generated[[i for i, r in enumerate(generated) if r["who"] == who]],
                   *([x_real[[i for i, r in enumerate(real) if r["who"] == who]]] if real else [])])
    final[who] = rc.Neuron(THRESHOLD[who]).fit(x, rc.labels_of(rows, who))
SAVED = rl.RESULTS / f"neurons-{datetime.now():%Y%m%d-%H%M}.pt"
rc.Rememberer(config, final["you"], final["her"]).save(SAVED)
print("saved", SAVED)
