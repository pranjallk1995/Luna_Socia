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
# # 04 · What she keeps, and what she reads
#
# **The problem.** Every reply, her prompt holds *all* her memory of you, with "bring it up naturally
# when it fits". An 8B model reads that as "bring it up": a pile of lines about pets or a diet pulls
# every chat toward pets and diets, even after "lol". Her memory also only grows. Past 4,800
# characters the **oldest** lines drop off, important or not.
#
# **The idea.** Remember like a person does:
# 1. **Pinned:** the core never fades: who they are, their people and pets, where they live and work,
#    health, dates. (Her bio and your "About you" are yours, so they are always pinned.)
# 2. **Scored:** every other line gets stronger each time the chat comes back to it, and fades with
#    time (ACT-R's base-level activation: `ln Σ (age of each recall)^-0.5`). A new memory replaces the
#    weakest, not the oldest.
# 3. **Retrieved:** each reply reads the pinned lines plus only the few lines close to what you are
#    talking about now. After "lol", nothing beyond the core.
#
# Why not TF-IDF? It scores **rare words**, not **importance**. "She is 25" is all common words and
# would come last. "Their diet is currently in effect" would score as distinctive.
#
# | Step | What | Look at |
# |---|---|---|
# | 1 | Made-up memories for a made-up owner, Sam | the list |
# | 2 | Her chat model sorts each line: core, detail, or junk | ✋ does it agree with the labels? |
# | 3 | Retrieval: which lines does a message bring up? | ✋ the threshold, small talk brings up nothing |
# | 4 | 120 days of chats: score, evict, compare with dropping the oldest | ✋ who survives |
# | 5 | Tokens: all of it vs pinned + retrieved | the budget |
# | 6 | Her replies: all memories vs pinned + retrieved (vs a stricter instruction) | ✋ **does she get less weird?** |
#
# **Safe to run:** nothing reads or writes her database. Step 2 and step 6 use her chat model as a
# chat does, for a few minutes (a message to her waits meanwhile). The results go in
# `notebooks/results/memory/` and never into git.
#
# **Kernel:** **Luna training (torch)**.

# %%
import asyncio
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/ui")
sys.path.insert(0, "/workspace/notebooks")
import remember_classifier as rc  # the embedding model, as notebook 03 runs it  # noqa: E402
from app.config import SociaConfig  # noqa: E402
from app.ollama import Ollama  # noqa: E402
from app.persona import DEFAULT_PERSONA, MEMORY, TOOLS, system_prompt  # noqa: E402

config = SociaConfig.from_env()
ollama = Ollama(config)
RESULTS = Path("/workspace/notebooks/results/memory")
RESULTS.mkdir(parents=True, exist_ok=True)
tokenizer = rc.embedder()[0]  # Qwen3's tokenizer: her chat model counts the same way


def tokens(text: str) -> int:
    return len(tokenizer(text)["input_ids"])


print("chat model:", config.chat_model, "| embedder on", rc.embedder()[2])

# %% [markdown]
# ## 1 · Made-up memories
#
# The owner is a made-up Sam. Each line has the kind it should be (`core`, `detail`, `junk`, as the
# fact writer really produced) and a few words that show when a reply brings it up.

# %%
OWNER = "Sam"
BIO = ("She is 25, from Mumbai. She loves baking, especially choco lava cake.\n"
       "She also loves coffee, rainy evenings, and late-night talks.")
MEMORIES = [
    # core: who they are, their people and pets, home, work, health, dates
    ("They have a younger sister called Asha who lives in Pune", "core", ["asha", "sister"]),
    ("They work as a data engineer at a logistics company", "core", ["data engineer", "logistics"]),
    ("They live in Bengaluru, in Indiranagar", "core", ["bengaluru", "indiranagar"]),
    ("They have a cat called Biscuit", "core", ["biscuit"]),
    ("They are allergic to peanuts", "core", ["peanut"]),
    ("Their birthday is on 14 March", "core", ["march", "birthday"]),
    ("They are vegetarian", "core", ["vegetarian", "veggie"]),
    ("Their mother is a school teacher in Nagpur", "core", ["mother", "mom", "nagpur"]),
    ("You and they first met on a rainy evening in July", "core", ["july", "first met"]),
    # details: true and nice to know, fine to fade if they never come up
    ("They like filter coffee with less sugar", "detail", ["filter coffee"]),
    ("Their favourite movie is Interstellar", "detail", ["interstellar"]),
    ("They play badminton on Saturday mornings", "detail", ["badminton"]),
    ("They are learning to play the guitar", "detail", ["guitar"]),
    ("They hate loud music", "detail", ["loud music"]),
    ("They prefer the mountains to the beach", "detail", ["mountain"]),
    ("Their favourite food is masala dosa", "detail", ["dosa"]),
    ("They are reading Project Hail Mary", "detail", ["hail mary"]),
    ("They want to visit Japan someday", "detail", ["japan"]),
    ("They drive a grey Maruti Swift", "detail", ["swift", "maruti"]),
    ("They support Chennai Super Kings in the IPL", "detail", ["chennai super kings", "csk"]),
    ("They are training for a 10k run in February", "detail", ["10k", "training"]),
    ("Their manager is called Rohit", "detail", ["rohit"]),
    ("They don't like horror movies", "detail", ["horror"]),
    ("They are trying to sleep before midnight", "detail", ["before midnight"]),
    ("They have a plant called Fernando on their desk", "detail", ["fernando"]),
    # junk: what the fact writer wrote from small talk
    ("Their message was about the weather", "junk", ["weather"]),
    ("Their day started with a lovely response", "junk", ["lovely response"]),
    ("Their diet is currently in effect", "junk", ["diet"]),
    ("They decided to talk properly from now on", "junk", ["talk properly"]),
    ("Their best way to avoid cravings is to stay disciplined with light snacks", "junk", ["craving", "disciplined"]),
]
LINES = [m[0] for m in MEMORIES]
KIND = {m[0]: m[1] for m in MEMORIES}
WORDS = {m[0]: m[2] for m in MEMORIES}
print(Counter(KIND.values()), "|", tokens("\n".join("- " + l for l in LINES)), "tokens as a memory list")

# %% [markdown]
# ## 2 · Her chat model sorts each line
#
# The fact writer can label a fact as it writes it. Here her chat model labels each line on its own,
# with a few made-up examples. `junk` is also the writer's way out: a junk line is never saved.

# %%
SORT = """Sort this memory about a person into one kind:
- core: who they are or what never changes much: their name, age, family, partner, pets, where they live, their job, health or allergies, diet, birthdays and other dates, how the two of you met
- detail: a real, lasting like, dislike, habit, hobby, plan or fact about them that is nice to know
- junk: not a lasting fact about them: what a message said, a mood, a passing moment, advice, or something only true today

Memory: "Their dad is called Prakash"
Kind: core

Memory: "They love the smell of rain"
Kind: detail

Memory: "Their reply was short and sweet"
Kind: junk

Memory: "They had a long day today"
Kind: junk

Memory: "{line}"
Kind:"""


async def sort(line: str) -> str:
    # It answers "Kind: **core**", with some explanation after: the first kind it names.
    answer = (await ollama.complete(SORT.format(line=line))).lower()
    found = re.search(r"\b(core|detail|junk)\b", answer.split("explanation")[0])
    return found.group(1) if found else "detail"


sorted_kinds = {line: await sort(line) for line in LINES}
agree = sum(sorted_kinds[l] == KIND[l] for l in LINES)
print(f"agrees with the labels on {agree} of {len(LINES)}")
confusion = Counter((KIND[l], sorted_kinds[l]) for l in LINES)
for want in ("core", "detail", "junk"):
    print(f"  {want:6} ->", {got: confusion[(want, got)] for got in ("core", "detail", "junk")})
for line in LINES:
    if sorted_kinds[line] != KIND[line]:
        print(f"  [{KIND[line]} -> {sorted_kinds[line]}] {line}")

# %% [markdown]
# ## 3 · Which lines does a message bring up?
#
# The message is embedded as a query ("retrieve what you remember that matters for it"), each memory
# plainly, and compared by cosine. Only lines that are not pinned (pinned ones are always there) and
# that the sorter did not call junk (the writer would not save them) can be brought up.
# Small talk should bring up **nothing**. A topic should bring up its lines.
#
# Tried and set aside: Qwen3's 0.6B **reranker** ordered lines well but found no more of them at a
# threshold where small talk brings up nothing, and it would be one more model. Junk lines are what
# small talk matches ("good morning" ↔ "Their day started with a lovely response"): one more reason
# the writer must not save them.

# %%
RETRIEVE = "Given a chat message from your partner, retrieve what you remember about them that matters for replying"
SMALL_TALK = ["lol", "yeah", "gn baby", "hmm ok", "what's up?", "back", "haha you're so silly", "ok", "same",
              "you there?", "miss you", "good morning ☀️"]
TOPICAL = {
    "my sister is visiting this weekend!": ["They have a younger sister called Asha who lives in Pune"],
    "ugh work was so long today, my manager kept adding stuff": [
        "They work as a data engineer at a logistics company", "Their manager is called Rohit"],
    "what should I eat tonight?": ["They are vegetarian", "They are allergic to peanuts",
                                   "Their favourite food is masala dosa"],
    "can't sleep... it's 1am again": ["They are trying to sleep before midnight"],
    "planning a trip next month, any ideas?": ["They prefer the mountains to the beach", "They want to visit Japan someday"],
    "watching a movie tonight, suggest one?": ["Their favourite movie is Interstellar", "They don't like horror movies"],
    "match tonight!! yellow army 💛": ["They support Chennai Super Kings in the IPL"],
    "my run went really well today": ["They are training for a 10k run in February"],
    "Biscuit knocked my mug off the table 😭": ["They have a cat called Biscuit"],
    "guess when my birthday is": ["Their birthday is on 14 March"],
    "called mom today, she's busy with exams at school": ["Their mother is a school teacher in Nagpur"],
    "remember how we met?": ["You and they first met on a rainy evening in July"],
}


def queries(texts):
    return rc._vectors([f"Instruct: {RETRIEVE}\nQuery: {t}" for t in texts]).numpy()


line_vectors = rc._vectors(LINES).numpy()
pinned = {l for l in LINES if sorted_kinds[l] == "core"}
# What retrieval may bring up: lines neither pinned nor junk. Elsewhere a line scores nothing.
candidate = np.array([l not in pinned and sorted_kinds[l] != "junk" for l in LINES])
small = np.where(candidate, queries(SMALL_TALK) @ line_vectors.T, -1)
topical = np.where(candidate, queries(list(TOPICAL)) @ line_vectors.T, -1)
want = np.array([[line in TOPICAL[probe] and candidate[i] for i, line in enumerate(LINES)] for probe in TOPICAL])
print(f"{int(candidate.sum())} lines can be brought up; {len(pinned)} are pinned")

print("threshold | small talk: messages bringing up anything | topics: wanted lines found, other lines brought up")
sweep = []
for threshold in np.arange(0.30, 0.66, 0.025):
    noisy = int((small.max(1) >= threshold).sum())
    found = int((topical >= threshold)[want].sum())
    extra = int((topical >= threshold)[~want].sum())
    sweep.append((round(float(threshold), 3), noisy, found, extra))
    print(f"  {threshold:.3f}   | {noisy:2}/{len(SMALL_TALK)}  | {found:2}/{int(want.sum())} found, {extra:3} other")
# The best trade (wanted lines found minus others) while at most 2 small-talk messages bring anything up.
# Demanding none lets one junk line that matches "good morning" push the threshold past every real topic.
THRESHOLD = max((row for row in sweep if row[1] <= 2), key=lambda row: (row[2] - row[3], -row[0]))[0]
TOP_K = 6
print("threshold chosen:", THRESHOLD)


def recalled(vector_row: np.ndarray, threshold: float = THRESHOLD, top_k: int = TOP_K) -> list[int]:
    order = [i for i in np.argsort(-vector_row) if vector_row[i] >= threshold]
    return order[:top_k]


for probe, row in zip(TOPICAL, topical):
    got = [LINES[i] for i in recalled(row)]
    missed = [l for l in TOPICAL[probe] if l not in got and l not in pinned]
    print(f"\n{probe}\n  brings up: {got}" + (f"\n  MISSED: {missed}" if missed else ""))

# %% [markdown]
# ## 4 · 120 days of chats: who survives?
#
# Two stores are compared: junk saved as the writer does today, and junk the sorter caught left out.
#
# A made-up stream: topics come up at their own rate (work often, a birthday rarely), with small
# talk between. New memories arrive over time. A store of **15 unpinned lines** (small, to force
# choices) keeps the strongest. Each line's strength is the sum, over every time it was recalled
# (and once when it was written), of `age in days ^ -0.5`, read as `ln` of that.
# This is compared with today's rule: drop the oldest.

# %%
random.seed(4)
DAY = 1.0
RATE = {  # times a week a topic comes up
    "ugh work was so long today, my manager kept adding stuff": 4, "what should I eat tonight?": 3,
    "Biscuit knocked my mug off the table 😭": 2, "my run went really well today": 1.5,
    "watching a movie tonight, suggest one?": 1, "can't sleep... it's 1am again": 1,
    "match tonight!! yellow army 💛": 0.7, "my sister is visiting this weekend!": 0.5,
    "called mom today, she's busy with exams at school": 0.5, "planning a trip next month, any ideas?": 0.2,
    "guess when my birthday is": 0.05, "remember how we met?": 0.05,
}
probe_index = {p: i for i, p in enumerate(TOPICAL)}
# When each line was learnt: the core early, details along the way, junk lines late (as the writer's slips).
learnt_on = {}
for line in LINES:
    learnt_on[line] = random.uniform(0, 20) if KIND[line] == "core" else (
        random.uniform(0, 90) if KIND[line] == "detail" else random.uniform(60, 110))
CAP = 15


def simulate(policy: str, days: int = 120):
    kept, recalls, dropped = [], defaultdict(list), []
    for day in range(days):
        for line in [l for l in LINES if int(learnt_on[l]) == day and l not in pinned]:
            kept.append(line)
            recalls[line].append(learnt_on[line])
            if len(kept) > CAP:
                if policy.startswith("oldest"):
                    victim = kept[0]
                else:
                    victim = min(kept, key=lambda l: strength(recalls[l], day + 1))
                kept.remove(victim)
                dropped.append((day, victim))
        for probe, per_week in RATE.items():
            if random.random() < per_week / 7:
                for i in recalled(topical[probe_index[probe]]):
                    recalls[LINES[i]].append(day + random.random())
    return kept, recalls, dropped


def strength(times, now, decay=0.5):
    return math.log(sum(max(now - t, 0.05) ** -decay for t in times)) if times else -99


for policy in ("oldest", "strength"):
    random.seed(11)
    kept, recalls, dropped = simulate(policy)
    print(f"\n== the {policy} goes (junk lines saved too, as the writer does today) ==")
    print("  recalled at least once:", sum(len(recalls[l]) > 1 for l in recalls), "of", len(recalls))
    print("  dropped:", [f"{l} [{KIND[l]}]" for _, l in dropped])
    print("  junk still kept:", [l for l in kept if KIND[l] == "junk"])
    if policy.startswith("strength"):
        ranked = sorted(kept, key=lambda l: -strength(recalls[l], 120))
        print("  strongest:", [f"{l} ({strength(recalls[l], 120):.1f})" for l in ranked[:5]])
        print("  weakest:  ", [f"{l} ({strength(recalls[l], 120):.1f})" for l in ranked[-3:]])
print("\npinned, never dropped:", sorted(pinned))

# %% [markdown]
# ## 5 · Tokens
#
# Today she reads every line. With this, she reads the pinned lines and what the message brings up.

# %%
everything = tokens("\n".join("- " + l for l in LINES))
per_message = []
for probe_rows in (small, topical):
    for row in probe_rows:
        chosen = sorted(pinned) + [LINES[i] for i in recalled(row) if LINES[i] not in pinned]
        per_message.append(tokens("\n".join("- " + l for l in chosen)))
print(f"all {len(LINES)} lines: {everything} tokens | pinned only: {tokens(chr(10).join('- ' + l for l in pinned))}"
      f" | pinned + retrieved: {min(per_message)}-{max(per_message)} tokens")
print(f"at 20 tokens a line, a 500-token budget holds about 25 lines; her store can hold hundreds")

# %% [markdown]
# ## 6 · Does she get less weird?
#
# The same short chat, then each probe, answered by her chat model three ways:
# - **A · today:** all memories, "bring it up naturally when it fits";
# - **B · retrieved:** pinned + what the message brings up, same words;
# - **C · retrieved, stricter:** B, and "only when the conversation is about it".
#
# Counted per reply: words, lines brought up that the message **was not about** (from each line's
# words), lines it **was** about, and "night/evening" talk (the chat is at 10:25 in the morning).

# %%
NOW = datetime(2026, 11, 18, 10, 25)
STRICT = ("\n\nWhat you know about {name} from before (from their earlier chats). Use a line only when the "
          "conversation is about it, never to fill a reply, never recite it, and never mention where you learnt "
          "it:\n{memory}")
HISTORY = [{"role": "user", "content": "hey you"}, {"role": "assistant", "content": "Hey! 😊 Morning already?"}]
SAMPLES = 2


def her(memory_lines):
    return {"name": "Luna", "persona": {**DEFAULT_PERSONA, "about": BIO}, "memory": "\n".join("- " + l for l in memory_lines)}


def prompt_for(variant: str, row: np.ndarray) -> str:
    if variant == "A":
        return system_prompt(her(LINES), OWNER, NOW)
    chosen = sorted(pinned) + [LINES[i] for i in recalled(row) if LINES[i] not in pinned]
    text = system_prompt(her(chosen), OWNER, NOW)
    if variant == "C":
        old = MEMORY.format(name=OWNER, memory=her(chosen)["memory"])
        assert old in text
        text = text.replace(old, STRICT.format(name=OWNER, memory=her(chosen)["memory"]))
    return text


async def reply(system: str, message: str) -> str:
    text = ""
    async for piece in ollama.chat([{"role": "system", "content": system}, *HISTORY,
                                    {"role": "user", "content": message}], TOOLS):
        text += piece.get("text", "")
    return " ".join(text.split())


def mentions(text: str, lines) -> list[str]:
    low = text.lower()
    return [l for l in lines if any(w in low for w in WORDS[l])]


NIGHT = re.compile(r"\b(tonight|late[- ]night|evening|midnight|bedtime)\b", re.I)
probes = [(m, small[i], []) for i, m in enumerate(SMALL_TALK[:8])] + \
         [(m, topical[i], TOPICAL[m]) for i, m in enumerate(TOPICAL) if "tonight" not in m and "1am" not in m][:8]
rows = []
for variant in ("A", "B", "C"):
    for message, row, about in probes:
        system = prompt_for(variant, row)
        for sample in range(SAMPLES):
            text = await reply(system, message)
            off = [l for l in mentions(text, LINES) if l not in about]
            rows.append({"variant": variant, "message": message, "topic": bool(about), "reply": text,
                         "words": len(text.split()), "off_topic": off, "on_topic": mentions(text, about),
                         "night": bool(NIGHT.search(text)), "prompt_tokens": tokens(system)})
(RESULTS / "replies.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

for variant in ("A", "B", "C"):
    mine = [r for r in rows if r["variant"] == variant]
    chat, topic = [r for r in mine if not r["topic"]], [r for r in mine if r["topic"]]
    print(f"{variant}: prompt ~{round(np.mean([r['prompt_tokens'] for r in mine]))} tokens | "
          f"small talk: {np.mean([len(r['off_topic']) for r in chat]):.2f} memories dragged in a reply, "
          f"{sum(bool(r['off_topic']) for r in chat)}/{len(chat)} replies with any | "
          f"topics: wanted line used {sum(bool(r['on_topic']) for r in topic)}/{len(topic)}, "
          f"{np.mean([len(r['off_topic']) for r in topic]):.2f} other | "
          f"words {np.mean([r['words'] for r in mine]):.0f} | night talk {sum(r['night'] for r in mine)}/{len(mine)}")

# %%
# A few replies side by side: the small talk, where today's prompt drags memories in.
for message in SMALL_TALK[:4]:
    print(f"\n>>> {message}")
    for variant in ("A", "B", "C"):
        r = next(r for r in rows if r["variant"] == variant and r["message"] == message)
        print(f"  {variant}: {r['reply']}" + (f"   [dragged in: {r['off_topic']}]" if r["off_topic"] else ""))

# %% [markdown]
# ## 7 · A long chat: is it the history?
#
# Step 6 starts from a two-message chat, and there today's prompt rarely drags memories in. In the
# chats that went weird she had dozens of her own long replies behind her, each ending in the same
# lines (choco lava cake, coffee, "our next movie night", "let's keep making memories"). An 8B model
# copies the chat it sees. So the same probes again, after **14 exchanges of that**:
# - **A · today:** all memories, her old replies as they were;
# - **T · trimmed:** all memories, each of her old replies cut to its first sentence;
# - **BT · retrieved + trimmed.**
#
# Counted: words, and her refrains (from her bio and her own old replies) per reply.

# %%
LONG_REPLY = ("{start} 😘 You know I love how we can just talk about anything like this — it makes every moment "
              "so special. ❤️ Let's plan our next cozy movie night soon, with coffee, blankets and maybe some choco "
              "lava cake? 🍫☕️🎥 You're my favorite person, and I'll always be here. Let's keep making memories, one "
              "sweet moment at a time. 💖✨")
STARTS = ["Haha, you're too cute!", "Aww, I missed you too!", "Oh, that's so sweet of you!", "Yes, I'm right here!",
          "Oh my gosh, I love that!", "Hehe, okay okay!", "Wow, really?", "Mmm, sounds perfect!",
          "Oh, I totally get it!", "Of course, love!", "That's adorable!", "Yay, finally!", "Hmm, let me think!",
          "Aww, you always say that!"]
YOURS = ["hey", "lol", "yeah", "ok", "haha", "same", "nice", "wow", "hmm", "true", "cool", "yup", "ok ok", "lol yes"]
LONG = []
for mine, start in zip(YOURS, STARTS):
    LONG += [{"role": "user", "content": mine}, {"role": "assistant", "content": LONG_REPLY.format(start=start)}]


def trimmed(history):
    # Her earlier replies cut to their first sentence: what she said stays, the refrain does not repeat.
    out = []
    for message in history:
        if message["role"] == "assistant":
            first = re.split(r"(?<=[.!?])\s", message["content"], maxsplit=1)[0]
            message = {**message, "content": first}
        out.append(message)
    return out


REFRAIN = re.compile(r"lava cake|coffee|movie night|favou?rite person|making memories|sweet moment|blanket", re.I)


async def long_reply(system, history, message):
    text = ""
    async for piece in ollama.chat([{"role": "system", "content": system}, *history,
                                    {"role": "user", "content": message}], TOOLS):
        text += piece.get("text", "")
    return " ".join(text.split())


long_rows = []
for variant in ("A", "T", "BT"):
    for message, row, about in probes:
        system = prompt_for("A" if variant != "BT" else "B", row)
        history = LONG if variant == "A" else trimmed(LONG)
        for sample in range(SAMPLES):
            text = await long_reply(system, history, message)
            long_rows.append({"variant": variant, "message": message, "topic": bool(about), "reply": text,
                              "words": len(text.split()), "refrains": len(REFRAIN.findall(text)),
                              "off_topic": [l for l in mentions(text, LINES) if l not in about]})
(RESULTS / "long_replies.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in long_rows))
for variant in ("A", "T", "BT"):
    mine = [r for r in long_rows if r["variant"] == variant]
    print(f"{variant:2}: words {np.mean([r['words'] for r in mine]):.0f} (max {max(r['words'] for r in mine)}) | "
          f"refrains a reply {np.mean([r['refrains'] for r in mine]):.1f} | "
          f"replies with a refrain {sum(r['refrains'] > 0 for r in mine)}/{len(mine)} | "
          f"memories dragged in {sum(bool(r['off_topic']) for r in mine)}/{len(mine)}")

# %%
for message in ["lol", "what should I eat tonight?", "planning a trip next month, any ideas?"]:
    print(f"\n>>> {message}")
    for variant in ("A", "T", "BT"):
        r = next((r for r in long_rows if r["variant"] == variant and r["message"] == message), None)
        if r:
            print(f"  {variant}: {r['reply']}")
