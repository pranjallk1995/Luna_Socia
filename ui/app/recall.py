"""What she reads of her memory for one reply, and what she forgets: from notebooks/04_memory_recall.

Her whole memory in every reply made an 8B model drag it into small talk ("lol" answered with her plans for
the two of you). So each line of it is sorted once by her chat model:
- core: who you are (name, age, family, pets, home, work, health, diet, dates, how you met): always read;
- detail: a like, habit, hobby or plan: read only when your message is about it (its embedding at least
  RECALL_THRESHOLD close to the message's, at most RECALL_LINES of them);
- junk: not a lasting fact ("Their message was..."): never read, and the first to go.
A line not sorted yet is read as core. Everything she reads stays within MEMORY_TOKENS.

Each time she reads a detail it grows stronger; with time it grows weaker: ACT-R's base-level activation,
ln(sum of age ** -0.5) over the times she learnt and read it, in days. Past MEMORY_DETAILS details, the junk and
then the weakest are forgotten, like anyone forgets. Core lines never are. The memory text stays what the owner
sees and edits; this only keeps, per line, its embedding, kind and the times she read it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.config import SociaConfig

logger = logging.getLogger(__name__)

RETRIEVE = "Given a chat message from your partner, retrieve what you remember about them that matters for replying"
KINDS = ("core", "detail", "junk")
# Sorted a few at a time after a turn, so a long memory does not hold her chat model up.
SORT_AT_ONCE = 8

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


def lines_of(memory: str) -> list[str]:
    """Her memory's lines, without their list dashes, once each."""
    seen, lines = set(), []
    for line in (memory or "").splitlines():
        line = line.strip().removeprefix("- ").strip()
        if line and line.casefold() not in seen:
            seen.add(line.casefold())
            lines.append(line)
    return lines


def kind_of(answer: str) -> str:
    """The kind the sorter named first (it answers "Kind: **core**", sometimes with an explanation after)."""
    found = re.search(rf"\b({'|'.join(KINDS)})\b", answer.lower().split("explanation")[0])
    return found.group(1) if found else "detail"


def strength(recalls: list[datetime], now: datetime) -> float:
    """ACT-R's base-level activation: higher the more often and the more recently she read it."""
    ages = [max((now - at).total_seconds() / 86400, 0.05) for at in recalls]
    return math.log(sum(age ** -0.5 for age in ages)) if ages else -99.0


def chosen(lines: list[str], kinds: dict[str, str], scores: dict[str, float], count, budget: int) -> list[str]:
    """The lines she reads: core (and unsorted) first, in her memory's order, then the details the message brought
    up, closest first, as many as fit in ``budget`` tokens."""
    pinned = [line for line in lines if kinds.get(line, "") in ("", "core")]
    found = sorted((line for line in scores if kinds.get(line) == "detail" and scores[line] >= SociaConfig.RECALL_THRESHOLD),
                   key=lambda line: -scores[line])[:SociaConfig.RECALL_LINES]
    read, used = [], 0
    for line in [*pinned, *found]:
        cost = count(f"- {line}\n")
        if used + cost > budget:
            continue
        read.append(line)
        used += cost
    return read


def fading(lines: list[str], kinds: dict[str, str], recalls: dict[str, list[datetime]], now: datetime,
           keep: int) -> list[str]:
    """What she forgets: junk first, then the weakest details, until at most ``keep`` details and junk are left."""
    loose = [line for line in lines if kinds.get(line) in ("detail", "junk")]
    loose.sort(key=lambda line: (kinds[line] != "junk", strength(recalls.get(line, []), now)))
    return loose[:max(len(loose) - keep, 0)]


class Recall:
    """Her memory, read by topic: the embedding model from remembering, and her chat model to sort the lines."""

    def __init__(self, store, embedder, complete) -> None:
        self.store, self.embedder, self.complete = store, embedder, complete
        self.lock = asyncio.Lock()

    def count(self, text: str) -> int:
        if self.embedder.tokenizer is None:
            with self.embedder.lock:
                self.embedder.load()
        return len(self.embedder.tokenizer.encode(text).ids)

    async def rows(self, companion_id: int, lines: list[str]) -> dict[str, dict[str, Any]]:
        """Each line's row, embedding the lines new to her (learnt, or written by the owner) first."""
        rows = {row["line"]: row for row in await asyncio.to_thread(self.store.memory_lines, companion_id)}
        missing = [line for line in lines if line not in rows]
        if missing:
            vectors = await asyncio.to_thread(self.embedder.vectors, missing)
            await asyncio.to_thread(self.store.add_memory_lines, companion_id,
                                    [(line, vector.astype(np.float32).tobytes()) for line, vector in zip(missing, vectors)])
            rows = {row["line"]: row for row in await asyncio.to_thread(self.store.memory_lines, companion_id)}
        return rows

    async def memory(self, companion: dict[str, Any], message: str) -> str:
        """The part of her memory she reads for a reply to ``message``."""
        lines = lines_of(companion.get("memory") or "")
        if not lines:
            return ""
        rows = await self.rows(companion["id"], lines)
        kinds = {line: rows[line]["kind"] for line in lines if line in rows}
        details = [line for line in lines if kinds.get(line) == "detail"]
        scores: dict[str, float] = {}
        if details and message.strip():
            query = await asyncio.to_thread(self.embedder.vectors, [f"Instruct: {RETRIEVE}\nQuery: {message[:800]}"])
            matrix = np.stack([np.frombuffer(rows[line]["vector"], dtype=np.float32) for line in details])
            scores = dict(zip(details, (matrix @ query[0]).tolist()))
        read = await asyncio.to_thread(chosen, lines, kinds, scores, self.count, SociaConfig.MEMORY_TOKENS)
        brought = [line for line in read if kinds.get(line) == "detail"]
        if brought:
            logger.info("Recalled for this message: %s", brought)
            await asyncio.to_thread(self.store.recalled, companion["id"], brought)
        return "\n".join(f"- {line}" for line in read)

    async def upkeep(self, companion_id: int) -> None:
        """After a turn: sort a few lines not sorted yet, and forget the weakest details past MEMORY_DETAILS."""
        async with self.lock:
            companion = await asyncio.to_thread(self.store.companion, companion_id)
            if companion is None:
                return
            lines = lines_of(companion.get("memory") or "")
            rows = await self.rows(companion_id, lines)
            kinds = {line: rows[line]["kind"] for line in lines if line in rows}
            for line in [line for line in lines if not kinds.get(line)][:SORT_AT_ONCE]:
                kinds[line] = kind_of(await self.complete(SORT.format(line=line)))
                await asyncio.to_thread(self.store.sort_memory_line, companion_id, line, kinds[line])
                logger.info("Memory line sorted as %s: %s", kinds[line], line)
            recalls = {line: list(rows[line]["recalls"]) for line in lines if line in rows}
            gone = fading(lines, kinds, recalls, datetime.now(timezone.utc), SociaConfig.MEMORY_DETAILS)
            if gone:
                logger.info("Forgotten, the weakest: %s", gone)
            stale = set(rows) - set(lines)
            if gone or stale:
                await asyncio.to_thread(self.store.fade, companion_id, gone, [l for l in lines if l not in gone])
