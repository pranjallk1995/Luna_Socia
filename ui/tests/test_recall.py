"""Her memory read by topic: core lines always, details when the message is about them, the weakest forgotten."""

import asyncio
import threading
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from app.config import SociaConfig
from app.recall import Recall, chosen, fading, kind_of, lines_of, strength

TOPICS = ["cat", "work", "movie", "food", "run"]


class Tokenizer:
    class Encoding:
        def __init__(self, text):
            self.ids = text.split()

    def encode(self, text):
        return self.Encoding(text)


class Embedder:
    """A line or a message points along the topics it names; nothing else."""

    def __init__(self):
        self.tokenizer, self.lock = Tokenizer(), threading.Lock()

    def load(self):
        pass

    def vectors(self, texts):
        out = np.zeros((len(texts), len(TOPICS) + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            said = text.split("Query:", 1)[-1].lower()
            for index, topic in enumerate(TOPICS):
                out[row, index] = topic in said
            out[row, -1] = 0.2
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Store:
    def __init__(self, memory):
        self.companion_ = {"id": 1, "memory": memory}
        self.rows = {}

    def companion(self, companion_id):
        return self.companion_

    def memory_lines(self, companion_id):
        return [{"line": line, **row} for line, row in self.rows.items()]

    def add_memory_lines(self, companion_id, lines):
        for line, vector in lines:
            self.rows.setdefault(line, {"kind": "", "vector": vector, "recalls": [datetime.now(timezone.utc)]})

    def sort_memory_line(self, companion_id, line, kind):
        self.rows[line]["kind"] = kind

    def recalled(self, companion_id, lines):
        for line in lines:
            self.rows[line]["recalls"].append(datetime.now(timezone.utc))

    def fade(self, companion_id, lines, keep):
        self.companion_["memory"] = "\n".join(l for l in self.companion_["memory"].splitlines()
                                              if l.removeprefix("- ") not in lines)
        self.rows = {line: row for line, row in self.rows.items() if line in keep and line not in lines}


KINDS = {"Their cat is called Biscuit": "core", "They work as a data engineer": "core",
         "Their favourite movie is Interstellar": "detail", "They love masala dosa food": "detail",
         "They run 10k on Sundays": "detail", "Their message was about the weather": "junk"}


async def sorter(prompt):
    line = prompt.rsplit('Memory: "', 1)[1].split('"', 1)[0]
    return f"Kind: **{KINDS[line]}**\nExplanation: a detail, maybe"


def memory():
    return "\n".join(f"- {line}" for line in KINDS)


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.store = Store(memory())
        self.recall = Recall(self.store, Embedder(), sorter)

    def test_before_sorting_she_reads_all_of_it(self):
        read = asyncio.run(self.recall.memory(self.store.companion_, "lol"))
        self.assertEqual(read, memory())

    def test_small_talk_brings_up_only_the_core(self):
        asyncio.run(self.recall.upkeep(1))
        read = asyncio.run(self.recall.memory(self.store.companion_, "lol"))
        self.assertEqual(read, "- Their cat is called Biscuit\n- They work as a data engineer")

    def test_a_topic_brings_up_its_detail_and_makes_it_stronger(self):
        asyncio.run(self.recall.upkeep(1))
        read = asyncio.run(self.recall.memory(self.store.companion_, "watching a movie tonight, suggest one?"))
        self.assertIn("- Their favourite movie is Interstellar", read)
        self.assertNotIn("dosa", read)
        self.assertNotIn("weather", read)
        self.assertEqual(len(self.store.rows["Their favourite movie is Interstellar"]["recalls"]), 2)

    def test_a_line_the_owner_removed_is_dropped(self):
        asyncio.run(self.recall.upkeep(1))
        self.store.companion_["memory"] = "- Their cat is called Biscuit"
        asyncio.run(self.recall.upkeep(1))
        self.assertEqual(list(self.store.rows), ["Their cat is called Biscuit"])

    def test_past_the_limit_junk_then_the_weakest_go_and_core_stays(self):
        asyncio.run(self.recall.upkeep(1))
        now = datetime.now(timezone.utc)
        self.store.rows["They run 10k on Sundays"]["recalls"] = [now - timedelta(days=90)]
        original = SociaConfig.MEMORY_DETAILS
        SociaConfig.MEMORY_DETAILS = 2
        try:
            asyncio.run(self.recall.upkeep(1))
        finally:
            SociaConfig.MEMORY_DETAILS = original
        left = lines_of(self.store.companion_["memory"])
        self.assertEqual(left, ["Their cat is called Biscuit", "They work as a data engineer",
                                "Their favourite movie is Interstellar", "They love masala dosa food"])


class PartsTest(unittest.TestCase):
    def test_kind_is_the_first_one_named(self):
        self.assertEqual(kind_of("Kind: **core**"), "core")
        self.assertEqual(kind_of("junk. Explanation: not core"), "junk")
        self.assertEqual(kind_of("hmm"), "detail")

    def test_often_and_lately_read_is_stronger(self):
        now = datetime.now(timezone.utc)
        old = strength([now - timedelta(days=60)], now)
        lately = strength([now - timedelta(days=60), now - timedelta(days=1)], now)
        self.assertGreater(lately, old)

    def test_everything_read_fits_the_budget(self):
        lines = ["a b c", "d e f", "g h i"]
        read = chosen(lines, {}, {}, lambda text: len(text.split()), 8)
        self.assertEqual(read, ["a b c", "d e f"])

    def test_fading_keeps_core(self):
        now = datetime.now(timezone.utc)
        kinds = {"a": "core", "b": "detail", "c": "junk"}
        self.assertEqual(fading(["a", "b", "c"], kinds, {"b": [now]}, now, 0), ["c", "b"])
        self.assertEqual(fading(["a", "b", "c"], kinds, {"b": [now]}, now, 1), ["c"])


if __name__ == "__main__":
    unittest.main()
