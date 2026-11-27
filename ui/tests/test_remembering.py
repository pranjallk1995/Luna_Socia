"""Remembering by herself: the neurons decide, her chat model writes the fact, and a fact she has is not saved again."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx
import numpy as np

from app.chat import Conversation
from app.images import ImageService
from app.ollama import Ollama
from app.persona import DEFAULT_PERSONA
from app.remembering import SHIPPED_NEURONS, Embedder, Neuron, Rememberer, fact_line, grounded, known_lines, load_neurons

try:
    from test_socia import MemoryStore, config, fake_services, png
except ImportError:  # run as a package
    from tests.test_socia import MemoryStore, config, fake_services, png

DIM = 1024


class FakeEmbedder:
    """Messages with "favourite" (and her lines with "my") point one way, anything else another; plain sentences
    get a vector from their words, so the same sentence is the same vector."""

    def __init__(self):
        self.calls = []

    def vectors(self, texts):
        self.calls.append(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            if text.startswith("Instruct:"):
                said = text.split("Query:", 1)[1].lower()
                out[row, 0 if ("favourite" in said or "my sister" in said) else 1] = 1
            else:
                for word in text.lower().split():
                    out[row, 2 + hash(word) % (DIM - 2)] += 1
                out[row] /= max(np.linalg.norm(out[row]), 1e-6)
        return out


def neurons():
    weight = np.zeros(2 * DIM, dtype=np.float32)
    weight[0] = 10
    sure = Neuron(np.zeros(2 * DIM), np.ones(2 * DIM), weight, -5, 0.5)
    return {"you": sure, "her": sure}


class RemembererTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.config = config(Path(self.folder.name))
        self.store = MemoryStore()
        self.her = self.store.create_companion("Luna", {"about": "She is 25, from Mumbai."}, "af_heart")
        self.prompts = []

    def tearDown(self):
        self.folder.cleanup()

    def rememberer(self, answer):
        async def complete(prompt):
            self.prompts.append(prompt)
            return answer(prompt)

        rememberer = Rememberer(self.config, self.store, complete, FakeEmbedder())
        rememberer.neurons = neurons()
        return rememberer

    def turn(self, rememberer, before, message, reply):
        return asyncio.run(rememberer.after_turn(self.her["id"], before, message, reply))

    def test_a_fact_about_them_is_written_and_kept(self):
        rememberer = self.rememberer(lambda prompt: "Fact: Their favourite food is biryani.")
        saved = self.turn(rememberer, "Biryani again? 🍛", "yes, my favourite", "Then let's order some! 🍛")
        self.assertEqual([(s["about"], s["fact"]) for s in saved], [("them", "Their favourite food is biryani")])
        self.assertIn("- Their favourite food is biryani", self.store.companion(self.her["id"])["memory"])
        # The writer is asked to word it from the message and the line before it, and from nothing else.
        self.assertIn('Message: "yes, my favourite"', self.prompts[0])
        self.assertIn('Previous message: "Biryani again? 🍛"', self.prompts[0])
        self.assertNotIn("She is 25", self.prompts[0])

    def test_small_talk_is_not_written(self):
        rememberer = self.rememberer(lambda prompt: "They said good night")
        self.assertEqual(self.turn(rememberer, "", "good night baby", "Sweet dreams 😘"), [])
        self.assertEqual(self.prompts, [])
        self.assertEqual(self.store.companion(self.her["id"])["memory"], "")

    def test_a_fact_she_already_has_is_not_saved_again(self):
        self.store.remember(self.her["id"], "Their favourite food is pizza", 4800)
        rememberer = self.rememberer(lambda prompt: "Their favourite food is pizza")
        self.assertEqual(self.turn(rememberer, "", "pizza is my favourite", "Yum!"), [])
        self.assertEqual(self.store.companion(self.her["id"])["memory"], "- Their favourite food is pizza")

    def test_facts_are_compared_by_meaning_with_what_she_has_and_each_other(self):
        self.store.update_companion(self.her["id"], {"memory": "- They have a cat named Mini"})

        def judge(prompt):
            # Her chat model: the same fact in other words is the same; a dog is not a cat.
            return "YES" if "A: They have a cat named Mini." in prompt else "NO"

        rememberer = self.rememberer(judge)
        her = self.store.companion(self.her["id"])
        kept = asyncio.run(rememberer.new_facts(her, "you", [
            "- They have a cat named Mini.", "They have a dog named Mini", "They work as a nurse",
            "they work as a nurse", "They love rain",
        ]))
        self.assertEqual(kept, ["They have a dog named Mini", "They work as a nurse", "They love rain"])
        asked = [prompt for prompt in self.prompts if prompt.startswith("Do these two lines")]
        self.assertTrue(any("A: They have a dog named Mini" in prompt for prompt in asked))
        self.assertFalse(any("A: they work as a nurse" in prompt for prompt in asked))  # a near copy needs no asking
        self.assertEqual(asyncio.run(rememberer.new_facts(her, "her", ["She is 25, from Mumbai.", "She paints"])), ["She paints"])

    def test_without_her_chat_model_closeness_alone_decides(self):
        self.store.update_companion(self.her["id"], {"memory": "- They have a cat named Mini"})
        rememberer = Rememberer(self.config, self.store, None, FakeEmbedder())
        her = self.store.companion(self.her["id"])
        self.assertEqual(asyncio.run(rememberer.new_facts(her, "you", ["They have a cat named Mini.", "They love rain"])),
                         ["They love rain"])

    def test_an_unreachable_embedding_model_keeps_the_facts(self):
        class Down:
            model = "qwen3-embedding:0.6b"

            def vectors(self, texts):
                raise httpx.ConnectError("Ollama is down")

        rememberer = Rememberer(self.config, self.store, None, Down())
        self.store.update_companion(self.her["id"], {"memory": "- They love rain"})
        with self.assertLogs("app.remembering", "ERROR"):
            kept = asyncio.run(rememberer.new_facts(self.store.companion(self.her["id"]), "you", ["They love rain."]))
        self.assertEqual(kept, ["They love rain."])

    def test_a_fact_about_herself_goes_into_her_bio(self):
        rememberer = self.rememberer(lambda prompt: "Her favourite colour is lavender")
        saved = self.turn(rememberer, "", "what's yours?", "My favourite colour is lavender 💜")
        self.assertEqual([(s["about"], s["fact"]) for s in saved], [("me", "Her favourite colour is lavender")])
        self.assertEqual(self.store.companion(self.her["id"])["persona"]["about"],
                         "She is 25, from Mumbai.\nHer favourite colour is lavender.")

    def test_nothing_lasting_is_none_and_not_saved(self):
        rememberer = self.rememberer(lambda prompt: "Fact: NONE")
        self.assertEqual(self.turn(rememberer, "", "my favourite part of today was that nap lol", "Haha cute"), [])
        self.assertEqual(self.store.companion(self.her["id"])["memory"], "")
        self.assertTrue(self.prompts)  # it was asked, and said there was nothing to keep

    def test_a_name_the_chat_never_said_is_dropped(self):
        # Copied from the writer's examples, not learnt from the chat.
        rememberer = self.rememberer(lambda prompt: "Their boss is called Oskarine")
        self.assertEqual(self.turn(rememberer, "How was work?", "my favourite boss quit today", "Oh no!"), [])
        self.assertEqual(self.store.companion(self.her["id"])["memory"], "")

    def test_names_places_and_numbers_must_be_in_the_chat(self):
        self.assertTrue(grounded("Their sister Asha lives in Pune", "my sister asha lives in pune"))
        self.assertTrue(grounded("They like the movie The Prestige", "watched prestige again", "the best one"))
        self.assertTrue(grounded("They have two older siblings", "two, both older", "Do you have siblings?"))
        self.assertTrue(grounded("Their cats are Maxi's daughters", "both are maxi's daughters"))
        self.assertFalse(grounded("Their boss is Neha", "ugh my boss made us redo the deck"))
        self.assertFalse(grounded("Their birthday is on 14 March", "guess when my birthday is in march"))

    def test_a_reply_without_words_is_not_read(self):
        rememberer = self.rememberer(lambda prompt: "Their sister is visiting")
        self.turn(rememberer, "", "my sister is visiting", "")
        # Your message and her line before it; nothing for an empty reply.
        self.assertEqual(len(rememberer.embedder.calls[0]), 2)

    def test_the_writers_answer_becomes_one_clean_line(self):
        self.assertEqual(fact_line('Fact: "They love pizza."\nThey also said...'), "They love pizza")
        self.assertEqual(fact_line("- Her favourite colour is lavender"), "Her favourite colour is lavender")
        self.assertEqual(fact_line(""), "")
        self.assertEqual(fact_line("Fact: NONE"), "")
        self.assertEqual(fact_line("**NONE**"), "")

    def test_what_she_has_is_her_memory_or_her_bio(self):
        her = {"memory": "- They love pizza\n- They have a cat", "persona": {"about": "She is 25. She loves chai!\nShe bakes."}}
        self.assertEqual(known_lines(her, "you"), ["They love pizza", "They have a cat"])
        self.assertEqual(known_lines(her, "her"), ["She is 25.", "She loves chai!", "She bakes."])

    def test_the_shipped_neurons_read_both_lines(self):
        shipped = load_neurons(SHIPPED_NEURONS)
        for who in ("you", "her"):
            self.assertEqual(shipped[who].weight.shape, (2 * DIM,))
            self.assertEqual(shipped[who].threshold, 0.5)

    def test_her_own_neurons_come_first(self):
        own = self.config.data_dir / "remember" / "neurons.npz"
        own.parent.mkdir(parents=True)
        own.write_bytes(SHIPPED_NEURONS.read_bytes())
        self.assertEqual(Rememberer(self.config, self.store, None, FakeEmbedder()).neurons_from, own)


class ConversationRemembersTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.config = config(Path(self.folder.name))
        self.store = MemoryStore()
        self.her = self.store.create_companion("Luna", dict(DEFAULT_PERSONA), "af_heart")
        profile = self.config.profile_path(self.her["id"])
        profile.parent.mkdir(parents=True)
        profile.write_bytes(png("pink"))

    def tearDown(self):
        self.folder.cleanup()

    def run_turn(self, reply, text, photo=None):
        async def complete(prompt):
            return "Their favourite food is pizza"

        rememberer = Rememberer(self.config, self.store, complete, FakeEmbedder())
        rememberer.neurons = neurons()
        transport, images = fake_services([[{"content": reply}]], {})
        conversation = Conversation(self.config, self.store, Ollama(self.config, transport),
                                    ImageService(self.config, images), rememberer=rememberer)

        async def collect():
            companion = self.store.companion(self.her["id"])
            return [e async for e in conversation.send(companion, None, text, photo, "Sam", "Asia/Kolkata")]

        return asyncio.run(collect())

    def test_she_remembers_without_being_asked(self):
        events = self.run_turn("Pizza? Let's order some! 🍕", "pizza is my favourite")
        remembered = next(e for e in events if e["type"] == "remembered")
        self.assertEqual(remembered["facts"], ["Their favourite food is pizza"])
        # Shown before the turn is done, as her remember tool's facts are.
        types = [e["type"] for e in events]
        self.assertLess(types.index("remembered"), types.index("done"))
        self.assertIn("- Their favourite food is pizza", self.store.companion(self.her["id"])["memory"])

    def test_her_remember_tool_skips_a_fact_she_has_in_other_words(self):
        self.store.update_companion(self.her["id"], {"memory": "- They have a cat named Mini"})
        rememberer = Rememberer(self.config, self.store, None, FakeEmbedder())
        transport, images = fake_services([], {})
        conversation = Conversation(self.config, self.store, Ollama(self.config, transport),
                                    ImageService(self.config, images), rememberer=rememberer)
        calls = [{"function": {"name": "remember", "arguments": {"fact": "They have a cat named Mini."}}},
                 {"function": {"name": "remember", "arguments": {"fact": "They work as a nurse"}}},
                 {"function": {"name": "remember", "arguments": {"fact": "I am from Mumbai, 25.", "about": "me"}}}]
        saved = asyncio.run(conversation.remember(self.her["id"], calls))
        self.assertEqual(saved, [{"about": "them", "fact": "They work as a nurse"},
                                 {"about": "me", "fact": "She is from Mumbai, 25."}])
        self.assertEqual(self.store.companion(self.her["id"])["memory"].count("Mini"), 1)

    def test_a_photo_turn_is_left_to_the_vision_model(self):
        events = self.run_turn("Cute! 😍", "my favourite photo", photo=png("red"))
        self.assertNotIn("remembered", [e["type"] for e in events])

    def test_she_reads_the_part_of_her_memory_recall_chose(self):
        self.store.update_companion(self.her["id"], {"memory": "- Their cat is Biscuit\n- They love rain"})

        class Recall:
            upkept = []

            async def memory(self, companion, message):
                self.message = message
                return "- Their cat is Biscuit"

            async def upkeep(self, companion_id):
                self.upkept.append(companion_id)

        recall = Recall()
        seen = {}
        transport, images = fake_services([[{"content": "Morning! ☀️"}]], seen)
        conversation = Conversation(self.config, self.store, Ollama(self.config, transport),
                                    ImageService(self.config, images), recall=recall)

        async def collect():
            companion = self.store.companion(self.her["id"])
            events = [e async for e in conversation.send(companion, None, "good morning", None, "Sam", "Asia/Kolkata")]
            await asyncio.gather(*conversation.upkeeps)
            return events

        asyncio.run(collect())
        prompt = next(c for c in seen["chats"] if c.get("stream", True))["messages"][0]["content"]
        self.assertIn("- Their cat is Biscuit", prompt)
        self.assertNotIn("They love rain", prompt)
        self.assertEqual(recall.message, "good morning")
        self.assertEqual(recall.upkept, [self.her["id"]])


if __name__ == "__main__":
    unittest.main()


class EmbedderTests(unittest.TestCase):
    def test_vectors_come_from_ollamas_embedding_model_at_unit_length(self):
        asked = []

        def answer(request):
            asked.append((request.url.path, request.headers.get("X-Luna-App"), __import__("json").loads(request.content)))
            return httpx.Response(200, json={"embeddings": [[3.0, 4.0], [0.0, 2.0]]})

        with tempfile.TemporaryDirectory() as folder:
            embedder = Embedder(config(Path(folder)), transport=httpx.MockTransport(answer))
            vectors = embedder.vectors(["a", "b"])
        self.assertEqual(asked, [("/api/embed", "socia", {"model": "qwen3-embedding:0.6b", "input": ["a", "b"]})])
        np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
