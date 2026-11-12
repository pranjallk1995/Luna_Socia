"""Tests for Luna Socia: the gate, girlfriends and their memory, what she reads, the image workflow, and full turns."""

import asyncio
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from starlette.testclient import TestClient

from app.chat import clean_name, Conversation, model_messages, without_written_pictures, written_picture
from app.memory import ExportError, MemoryReader, fact_lines, parse_export, their_words
from app.config import SociaConfig
from app.images import ComfyUI, workflow
from app.ollama import Ollama
from app.persona import DEFAULT_PERSONA, clean_persona, image_prompt, system_prompt
from app.voice import language, speakable
from app.web import create_app

NOW = datetime(2026, 11, 10, 20, 30, tzinfo=timezone.utc)


def config(data_dir: Path = Path("/tmp/socia-test")) -> SociaConfig:
    return SociaConfig(database_url="postgresql://unused", auth_internal_url="http://auth",
                       auth_public_url="https://auth.example.com", public_url="https://socia.example.com",
                       ollama_url="http://ollama", comfyui_url="http://comfy", data_dir=data_dir)


def png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, "PNG")
    return buffer.getvalue()


class MemoryStore:
    """The store's methods over plain lists."""

    def __init__(self) -> None:
        self.companions_: list[dict] = []
        self.sources: list[dict] = []
        self.chats_: list[dict] = []
        self.images: list[dict] = []
        self.messages: list[dict] = []

    def create_companion(self, name, persona, voice):
        row = {"id": len(self.companions_) + 1, "name": name, "persona": persona, "voice": voice, "memory": "",
               "created_at": NOW}
        self.companions_.append(row)
        return self.companion(row["id"])

    def companion(self, companion_id):
        row = next((c for c in self.companions_ if c["id"] == companion_id), None)
        if row is None:
            return None
        chats = [c for c in self.chats_ if c["companion_id"] == companion_id]
        sources = [s for s in self.sources if s["companion_id"] == companion_id]
        return {**row, "chats": len(chats), "last_active": max((c["updated_at"] for c in chats), default=None),
                "sources": len(sources), "read": sum(s["facts"] is not None for s in sources)}

    def companions(self):
        return [self.companion(c["id"]) for c in reversed(self.companions_)]

    def update_companion(self, companion_id, fields):
        row = next((c for c in self.companions_ if c["id"] == companion_id), None)
        if row:
            row.update({k: v for k, v in fields.items() if k in ("name", "persona", "voice", "memory")})
        return self.companion(companion_id)

    def delete_companion(self, companion_id):
        if not self.companion(companion_id):
            return None
        files = []
        for chat in [c for c in self.chats_ if c["companion_id"] == companion_id]:
            files += self.delete_chat(chat["id"])
        self.companions_ = [c for c in self.companions_ if c["id"] != companion_id]
        self.sources = [s for s in self.sources if s["companion_id"] != companion_id]
        return files

    def oldest_companion(self):
        return self.companion(self.companions_[0]["id"]) if self.companions_ else None

    def add_memory_sources(self, companion_id, source, conversations):
        known = {(s["companion_id"], s["source"], s["external_id"]) for s in self.sources}
        added = 0
        for conversation in conversations:
            if (companion_id, source, conversation["id"]) in known:
                continue
            self.sources.append({"id": len(self.sources) + 1, "companion_id": companion_id, "source": source,
                                 "external_id": conversation["id"], "title": conversation["title"],
                                 "messages": conversation["messages"], "started_at": conversation.get("started_at"),
                                 "facts": None, "merged": False})
            added += 1
        return added

    def unread_source(self):
        return next((dict(s) for s in self.sources if s["facts"] is None), None)

    def set_source_facts(self, source_id, facts):
        next(s for s in self.sources if s["id"] == source_id)["facts"] = facts

    def unmerged_facts(self, companion_id):
        return [{"id": s["id"], "facts": s["facts"]} for s in self.sources
                if s["companion_id"] == companion_id and s["facts"] is not None and not s["merged"]]

    def merge_memory(self, companion_id, memory, source_ids):
        self.update_companion(companion_id, {"memory": memory})
        for source in self.sources:
            if source["id"] in source_ids:
                source["merged"] = True

    def forget_sources(self, companion_id):
        before = len(self.sources)
        self.sources = [s for s in self.sources if s["companion_id"] != companion_id]
        return before - len(self.sources)

    def create_chat(self, companion_id, title=""):
        chat = {"id": len(self.chats_) + 1, "companion_id": companion_id, "title": title, "named": "",
                "created_at": NOW, "updated_at": NOW}
        self.chats_.append(chat)
        return dict(chat)

    def chat(self, chat_id):
        return next((dict(c) for c in self.chats_ if c["id"] == chat_id), None)

    def chats(self, companion_id):
        rows = []
        for chat in reversed([c for c in self.chats_ if c["companion_id"] == companion_id]):
            last = next((m for m in reversed(self.messages) if m["chat_id"] == chat["id"]), None)
            rows.append({**chat, "last_role": last and last["role"], "last_content": last and last["content"],
                         "last_image_id": last and last["image_id"]})
        return rows

    def rename_chat(self, chat_id, title):
        chat = next((c for c in self.chats_ if c["id"] == chat_id), None)
        if chat:
            chat.update(title=title, named="owner")
        return chat

    def name_chat(self, chat_id, title):
        chat = next(c for c in self.chats_ if c["id"] == chat_id)
        if chat["named"] == "owner":
            return None
        chat.update(title=title, named="auto")
        return dict(chat)

    def keep_name(self, chat_id):
        next(c for c in self.chats_ if c["id"] == chat_id)["named"] = "owner"

    def unnamed_chats(self, limit=20):
        return [dict(c) for c in self.chats_ if c["named"] == ""
                and any(m["chat_id"] == c["id"] and m["role"] == "luna" for m in self.messages)][:limit]

    def opening(self, chat_id, count=6):
        return [m for m in self.messages if m["chat_id"] == chat_id][:count]

    def title_chat(self, chat_id, title):
        chat = next(c for c in self.chats_ if c["id"] == chat_id)
        chat["title"] = chat["title"] or title

    def delete_chat(self, chat_id):
        if not self.chat(chat_id):
            return None
        files = [i["file"] for i in self.images if i["chat_id"] == chat_id]
        self.chats_ = [c for c in self.chats_ if c["id"] != chat_id]
        self.images = [i for i in self.images if i["chat_id"] != chat_id]
        self.messages = [m for m in self.messages if m["chat_id"] != chat_id]
        return files

    def add_image(self, chat_id, kind, file, prompt=None):
        image_id = len(self.images) + 1
        self.images.append({"id": image_id, "chat_id": chat_id, "kind": kind, "file": file, "prompt": prompt})
        return image_id

    def image(self, image_id):
        return next((i for i in self.images if i["id"] == image_id), None)

    def add_message(self, chat_id, role, content, image_id=None, photo_note=None):
        image = self.image(image_id) if image_id else None
        row = {"id": len(self.messages) + 1, "chat_id": chat_id, "role": role, "content": content,
               "image_id": image_id, "photo_note": photo_note, "created_at": NOW,
               "image_kind": image and image["kind"], "image_prompt": image and image["prompt"]}
        self.messages.append(row)
        return row

    def set_photo_note(self, message_id, note):
        next(m for m in self.messages if m["id"] == message_id)["photo_note"] = note

    def recent(self, chat_id, limit, before=None):
        rows = [m for m in self.messages if m["chat_id"] == chat_id and (before is None or m["id"] < before)]
        return rows[-limit:]

    def message(self, message_id):
        row = next((m for m in self.messages if m["id"] == message_id), None)
        if row is None:
            return None
        chat = self.chat(row["chat_id"])
        return {**row, "companion_id": chat["companion_id"], "voice": self.companion(chat["companion_id"])["voice"]}

    def delete_message(self, message_id):
        row = self.message(message_id)
        if row is None:
            return None
        self.messages = [m for m in self.messages if m["id"] != message_id]
        image = self.image(row["image_id"]) if row["image_id"] else None
        if image is None or any(m["image_id"] == image["id"] for m in self.messages):
            return []
        self.images = [i for i in self.images if i["id"] != image["id"]]
        return [image["file"]]

    def last_photo(self, chat_id):
        return next((i for i in reversed(self.images) if i["kind"] == "photo" and i["chat_id"] == chat_id), None)


class PromptTests(unittest.TestCase):
    def test_luna_reads_photos_and_her_own_pictures_as_notes(self):
        rows = [
            {"role": "user", "content": "look", "photo_note": "A beach at sunset.", "image_kind": "photo"},
            {"role": "luna", "content": "wow 😍", "image_kind": "generated", "image_prompt": "Luna on the beach"},
            {"role": "luna", "content": "hi", "image_kind": None},
        ]
        self.assertEqual(model_messages(rows), [
            {"role": "user", "content": "look\n[Photo: A beach at sunset.]"},
            {"role": "assistant", "content": "wow 😍", "tool_calls": [{"function": {"name": "create_image", "arguments": {
                "description": "Luna on the beach", "include_luna": True, "use_their_photo": False}}}]},
            {"role": "tool", "content": "The picture was sent."},
            {"role": "assistant", "content": "hi"},
        ])

    def test_a_picture_she_wrote_out_is_taken_out_and_taken_as_a_call(self):
        text = "Okay, here's a little more…\n[You sent a picture: Luna by the window, in a sheer blouse.]"
        self.assertEqual(without_written_pictures(text), "Okay, here's a little more…")
        self.assertEqual(written_picture(text)["function"]["arguments"]["description"],
                         "Luna by the window, in a sheer blouse.")
        self.assertIsNone(written_picture("[hugs you] no picture here"))
        self.assertEqual(without_written_pictures("[hugs you] hi"), "[hugs you] hi")
        rows = [{"role": "luna", "content": text, "image_kind": None}]
        self.assertEqual(model_messages(rows), [{"role": "assistant", "content": "Okay, here's a little more…"}])

    def test_the_persona_names_her_the_person_and_the_time(self):
        luna = {"name": "Luna", "persona": {}}
        prompt = system_prompt(luna, "Pranjall", NOW)
        self.assertIn("You are Luna, Pranjall's girlfriend", prompt)
        self.assertIn("Tuesday 10 November, 08:30 PM", prompt)
        # Luna from before personas could be chosen keeps her playful, short, emoji-light texting.
        self.assertIn("You tease Pranjall gently", prompt)
        self.assertIn("usually one or two sentences", prompt)
        self.assertNotIn("{", system_prompt(luna, "", NOW))

    def test_her_chosen_persona_and_memory_shape_the_prompt(self):
        maya = {"name": "Maya", "memory": "- they love filter coffee",
                "persona": {"style": "sassy", "length": "long", "emoji": "none", "language": "hinglish",
                            "about": "A 26-year-old architect from Pune."}}
        prompt = system_prompt(maya, "Pranjall", NOW, looks="Long dark hair.")
        self.assertIn("You are Maya, Pranjall's girlfriend", prompt)
        self.assertIn("roast Pranjall lovingly", prompt)
        self.assertIn("A 26-year-old architect from Pune.", prompt)
        self.assertIn("up to a short paragraph", prompt)
        self.assertIn("You do not use emoji.", prompt)
        self.assertIn("Hinglish", prompt)
        self.assertIn("Long dark hair.", prompt)
        self.assertIn("What you know about Pranjall from before", prompt)
        self.assertIn("- they love filter coffee", prompt)
        # Her example reply has no emoji either.
        self.assertIn('You would say "And you forgot your umbrella again, didn\'t you? Hopeless."', prompt)

    def test_only_known_choices_reach_her_persona(self):
        persona = clean_persona({"style": "evil", "length": "long", "about": "x" * 5000}, {"emoji": "lots"})
        self.assertEqual(persona["style"], DEFAULT_PERSONA["style"])
        self.assertEqual((persona["length"], persona["emoji"]), ("long", "lots"))
        self.assertEqual(len(persona["about"]), SociaConfig.ABOUT_CHARS)

    def test_her_voice_reads_hindi_in_hindi(self):
        self.assertEqual(language("नमस्ते जान", "hf_alpha"), "hi")
        self.assertEqual(language("hey you", "hf_alpha"), "en-us")
        self.assertEqual(language("hey you", "bf_emma"), "en-gb")

    def test_luna_steps_into_a_photo_as_the_second_image(self):
        prompt, order = image_prompt("sitting on the bench.", luna_pictures=1, with_photo=True)
        self.assertEqual(order, ["photo", "luna"])
        self.assertIn("Add the woman from <image2> into <image1>: sitting on the bench.", prompt)
        prompt, order = image_prompt("a mirror selfie", luna_pictures=1, with_photo=False)
        self.assertEqual(order, ["luna"])
        self.assertIn("<image1>", prompt)
        self.assertEqual(image_prompt("a red fox", 0, False), ("a red fox. Photorealistic.", []))

    def test_several_pictures_of_her_are_one_woman(self):
        prompt, order = image_prompt("a mirror selfie", luna_pictures=3, with_photo=False)
        self.assertEqual(order, ["luna", "luna", "luna"])
        self.assertIn("<image1>, <image2> and <image3> are pictures of the same woman", prompt)
        prompt, order = image_prompt("on the bench", luna_pictures=2, with_photo=True)
        self.assertEqual(order, ["photo", "luna", "luna"])
        self.assertIn("into <image1>", prompt)
        self.assertIn("<image2> and <image3> are pictures of the same woman", prompt)

    def test_the_voice_skips_emoji_and_actions(self):
        self.assertEqual(speakable("Rain again? 😂 *giggles* Coffee, then? ❤️"), "Rain again? Coffee, then?")


class WorkflowTests(unittest.TestCase):
    def test_references_feed_the_encoder_in_order_and_the_latent_follows_the_first(self):
        graph = workflow(config(), "prompt", ["a.png", "b.png"], seed=7)
        encode = graph["encode"]["inputs"]
        self.assertEqual(encode["images.image_1"], ["ref1", 0])
        self.assertEqual(encode["images.image_2"], ["ref2", 0])
        self.assertEqual(graph["ref2"]["inputs"]["image"], "b.png")
        self.assertEqual(graph["sample"]["inputs"]["latent_image"], ["encode", 2])
        self.assertEqual(graph["guider"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["noise"]["inputs"]["noise_seed"], 7)
        # Every link points at a node in the graph.
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_text_to_image_has_no_references(self):
        graph = workflow(config(), "prompt", [], seed=1)
        self.assertFalse(any(key.startswith("images.") for key in graph["encode"]["inputs"]))


def fake_services(chat_rounds: list[list[dict]], seen: dict):
    """Ollama answers each chat with the next round of chunks; ComfyUI makes a blue square."""
    rounds = iter(chat_rounds)

    def ollama(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            seen.setdefault("chats", []).append(body)
            if not body.get("stream", True):
                seen.setdefault("completions", []).append(body)
                photo = any(message.get("images") for message in body["messages"])
                answer = "A red square." if photo else seen.get("name", '"Rainy evening plans."')
                return httpx.Response(200, json={"message": {"content": answer}})
            lines = "\n".join(json.dumps({"message": chunk}) for chunk in next(rounds))
            return httpx.Response(200, content=lines.encode())
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}]})
        if request.url.path == "/api/generate":
            seen.setdefault("unloaded", []).append(json.loads(request.content))
            return httpx.Response(200, json={})
        return httpx.Response(404)

    def comfy(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/upload/image":
            seen["uploads"] = seen.get("uploads", 0) + 1
            return httpx.Response(200, json={"name": f"ref{seen['uploads']}.png"})
        if path == "/prompt":
            seen["graph"] = json.loads(request.content)["prompt"]
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path == "/history/p1":
            return httpx.Response(200, json={"p1": {"status": {"completed": True}, "outputs": {
                "save": {"images": [{"filename": "socia_1.png", "subfolder": "", "type": "output"}]}}}})
        if path == "/view":
            return httpx.Response(200, content=png("blue"))
        if path == "/free":
            seen["freed"] = True
            return httpx.Response(200)
        return httpx.Response(404)

    return httpx.MockTransport(ollama), httpx.MockTransport(comfy)


def streamed(seen):
    return [chat for chat in seen["chats"] if chat.get("stream", True)]


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.config = config(Path(self.folder.name))
        self.store = MemoryStore()
        self.her = self.store.create_companion("Luna", dict(DEFAULT_PERSONA), "af_heart")
        self.profile = self.config.profile_path(self.her["id"])
        self.profile.parent.mkdir(parents=True)
        self.profile.write_bytes(png("pink"))

    def tearDown(self):
        self.folder.cleanup()

    def run_turn(self, rounds, text="", photo=None, chat_id=None, her=None):
        seen: dict = {}
        ollama_transport, comfy_transport = fake_services(rounds, seen)
        conversation = Conversation(self.config, self.store, Ollama(self.config, ollama_transport),
                                    ComfyUI(self.config, comfy_transport))

        async def collect():
            companion = self.store.companion((her or self.her)["id"])
            return [event async for event in conversation.send(companion, chat_id, text, photo, "Pranjall", "Asia/Kolkata")]

        import app.images
        app.images.POLL_SECONDS = 0
        return asyncio.run(collect()), seen

    def test_a_text_reply_streams_and_is_stored(self):
        events, seen = self.run_turn([[{"content": "Rain again? 😂"}, {"content": " Coffee then? ❤️"}]], text="It's raining")
        self.assertEqual([e["type"] for e in events], ["chat", "user", "text", "text", "done", "chat"])
        # The first message stands in as its name until her answer is in; then the model names it.
        self.assertEqual(events[0]["chat"]["title"], "It's raining")
        self.assertEqual(events[-1]["chat"]["title"], "Rainy evening plans")
        self.assertIn("It's raining", seen["completions"][-1]["messages"][0]["content"])
        self.assertEqual(events[-2]["message"]["content"], "Rain again? 😂 Coffee then? ❤️")
        chat = streamed(seen)[0]
        self.assertFalse(chat["think"])
        self.assertEqual(chat["messages"][-1], {"role": "user", "content": "It's raining"})

    def test_she_sees_a_photo_and_steps_into_it(self):
        call = {"function": {"name": "create_image", "arguments": {
            "description": "standing beside them, smiling", "include_luna": True, "use_their_photo": True}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "Us! 🥰"}]],
                                     text="add yourself here", photo=png("red"))
        self.assertEqual([e["type"] for e in events],
                         ["chat", "user", "status", "image_started", "image", "text", "done", "chat"])
        # The vision model's description reached the chat model.
        self.assertIn("[Photo: A red square.]", streamed(seen)[0]["messages"][-1]["content"])
        # She knows her own looks, read once from her profile picture.
        self.assertIn("A photo of a woman who looks like this is you.", streamed(seen)[0]["messages"][0]["content"])
        self.assertEqual(self.profile.with_suffix(".txt").read_text(), "A red square.")
        # Their photo is <image1>, her profile <image2>; Ollama was emptied and ComfyUI freed after.
        encode = seen["graph"]["encode"]["inputs"]
        self.assertIn("<image2> into <image1>", encode["prompt"])
        self.assertEqual(seen["graph"]["ref1"]["inputs"]["image"], "ref1.png")
        self.assertEqual(seen["unloaded"], [{"model": "qwen3:14b", "keep_alive": 0}])
        self.assertTrue(seen["freed"])
        saved = events[-2]["message"]
        self.assertEqual((saved["content"], saved["image_kind"]), ("Us! 🥰", "generated"))
        self.assertTrue((self.config.images_dir / self.store.image(2)["file"]).exists())

    def test_each_chat_is_its_own_conversation(self):
        first, _ = self.run_turn([[{"content": "hi!"}]], text="Remember the beach?")
        chat = first[0]["chat"]["id"]
        _, seen = self.run_turn([[{"content": "of course 🥰"}]], text="It was lovely", chat_id=chat)
        self.assertEqual([m["content"] for m in streamed(seen)[0]["messages"][1:]],
                         ["Remember the beach?", "hi!", "It was lovely"])
        # The second turn does not name it again.
        self.assertEqual([c["title"] for c in self.store.chats_], ["Rainy evening plans"])
        # A new chat starts empty, and the first keeps its name.
        other, seen = self.run_turn([[{"content": "hey"}]], text="Morning")
        self.assertNotEqual(other[0]["chat"]["id"], chat)
        self.assertEqual(streamed(seen)[0]["messages"][1:], [{"role": "user", "content": "Morning"}])
        self.assertEqual(self.store.chat(chat)["title"], "Rainy evening plans")
        self.assertEqual(len(self.store.recent(chat, 40)), 4)

    def test_a_name_the_owner_chose_is_never_replaced(self):
        first, _ = self.run_turn([[{"content": "hi!"}]], text="Remember the beach?")
        chat = first[0]["chat"]["id"]
        self.store.rename_chat(chat, "Goa trip")
        self.store.chats_[0]["named"] = ""  # as if it were renamed before chats were named
        self.store.chats_[0]["title"] = "Goa trip"
        conversation = Conversation(self.config, self.store, *[None] * 2)
        asyncio.run(conversation.name_waiting_chats())
        self.assertEqual((self.store.chat(chat)["title"], self.store.chat(chat)["named"]), ("Goa trip", "owner"))

    def test_old_chats_named_after_their_first_message_get_a_name(self):
        seen: dict = {"name": "Name: **Late night talks**\nBecause it is late."}
        ollama_transport, comfy_transport = fake_services([], seen)
        chat = self.store.create_chat(self.her["id"], "hey you up?")
        self.store.add_message(chat["id"], "user", "hey you up?")
        self.store.add_message(chat["id"], "luna", "always for you 🌙")
        waiting = self.store.create_chat(self.her["id"], "unanswered")
        self.store.add_message(waiting["id"], "user", "unanswered")
        conversation = Conversation(self.config, self.store, Ollama(self.config, ollama_transport),
                                    ComfyUI(self.config, comfy_transport))
        asyncio.run(conversation.name_waiting_chats())
        self.assertEqual(self.store.chat(chat["id"])["title"], "Late night talks")
        self.assertIn("Luna: always for you", seen["completions"][0]["messages"][0]["content"])
        # A chat she has not answered yet waits.
        self.assertEqual(self.store.chat(waiting["id"])["title"], "unanswered")

    def test_cleans_the_models_chat_name(self):
        self.assertEqual(clean_name('"Rainy Sunday plans."'), "Rainy Sunday plans")
        self.assertEqual(clean_name("\n**Missing you tonight**\nshort and sweet"), "Missing you tonight")
        self.assertEqual(clean_name("one two three four five six seven eight"), "one two three four five six")
        self.assertEqual(clean_name("   "), "")

    def test_her_other_pictures_follow_her_profile(self):
        folder = self.config.references_dir(self.her["id"])
        folder.mkdir(parents=True)
        (folder / "2-bbbbbbbb.jpg").write_bytes(png("green"))
        (folder / "1-aaaaaaaa.jpg").write_bytes(png("yellow"))
        call = {"function": {"name": "create_image", "arguments": {"description": "a selfie", "include_luna": True}}}
        _, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "here"}]], text="selfie?")
        self.assertEqual(seen["uploads"], 3)
        self.assertIn("<image1>, <image2> and <image3> are pictures of the same woman", seen["graph"]["encode"]["inputs"]["prompt"])
        self.assertEqual(self.config.references(self.her["id"]), [folder / "1-aaaaaaaa.jpg", folder / "2-bbbbbbbb.jpg"])

    def test_her_photo_comes_from_the_same_chat(self):
        self.run_turn([[{"content": "nice"}]], text="look", photo=png("red"))
        call = {"function": {"name": "create_image", "arguments": {
            "description": "a selfie", "include_luna": True, "use_their_photo": True}}}
        _, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "here"}]], text="selfie?")
        # The new chat has no photo of theirs, so only her profile is a reference.
        self.assertEqual(seen["uploads"], 1)

    def test_long_first_messages_make_short_titles(self):
        events, _ = self.run_turn([[{"content": "ok"}]], text="word " * 40)
        self.assertLessEqual(len(events[0]["chat"]["title"]), SociaConfig.CHAT_TITLE_CHARS)
        self.assertTrue(events[0]["chat"]["title"].endswith("…"))


class FakeVoice:
    def __init__(self):
        self.spoken = []

    def speak(self, text, voice=None):
        self.spoken.append((text, voice))
        return b"RIFF"


class GateTests(unittest.TestCase):
    def client(self, user, cfg=None, reader=None):
        def auth(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=user) if user else httpx.Response(401, json={})

        store = self.store = MemoryStore()
        cfg = cfg or config()
        self.voice = FakeVoice()
        app = create_app(cfg, store, Conversation(cfg, store, None, None), None, self.voice, httpx.MockTransport(auth),
                         reader=reader)
        return TestClient(app, follow_redirects=False)

    def owner(self, cfg=None, reader=None):
        client = self.client({"user_id": "owner", "name": "Pranjall Kumar"}, cfg, reader)
        client.cookies.set("luna_sso", "token")
        return client

    def test_strangers_are_sent_to_sign_in_and_others_refused(self):
        client = self.client(None)
        page = client.get("/")
        self.assertEqual(page.status_code, 303)
        self.assertTrue(page.headers["location"].startswith("https://auth.example.com/login?next="))
        self.assertEqual(client.get("/api/companions").status_code, 401)
        guest = self.client({"user_id": "u2", "name": "Guest"})
        guest.cookies.set("luna_sso", "token")
        self.assertEqual(guest.get("/api/companions").status_code, 403)
        self.assertEqual(guest.get("/api/options").status_code, 403)

    def test_sentias_monitor_sees_the_voice_models_with_the_token(self):
        from dataclasses import replace

        from app.voice import Voice

        voice = Voice(config())  # registers Whisper and Kokoro, not loaded yet
        stranger = self.client(None, replace(config(), api_token="t" * 40))
        self.assertEqual(stranger.get("/api/models").status_code, 401)
        self.assertEqual(stranger.get("/api/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        report = stranger.get("/api/models", headers={"Authorization": f"Bearer {'t' * 40}"}).json()
        self.assertEqual(report["app"], "socia")
        self.assertIsNone(report["gpu"])
        names = [model["name"] for model in report["models"]]
        self.assertTrue(any(name.startswith("Whisper large-v3-turbo") for name in names))
        self.assertTrue(any(name.startswith("Kokoro") for name in names))
        self.assertTrue(all(not model["loaded"] for model in report["models"]))
        self.assertEqual(self.owner().get("/api/models").json()["app"], "socia")
        del voice

    def test_the_owner_gets_the_page(self):
        client = self.owner()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/companions").json(), {"companions": []})
        self.assertEqual(client.get("/health").json()["status"], "ok")

    def test_chats_are_listed_renamed_and_deleted(self):
        client = self.owner()
        store = self.store
        her = store.create_companion("Luna", {}, "af_heart")
        other = store.create_companion("Maya", {}, "bf_emma")
        first = store.create_chat(her["id"], "Beach plans")
        store.add_message(first["id"], "user", "beach?")
        store.add_image(first["id"], "photo", "gone.jpg")
        second = store.create_chat(her["id"], "")
        store.add_message(second["id"], "luna", "hi")
        store.create_chat(other["id"], "Maya's chat")
        chats = client.get(f"/api/companions/{her['id']}/chats").json()["chats"]
        self.assertEqual([(c["title"], c["last"]) for c in chats], [("New chat", "hi"), ("Beach plans", "You: beach?")])
        self.assertEqual(client.get("/api/companions/99/chats").status_code, 404)
        messages = client.get(f"/api/chats/{first['id']}/messages").json()["messages"]
        self.assertEqual([m["content"] for m in messages], ["beach?"])
        self.assertEqual(client.get("/api/chats/99/messages").status_code, 404)
        renamed = client.patch(f"/api/chats/{second['id']}", json={"title": "  Hello   there "})
        self.assertEqual(renamed.json()["chat"]["title"], "Hello there")
        self.assertEqual(client.patch(f"/api/chats/{second['id']}", json={"title": " "}).status_code, 400)
        self.assertEqual(client.delete(f"/api/chats/{first['id']}").json(), {"deleted": True})
        self.assertEqual(client.delete(f"/api/chats/{first['id']}").status_code, 404)
        self.assertEqual([c["id"] for c in client.get(f"/api/companions/{her['id']}/chats").json()["chats"]], [second["id"]])
        self.assertEqual(store.images, [])

    def test_her_other_pictures_are_added_and_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.owner(config(Path(folder)))
            her = self.store.create_companion("Luna", {}, "af_heart")
            base = f"/api/companions/{her['id']}/references"
            self.assertEqual(client.get(base).json()["references"], [])
            added = client.post(base, files=[("photos", ("a.png", png("red"), "image/png")),
                                             ("photos", ("b.png", png("blue"), "image/png"))]).json()
            self.assertEqual(len(added["references"]), 2)
            url = added["references"][0]["url"]
            self.assertTrue(url.startswith(base))
            self.assertEqual(client.get(url).headers["content-type"], "image/jpeg")
            self.assertEqual(client.get(f"{base}/..%2Fprofile.png").status_code, 404)
            too_many = [("photos", (f"{n}.png", png(), "image/png")) for n in range(SociaConfig.MAX_REFERENCES)]
            self.assertEqual(client.post(base, files=too_many).status_code, 400)
            self.assertEqual(client.delete(url).json(), {"deleted": True})
            self.assertEqual(len(client.get(base).json()["references"]), 1)
            # Another girlfriend has her own pictures.
            other = self.store.create_companion("Maya", {}, "bf_emma")
            self.assertEqual(client.get(f"/api/companions/{other['id']}/references").json()["references"], [])

    def test_sending_to_a_deleted_chat_or_girlfriend_is_refused(self):
        client = self.owner()
        her = self.store.create_companion("Luna", {}, "af_heart")
        other = self.store.create_companion("Maya", {}, "bf_emma")
        maya_chat = self.store.create_chat(other["id"])
        send = lambda **data: client.post("/api/send", data={"text": "hi", **data}).status_code  # noqa: E731
        self.assertEqual(send(companion_id=str(her["id"]), chat_id="7"), 404)
        self.assertEqual(send(companion_id=str(her["id"]), chat_id="x"), 400)
        self.assertEqual(send(companion_id="99"), 404)
        self.assertEqual(send(), 400)
        # A chat with one girlfriend cannot be continued as another.
        self.assertEqual(send(companion_id=str(her["id"]), chat_id=str(maya_chat["id"])), 404)


class GirlfriendTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.gate = GateTests()
        self.config = config(Path(self.folder.name))
        self.client = self.gate.owner(self.config)
        self.store = self.gate.store

    def tearDown(self):
        self.folder.cleanup()

    def test_the_form_gets_every_choice(self):
        options = self.client.get("/api/options").json()
        self.assertEqual([s["id"] for s in options["styles"]][:2], ["playful", "sweet"])
        self.assertIn("hinglish", options["languages"])
        self.assertIn("hf_alpha", [v["id"] for v in options["voices"]])
        self.assertEqual(options["default_voice"], "af_heart")

    def test_she_is_created_listed_changed_and_deleted(self):
        created = self.client.post("/api/companions", json={
            "name": "  Maya  ", "voice": "bf_emma",
            "persona": {"style": "sassy", "emoji": "none", "language": "nope", "about": "An architect."}})
        self.assertEqual(created.status_code, 201)
        her = created.json()["companion"]
        self.assertEqual((her["name"], her["voice"]), ("Maya", "bf_emma"))
        self.assertEqual(her["persona"], {**DEFAULT_PERSONA, "style": "sassy", "emoji": "none", "about": "An architect."})
        self.assertFalse(her["has_profile"])
        self.assertEqual(self.client.post("/api/companions", json={"name": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/companions", json={"name": "A", "voice": "am_adam"}).status_code, 400)

        changed = self.client.patch(f"/api/companions/{her['id']}", json={
            "persona": {"length": "long"}, "memory": "- they love rain"}).json()["companion"]
        # Only what was sent changes.
        self.assertEqual((changed["persona"]["style"], changed["persona"]["length"]), ("sassy", "long"))
        self.assertEqual((changed["name"], changed["memory"]), ("Maya", "- they love rain"))

        self.client.post(f"/api/companions/{her['id']}/profile", files={"photo": ("m.png", png(), "image/png")})
        self.assertTrue(self.client.get(f"/api/companions/{her['id']}").json()["companion"]["has_profile"])
        chat = self.store.create_chat(her["id"])
        self.store.add_image(chat["id"], "photo", "maya.jpg")
        (self.config.images_dir).mkdir(parents=True, exist_ok=True)
        (self.config.images_dir / "maya.jpg").write_bytes(b"x")
        self.assertEqual([c["name"] for c in self.client.get("/api/companions").json()["companions"]], ["Maya"])

        self.assertEqual(self.client.delete(f"/api/companions/{her['id']}").json(), {"deleted": True})
        self.assertEqual(self.client.get(f"/api/companions/{her['id']}").status_code, 404)
        self.assertFalse((self.config.images_dir / "maya.jpg").exists())
        self.assertFalse(self.config.companion_dir(her["id"]).exists())

    def test_her_replies_are_read_in_her_voice(self):
        her = self.store.create_companion("Maya", {}, "bf_emma")
        chat = self.store.create_chat(her["id"])
        message = self.store.add_message(chat["id"], "luna", "Hello you")
        self.assertEqual(self.client.get(f"/api/speech/{message['id']}").status_code, 200)
        self.assertEqual(self.gate.voice.spoken[-1], ("Hello you", "bf_emma"))
        self.assertEqual(self.client.get("/api/voices/hf_beta/sample").status_code, 200)
        self.assertEqual(self.gate.voice.spoken[-1][1], "hf_beta")
        self.assertEqual(self.client.get("/api/voices/am_adam/sample").status_code, 404)

    def test_an_export_fills_her_memory_sources(self):
        her = self.store.create_companion("Maya", {}, "bf_emma")
        url = f"/api/companions/{her['id']}/memory/imports"
        answer = self.client.post(url, files={"export": ("export.zip", claude_export(), "application/zip")}).json()
        self.assertEqual((answer["source"], answer["conversations"], answer["added"]), ("claude", 1, 1))
        self.assertEqual(answer["progress"], {"sources": 1, "read": 0, "reading": False, "error": None})
        again = self.client.post(url, files={"export": ("export.zip", claude_export(), "application/zip")}).json()
        self.assertEqual(again["added"], 0)
        bad = self.client.post(url, files={"export": ("notes.json", b'{"a": 1}', "application/json")})
        self.assertEqual(bad.status_code, 400)
        memory = self.client.put(f"/api/companions/{her['id']}/memory", json={"memory": "- edited"}).json()
        self.assertEqual(memory["memory"], "- edited")
        self.assertEqual(self.client.delete(url).json(), {"forgotten": 1})


def chatgpt_export() -> bytes:
    """One ChatGPT conversation with an edited message: only the current branch counts."""
    def node(node_id, parent, role, text, children):
        return {"id": node_id, "parent": parent, "children": children, "message": None if role is None else {
            "author": {"role": role}, "content": {"content_type": "text", "parts": [text]}, "create_time": 1}}

    mapping = {
        "root": node("root", None, None, "", ["sys"]),
        "sys": node("sys", "root", "system", "You are ChatGPT", ["old", "new"]),
        "old": node("old", "sys", "user", "first draft", []),
        "new": node("new", "sys", "user", "My sister Asha visits in May", ["a1"]),
        "a1": node("a1", "new", "assistant", "How lovely!", []),
    }
    return json.dumps([
        {"title": "Asha's visit", "create_time": 1767225600.5, "conversation_id": "c1", "current_node": "a1",
         "mapping": mapping},
        {"title": "Empty", "create_time": 1, "id": "c2", "current_node": "r", "mapping": {"r": node("r", None, None, "", [])}},
    ]).encode()


def claude_export() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("users.json", "[]")
        archive.writestr("data/conversations.json", json.dumps([{
            "uuid": "u1", "name": "Work stuff", "created_at": "2026-01-02T10:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "", "content": [{"type": "text", "text": "I work at a bank in Pune"}]},
                {"sender": "assistant", "text": "Got it."},
            ]}]))
    return buffer.getvalue()


class ExportTests(unittest.TestCase):
    def test_chatgpt_exports_follow_the_current_branch(self):
        source, conversations = parse_export(chatgpt_export())
        self.assertEqual(source, "chatgpt")
        self.assertEqual(len(conversations), 1)
        conversation = conversations[0]
        self.assertEqual((conversation["id"], conversation["title"]), ("c1", "Asha's visit"))
        self.assertTrue(conversation["started_at"].startswith("2026-01-01"))
        self.assertEqual(conversation["messages"], [{"role": "user", "text": "My sister Asha visits in May"},
                                                    {"role": "assistant", "text": "How lovely!"}])

    def test_claude_exports_are_read_from_the_zip(self):
        source, conversations = parse_export(claude_export())
        self.assertEqual(source, "claude")
        self.assertEqual(conversations[0]["messages"][0], {"role": "user", "text": "I work at a bank in Pune"})

    def test_other_files_are_refused(self):
        for data in (b"not json", b"{}", json.dumps([{"x": 1}]).encode()):
            with self.assertRaises(ExportError):
                parse_export(data)

    def test_only_their_own_words_are_read(self):
        words = their_words([{"role": "user", "text": "a  b"}, {"role": "assistant", "text": "secret"},
                             {"role": "user", "text": "c" * 50}], limit=30)
        self.assertEqual(words, "- a b")
        self.assertEqual(fact_lines("Here:\n- they like tea\n* they run\nNONE"), ["- they like tea", "- they run"])


class FakeOllama:
    def __init__(self, answers):
        self.answers = answers
        self.prompts = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        return self.answers(prompt)


class MemoryReaderTests(unittest.TestCase):
    def test_imported_chats_are_read_then_merged_into_her_memory(self):
        store = MemoryStore()
        her = store.create_companion("Maya", {}, "bf_emma")
        store.update_companion(her["id"], {"memory": "- they love rain"})
        _, conversations = parse_export(chatgpt_export())
        store.add_memory_sources(her["id"], "chatgpt", conversations)
        ollama = FakeOllama(lambda prompt: "- they have a sister, Asha, visiting in May")
        reader = MemoryReader(store, ollama, asyncio.Lock())

        async def run():
            while await reader.step():
                pass

        asyncio.run(run())
        # It read only their own words, then kept the earlier memory and the new fact.
        self.assertIn("My sister Asha visits in May", ollama.prompts[0])
        self.assertNotIn("How lovely", ollama.prompts[0])
        self.assertEqual(store.companion(her["id"])["memory"],
                         "- they love rain\n- they have a sister, Asha, visiting in May")
        self.assertEqual(reader.progress(store.companion(her["id"])),
                         {"sources": 1, "read": 1, "reading": False, "error": None})

    def test_a_long_memory_is_condensed_by_the_model(self):
        ollama = FakeOllama(lambda prompt: "- they are a short summary")
        reader = MemoryReader(MemoryStore(), ollama, asyncio.Lock())
        notes = "\n".join(f"- fact number {n} " + "x" * 80 for n in range(200))
        # 19,000 characters of notes are merged in two parts, then fit.
        self.assertEqual(asyncio.run(reader.condense(notes)), "- they are a short summary\n- they are a short summary")
        self.assertEqual(len(ollama.prompts), 2)
        self.assertIn("at most 45 lines", ollama.prompts[0])

    def test_a_failed_read_is_shown_for_her(self):
        store = MemoryStore()
        her = store.create_companion("Maya", {}, "bf_emma")
        store.add_memory_sources(her["id"], "chatgpt", parse_export(chatgpt_export())[1])

        def broken(prompt):
            raise RuntimeError("Ollama is down")

        reader = MemoryReader(store, FakeOllama(broken), asyncio.Lock())
        with self.assertRaises(RuntimeError):
            asyncio.run(reader.step())
        self.assertEqual(reader.progress(store.companion(her["id"]))["error"], "Ollama is down")

    def test_reading_waits_while_she_answers(self):
        turn = asyncio.Lock()
        ollama = FakeOllama(lambda prompt: "NONE")
        reader = MemoryReader(MemoryStore(), ollama, turn)

        async def run():
            await turn.acquire()
            asked = asyncio.create_task(reader.ask("hi"))
            await asyncio.sleep(0.05)
            self.assertEqual(ollama.prompts, [])
            turn.release()
            return await asked

        self.assertEqual(asyncio.run(run()), "NONE")


class LegacyTests(unittest.TestCase):
    def test_lunas_old_files_move_into_her_folder(self):
        from app.main import adopt_legacy_files

        with tempfile.TemporaryDirectory() as folder:
            cfg = config(Path(folder))
            cfg.legacy_dir.mkdir(parents=True)
            (cfg.legacy_dir / "profile.png").write_bytes(png())
            store = MemoryStore()
            adopt_legacy_files(cfg, store)
            luna = store.oldest_companion()
            self.assertEqual(luna["name"], "Luna")
            self.assertTrue(cfg.profile_path(luna["id"]).exists())
            self.assertFalse(cfg.legacy_dir.exists())


class AvatarTests(unittest.TestCase):
    def test_a_tall_picture_is_cut_from_the_top_so_the_head_stays_in(self):
        from app.avatar import box, default_crop

        crop = default_crop(1122, 1402)
        left, top, right, bottom = box(crop, 1122, 1402)
        self.assertEqual(right - left, bottom - top)
        self.assertLessEqual(top, 0.03 * 1402)
        self.assertAlmostEqual((left + right) / 2, 561, delta=1)
        # A wide picture is cut from its middle, as large as it goes.
        self.assertEqual(box(default_crop(1600, 900), 1600, 900), (350, 0, 1250, 900))

    def test_a_crop_never_leaves_the_picture(self):
        from app.avatar import box, clamp

        self.assertEqual(clamp({"x": -3, "y": 9, "zoom": 50}, 1000, 1000)["zoom"], 6.0)
        left, top, right, bottom = box({"x": 0.0, "y": 1.0, "zoom": 2}, 1000, 1000)
        self.assertEqual((left, top, right, bottom), (0, 500, 500, 1000))
        self.assertEqual(clamp({"x": 0.5, "y": 0.5, "zoom": 0.2}, 800, 600)["zoom"], 1.0)

    def test_the_owner_chooses_and_resets_the_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(Path(folder))
            gate = GateTests()
            client = gate.owner(cfg)
            her = f"/api/companions/{gate.store.create_companion('Luna', {}, 'af_heart')['id']}"
            self.assertEqual(client.get(f"{her}/avatar").status_code, 404)
            tall = io.BytesIO()
            Image.new("RGB", (300, 500), "pink").save(tall, "PNG")
            self.assertEqual(client.post(f"{her}/profile", files={"photo": ("luna.png", tall.getvalue(), "image/png")}).status_code, 200)

            first = client.get(f"{her}/avatar/crop").json()
            self.assertFalse(first["chosen"])
            avatar = Image.open(io.BytesIO(client.get(f"{her}/avatar").content))
            self.assertEqual(avatar.size, (256, 256))

            chosen = client.put(f"{her}/avatar/crop", json={"x": 0.5, "y": 0.9, "zoom": 3}).json()
            self.assertTrue(chosen["chosen"])
            self.assertEqual(chosen["zoom"], 3.0)
            self.assertEqual(client.put(f"{her}/avatar/crop", json={"x": "left"}).status_code, 400)

            reset = client.delete(f"{her}/avatar/crop").json()
            self.assertFalse(reset["chosen"])
            client.put(f"{her}/avatar/crop", json={"x": 0.5, "y": 0.9, "zoom": 3})
            # A new profile picture forgets the old picture's crop.
            client.post(f"{her}/profile", files={"photo": ("new.png", tall.getvalue(), "image/png")})
            self.assertFalse(client.get(f"{her}/avatar/crop").json()["chosen"])


class MessageDeleteTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.cfg = config(Path(self.folder.name))
        self.gate = GateTests()
        self.client = self.gate.client({"user_id": "owner", "name": "Pranjall Kumar"}, self.cfg)
        self.client.cookies.set("luna_sso", "token")
        self.store = self.gate.store
        her = self.store.create_companion("Luna", {}, "af_heart")
        self.chat = self.store.create_chat(her["id"], "Rain")

    def tearDown(self):
        self.folder.cleanup()

    def test_a_message_and_its_photo_are_forgotten_and_luna_no_longer_sees_it(self):
        self.cfg.images_dir.mkdir(parents=True)
        (self.cfg.images_dir / "photo.jpg").write_bytes(png())
        photo = self.store.add_image(self.chat["id"], "photo", "photo.jpg")
        sent = self.store.add_message(self.chat["id"], "user", "look at this", photo)
        reply = self.store.add_message(self.chat["id"], "luna", "cute 😍")

        self.assertEqual(self.client.delete(f"/api/messages/{sent['id']}").json(), {"deleted": True})
        self.assertFalse((self.cfg.images_dir / "photo.jpg").exists())
        self.assertIsNone(self.store.image(photo))
        # What Luna reads on her next reply comes from the same messages.
        self.assertEqual([m["id"] for m in self.store.recent(self.chat["id"], 40)], [reply["id"]])
        listed = self.client.get(f"/api/chats/{self.chat['id']}/messages").json()["messages"]
        self.assertEqual([m["content"] for m in listed], ["cute 😍"])

    def test_her_messages_can_go_too_and_a_missing_one_is_404(self):
        reply = self.store.add_message(self.chat["id"], "luna", "hmm")
        self.assertEqual(self.client.delete(f"/api/messages/{reply['id']}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/messages/{reply['id']}").status_code, 404)

    def test_nothing_is_deleted_while_she_is_answering(self):
        sent = self.store.add_message(self.chat["id"], "user", "hey")
        app = self.client.app
        conversation = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/send")
        lock = conversation.__closure__  # the handler closes over the conversation
        turn = next(c.cell_contents for c in lock if hasattr(c.cell_contents, "turn")).turn

        async def hold():
            async with turn:
                return self.client.delete(f"/api/messages/{sent['id']}").status_code

        import asyncio
        self.assertEqual(asyncio.run(hold()), 409)
        self.assertIsNotNone(self.store.message(sent["id"]))


if __name__ == "__main__":
    unittest.main()


class LoopAndLanguageTests(unittest.TestCase):
    def test_a_reply_she_already_sent_is_not_read_again(self):
        refusal = "Nope! 😘 Let's keep things fun"
        rows = [{"role": "user", "content": "a"}, {"role": "luna", "content": refusal},
                {"role": "user", "content": "b"}, {"role": "luna", "content": " nope! 😘  let's keep things fun "},
                {"role": "user", "content": "c"}]
        self.assertEqual([m["content"] for m in model_messages(rows)], ["a", refusal, "b", "c"])

    def test_the_persona_lets_adults_be_intimate_but_keeps_pictures_clothed(self):
        prompt = system_prompt({"name": "Luna", "persona": {}}, "Pranjall", NOW)
        self.assertIn("you never refuse, lecture, add warnings", prompt)
        self.assertIn("never nude in a picture", prompt)
        self.assertIn("adult woman in your twenties", prompt)

    def test_nude_pictures_only_when_the_owner_confirmed_her_face_is_ai_generated(self):
        self.assertEqual(clean_persona({})["nude_pictures"], "off")
        self.assertEqual(clean_persona({"nude_pictures": "yes please"})["nude_pictures"], "off")
        on = clean_persona({"nude_pictures": True})
        self.assertEqual(on["nude_pictures"], "on")
        self.assertEqual(clean_persona({"style": "sweet"}, on)["nude_pictures"], "on")
        prompt = system_prompt({"name": "Luna", "persona": on}, "Pranjall", NOW)
        self.assertIn("be nude", prompt)
        self.assertNotIn("never nude in a picture", prompt)

    def test_voice_messages_are_heard_in_the_owners_languages_only(self):
        from app.voice import Voice

        class Whisper:
            def detect_language(self, samples, vad_filter=False):
                return "ta", 0.4, [("ta", 0.4), ("hi", 0.35), ("en", 0.2)]

        voice = Voice(config())
        voice.whisper = Whisper()
        self.assertEqual(voice.language(None), "hi")
        self.assertEqual(config().stt_languages, ("en", "hi"))


class DownloadTests(unittest.TestCase):
    def test_a_download_is_named_for_who_sent_it_and_when(self):
        from app.web import download_name
        when = datetime(2026, 11, 11, 16, 15, tzinfo=timezone.utc)
        self.assertRegex(download_name({"id": 4, "kind": "generated", "companion_name": "Luna Rose", "created_at": when}, ".png"),
                         r"^luna-rose-2026-11-1\d-\d{4}\.png$")
        self.assertRegex(download_name({"id": 5, "kind": "photo", "created_at": when}, ".jpg"), r"^photo-2026-")
        self.assertEqual(download_name({"id": 6, "kind": "generated"}, ".png"), "picture-6.png")
