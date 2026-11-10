"""Tests for Luna Socia: the gate, what Luna reads, the image workflow, and one full turn."""

import asyncio
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from starlette.testclient import TestClient

from app.chat import Conversation, model_messages
from app.config import SociaConfig
from app.images import ComfyUI, workflow
from app.ollama import Ollama
from app.persona import image_prompt, system_prompt
from app.voice import speakable
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
        self.images: list[dict] = []
        self.messages: list[dict] = []

    def add_image(self, kind, file, prompt=None):
        self.images.append({"id": len(self.images) + 1, "kind": kind, "file": file, "prompt": prompt})
        return len(self.images)

    def image(self, image_id):
        return next((i for i in self.images if i["id"] == image_id), None)

    def add_message(self, role, content, image_id=None, photo_note=None):
        image = self.image(image_id) if image_id else None
        row = {"id": len(self.messages) + 1, "role": role, "content": content, "image_id": image_id,
               "photo_note": photo_note, "created_at": NOW, "image_kind": image and image["kind"],
               "image_prompt": image and image["prompt"]}
        self.messages.append(row)
        return row

    def set_photo_note(self, message_id, note):
        self.messages[message_id - 1]["photo_note"] = note

    def recent(self, limit, before=None):
        return self.messages[-limit:]

    def message(self, message_id):
        return self.messages[message_id - 1]

    def last_photo(self):
        return next((i for i in reversed(self.images) if i["kind"] == "photo"), None)

    def clear(self):
        files = [i["file"] for i in self.images]
        self.images, self.messages = [], []
        return files


class PromptTests(unittest.TestCase):
    def test_luna_reads_photos_and_her_own_pictures_as_notes(self):
        rows = [
            {"role": "user", "content": "look", "photo_note": "A beach at sunset.", "image_kind": "photo"},
            {"role": "luna", "content": "wow 😍", "image_kind": "generated", "image_prompt": "Luna on the beach"},
            {"role": "luna", "content": "hi", "image_kind": None},
        ]
        self.assertEqual(model_messages(rows), [
            {"role": "user", "content": "look\n[Photo: A beach at sunset.]"},
            {"role": "assistant", "content": "wow 😍\n[You sent a picture: Luna on the beach]"},
            {"role": "assistant", "content": "hi"},
        ])

    def test_the_persona_names_the_person_and_the_time(self):
        prompt = system_prompt("Pranjall", NOW)
        self.assertIn("You are Luna, Pranjall's girlfriend", prompt)
        self.assertIn("Tuesday 10 November, 08:30 PM", prompt)
        self.assertNotIn("{name}", system_prompt("", NOW))

    def test_luna_steps_into_a_photo_as_the_second_image(self):
        prompt, order = image_prompt("sitting on the bench.", with_luna=True, with_photo=True)
        self.assertEqual(order, ["photo", "luna"])
        self.assertIn("Add the woman from <image2> into <image1>: sitting on the bench.", prompt)
        prompt, order = image_prompt("a mirror selfie", with_luna=True, with_photo=False)
        self.assertEqual(order, ["luna"])
        self.assertIn("<image1>", prompt)
        self.assertEqual(image_prompt("a red fox", False, False), ("a red fox. Photorealistic.", []))

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
                return httpx.Response(200, json={"message": {"content": "A red square."}})
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
        self.config.profile_path.parent.mkdir(parents=True)
        self.config.profile_path.write_bytes(png("pink"))
        self.store = MemoryStore()

    def tearDown(self):
        self.folder.cleanup()

    def run_turn(self, rounds, text="", photo=None):
        seen: dict = {}
        ollama_transport, comfy_transport = fake_services(rounds, seen)
        conversation = Conversation(self.config, self.store, Ollama(self.config, ollama_transport),
                                    ComfyUI(self.config, comfy_transport))

        async def collect():
            return [event async for event in conversation.send(text, photo, "Pranjall", "Asia/Kolkata")]

        import app.images
        app.images.POLL_SECONDS = 0
        return asyncio.run(collect()), seen

    def test_a_text_reply_streams_and_is_stored(self):
        events, seen = self.run_turn([[{"content": "Rain again? 😂"}, {"content": " Coffee then? ❤️"}]], text="It's raining")
        self.assertEqual([e["type"] for e in events], ["user", "text", "text", "done"])
        self.assertEqual(events[-1]["message"]["content"], "Rain again? 😂 Coffee then? ❤️")
        chat = streamed(seen)[0]
        self.assertFalse(chat["think"])
        self.assertEqual(chat["messages"][-1], {"role": "user", "content": "It's raining"})

    def test_she_sees_a_photo_and_steps_into_it(self):
        call = {"function": {"name": "create_image", "arguments": {
            "description": "standing beside them, smiling", "include_luna": True, "use_their_photo": True}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "Us! 🥰"}]],
                                     text="add yourself here", photo=png("red"))
        self.assertEqual([e["type"] for e in events],
                         ["user", "status", "image_started", "image", "text", "done"])
        # The vision model's description reached the chat model.
        self.assertIn("[Photo: A red square.]", streamed(seen)[0]["messages"][-1]["content"])
        # She knows her own looks, read once from her profile picture.
        self.assertIn("A photo of a woman who looks like this is you.", streamed(seen)[0]["messages"][0]["content"])
        self.assertEqual(self.config.profile_path.with_suffix(".txt").read_text(), "A red square.")
        # Their photo is <image1>, her profile <image2>; Ollama was emptied and ComfyUI freed after.
        encode = seen["graph"]["encode"]["inputs"]
        self.assertIn("<image2> into <image1>", encode["prompt"])
        self.assertEqual(seen["graph"]["ref1"]["inputs"]["image"], "ref1.png")
        self.assertEqual(seen["unloaded"], [{"model": "qwen3:14b", "keep_alive": 0}])
        self.assertTrue(seen["freed"])
        saved = events[-1]["message"]
        self.assertEqual((saved["content"], saved["image_kind"]), ("Us! 🥰", "generated"))
        self.assertTrue((self.config.images_dir / self.store.image(2)["file"]).exists())


class GateTests(unittest.TestCase):
    def client(self, user):
        def auth(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=user) if user else httpx.Response(401, json={})

        store = MemoryStore()
        cfg = config()
        app = create_app(cfg, store, Conversation(cfg, store, None, None), None, None, httpx.MockTransport(auth))
        return TestClient(app, follow_redirects=False)

    def test_strangers_are_sent_to_sign_in_and_others_refused(self):
        client = self.client(None)
        page = client.get("/")
        self.assertEqual(page.status_code, 303)
        self.assertTrue(page.headers["location"].startswith("https://auth.example.com/login?next="))
        self.assertEqual(client.get("/api/messages").status_code, 401)
        guest = self.client({"user_id": "u2", "name": "Guest"})
        guest.cookies.set("luna_sso", "token")
        self.assertEqual(guest.get("/api/messages").status_code, 403)

    def test_the_owner_gets_the_page(self):
        client = self.client({"user_id": "owner", "name": "Pranjall Kumar"})
        client.cookies.set("luna_sso", "token")
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/messages").json(), {"messages": []})
        self.assertEqual(client.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
