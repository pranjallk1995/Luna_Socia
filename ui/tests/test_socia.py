"""Tests for Luna Socia: the gate, girlfriends and their memory, what she reads, her pictures, and full turns."""

import asyncio
import io
import json
import tempfile
import unittest
from unittest import mock
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from starlette.testclient import TestClient

from app.chat import (ASKS_FOR_PICTURE, asks_for_picture, PICTURE_PROMISE, written_calls, written_reactions, written_remembers, clean_name, Conversation, model_messages, with_secret_nudge, without_written_pictures,
                      written_notes, written_picture)
from app.memory import ExportError, MemoryReader, fact_lines, facts_about_her, facts_about_them, parse_export, their_words
from app.config import SociaConfig
from app.store import without_fact
from app.images import ImageError, ImageService
from app.ollama import Ollama
from app.persona import DEFAULT_PERSONA, clean_persona, image_prompt, system_prompt
from app.voice import language, speakable
from app.web import create_app

NOW = datetime(2026, 11, 10, 20, 30, tzinfo=timezone.utc)


def config(data_dir: Path = Path("/tmp/socia-test")) -> SociaConfig:
    return SociaConfig(database_url="postgresql://unused", auth_internal_url="http://auth",
                       auth_public_url="https://auth.example.com", public_url="https://socia.example.com",
                       ollama_url="http://ollama", imagegen_url="http://imagegen", data_dir=data_dir)


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
        self.apps: dict | None = None
        self.owner = {"name": "", "about": ""}
        self.picture_references: list[dict] = []

    def owner_profile(self):
        return dict(self.owner)

    def set_owner_profile(self, name, about):
        self.owner = {"name": name, "about": about}
        return dict(self.owner)

    def apps_connection(self):
        return dict(self.apps) if self.apps else None

    def set_apps_client(self, client_id, client_secret, redirect_uri):
        self.apps = {"client_id": client_id, "client_secret": client_secret, "redirect_uri": redirect_uri,
                     "access_token": None, "refresh_token": None, "access_expires": None}
        return dict(self.apps)

    def set_apps_tokens(self, access, refresh, expires):
        self.apps.update(access_token=access, refresh_token=refresh, access_expires=expires)

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

    def quiet_chats(self, quiet_seconds):
        return [{"id": c["id"], "companion_id": c["companion_id"], "title": c["title"],
                 "remembered_to": c.get("remembered_to", 0)} for c in self.chats_
                if any(m["chat_id"] == c["id"] and m["role"] == "user" and m["id"] > c.get("remembered_to", 0)
                       for m in self.messages)]

    def chat_source(self, chat, limit):
        rows = [m for m in self.messages if m["chat_id"] == chat["id"] and m["id"] > chat["remembered_to"]][:limit]
        if not rows:
            return 0
        messages = [{"role": m["role"], "text": " ".join(p for p in [m["content"] or "",
                     f"[with a photo: {m['photo_note']}]" if m["photo_note"] else ""] if p)}
                    for m in rows if m["content"] or m["photo_note"]]
        if messages:
            self.sources.append({"id": len(self.sources) + 1, "companion_id": chat["companion_id"], "source": "socia",
                                 "external_id": f"chat-{chat['id']}-{rows[-1]['id']}", "title": chat["title"],
                                 "messages": messages, "started_at": None, "facts": None, "merged": False})
        next(c for c in self.chats_ if c["id"] == chat["id"])["remembered_to"] = rows[-1]["id"]
        return len(messages)

    def remember(self, companion_id, fact, limit):
        her = next(c for c in self.companions_ if c["id"] == companion_id)
        line = "- " + fact.removeprefix("- ").strip()
        lines = [l for l in her["memory"].splitlines() if l.strip()]
        if line.casefold() in (l.casefold() for l in lines):
            return False
        her["memory"] = "\n".join([*lines, line])
        return True

    def add_to_bio(self, companion_id, facts, limit):
        her = next(c for c in self.companions_ if c["id"] == companion_id)
        about, added = str(her["persona"].get("about") or "").strip(), []
        for fact in facts:
            sentence = fact.removeprefix("- ").strip()
            sentence = sentence[0].upper() + sentence[1:] + ("" if sentence.endswith((".", "!", "?")) else ".")
            if sentence.casefold() in about.casefold() or len(about) + len(sentence) + 1 > limit:
                continue
            about = f"{about}\n{sentence}".strip()
            added.append(sentence)
        if added:
            her["persona"] = {**her["persona"], "about": about}
        return added

    def forget_sources(self, companion_id):
        before = len(self.sources)
        self.sources = [s for s in self.sources if s["companion_id"] != companion_id]
        return before - len(self.sources)

    def create_chat(self, companion_id, title=""):
        chat = {"id": len(self.chats_) + 1, "companion_id": companion_id, "title": title, "named": "", "notes": "",
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

    def add_note(self, chat_id, note, limit):
        chat = next(c for c in self.chats_ if c["id"] == chat_id)
        lines = [*chat["notes"].splitlines(), note]
        while len(lines) > 1 and len("\n".join(lines)) > limit:
            lines.pop(0)
        chat["notes"] = "\n".join(lines)
        return chat["notes"]

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

    def add_picture_references(self, companion_id, image_id, references):
        for reference in references:
            self.picture_references.append({"companion_id": companion_id, "reference": reference,
                                            "image_id": image_id, "used_at": datetime.now(timezone.utc)})

    def reference_history(self, companion_id, liked):
        history = {}
        for row in self.picture_references:
            if row["companion_id"] != companion_id:
                continue
            past = history.setdefault(row["reference"], {"uses": [], "likes": 0})
            past["uses"].append(row["used_at"])
            past["likes"] += any(m["image_id"] == row["image_id"] and m["reaction"] in liked for m in self.messages)
        return history

    def add_message(self, chat_id, role, content, image_id=None, photo_note=None, apps=None, remembered=None):
        image = self.image(image_id) if image_id else None
        row = {"id": len(self.messages) + 1, "chat_id": chat_id, "role": role, "content": content,
               "image_id": image_id, "photo_note": photo_note, "reaction": None, "apps": apps or None, "created_at": NOW, "edited_at": None,
               "remembered": remembered or None,
               "image_kind": image and image["kind"], "image_prompt": image and image["prompt"],
               "image_file": image and image["file"]}
        self.messages.append(row)
        return row

    def forget_remembered(self, message_id, index):
        row = next((m for m in self.messages if m["id"] == message_id), None)
        items = list((row or {}).get("remembered") or [])
        if not 0 <= index < len(items) or items[index].get("forgotten"):
            return None
        item = items[index]
        her = next(c for c in self.companions_ if c["id"] == self.chat(row["chat_id"])["companion_id"])
        if item["about"] == "them":
            her["memory"], found = without_fact(her["memory"], "them", item["fact"])
        else:
            about, found = without_fact(str(her["persona"].get("about") or ""), "me", item["fact"])
            her["persona"] = {**her["persona"], "about": about}
        items[index] = {**item, "forgotten": True}
        row["remembered"] = items
        return row, found

    def edit_message(self, message_id, content):
        row = next((m for m in self.messages if m["id"] == message_id), None)
        if row is None:
            return None
        row.update(content=content, edited_at=NOW)
        return row

    def set_reaction(self, message_id, reaction):
        row = next((m for m in self.messages if m["id"] == message_id), None)
        if row is not None:
            row["reaction"] = reaction
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
        prompt = system_prompt(maya, "Pranjall", NOW, looks="Long dark hair.", owner_about="I make apps.")
        self.assertIn("What Pranjall wrote about themselves:\nI make apps.", prompt)
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


class PoseTests(unittest.TestCase):
    def test_used_lately_scores_low_and_likes_raise_it_a_little(self):
        from datetime import timedelta
        import random
        from app import poses
        now = datetime(2026, 11, 22, tzinfo=timezone.utc)
        fresh, hour_ago, week_ago = Path("fresh.jpg"), Path("hour.jpg"), Path("week.jpg")
        history = {"hour.jpg": {"uses": [now - timedelta(hours=1)], "likes": 0},
                   "week.jpg": {"uses": [now - timedelta(days=7)], "likes": 0}}
        pictures = [hour_ago, week_ago, fresh]
        self.assertEqual(poses.chosen(pictures, history, now, 3, random.Random(1)), [fresh, week_ago, hour_ago])
        # Liked, the same use still scores lower than one never used, but higher than without the like.
        self.assertGreater(poses.score([now - timedelta(days=1)], 2, now), poses.score([now - timedelta(days=1)], 0, now))
        self.assertLess(poses.score([now - timedelta(days=1)], 5, now), poses.score([], 0, now))
        # Used more often scores lower than used once at the same time.
        self.assertLess(poses.score([now - timedelta(days=2)] * 3, 0, now), poses.score([now - timedelta(days=2)], 0, now))


class ImageServiceTests(unittest.TestCase):
    def test_sends_the_prompt_and_references_then_fetches_the_picture(self):
        import app.images

        sent = []

        def imagegen(request):
            sent.append((request.method, request.url.path))
            if request.url.path == "/pictures":
                sent.append(json.loads(request.content))
                return httpx.Response(202, json={"id": "p1", "status": "queued"})
            if request.url.path == "/pictures/p1":
                return httpx.Response(200, json={"id": "p1", "status": "done"})
            return httpx.Response(200, content=png("blue"))

        poll, app.images.POLL_SECONDS = app.images.POLL_SECONDS, 0.0
        try:
            image = asyncio.run(ImageService(config(), httpx.MockTransport(imagegen)).generate("a selfie", [b"face"]))
        finally:
            app.images.POLL_SECONDS = poll
        self.assertEqual(image, png("blue"))
        self.assertEqual(sent[1], {"prompt": "a selfie", "resolution": 1024, "references": ["ZmFjZQ=="]})
        self.assertEqual(sent[-1], ("GET", "/pictures/p1/image"))

    def test_a_refused_or_failed_picture_says_why(self):
        import app.images

        refused = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "still being prepared"}))
        with self.assertRaisesRegex(ImageError, "still being prepared"):
            asyncio.run(ImageService(config(), refused).generate("a selfie", []))

        def failing(request):
            if request.method == "POST":
                return httpx.Response(202, json={"id": "p2"})
            return httpx.Response(200, json={"id": "p2", "status": "failed", "error": "the draw pass stopped"})
        poll, app.images.POLL_SECONDS = app.images.POLL_SECONDS, 0.0
        try:
            with self.assertRaisesRegex(ImageError, "draw pass"):
                asyncio.run(ImageService(config(), httpx.MockTransport(failing)).generate("a selfie", []))
        finally:
            app.images.POLL_SECONDS = poll


def fake_services(chat_rounds: list[list[dict]], seen: dict):
    """Ollama answers each chat with the next round of chunks; the image service makes a blue square."""
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

    def imagegen(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/pictures":
            seen["picture"] = json.loads(request.content)
            return httpx.Response(202, json={"id": "p1", "status": "queued"})
        if path == "/pictures/p1":
            return httpx.Response(200, json={"id": "p1", "status": "done"})
        if path == "/pictures/p1/image":
            return httpx.Response(200, content=png("blue"))
        return httpx.Response(404)

    return httpx.MockTransport(ollama), httpx.MockTransport(imagegen)


def streamed(seen):
    return [chat for chat in seen["chats"] if chat.get("stream", True)]



class PictureTimeoutTests(unittest.TestCase):
    def test_a_picture_given_up_on_is_stopped(self):
        import app.images
        from app.config import SociaConfig as Config

        sent = []

        def imagegen(request):
            sent.append((request.method, request.url.path))
            if request.method == "POST":
                return httpx.Response(202, json={"id": "p9", "status": "queued"})
            return httpx.Response(200, json={"id": "p9", "status": "drawing"})

        with tempfile.TemporaryDirectory() as folder:
            images = ImageService(config(Path(folder)), httpx.MockTransport(imagegen))
            limit, poll = Config.IMAGE_TIMEOUT_SECONDS, app.images.POLL_SECONDS
            Config.IMAGE_TIMEOUT_SECONDS, app.images.POLL_SECONDS = 0.05, 0.01
            try:
                with self.assertRaises((TimeoutError, asyncio.TimeoutError)):
                    asyncio.run(images.generate("a selfie", []))
            finally:
                Config.IMAGE_TIMEOUT_SECONDS, app.images.POLL_SECONDS = limit, poll
        self.assertEqual(sent[-1], ("DELETE", "/pictures/p9"))


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

    def run_turn(self, rounds, text="", photo=None, chat_id=None, her=None, completion=None, apps=None, faces=None):
        seen: dict = {"name": completion} if completion else {}
        ollama_transport, images_transport = fake_services(rounds, seen)
        conversation = Conversation(self.config, self.store, Ollama(self.config, ollama_transport),
                                    ImageService(self.config, images_transport), apps, faces)

        async def collect():
            companion = self.store.companion((her or self.her)["id"])
            events = [event async for event in conversation.send(companion, chat_id, text, photo, "Pranjall", "Asia/Kolkata")]
            await asyncio.gather(*conversation.background)
            return events

        import app.images
        app.images.POLL_SECONDS = 0
        return asyncio.run(collect()), seen

    def test_a_picture_she_promises_but_does_not_take_is_taken(self):
        description = "I lie on my bed under a warm lamp, a mirror selfie, smiling at you."
        events, seen = self.run_turn([[{"content": "Here you go 😘"}]], text="send me one please", completion=description)
        checks = [c for c in seen["completions"] if "Does Luna's last message send a picture" in c["messages"][0]["content"]]
        self.assertEqual(len(checks), 1)
        self.assertIn("Luna: Here you go 😘", checks[0]["messages"][0]["content"])
        self.assertIn("image", [event["type"] for event in events])
        self.assertIn("picture", seen)

    def test_only_her_own_promises_of_a_picture_count(self):
        # Her replies on 16 November that got a picture nobody asked for, and ones that promised one.
        for reply in ["Aww, Melon! That's such a cute photo 😊", "I'm ready when you are! Bring it on! 🥰",
                      "You've been busy taking pictures of them all day, haven't you?",
                      "Oh no—this one looks like Melon!"]:
            self.assertIsNone(PICTURE_PROMISE.search(reply), reply)
        for reply in ["Here it is—my titties, just for you!", "Okay… let me get that for you. ❤️", "There! 😘",
                      "Let me resend it right away", "Here you go—just for you. 💞", "I just sent you my titties"]:
            self.assertIsNotNone(PICTURE_PROMISE.search(reply), reply)

    def test_no_picture_when_she_promises_none_or_nothing_hints_at_one(self):
        events, seen = self.run_turn([[{"content": "Not now, later 😘"}]], text="send me one", completion="NONE")
        self.assertNotIn("image", [event["type"] for event in events])
        self.assertNotIn("picture", seen)
        events, seen = self.run_turn([[{"content": "Coffee then? ❤️"}]], text="It's raining")
        self.assertFalse(any("Does Luna's last message" in c["messages"][0]["content"] for c in seen.get("completions", [])))

    def test_she_uses_a_luna_app_and_answers_from_what_it_said(self):
        from app.apps import APPS

        class Apps:
            used = []

            async def running(self):
                return [APPS["vibia"], APPS["lumia"]]

            async def handle(self, name, arguments):
                self.used.append((name, arguments))
                return "Playing Kesariya on the speakers." if name == "use_app" else "- play_tracks: Play tracks."

        listing = {"function": {"name": "app_tools", "arguments": {"app": "vibia"}}}
        play = {"function": {"name": "use_app", "arguments": json.dumps(
            {"app": "vibia", "tool": "play_tracks", "arguments": {"names": ["Kesariya.mp3"]}})}}
        apps = Apps()
        events, seen = self.run_turn([[{"content": "", "tool_calls": [listing]}], [{"content": "", "tool_calls": [play]}],
                                      [{"content": "Done, it's playing ❤️"}]], text="play kesariya", apps=apps)
        self.assertEqual([name for name, _ in apps.used], ["app_tools", "use_app"])
        self.assertEqual(apps.used[1][1]["arguments"], {"names": ["Kesariya.mp3"]})
        self.assertIn({"type": "status", "text": "Using Luna Vibia"}, events)
        self.assertEqual(events[-2]["message"]["content"], "Done, it's playing ❤️")
        # Saved with her reply, so the page shows what she did; listing tools is not shown.
        self.assertEqual(events[-2]["message"]["apps"], [{"app": "vibia", "tool": "play_tracks"}])
        first, _, last = streamed(seen)
        self.assertIn("- vibia: their music", first["messages"][0]["content"])
        self.assertEqual({tool["function"]["name"] for tool in first["tools"]} & {"app_tools", "use_app"},
                         {"app_tools", "use_app"})
        self.assertEqual(last["messages"][-1], {"role": "tool", "content": "Playing Kesariya on the speakers."})

    def test_a_tool_call_she_writes_out_is_made_and_never_shown(self):
        from app.apps import APPS

        class Apps:
            used = []

            async def running(self):
                return [APPS["visia"]]

            async def handle(self, name, arguments):
                self.used.append((name, arguments))
                return "- list_images: List images."

        # Her reply in chat 9 on 16 November, streamed in pieces.
        written = [{"content": "</tool_"}, {"content": 'call>\n{"name": "app_tools", "arguments": {"app": "visia"}}\n'},
                   {"content": "</tool_call>"}]
        apps = Apps()
        events, _ = self.run_turn([written, [{"content": "Found your favourites ❤️"}]], text="they are in visia!", apps=apps)
        self.assertEqual(apps.used, [("app_tools", {"app": "visia"})])
        shown = "".join(e["text"] for e in events if e["type"] == "text")
        self.assertEqual(shown, "Found your favourites ❤️")
        self.assertEqual(events[-2]["message"]["content"], "Found your favourites ❤️")
        self.assertEqual(written_calls("Sure 😘 <tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>"),
                         ([{"function": {"name": "x", "arguments": {}}}], "Sure 😘"))
        self.assertEqual(written_calls("Just text {not a call}"), ([], "Just text {not a call}"))

    def test_the_app_and_tools_a_message_needs_are_handed_to_her(self):
        from app.apps import APPS

        class Apps:
            used = []

            async def running(self):
                return [APPS["visia"], APPS["vibia"]]

            async def tool_lines(self, key):
                return "- search_images: Search.\n- list_images: List."

            async def direct_specs(self, key, names):
                self.picked = (key, names)
                return [{"type": "function", "function": {"name": f"{key}__{n}", "description": "", "parameters": {}}}
                        for n in names]

            async def handle(self, name, arguments):
                self.used.append((name, arguments))
                return "3 favourite photos: beach.jpg, cats.jpg, cake.jpg"

        class Model(Ollama):
            answers = iter(["visia", "search_images, list_images"])

            async def complete(self, prompt):
                return next(self.answers, "Favourite photos")

        search = {"function": {"name": "visia__search_images", "arguments": {"query": "favourite"}}}
        ollama_transport, images_transport = fake_services([[{"content": "", "tool_calls": [search]}],
                                                          [{"content": "You have 3 favourites 😍"}]], seen := {})
        apps = Apps()
        conversation = Conversation(self.config, self.store, Model(self.config, ollama_transport),
                                    ImageService(self.config, images_transport), apps)

        async def collect():
            return [e async for e in conversation.send(self.store.companion(self.her["id"]), None,
                                                       "you see my fav images?", None, "Pranjall", "Asia/Kolkata")]

        events = asyncio.run(collect())
        self.assertEqual(apps.picked, ("visia", ["search_images", "list_images"]))
        first = streamed(seen)[0]
        self.assertIn("visia__search_images", {tool["function"]["name"] for tool in first["tools"]})
        self.assertIn("use Luna Visia", first["messages"][0]["content"])
        self.assertEqual(apps.used, [("visia__search_images", {"query": "favourite"})])
        self.assertEqual(events[-2]["message"]["apps"], [{"app": "visia", "tool": "search_images"}])
        self.assertEqual(events[-2]["message"]["content"], "You have 3 favourites 😍")

    def test_no_app_is_chosen_for_a_message_that_needs_none(self):
        from app.apps import APPS

        class Apps:
            async def running(self):
                return [APPS["visia"]]

        _, seen = self.run_turn([[{"content": "Hi ❤️"}]], text="hi", apps=Apps(), completion="NONE")
        first = streamed(seen)[0]
        self.assertFalse(any("__" in tool["function"]["name"] for tool in first["tools"]))
        self.assertNotIn("For Pranjall's last message, use", first["messages"][0]["content"])

    def test_without_the_apps_she_has_no_app_tools(self):
        _, seen = self.run_turn([[{"content": "Hi ❤️"}]], text="hi")
        self.assertNotIn("app_tools", {tool["function"]["name"] for tool in streamed(seen)[0]["tools"]})
        self.assertNotIn("Luna apps", streamed(seen)[0]["messages"][0]["content"])

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

    def test_she_reacts_to_what_was_sent_and_still_answers(self):
        react = {"function": {"name": "react", "arguments": {"emoji": "❤"}}}
        events, seen = self.run_turn([[{"content": "Aww you remembered", "tool_calls": [react]}]], text="I got you flowers")
        reaction = next(e for e in events if e["type"] == "reaction")
        sent = next(e for e in events if e["type"] == "user")["message"]
        # A heart without its variation selector is still the heart.
        self.assertEqual(reaction, {"type": "reaction", "message_id": sent["id"], "reaction": "❤️"})
        self.assertEqual(self.store.message(sent["id"])["reaction"], "❤️")
        self.assertEqual(events[-2]["message"]["content"], "Aww you remembered")
        self.assertIn("react", [tool["function"]["name"] for tool in streamed(seen)[0]["tools"]])

    def test_a_reaction_alone_gets_words_and_an_unknown_one_is_ignored(self):
        react = {"function": {"name": "react", "arguments": {"emoji": "😂"}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [react]}], [{"content": "Stoppp 😂"}]],
                                     text="I fell in the rain")
        self.assertEqual(events[-2]["message"]["content"], "Stoppp 😂")
        follow = streamed(seen)[1]["messages"]
        self.assertEqual(follow[-2]["tool_calls"][0]["function"]["name"], "react")
        odd = {"function": {"name": "react", "arguments": {"emoji": "🍆"}}}
        events, _ = self.run_turn([[{"content": "hmm", "tool_calls": [odd]}]], text="guess what")
        self.assertNotIn("reaction", [e["type"] for e in events])

    def test_she_writes_down_her_secret_and_reads_it_on_every_reply(self):
        note = {"function": {"name": "note", "arguments": {"text": "Secret: a penguin"}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [note]}], [{"content": "Got one! Ask away 😘"}]],
                                     text="Let's play 20 questions, you think of something")
        chat_id = events[0]["chat"]["id"]
        self.assertEqual(self.store.chat(chat_id)["notes"], "Secret: a penguin")
        self.assertEqual(events[-2]["message"]["content"], "Got one! Ask away 😘")
        # Asked to pick a secret, she was reminded to write it down; the next reply reads it.
        self.assertIn("call note", streamed(seen)[0]["messages"][-1]["content"])
        events, seen = self.run_turn([[{"content": "Nope! 🙈"}]], text="Is it a cat?", chat_id=chat_id)
        system = streamed(seen)[0]["messages"][0]["content"]
        self.assertIn("Your private notes for this chat", system)
        self.assertIn("- Secret: a penguin", system)

    def test_a_note_she_writes_out_is_kept_and_never_shown(self):
        events, _ = self.run_turn([[{"content": 'Ready! note("Secret: the moon") Ask me 😘'}]],
                                  text="your turn to think of something, I'll ask")
        self.assertEqual(self.store.chat(events[0]["chat"]["id"])["notes"], "Secret: the moon")
        self.assertEqual(events[-2]["message"]["content"], "Ready!  Ask me 😘")

    def test_a_vision_model_of_her_own_reads_photos_at_its_own_context(self):
        from dataclasses import replace
        asked = []

        def ollama(request: httpx.Request) -> httpx.Response:
            asked.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "A cat."}})

        own = replace(self.config, vision_model="huihui_ai/qwen3-vl-abliterated:4b-instruct", vision_context_tokens=4096)
        self.assertEqual(asyncio.run(Ollama(own, httpx.MockTransport(ollama)).describe_photo(png())), "A cat.")
        self.assertEqual((asked[0]["model"], asked[0]["options"]["num_ctx"]),
                         ("huihui_ai/qwen3-vl-abliterated:4b-instruct", 4096))

    def test_she_looks_again_when_asked_about_a_recent_photo_and_chats_as_usual_otherwise(self):
        events, _ = self.run_turn([[{"content": "Cute! 😍"}]], text="Look...", photo=png("red"))
        chat_id = events[0]["chat"]["id"]
        _, seen = self.run_turn([[{"content": "A red circle around your hand 👀"}]],
                                text="what did I mark in the red circle?", chat_id=chat_id)
        asked = streamed(seen)[0]
        self.assertEqual(asked["model"], self.config.vision_model)
        self.assertIn("1) the photo Pranjall sent 2 messages ago", asked["messages"][-1]["content"])
        # The earlier photo is still in the chat as its description, for the context around it.
        self.assertIn("[Photo: A red square.]", asked["messages"][-3]["content"])
        _, seen = self.run_turn([[{"content": "Rain again? ☔"}]], text="It's raining", chat_id=chat_id)
        self.assertEqual(streamed(seen)[0]["model"], self.config.chat_model)
        self.assertNotIn("unloaded", seen)
        self.assertNotIn("images", streamed(seen)[0]["messages"][-1])

    def test_face_recognition_names_the_faces_the_vision_model_sees(self):
        from app.faces import Match

        class Faces:
            asked = []
            unloads = 0

            def who(self, picture, people):
                self.asked.append(sorted(people))
                return [Match("Luna", 0.81, "left"), Match(None, 0.41, "right")]

            def unload(self):
                self.unloads += 1

        faces = Faces()
        events, seen = self.run_turn([[{"content": "That's me with someone 👀"}]], text="who is this?",
                                     photo=png("red"), faces=faces)
        legend = streamed(seen)[0]["messages"][-1]["content"]
        self.assertIn("1) the photo Pranjall sent just now with \"who is this?\". "
                      "Faces: you, Luna (left); someone who is neither you nor Pranjall (right).", legend)
        self.assertIn('Under each picture, "Faces:" says whose faces are in it', streamed(seen)[0]["messages"][0]["content"])
        self.assertEqual(faces.asked[0], ["Luna", "Pranjall"])
        # The face models came with the vision model, and leave when her chat model is back.
        self.assertEqual(faces.unloads, 1)
        # Her later replies read the faces with the photo's description.
        sent = next(m for m in self.store.messages if m["role"] == "user")
        self.assertEqual(sent["photo_note"], "A red square. Faces: you, Luna (left); someone who is neither you nor Pranjall (right).")

    def test_the_faces_line(self):
        from app.faces import Match, faces_line

        self.assertEqual(faces_line([], "Luna", "Pranjall"), "Faces: none visible, so whoever it is cannot be told from a face: say you cannot see their face.")
        self.assertEqual(faces_line([Match("Pranjall", 0.7, "middle")], "Luna", "Pranjall"), "Faces: Pranjall.")
        many = [Match(None, 0.2, "left")] * (SociaConfig.FACES_NAMED + 2)
        self.assertTrue(faces_line(many, "Luna", "Pranjall").endswith("; and 2 more."))

    def test_faces_are_aligned_and_kept_once_by_their_best_box(self):
        import numpy as np

        from app.faces import ARCFACE_POINTS, overlap_filter, place, similarity_transform

        angle, scale, shift = 0.3, 2.5, np.array([40.0, -12.0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        source = (ARCFACE_POINTS - shift) @ np.linalg.inv(scale * rotation).T
        matrix = similarity_transform(source.astype(np.float32), ARCFACE_POINTS)
        moved = source @ matrix[:, :2].T + matrix[:, 2]
        self.assertTrue(np.allclose(moved, ARCFACE_POINTS, atol=1e-3))
        boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [200, 200, 260, 260]], dtype=float)
        self.assertEqual(overlap_filter(boxes, np.array([0.6, 0.9, 0.7]), 0.4), [1, 2])
        self.assertEqual([place(b, 300) for b in boxes], ["left", "left", "right"])

    def test_which_messages_let_her_see_the_pictures(self):
        from app.chat import sees_pictures

        photo = {"role": "user", "content": "look", "image_file": "a.jpg", "image_id": 1}
        her = {"role": "luna", "content": "wow", "image_file": None}
        picture = {"role": "luna", "content": "me!", "image_file": "b.png", "image_id": 2}
        ask = lambda text: {"role": "user", "content": text, "image_file": None}  # noqa: E731
        self.assertTrue(sees_pictures([photo]))
        self.assertTrue(sees_pictures([photo, her, ask("which one is better?")]))
        self.assertTrue(sees_pictures([picture, ask("is this one you?")]))
        self.assertFalse(sees_pictures([photo, her, ask("ok going to the gym")]))
        self.assertFalse(sees_pictures([ask("send me a photo"), her]))
        old = [photo, *[her, ask("hmm")] * 3, ask("what's in the photo?")]
        self.assertFalse(sees_pictures(old))

    def test_several_photos_are_a_message_each_and_she_answers_them_together(self):
        events, seen = self.run_turn([[{"content": "Both are lovely, the second more 😍"}]], text="which one?",
                                     photo=[png("red"), png("blue")])
        users = [event["message"] for event in events if event["type"] == "user"]
        self.assertEqual([(m["content"], bool(m["image_url"])) for m in users], [("", True), ("which one?", True)])
        self.assertIn({"type": "status", "text": "Looking at your photos"}, events)
        # One look at both, each sent just now; then one reply.
        looked = streamed(seen)[0]["messages"][-1]
        self.assertEqual(len(looked["images"]), 2)
        self.assertEqual(looked["content"].count("sent just now"), 2)
        self.assertEqual(len(streamed(seen)), 1)
        # Each photo is described for her later replies.
        notes = [m["photo_note"] for m in self.store.messages if m["role"] == "user"]
        self.assertEqual(notes, ["A red square.", "A red square."])

    def test_she_remembers_facts_about_them_and_herself_for_every_chat(self):
        about_them = {"function": {"name": "remember", "arguments": {"fact": "Their cats are Mini, Maxi and Melon"}}}
        about_her = {"function": {"name": "remember", "arguments": {"fact": "My favourite colour is lavender",
                                                                    "about": "me"}}}
        events, _ = self.run_turn([[{"content": "Lavender, always 💜", "tool_calls": [about_them, about_her]}]],
                                  text="my cats are Mini, Maxi and Melon. fav colour?")
        her = self.store.companion(self.her["id"])
        self.assertIn("- Their cats are Mini, Maxi and Melon", her["memory"])
        self.assertTrue(her["persona"]["about"].endswith("Her favourite colour is lavender."))
        remembered = next(e for e in events if e["type"] == "remembered")
        self.assertEqual(len(remembered["facts"]), 2)
        # Kept with her reply, each with where it went, so it can be taken back from the chat.
        self.assertEqual([(i["index"], i["about"]) for i in next(e for e in events if e["type"] == "done")["message"]["remembered"]], [(0, "them"), (1, "me")])
        # What she remembers is in her next prompt, in any chat.
        _, seen = self.run_turn([[{"content": "Mini, Maxi and Melon 🐱"}]], text="what are my cats called?")
        prompt = streamed(seen)[0]["messages"][0]["content"]
        self.assertIn("Their cats are Mini, Maxi and Melon", prompt)
        self.assertIn("Her favourite colour is lavender", prompt)

    def test_a_reaction_she_types_out_becomes_a_reaction(self):
        # 17 November, her reply to a pool selfie began with the reaction as text.
        calls, text = written_reactions("react: ❤️\n\nLook at you, Mr. Poolside Prince 😉")
        self.assertEqual(calls, [{"function": {"name": "react", "arguments": {"emoji": "❤️"}}}])
        self.assertEqual(text, "Look at you, Mr. Poolside Prince 😉")
        self.assertEqual(written_reactions("I react: fast!")[0], [])
        events, _ = self.run_turn([[{"content": "react: 🔥\nWow, the pool!"}]], text="LOOK!!!", photo=png("red"))
        self.assertIn("reaction", [e["type"] for e in events])
        saved = next(e for e in events if e["type"] == "done")["message"]
        self.assertEqual(saved["content"], "Wow, the pool!")

    def test_her_secret_notes_stay_out_of_a_photo_turn(self):
        chat = self.store.create_chat(self.her["id"])
        self.store.add_note(chat["id"], "Secret: chocolate with pink sea salt", 1000)
        _, seen = self.run_turn([[{"content": "Nice!"}]], text="LOOK!!!", photo=png("red"), chat_id=chat["id"])
        self.assertNotIn("pink sea salt", streamed(seen)[0]["messages"][0]["content"])
        _, seen = self.run_turn([[{"content": "Hmm, guess again"}]], text="is it fruit?", chat_id=chat["id"])
        self.assertIn("pink sea salt", streamed(seen)[0]["messages"][0]["content"])

    def test_a_text_turn_gets_the_camera_only_when_a_picture_is_asked_for(self):
        for words in ["send me a selfie", "can you send a pic of you?", "show me yourself", "pic of us together"]:
            self.assertTrue(asks_for_picture(words), words)
        # 20 November: "imma send you two images" got a picture nobody asked for, which then took 10 minutes.
        for words in ["ok imma send you two images... tell me who looks better ok...", "I'll send you a pic later",
                      "let me show you a photo", "good morning", "i'm gonna share some pictures of my trip"]:
            self.assertFalse(asks_for_picture(words), words)
        call = {"function": {"name": "create_image", "arguments": {"description": "me waving"}}}
        events, seen = self.run_turn([[{"content": "Okay, send them! Here you go 📷", "tool_calls": [call]}]],
                                     text="ok imma send you two images... tell me who looks better ok...")
        self.assertNotIn("create_image", [t["function"]["name"] for t in streamed(seen)[0].get("tools", [])])
        self.assertNotIn("image_started", [e["type"] for e in events])

    def test_a_photo_gets_words_unless_a_picture_is_asked_for(self):
        for words in ["add yourself here", "send me a selfie", "put yourself next to me", "pic of you please",
                      "improve the lighting", "wish you could be in this photo", "take a photo of us", "you and me here?"]:
            self.assertTrue(ASKS_FOR_PICTURE.search(words), words)
        for words in ["LOOK!!!", "how do I look?", "who is this?", "which one is better?", "my new haircut 😎",
                      "guess where I am", "the pool was so cold", "thoughts?"]:
            self.assertFalse(ASKS_FOR_PICTURE.search(words), words)
        # 17 November: "LOOK!!!" with a pool selfie, and the vision model took a picture of its own.
        call = {"function": {"name": "create_image", "arguments": {"description": "me at the pool"}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "Look at you! 😍"}]],
                                     text="LOOK!!!", photo=png("red"))
        self.assertNotIn("create_image", [t["function"]["name"] for t in streamed(seen)[0].get("tools", [])])
        self.assertNotIn("image_started", [e["type"] for e in events])
        self.assertNotIn("picture", seen)
        self.assertEqual(events[-2]["message"]["content"] if events[-1]["type"] == "chat" else events[-1]["message"]["content"],
                         "Look at you! 😍")

    def test_she_sees_a_photo_and_steps_into_it(self):
        call = {"function": {"name": "create_image", "arguments": {
            "description": "standing beside them, smiling", "include_luna": True, "use_their_photo": True}}}
        events, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "Us! 🥰"}]],
                                     text="add yourself here", photo=png("red"))
        self.assertEqual([e["type"] for e in events],
                         ["chat", "user", "status", "image_started", "image", "text", "done", "chat"])
        # The vision model answers as her, seeing the photo itself, at its own context. Not her profile picture:
        # as picture 1 she took it for one of his photos and said "the first one is me" every time (17 November).
        first = streamed(seen)[0]
        self.assertEqual((first["model"], first["options"]["num_ctx"]), (self.config.vision_model, self.config.vision_context_tokens))
        self.assertEqual(len(first["messages"][-1]["images"]), 1)
        self.assertIn("[Pictures you see now: 1) the photo Pranjall sent just now", first["messages"][-1]["content"])
        self.assertNotIn("profile picture", first["messages"][-1]["content"])
        self.assertIn('You are in a picture only when its "Faces:" says "you"', first["messages"][0]["content"])
        self.assertIn("Right now you see the pictures themselves", first["messages"][0]["content"])
        # Her caption after the picture comes from the same model, so nothing is swapped in between.
        self.assertEqual(streamed(seen)[1]["model"], self.config.vision_model)
        # Afterwards the photo is described once for her later replies, which read only text.
        sent = next(m for m in self.store.messages if m["role"] == "user")
        self.assertEqual(sent["photo_note"], "A red square.")
        looks = [c for c in seen["completions"] if c["model"] == self.config.vision_model]
        self.assertEqual({c["options"]["num_ctx"] for c in looks}, {self.config.vision_context_tokens})
        # The vision model starts loading first, while the pictures are got ready.
        self.assertEqual(seen["unloaded"][0], {"model": self.config.vision_model, "options": {"num_ctx": self.config.vision_context_tokens}})
        # Then the vision model leaves the GPU and her chat model comes back, before her next reply.
        self.assertEqual(seen["unloaded"][-2:], [{"model": self.config.vision_model, "keep_alive": 0},
                                                 {"model": self.config.chat_model, "options": {"num_ctx": 8192}}])
        # She knows her own looks, read once from her profile picture.
        self.assertIn("A woman who looks like this is not always you", streamed(seen)[0]["messages"][0]["content"])
        self.assertEqual(self.profile.with_suffix(".txt").read_text(), "A red square.")
        # Their photo is <image1>, her profile <image2>, sent to the image service in that order.
        picture = seen["picture"]
        self.assertIn("<image2> into <image1>", picture["prompt"])
        self.assertEqual(len(picture["references"]), 2)
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
        ollama_transport, images_transport = fake_services([], seen)
        chat = self.store.create_chat(self.her["id"], "hey you up?")
        self.store.add_message(chat["id"], "user", "hey you up?")
        self.store.add_message(chat["id"], "luna", "always for you 🌙")
        waiting = self.store.create_chat(self.her["id"], "unanswered")
        self.store.add_message(waiting["id"], "user", "unanswered")
        conversation = Conversation(self.config, self.store, Ollama(self.config, ollama_transport),
                                    ImageService(self.config, images_transport))
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
        self.assertEqual(len(seen["picture"]["references"]), 3)
        self.assertIn("<image1>, <image2> and <image3> are pictures of the same woman", seen["picture"]["prompt"])
        self.assertEqual(self.config.references(self.her["id"]), [folder / "1-aaaaaaaa.jpg", folder / "2-bbbbbbbb.jpg"])

    def test_her_pictures_least_used_lately_go_next(self):
        folder = self.config.references_dir(self.her["id"])
        folder.mkdir(parents=True)
        for number in range(1, 7):
            (folder / f"{number}-{number:08x}.jpg").write_bytes(png("green"))
        call = {"function": {"name": "create_image", "arguments": {"description": "a selfie", "include_luna": True}}}
        self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "here"}]], text="selfie?")
        first = {row["reference"] for row in self.store.picture_references}
        self.assertEqual(len(first), SociaConfig.PICTURE_REFERENCES)
        _, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "here"}]], text="another selfie?")
        self.assertEqual(len(seen["picture"]["references"]), 1 + SociaConfig.PICTURE_REFERENCES)
        second = [row["reference"] for row in self.store.picture_references[SociaConfig.PICTURE_REFERENCES:]]
        # The two never used go first, then two of the first picture's.
        self.assertEqual(set(second) - first, {path.name for path in self.config.references(self.her["id"])} - first)

    def test_her_photo_comes_from_the_same_chat(self):
        self.run_turn([[{"content": "nice"}]], text="look", photo=png("red"))
        call = {"function": {"name": "create_image", "arguments": {
            "description": "a selfie", "include_luna": True, "use_their_photo": True}}}
        _, seen = self.run_turn([[{"content": "", "tool_calls": [call]}], [{"content": "here"}]], text="selfie?")
        # The new chat has no photo of theirs, so only her profile is a reference.
        self.assertEqual(len(seen["picture"]["references"]), 1)

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
        self.conversation = Conversation(cfg, store, None, None)
        app = create_app(cfg, store, self.conversation, None, self.voice, httpx.MockTransport(auth), reader=reader)
        return TestClient(app, follow_redirects=False)

    def owner(self, cfg=None, reader=None):
        client = self.client({"user_id": "owner", "name": "Pranjall Kumar"}, cfg, reader)
        client.cookies.set("luna_sso", "token")
        return client

    def test_only_the_owner_sees_or_connects_the_luna_apps(self):
        self.assertEqual(self.client(None).get("/api/apps").status_code, 401)
        self.assertEqual(self.client(None).get("/apps/connect").status_code, 303)  # to sign in
        self.assertEqual(self.owner().get("/api/apps").json(), {"connected": False, "apps": []})

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

    def test_photos_of_the_owner_need_a_face_and_are_owner_only(self):
        from app.faces import Face

        class Faces:
            unloads = 0

            def unload(self):
                self.unloads += 1

            def read(self, data):
                # A red picture has a face, any other none.
                from PIL import Image as Picture
                red = Picture.open(io.BytesIO(data)).convert("RGB").getpixel((5, 5))[0] > 200
                return ([Face((0, 0, 1, 1), 0.9, None)] if red else []), 64

        with tempfile.TemporaryDirectory() as folder:
            client = self.owner(config(Path(folder)))
            self.conversation.faces = Faces()
            self.assertEqual(client.get("/api/owner/photos").json()["photos"], [])
            added = client.post("/api/owner/photos", files=[("photos", ("a.png", png("red"), "image/png")),
                                                            ("photos", ("b.png", png("blue"), "image/png"))]).json()
            self.assertEqual(len(added["photos"]), 1)
            self.assertIn("1 of them had no face", added["note"])
            url = added["photos"][0]["url"]
            self.assertEqual(client.get(url).headers["content-type"], "image/jpeg")
            self.assertEqual(client.get("/api/owner/photos/..%2F..%2Fprofile.png").status_code, 404)
            too_many = [("photos", (f"{n}.png", png("red"), "image/png")) for n in range(SociaConfig.MAX_OWNER_PHOTOS)]
            self.assertEqual(client.post("/api/owner/photos", files=too_many).status_code, 400)
            self.assertEqual(client.delete(url).json(), {"deleted": True})
            self.assertEqual(client.get("/api/owner/photos").json()["photos"], [])
        self.assertEqual(self.client(None).get("/api/owner/photos").status_code, 401)

    def test_the_owner_has_a_picture_a_name_she_calls_them_and_details(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = config(Path(folder))
            client = self.owner(settings)
            me = client.get("/api/owner").json()
            self.assertEqual((me["name"], me["about"], me["has_picture"]), ("", "", False))
            self.assertEqual(me["called"], me["signed_in_as"])
            saved = client.put("/api/owner", json={"name": "  Pran  ", "about": "I make apps. " + "x" * 3000}).json()
            self.assertEqual((saved["name"], saved["called"]), ("Pran", "Pran"))
            self.assertEqual(len(saved["about"]), SociaConfig.ABOUT_CHARS)
            self.assertEqual(client.get("/api/owner/picture").status_code, 404)
            self.assertEqual(client.post("/api/owner/picture", files={"photo": ("me.png", png("red"), "image/png")}).json(),
                             {"saved": True})
            self.assertTrue(client.get("/api/owner").json()["has_picture"])
            self.assertEqual(client.get("/api/owner/avatar").headers["content-type"], "image/png")
            # Their avatar is cut where they choose, as hers is.
            default = client.get("/api/owner/avatar/crop").json()
            moved = client.put("/api/owner/avatar/crop", json={"x": 0.3, "y": 0.4, "zoom": 2}).json()
            self.assertEqual((moved["zoom"], moved["x"]), (2, 0.3))
            self.assertEqual(client.delete("/api/owner/avatar/crop").json(), default)
            self.assertEqual(client.put("/api/owner/avatar/crop", json={"x": "left"}).status_code, 400)
            # Face recognition knows them by their picture too, before their other photos.
            self.assertEqual(settings.owner_faces(), [settings.owner_picture_path])
            self.assertEqual(client.delete("/api/owner/picture").json(), {"deleted": True})
            self.assertFalse(client.get("/api/owner").json()["has_picture"])
        self.assertEqual(self.client(None).get("/api/owner").status_code, 401)

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
            # One of them becomes her profile picture, and the old profile picture takes its place.
            settings = config(Path(folder))
            profile = settings.profile_path(her["id"])
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_bytes(b"old profile")
            self.store.update_companion(her["id"], {"persona": {"nude_pictures": "on"}})
            left = client.get(base).json()["references"][0]
            chosen = settings.references_dir(her["id"]) / left["name"]
            picture = chosen.read_bytes()
            self.assertEqual(client.post(f"{left['url']}/profile").json(), {"saved": True})
            self.assertEqual(profile.read_bytes(), picture)
            self.assertEqual(chosen.read_bytes(), b"old profile")
            self.assertEqual(len(client.get(base).json()["references"]), 1)
            # A new face: nude pictures wait for the owner's confirmation again.
            self.assertEqual(self.store.companion(her["id"])["persona"]["nude_pictures"], "off")
            self.assertEqual(client.post(f"{base}/..%2Fprofile.png/profile").status_code, 404)
            # Another girlfriend has her own pictures.
            other = self.store.create_companion("Maya", {}, "bf_emma")
            self.assertEqual(client.get(f"/api/companions/{other['id']}/references").json()["references"], [])

    def test_a_slow_answer_keeps_the_page_connected_and_a_failed_one_says_so(self):
        import app.web as web

        async def slow(*args):
            yield {"type": "image_started"}
            await asyncio.sleep(0.2)
            yield {"type": "done"}

        async def broken(*args):
            yield {"type": "image_started"}
            raise RuntimeError("camera fell")

        client = self.owner()
        her = self.store.create_companion("Luna", {}, "af_heart")
        conversation = next(r.endpoint for r in client.app.routes if getattr(r, "path", "") == "/api/send")
        cells = dict(zip(conversation.__code__.co_freevars, (c.cell_contents for c in conversation.__closure__)))
        keepalive, web.KEEPALIVE_SECONDS = web.KEEPALIVE_SECONDS, 0.05
        try:
            cells["conversation"].send = slow
            body = client.post("/api/send", data={"text": "a picture?", "companion_id": str(her["id"])}).text
            self.assertIn(": still here", body)
            self.assertTrue(body.endswith('data: {"type": "done"}\n\n'))
            cells["conversation"].send = broken
            body = client.post("/api/send", data={"text": "again?", "companion_id": str(her["id"])}).text
            self.assertIn("Luna could not answer: camera fell", body)
        finally:
            web.KEEPALIVE_SECONDS = keepalive

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
    def test_her_chats_are_read_for_what_they_and_she_said_once_quiet(self):
        store = MemoryStore()
        her = store.create_companion("Luna", {"about": "She is 25, from Mumbai."}, "af_heart")
        chat = store.create_chat(her["id"])
        store.add_message(chat["id"], "user", "this is Melon, my cat", None, "A tabby cat with a green collar.")
        store.add_message(chat["id"], "luna", "Aww! My favourite colour is lavender, like her blanket 💜")
        def answers(prompt):
            if "Her bio so far" in prompt:
                self.assertIn("lavender", prompt)
                self.assertNotIn("this is Melon", prompt)
                return "- Her favourite colour is lavender"
            if "Answer only YES or NO" in prompt:
                return "YES"
            if "their girlfriend" in prompt:
                # Their own words only: what a photo showed made "facts" such as "They have a lamp".
                self.assertIn("this is Melon, my cat", prompt)
                self.assertNotIn("tabby", prompt)
                self.assertNotIn("lavender", prompt)
                return "- Their cat is called Melon"
            return "NONE"
        reader = MemoryReader(store, FakeOllama(answers), asyncio.Lock())

        async def run():
            while await reader.step():
                pass

        asyncio.run(run())
        her = store.companion(her["id"])
        self.assertEqual(her["memory"], "- Their cat is called Melon")
        self.assertEqual(her["persona"]["about"], "She is 25, from Mumbai.\nHer favourite colour is lavender.")
        # Read once: nothing new, nothing read again.
        self.assertEqual(store.quiet_chats(0), [])

    def test_only_lasting_facts_are_kept(self):
        # What the reader listed from a real chat on 17 November.
        them = "\n".join(["- they have three cats named Mini, Maxi, and Melon", "- they hate olives", "- they are tired",
                          "- they say good night", "- their name is they", "- they had maggi tonight"])
        her = "\n".join(["- Her favourite colour is lavender", "- She grew up in Mumbai", "- She likes to miss people",
                         "- She likes cozy chats", "- She loves baking", "- Her favourite snack is a samosa"])

        async def ask(prompt):
            # Each fact about them is then checked against their words; here every check says yes.
            if "Answer only YES or NO" in prompt:
                return "YES"
            return them if "their girlfriend" in prompt else her

        bio = "She loves baking. Her favourite snack is chocolate with pink sea salt."
        chat = [{"role": "user", "text": "hi"}, {"role": "luna", "text": "hey"}]
        self.assertEqual(asyncio.run(facts_about_them(ask, "Luna", chat)),
                         ["- they have three cats named Mini, Maxi, and Melon", "- they hate olives"])
        self.assertEqual(asyncio.run(facts_about_her(ask, "Luna", bio, chat)),
                         ["- Her favourite colour is lavender", "- She grew up in Mumbai"])

    def test_a_remember_she_types_out_is_made(self):
        calls, text = written_remembers('remember {"about":"them","fact":"Their cats are Mini, Maxi and Melon"} So cute!')
        self.assertEqual(calls[0]["function"]["arguments"]["fact"], "Their cats are Mini, Maxi and Melon")
        self.assertEqual(text, "So cute!")

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

    def test_the_owner_reacts_to_her_messages_only(self):
        mine = self.store.add_message(self.chat["id"], "user", "hey")
        hers = self.store.add_message(self.chat["id"], "luna", "hiii")
        url = f"/api/messages/{hers['id']}/reaction"
        self.assertEqual(self.client.put(url, json={"reaction": "🔥"}).json()["message"]["reaction"], "🔥")
        self.assertEqual(self.client.get(f"/api/chats/{self.chat['id']}/messages").json()["messages"][1]["reaction"], "🔥")
        self.assertIsNone(self.client.put(url, json={"reaction": None}).json()["message"]["reaction"])
        self.assertEqual(self.client.put(url, json={"reaction": "🍆"}).status_code, 400)
        self.assertEqual(self.client.put(f"/api/messages/{mine['id']}/reaction", json={"reaction": "🔥"}).status_code, 400)
        self.assertEqual(self.client.put("/api/messages/999/reaction", json={"reaction": "🔥"}).status_code, 404)
        self.assertIn("😂", self.client.get("/api/options").json()["reactions"])

    def test_your_message_can_be_edited_and_she_reads_the_new_words(self):
        mine = self.store.add_message(self.chat["id"], "user", "my favourite is pizz")
        hers = self.store.add_message(self.chat["id"], "luna", "Pizza? 🍕")
        url = f"/api/messages/{mine['id']}"
        edited = self.client.put(url, json={"content": "  my favourite is pizza "}).json()["message"]
        self.assertEqual((edited["content"], edited["edited"]), ("my favourite is pizza", True))
        listed = self.client.get(f"/api/chats/{self.chat['id']}/messages").json()["messages"]
        self.assertEqual([(m["content"], m["edited"]) for m in listed], [("my favourite is pizza", True), ("Pizza? 🍕", False)])
        # What she reads on her next reply comes from the same messages.
        self.assertEqual(self.store.recent(self.chat["id"], 40)[0]["content"], "my favourite is pizza")
        self.assertEqual(self.client.put(url, json={"content": "  "}).status_code, 400)
        self.assertEqual(self.client.put(f"/api/messages/{hers['id']}", json={"content": "no"}).status_code, 400)
        self.assertEqual(self.client.put("/api/messages/999", json={"content": "hi"}).status_code, 404)
        photo = self.store.add_image(self.chat["id"], "photo", "p.jpg")
        bare = self.store.add_message(self.chat["id"], "user", "", photo)
        self.assertEqual(self.client.put(f"/api/messages/{bare['id']}", json={"content": "hi"}).status_code, 400)

    def test_a_wrong_fact_is_taken_back_from_the_chat(self):
        her = self.store.companions_[0]
        her["memory"] = "- They love pizza\n- Their boss is Neha"
        her["persona"] = {"about": "She is 25.\nHer favourite fruit is the Alphonso mango. She bakes."}
        reply = self.store.add_message(self.chat["id"], "luna", "Got it 😘", remembered=[
            {"about": "them", "fact": "Their boss is Neha"}, {"about": "me", "fact": "Her favourite fruit is the Alphonso mango"}])
        listed = self.client.get(f"/api/chats/{self.chat['id']}/messages").json()["messages"][0]
        self.assertEqual([i["fact"] for i in listed["remembered"]], ["Their boss is Neha", "Her favourite fruit is the Alphonso mango"])
        url = f"/api/messages/{reply['id']}/remembered"
        answer = self.client.delete(f"{url}/0").json()
        self.assertTrue(answer["found"])
        self.assertEqual(her["memory"], "- They love pizza")
        # Gone from the chat too; the other fact keeps its own number.
        self.assertEqual([(i["index"], i["fact"]) for i in answer["message"]["remembered"]],
                         [(1, "Her favourite fruit is the Alphonso mango")])
        self.assertTrue(self.client.delete(f"{url}/1").json()["found"])
        self.assertEqual(her["persona"]["about"], "She is 25.\nShe bakes.")
        # Kept on the message as forgotten: a wrong fact to learn from.
        self.assertTrue(all(item["forgotten"] for item in self.store.message(reply["id"])["remembered"]))
        self.assertEqual(self.client.delete(f"{url}/1").status_code, 404)
        self.assertEqual(self.client.delete(f"{url}/5").status_code, 404)
        self.assertEqual(self.client.delete("/api/messages/999/remembered/0").status_code, 404)

    def test_a_fact_is_found_as_it_was_saved(self):
        self.assertEqual(without_fact("- A\n- they LOVE pizza\n- B", "them", "They love pizza"), ("- A\n- B", True))
        self.assertEqual(without_fact("- A", "them", "They love pizza"), ("- A", False))
        self.assertEqual(without_fact("She is 25. She bakes!", "me", "she is 25"), ("She bakes!", True))
        self.assertEqual(without_fact("She bakes.", "me", ""), ("She bakes.", False))

    def test_nothing_is_deleted_while_she_is_answering(self):
        sent = self.store.add_message(self.chat["id"], "user", "hey")
        app = self.client.app
        conversation = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/send")
        lock = conversation.__closure__  # the handler closes over the conversation
        turn = next(c.cell_contents for c in lock if hasattr(c.cell_contents, "turn")).turn

        async def hold():
            async with turn:
                return (self.client.delete(f"/api/messages/{sent['id']}").status_code,
                        self.client.put(f"/api/messages/{sent['id']}", json={"content": "hi"}).status_code)

        import asyncio
        self.assertEqual(asyncio.run(hold()), (409, 409))
        self.assertEqual(self.store.message(sent["id"])["content"], "hey")
        self.assertIsNotNone(self.store.message(sent["id"]))


class LunaAppsTests(unittest.TestCase):
    """Socia's sign-in to the owner's Luna apps, as an OAuth client of Luna Sentia's sign-on."""

    def setUp(self):
        self.seen = []
        self.refuse = False

        def auth(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            if request.url.path == "/oauth/register":
                return httpx.Response(201, json={"client_id": "socia", "client_secret": "s3cret"})
            if request.url.path == "/oauth/token":
                if self.refuse:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                n = len([r for r in self.seen if r.url.path == "/oauth/token"])
                return httpx.Response(200, json={"access_token": f"access{n}", "refresh_token": f"refresh{n}",
                                                 "expires_in": 3600})
            return httpx.Response(404)

        from app.apps import LunaApps
        self.store = MemoryStore()
        self.apps = LunaApps(config(), self.store, httpx.MockTransport(auth))

    def test_the_owner_approves_socia_with_pkce_once_for_every_app(self):
        import base64
        import hashlib
        from urllib.parse import parse_qs, urlparse

        url = asyncio.run(self.apps.start())
        query = parse_qs(urlparse(url).query)
        self.assertTrue(url.startswith("https://auth.example.com/oauth/authorize?"))
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["redirect_uri"], ["https://socia.example.com/apps/callback"])
        self.assertNotIn("resource", query)  # one token for every Luna app
        asyncio.run(self.apps.finish("code1", query["state"][0]))
        exchange = parse_qs(self.seen[-1].content.decode())
        verifier = exchange["code_verifier"][0]
        digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self.assertEqual(digest, query["code_challenge"][0])
        self.assertEqual(exchange["client_secret"], ["s3cret"])
        self.assertTrue(asyncio.run(self.apps.connected()))
        self.assertEqual(asyncio.run(self.apps.access_token()), "access1")
        with self.assertRaises(ValueError):
            asyncio.run(self.apps.finish("code1", query["state"][0]))  # each approval works once

    def test_an_expiring_token_is_refreshed_and_a_refused_refresh_disconnects(self):
        from app.apps import NotConnected

        self.store.set_apps_client("socia", "s3cret", self.apps.redirect_uri)
        self.store.set_apps_tokens("old", "refresh0", 0)
        self.assertEqual(asyncio.run(self.apps.access_token()), "access1")
        self.assertEqual(self.store.apps["refresh_token"], "refresh1")
        self.store.set_apps_tokens("old", "refresh1", 0)
        self.refuse = True
        with self.assertRaises(NotConnected):
            asyncio.run(self.apps.access_token())
        self.assertFalse(asyncio.run(self.apps.connected()))

    def test_her_tool_calls_name_real_apps_and_tools(self):
        from types import SimpleNamespace

        tool = SimpleNamespace(name="play_tracks", description="Play tracks on the speakers. Owner only.",
                               inputSchema={"type": "object", "properties": {"names": {"type": "array"}}})

        async def tools(key):
            return [tool]

        self.apps.tools = tools
        handle = lambda name, arguments: asyncio.run(self.apps.handle(name, arguments))
        self.assertIn("no Luna app called", handle("app_tools", {"app": "spotify"}))
        self.assertEqual(handle("app_tools", {"app": "Luna Vibia"}), "Luna Vibia can:\n- play_tracks: Play tracks on the speakers.")
        self.assertIn('"names"', handle("app_tool_help", {"app": "vibia", "tool": "play_tracks"}))
        self.assertIn("has no tool", handle("use_app", {"app": "vibia", "tool": "dance", "arguments": {}}))

    def test_the_page_sees_every_app_and_which_answer(self):
        from app.apps import APPS

        async def running():
            return [APPS["vibia"]]

        self.apps.running = running
        self.assertEqual(asyncio.run(self.apps.status())["connected"], False)
        self.store.set_apps_client("socia", "s3cret", self.apps.redirect_uri)
        self.store.set_apps_tokens("a", "r", 9e9)
        status = asyncio.run(self.apps.status())
        self.assertTrue(status["connected"])
        self.assertEqual(len(status["apps"]), 7)
        self.assertEqual([app["key"] for app in status["apps"] if app["running"]], ["vibia"])

    def test_a_long_answer_is_shortened_for_her(self):
        from types import SimpleNamespace

        from app.apps import RESULT_CHARS, result_text

        long = SimpleNamespace(structuredContent=None, isError=False, content=[SimpleNamespace(type="text", text="x" * 5000)])
        self.assertEqual(len(result_text(long)), RESULT_CHARS)
        failed = SimpleNamespace(structuredContent={"error": "no"}, isError=True, content=[])
        self.assertTrue(result_text(failed).startswith("The app answered with an error"))


if __name__ == "__main__":
    unittest.main()


class LoopAndLanguageTests(unittest.TestCase):
    def test_she_reads_the_owners_reactions_before_their_next_message(self):
        rows = [{"role": "user", "content": "hi"}, {"role": "luna", "content": "Missed you all day", "reaction": "❤️"},
                {"role": "user", "content": "me too"}]
        self.assertEqual(model_messages(rows)[-1]["content"],
                         '[Reacted ❤️ to your message: "Missed you all day"]\nme too')

    def test_she_is_reminded_to_note_a_secret_only_when_she_picks_one(self):
        def nudged(text, notes=""):
            return "call note" in with_secret_nudge([{"role": "user", "content": text}], notes)[-1]["content"]
        self.assertTrue(nudged("Let's play 20 questions! You think of something"))
        self.assertTrue(nudged("pick an animal and I'll guess"))
        self.assertFalse(nudged("I'm thinking of something, you ask me questions"))
        self.assertFalse(nudged("Let's play a guessing game", notes="Secret: a penguin"))
        self.assertFalse(nudged("It's raining again"))
        self.assertEqual(written_notes('ok note(text="a guitar")')[0]["function"]["arguments"], {"text": "a guitar"})
        self.assertEqual(without_written_pictures('Ready! note("the moon") 😘'), "Ready!  😘")

    def test_her_notes_keep_their_newest_lines(self):
        store = MemoryStore()
        chat = store.create_chat(store.create_companion("Luna", {}, "af_heart")["id"])
        for n in range(5):
            store.add_note(chat["id"], f"note {n} " + "x" * 20, 60)
        self.assertEqual(store.chat(chat["id"])["notes"].splitlines()[0][:6], "note 3")
        prompt = system_prompt({"name": "Luna", "persona": dict(DEFAULT_PERSONA)}, "Pranjall", NOW, notes="Secret: a penguin")
        self.assertTrue(prompt.endswith("- Secret: a penguin"))

    def test_she_knows_how_to_react(self):
        prompt = system_prompt({"name": "Luna", "persona": dict(DEFAULT_PERSONA)}, "Pranjall", NOW)
        self.assertIn("calling react", prompt)
        self.assertIn("😂", prompt)

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


class VoiceDeviceTests(unittest.TestCase):
    def test_her_voice_runs_on_the_gpu_and_falls_back_to_the_cpu(self):
        import sys
        import types
        from dataclasses import replace

        from app.voice import Voice

        made = []

        class WhisperModel:
            def __init__(self, name, device, compute_type, **kwargs):
                made.append((device, compute_type))
                if device == "cuda" and broken:
                    raise RuntimeError("CUDA failed with error no CUDA-capable device is detected")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = WhisperModel
        with mock.patch.dict(sys.modules, {"faster_whisper": fake}):
            broken = False
            voice = Voice(replace(config(), voice_device="cuda"))
            voice._load_whisper()
            self.assertEqual((made[-1], voice.whisper_device), (("cuda", "int8_float16"), "GPU (int8/float16)"))
            broken = True
            voice = Voice(replace(config(), voice_device="cuda"))
            voice._load_whisper()
            self.assertEqual((made[-1], voice.whisper_device), (("cpu", "int8"), "CPU (int8)"))
            voice = Voice(replace(config(), voice_device="cpu"))
            self.assertEqual(voice.hearing.state(), (False, "loads at start (CPU); stays loaded"))
            voice._load_whisper()
            self.assertEqual(made[-1], ("cpu", "int8"))


class DownloadTests(unittest.TestCase):
    def test_a_download_is_named_for_who_sent_it_and_when(self):
        from app.web import download_name
        when = datetime(2026, 11, 11, 16, 15, tzinfo=timezone.utc)
        self.assertRegex(download_name({"id": 4, "kind": "generated", "companion_name": "Luna Rose", "created_at": when}, ".png"),
                         r"^luna-rose-2026-11-1\d-\d{4}\.png$")
        self.assertRegex(download_name({"id": 5, "kind": "photo", "created_at": when}, ".jpg"), r"^photo-2026-")
        self.assertEqual(download_name({"id": 6, "kind": "generated"}, ".png"), "picture-6.png")
