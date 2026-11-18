"""The girlfriends and every conversation with them, kept in Socia's own database (lunasocia).

Each girlfriend (a companion) has her own persona, voice, memory and chats. Every chat is
kept, with all its messages, so her memory can draw on them. Chats imported from ChatGPT or
Claude are kept too: she reads them for what to remember, and they are the training data
for her own fine-tuned model later. Image files live on the data volume; the database keeps
their names, their chat and how they came about.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS images (
    id BIGSERIAL PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('photo', 'generated')),
    file TEXT NOT NULL,
    prompt TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    role TEXT NOT NULL CHECK (role IN ('user', 'luna')),
    content TEXT NOT NULL DEFAULT '',
    image_id BIGINT REFERENCES images(id) ON DELETE SET NULL,
    -- What the vision model saw in a photo sent to Luna, which is how she "sees" it.
    photo_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE images ADD COLUMN IF NOT EXISTS chat_id BIGINT REFERENCES chats(id) ON DELETE CASCADE;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS chat_id BIGINT REFERENCES chats(id) ON DELETE CASCADE;
-- Before chats, there was one continuing conversation: it becomes the first chat.
DO $$
DECLARE earlier BIGINT;
BEGIN
    IF EXISTS (SELECT 1 FROM messages WHERE chat_id IS NULL) OR EXISTS (SELECT 1 FROM images WHERE chat_id IS NULL) THEN
        INSERT INTO chats (title, created_at, updated_at) VALUES (
            COALESCE((SELECT left(content, 60) FROM messages
                      WHERE chat_id IS NULL AND role = 'user' AND content <> '' ORDER BY id LIMIT 1),
                     'Earlier conversation'),
            COALESCE((SELECT min(created_at) FROM messages WHERE chat_id IS NULL), now()),
            COALESCE((SELECT max(created_at) FROM messages WHERE chat_id IS NULL), now()))
        RETURNING id INTO earlier;
        UPDATE messages SET chat_id = earlier WHERE chat_id IS NULL;
        UPDATE images SET chat_id = earlier WHERE chat_id IS NULL;
    END IF;
END $$;
ALTER TABLE images ALTER COLUMN chat_id SET NOT NULL;
ALTER TABLE messages ALTER COLUMN chat_id SET NOT NULL;
DROP INDEX IF EXISTS messages_created;
CREATE INDEX IF NOT EXISTS messages_chat ON messages (chat_id, id DESC);
CREATE INDEX IF NOT EXISTS images_chat ON images (chat_id, id DESC);
CREATE INDEX IF NOT EXISTS chats_updated ON chats (updated_at DESC);

CREATE TABLE IF NOT EXISTS companions (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    -- How she texts: style, reply length, emoji, language and what the owner wrote about her.
    persona JSONB NOT NULL DEFAULT '{}',
    voice TEXT NOT NULL DEFAULT 'af_heart',
    -- What she remembers about the owner, distilled from imported chats; the owner can edit it.
    memory TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE chats ADD COLUMN IF NOT EXISTS companion_id BIGINT REFERENCES companions(id) ON DELETE CASCADE;
-- Before there could be several, every chat was with Luna: she becomes the first girlfriend.
DO $$
DECLARE luna BIGINT;
BEGIN
    IF EXISTS (SELECT 1 FROM chats WHERE companion_id IS NULL) THEN
        SELECT id INTO luna FROM companions ORDER BY id LIMIT 1;
        IF luna IS NULL THEN
            INSERT INTO companions (name, created_at) VALUES ('Luna', (SELECT min(created_at) FROM chats))
            RETURNING id INTO luna;
        END IF;
        UPDATE chats SET companion_id = luna WHERE companion_id IS NULL;
    END IF;
END $$;
ALTER TABLE chats ALTER COLUMN companion_id DROP DEFAULT;
ALTER TABLE chats ALTER COLUMN companion_id SET NOT NULL;
-- Who named a chat: '' not yet (its first message stands in), 'auto' the model, 'owner' by hand.
ALTER TABLE chats ADD COLUMN IF NOT EXISTS named TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS chats_companion ON chats (companion_id, updated_at DESC);

-- Conversations imported from ChatGPT or Claude: read once for facts, kept for training.
CREATE TABLE IF NOT EXISTS memory_sources (
    id BIGSERIAL PRIMARY KEY,
    companion_id BIGINT NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('chatgpt', 'claude')),
    external_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    messages JSONB NOT NULL,
    started_at TIMESTAMPTZ,
    -- What the model found in it; NULL until read. merged: folded into the companion's memory.
    facts TEXT,
    merged BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (companion_id, source, external_id)
);
-- What she wrote down to remember in a chat (a secret she picked for a game, the score, a plan); only she reads it.
ALTER TABLE chats ADD COLUMN IF NOT EXISTS notes TEXT NOT NULL DEFAULT '';
-- An emoji the other one put on this message: the owner on hers, she on the owner's.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS reaction TEXT;
CREATE INDEX IF NOT EXISTS memory_sources_unread ON memory_sources (companion_id, id) WHERE facts IS NULL;
-- The Luna apps she used for a reply: [{"app": "vibia", "tool": "play_tracks"}, ...].
ALTER TABLE messages ADD COLUMN IF NOT EXISTS apps JSONB;
-- When the owner last changed the words of one of their messages; she reads the new words.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS edited_at TIMESTAMPTZ;
-- Socia's sign-in to the owner's Luna apps: its OAuth client at Sentia's sign-on and the
-- tokens the owner approved. One row; no refresh token means not connected.
CREATE TABLE IF NOT EXISTS apps_connection (
    singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    client_id TEXT NOT NULL,
    client_secret TEXT,
    redirect_uri TEXT NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    access_expires DOUBLE PRECISION,
    connected_at TIMESTAMPTZ
);
-- Her own chats are read for lasting facts too, once quiet: 'socia' sources, and each chat's last message read.
ALTER TABLE memory_sources DROP CONSTRAINT IF EXISTS memory_sources_source_check;
ALTER TABLE memory_sources ADD CONSTRAINT memory_sources_source_check CHECK (source IN ('chatgpt', 'claude', 'socia'));
ALTER TABLE chats ADD COLUMN IF NOT EXISTS remembered_to BIGINT NOT NULL DEFAULT 0;
-- The owner as every girlfriend knows them: the name she calls them ('' for their sign-in's first
-- name) and what they wrote about themselves. One row.
CREATE TABLE IF NOT EXISTS owner_profile (
    singleton BOOLEAN PRIMARY KEY DEFAULT true CHECK (singleton),
    name TEXT NOT NULL DEFAULT '',
    about TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

COMPANION_COLUMNS = """c.*, stats.chats, stats.last_active, mem.sources, mem.read"""
COMPANION_STATS = """
LEFT JOIN LATERAL (SELECT count(*) AS chats, max(updated_at) AS last_active FROM chats
                   WHERE companion_id = c.id) stats ON true
LEFT JOIN LATERAL (SELECT count(*) AS sources, count(facts) AS read FROM memory_sources
                   WHERE companion_id = c.id AND source <> 'socia') mem ON true"""

MESSAGE_COLUMNS = """m.id, m.chat_id, m.role, m.content, m.image_id, m.photo_note, m.reaction, m.apps, m.created_at, m.edited_at,
                     i.kind AS image_kind, i.prompt AS image_prompt, i.file AS image_file"""


class SociaStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row, autocommit=True)

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute(SCHEMA)

    # Girlfriends

    def create_companion(self, name: str, persona: dict[str, Any], voice: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("INSERT INTO companions (name, persona, voice) VALUES (%s, %s, %s) RETURNING id",
                             (name, Jsonb(persona), voice)).fetchone()
        return self.companion(row["id"])

    def companion(self, companion_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute(f"SELECT {COMPANION_COLUMNS} FROM companions c {COMPANION_STATS} WHERE c.id = %s",
                              (companion_id,)).fetchone()

    def companions(self) -> list[dict[str, Any]]:
        """Every girlfriend, the one talked to most recently first."""
        with self.connect() as db:
            return db.execute(f"SELECT {COMPANION_COLUMNS} FROM companions c {COMPANION_STATS} "
                              "ORDER BY COALESCE(stats.last_active, c.created_at) DESC, c.id DESC").fetchall()

    def update_companion(self, companion_id: int, fields: dict[str, Any]) -> dict[str, Any] | None:
        """Change her name, persona, voice or memory; fields holds only what changes."""
        allowed = {key: Jsonb(value) if key == "persona" else value for key, value in fields.items()
                   if key in ("name", "persona", "voice", "memory")}
        if allowed:
            sets = ", ".join(f"{key} = %({key})s" for key in allowed)
            with self.connect() as db:
                db.execute(f"UPDATE companions SET {sets}, updated_at = now() WHERE id = %(id)s",
                           {**allowed, "id": companion_id})
        return self.companion(companion_id)

    def delete_companion(self, companion_id: int) -> list[str] | None:
        """Forget her and all her chats; returns their image files, or None if there is no such companion."""
        with self.connect() as db:
            files = [row["file"] for row in db.execute(
                "SELECT i.file FROM images i JOIN chats c ON c.id = i.chat_id WHERE c.companion_id = %s",
                (companion_id,)).fetchall()]
            deleted = db.execute("DELETE FROM companions WHERE id = %s RETURNING id", (companion_id,)).fetchone()
        return files if deleted else None

    def oldest_companion(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT id FROM companions ORDER BY id LIMIT 1").fetchone()
        return self.companion(row["id"]) if row else None

    # What she remembers: imported chats, read one at a time for facts.

    def add_memory_sources(self, companion_id: int, source: str, conversations: list[dict[str, Any]]) -> int:
        """Keep imported conversations; ones imported before are skipped. Returns how many are new."""
        added = 0
        with self.connect() as db:
            for conversation in conversations:
                row = db.execute(
                    "INSERT INTO memory_sources (companion_id, source, external_id, title, messages, started_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING id",
                    (companion_id, source, conversation["id"], conversation["title"],
                     Jsonb(conversation["messages"]), conversation.get("started_at"))).fetchone()
                added += row is not None
        return added

    def quiet_chats(self, quiet_seconds: float) -> list[dict[str, Any]]:
        """Chats with owner messages not yet read for her memory, quiet for ``quiet_seconds``, oldest first."""
        with self.connect() as db:
            return db.execute(
                """SELECT c.id, c.companion_id, c.title, c.remembered_to FROM chats c
                   WHERE EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = c.id AND m.role = 'user'
                                 AND m.id > c.remembered_to)
                     AND (SELECT max(created_at) FROM messages m WHERE m.chat_id = c.id)
                         < now() - make_interval(secs => %s)
                   ORDER BY c.updated_at""", (quiet_seconds,)).fetchall()

    def chat_source(self, chat: dict[str, Any], limit: int) -> int:
        """The owner's messages after the last ones read, with what their photos showed, kept as a 'socia' source
        to read for facts; the chat is marked read up to its last message. Returns how many messages."""
        with self.connect() as db, db.transaction():
            rows = db.execute(
                "SELECT id, role, content, photo_note, created_at FROM messages WHERE chat_id = %s AND id > %s "
                "ORDER BY id LIMIT %s", (chat["id"], chat["remembered_to"], limit)).fetchall()
            if not rows:
                return 0
            # Both sides: the owner's words (with what their photos showed) and hers, for what she says of herself.
            messages = [{"role": row["role"], "text": " ".join(part for part in [
                row["content"] or "", f"[with a photo: {row['photo_note']}]" if row["photo_note"] else ""] if part)}
                for row in rows if row["content"] or row["photo_note"]]
            last = rows[-1]
            if messages:
                db.execute(
                    "INSERT INTO memory_sources (companion_id, source, external_id, title, messages, started_at) "
                    "VALUES (%s, 'socia', %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (chat["companion_id"], f"chat-{chat['id']}-{last['id']}", chat["title"] or "",
                     Jsonb(messages), last["created_at"]))
            db.execute("UPDATE chats SET remembered_to = %s WHERE id = %s", (last["id"], chat["id"]))
            return len(messages)

    def remember(self, companion_id: int, fact: str, limit: int) -> bool:
        """A fact she chose to remember, added to her memory unless she has it; the oldest lines go past
        ``limit``. False when she already had it."""
        line = "- " + fact.removeprefix("- ").strip()
        with self.connect() as db, db.transaction():
            row = db.execute("SELECT memory FROM companions WHERE id = %s FOR UPDATE", (companion_id,)).fetchone()
            if row is None:
                return False
            lines = [l for l in row["memory"].splitlines() if l.strip()]
            if line.casefold() in (l.casefold() for l in lines):
                return False
            lines.append(line)
            while len(lines) > 1 and len("\n".join(lines)) > limit:
                lines.pop(0)
            db.execute("UPDATE companions SET memory = %s, updated_at = now() WHERE id = %s",
                       ("\n".join(lines), companion_id))
            return True

    def add_to_bio(self, companion_id: int, facts: list[str], limit: int) -> list[str]:
        """What she said about herself, added to her bio (persona "about") as sentences, unless already there and
        while it stays under ``limit``; the sentences added."""
        with self.connect() as db, db.transaction():
            row = db.execute("SELECT persona FROM companions WHERE id = %s FOR UPDATE", (companion_id,)).fetchone()
            if row is None:
                return []
            about = str((row["persona"] or {}).get("about") or "").strip()
            added = []
            for fact in facts:
                sentence = fact.removeprefix("- ").strip()
                if not sentence:
                    continue
                sentence = sentence[0].upper() + sentence[1:]
                if not sentence.endswith((".", "!", "?")):
                    sentence += "."
                if sentence.casefold() in about.casefold() or len(about) + len(sentence) + 1 > limit:
                    continue
                about = f"{about}\n{sentence}".strip()
                added.append(sentence)
            if added:
                db.execute("UPDATE companions SET persona = persona || jsonb_build_object('about', %s::text), "
                           "updated_at = now() WHERE id = %s", (about, companion_id))
            return added

    def unread_source(self) -> dict[str, Any] | None:
        """The next imported conversation to read, of any girlfriend."""
        with self.connect() as db:
            return db.execute("SELECT * FROM memory_sources WHERE facts IS NULL ORDER BY id LIMIT 1").fetchone()

    def set_source_facts(self, source_id: int, facts: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE memory_sources SET facts = %s WHERE id = %s", (facts, source_id))

    def unmerged_facts(self, companion_id: int) -> list[dict[str, Any]]:
        """Facts read from her imported chats but not yet in her memory, oldest chat first."""
        with self.connect() as db:
            return db.execute("SELECT id, facts FROM memory_sources WHERE companion_id = %s AND facts IS NOT NULL "
                              "AND NOT merged ORDER BY started_at NULLS FIRST, id", (companion_id,)).fetchall()

    def merge_memory(self, companion_id: int, memory: str, source_ids: list[int]) -> None:
        """Her new memory, with the chats it was made from marked as merged, in one go."""
        with self.connect() as db, db.transaction():
            db.execute("UPDATE companions SET memory = %s, updated_at = now() WHERE id = %s", (memory, companion_id))
            db.execute("UPDATE memory_sources SET merged = true WHERE id = ANY(%s)", (source_ids,))

    def forget_sources(self, companion_id: int) -> int:
        """Delete her imported chats; what she remembers from them stays until edited."""
        with self.connect() as db:
            return db.execute("DELETE FROM memory_sources WHERE companion_id = %s", (companion_id,)).rowcount

    # Chats

    def create_chat(self, companion_id: int, title: str = "") -> dict[str, Any]:
        with self.connect() as db:
            return db.execute("INSERT INTO chats (companion_id, title) VALUES (%s, %s) RETURNING *",
                              (companion_id, title)).fetchone()

    def chat(self, chat_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM chats WHERE id = %s", (chat_id,)).fetchone()

    def chats(self, companion_id: int) -> list[dict[str, Any]]:
        """Her chats, the most recently active first, with its last message."""
        with self.connect() as db:
            return db.execute(
                "SELECT c.*, last.role AS last_role, last.content AS last_content, last.image_id AS last_image_id "
                "FROM chats c LEFT JOIN LATERAL (SELECT role, content, image_id FROM messages "
                "WHERE chat_id = c.id ORDER BY id DESC LIMIT 1) last ON true "
                "WHERE c.companion_id = %s ORDER BY c.updated_at DESC, c.id DESC", (companion_id,)).fetchall()

    def rename_chat(self, chat_id: int, title: str) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute("UPDATE chats SET title = %s, named = 'owner' WHERE id = %s RETURNING *",
                              (title, chat_id)).fetchone()

    def name_chat(self, chat_id: int, title: str) -> dict[str, Any] | None:
        """The model's name for a chat; never over one the owner chose."""
        with self.connect() as db:
            return db.execute("UPDATE chats SET title = %s, named = 'auto' WHERE id = %s AND named <> 'owner' RETURNING *",
                              (title, chat_id)).fetchone()

    def keep_name(self, chat_id: int) -> None:
        """A name the owner gave before chats were named by the model."""
        with self.connect() as db:
            db.execute("UPDATE chats SET named = 'owner' WHERE id = %s", (chat_id,))

    def unnamed_chats(self, limit: int = 20) -> list[dict[str, Any]]:
        """Chats not named yet that she has answered in, oldest first."""
        with self.connect() as db:
            return db.execute(
                "SELECT c.* FROM chats c WHERE c.named = '' AND EXISTS "
                "(SELECT 1 FROM messages m WHERE m.chat_id = c.id AND m.role = 'luna') ORDER BY c.id LIMIT %s",
                (limit,)).fetchall()

    def opening(self, chat_id: int, count: int = 6) -> list[dict[str, Any]]:
        """A chat's first messages, to name it by."""
        with self.connect() as db:
            return db.execute("SELECT role, content, image_id FROM messages WHERE chat_id = %s ORDER BY id LIMIT %s",
                              (chat_id, count)).fetchall()

    def add_note(self, chat_id: int, note: str, limit: int) -> str:
        """Add a line to her notes for a chat, keeping the last ``limit`` characters' worth of lines; her notes."""
        with self.connect() as db:
            row = db.execute("SELECT notes FROM chats WHERE id = %s FOR UPDATE", (chat_id,)).fetchone()
            lines = [*(row["notes"].splitlines() if row else []), note]
            while len(lines) > 1 and len("\n".join(lines)) > limit:
                lines.pop(0)
            notes = "\n".join(lines)
            db.execute("UPDATE chats SET notes = %s WHERE id = %s", (notes, chat_id))
        return notes

    def title_chat(self, chat_id: int, title: str) -> None:
        """Name a chat after its first message, unless it has a name already."""
        with self.connect() as db:
            db.execute("UPDATE chats SET title = %s WHERE id = %s AND title = ''", (title, chat_id))

    def delete_chat(self, chat_id: int) -> list[str] | None:
        """Forget one chat; returns its image files to delete, or None if there is no such chat."""
        with self.connect() as db:
            files = [row["file"] for row in db.execute("SELECT file FROM images WHERE chat_id = %s",
                                                        (chat_id,)).fetchall()]
            deleted = db.execute("DELETE FROM chats WHERE id = %s RETURNING id", (chat_id,)).fetchone()
        return files if deleted else None

    # Messages and images

    def add_image(self, chat_id: int, kind: str, file: str, prompt: str | None = None) -> int:
        with self.connect() as db:
            row = db.execute("INSERT INTO images (chat_id, kind, file, prompt) VALUES (%s, %s, %s, %s) RETURNING id",
                             (chat_id, kind, file, prompt)).fetchone()
        return row["id"]

    def image(self, image_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            # With the name of the girlfriend whose chat it is in, for naming a download.
            return db.execute("SELECT i.*, co.name AS companion_name FROM images i "
                              "LEFT JOIN chats c ON c.id = i.chat_id LEFT JOIN companions co ON co.id = c.companion_id "
                              "WHERE i.id = %s", (image_id,)).fetchone()

    def add_message(self, chat_id: int, role: str, content: str, image_id: int | None = None,
                    photo_note: str | None = None, apps: list[dict[str, str]] | None = None) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "INSERT INTO messages (chat_id, role, content, image_id, photo_note, apps) VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING id", (chat_id, role, content, image_id, photo_note, Jsonb(apps) if apps else None)).fetchone()
            db.execute("UPDATE chats SET updated_at = now() WHERE id = %s", (chat_id,))
            return db.execute(f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                              "WHERE m.id = %s", (row["id"],)).fetchone()

    def set_photo_note(self, message_id: int, note: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE messages SET photo_note = %s WHERE id = %s", (note, message_id))

    def recent(self, chat_id: int, limit: int, before: int | None = None) -> list[dict[str, Any]]:
        """A chat's latest messages, oldest first; ``before`` pages further back."""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                "WHERE m.chat_id = %(chat)s AND (%(before)s::bigint IS NULL OR m.id < %(before)s) "
                "ORDER BY m.id DESC LIMIT %(limit)s",
                {"chat": chat_id, "before": before, "limit": limit}).fetchall()
        return list(reversed(rows))

    def message(self, message_id: int) -> dict[str, Any] | None:
        """A message, with whose chat it is (companion_id) and her voice."""
        with self.connect() as db:
            return db.execute("SELECT m.*, c.companion_id, g.voice FROM messages m JOIN chats c ON c.id = m.chat_id "
                              "JOIN companions g ON g.id = c.companion_id WHERE m.id = %s", (message_id,)).fetchone()

    def delete_message(self, message_id: int) -> list[str] | None:
        """Forget one message, and its photo or picture; returns the image files to delete,
        or None if there is no such message. Luna no longer sees it either."""
        with self.connect() as db, db.transaction():
            row = db.execute("DELETE FROM messages WHERE id = %s RETURNING image_id", (message_id,)).fetchone()
            if row is None:
                return None
            if row["image_id"] is None:
                return []
            image = db.execute(
                "DELETE FROM images WHERE id = %s AND NOT EXISTS (SELECT 1 FROM messages WHERE image_id = %s) "
                "RETURNING file", (row["image_id"], row["image_id"])).fetchone()
        return [image["file"]] if image else []

    def edit_message(self, message_id: int, content: str) -> dict[str, Any] | None:
        """New words for a message, marked edited; the message, or None if there is none."""
        with self.connect() as db:
            row = db.execute("UPDATE messages SET content = %s, edited_at = now() WHERE id = %s RETURNING id",
                             (content, message_id)).fetchone()
            if row is None:
                return None
            return db.execute(f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                              "WHERE m.id = %s", (message_id,)).fetchone()

    def set_reaction(self, message_id: int, reaction: str | None) -> dict[str, Any] | None:
        """Put a reaction on a message, or take it off with None; the message, or None if there is none."""
        with self.connect() as db:
            row = db.execute("UPDATE messages SET reaction = %s WHERE id = %s RETURNING id", (reaction, message_id)).fetchone()
            if row is None:
                return None
            return db.execute(f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                              "WHERE m.id = %s", (message_id,)).fetchone()

    def last_photo(self, chat_id: int) -> dict[str, Any] | None:
        """The last photo sent to Luna in this chat, which she can edit or step into."""
        with self.connect() as db:
            return db.execute("SELECT * FROM images WHERE chat_id = %s AND kind = 'photo' ORDER BY id DESC LIMIT 1",
                              (chat_id,)).fetchone()

    # The owner

    def owner_profile(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT name, about FROM owner_profile").fetchone()
        return row or {"name": "", "about": ""}

    def set_owner_profile(self, name: str, about: str) -> dict[str, Any]:
        with self.connect() as db:
            return db.execute(
                """INSERT INTO owner_profile (name, about) VALUES (%s, %s)
                   ON CONFLICT (singleton) DO UPDATE SET name = EXCLUDED.name, about = EXCLUDED.about,
                     updated_at = now()
                   RETURNING name, about""", (name, about)).fetchone()

    # The Luna apps

    def apps_connection(self) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM apps_connection").fetchone()

    def set_apps_client(self, client_id: str, client_secret: str | None, redirect_uri: str) -> dict[str, Any]:
        """A newly registered client; tokens of an earlier one no longer work with it."""
        with self.connect() as db:
            return db.execute(
                """INSERT INTO apps_connection (client_id, client_secret, redirect_uri) VALUES (%s, %s, %s)
                   ON CONFLICT (singleton) DO UPDATE SET client_id = EXCLUDED.client_id,
                     client_secret = EXCLUDED.client_secret, redirect_uri = EXCLUDED.redirect_uri,
                     access_token = NULL, refresh_token = NULL, access_expires = NULL, connected_at = NULL
                   RETURNING *""", (client_id, client_secret, redirect_uri)).fetchone()

    def set_apps_tokens(self, access: str | None, refresh: str | None, expires: float | None) -> None:
        with self.connect() as db:
            db.execute("""UPDATE apps_connection SET access_token = %s, refresh_token = %s, access_expires = %s,
                          connected_at = CASE WHEN %s::text IS NULL THEN NULL ELSE coalesce(connected_at, now()) END""",
                       (access, refresh, expires, refresh))
