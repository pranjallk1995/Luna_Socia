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
CREATE INDEX IF NOT EXISTS memory_sources_unread ON memory_sources (companion_id, id) WHERE facts IS NULL;
"""

COMPANION_COLUMNS = """c.*, stats.chats, stats.last_active, mem.sources, mem.read"""
COMPANION_STATS = """
LEFT JOIN LATERAL (SELECT count(*) AS chats, max(updated_at) AS last_active FROM chats
                   WHERE companion_id = c.id) stats ON true
LEFT JOIN LATERAL (SELECT count(*) AS sources, count(facts) AS read FROM memory_sources
                   WHERE companion_id = c.id) mem ON true"""

MESSAGE_COLUMNS = """m.id, m.chat_id, m.role, m.content, m.image_id, m.photo_note, m.created_at,
                     i.kind AS image_kind, i.prompt AS image_prompt"""


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
                    photo_note: str | None = None) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "INSERT INTO messages (chat_id, role, content, image_id, photo_note) VALUES (%s, %s, %s, %s, %s) "
                "RETURNING id", (chat_id, role, content, image_id, photo_note)).fetchone()
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

    def last_photo(self, chat_id: int) -> dict[str, Any] | None:
        """The last photo sent to Luna in this chat, which she can edit or step into."""
        with self.connect() as db:
            return db.execute("SELECT * FROM images WHERE chat_id = %s AND kind = 'photo' ORDER BY id DESC LIMIT 1",
                              (chat_id,)).fetchone()
