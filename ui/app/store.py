"""The conversation with Luna, kept in Socia's own database (lunasocia).

One continuing conversation, as on a phone. Image files live on the data volume; the
database keeps their names and how they came about.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
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
CREATE INDEX IF NOT EXISTS messages_created ON messages (id DESC);
"""

MESSAGE_COLUMNS = """m.id, m.role, m.content, m.image_id, m.photo_note, m.created_at,
                     i.kind AS image_kind, i.prompt AS image_prompt"""


class SociaStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row, autocommit=True)

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute(SCHEMA)

    def add_image(self, kind: str, file: str, prompt: str | None = None) -> int:
        with self.connect() as db:
            row = db.execute("INSERT INTO images (kind, file, prompt) VALUES (%s, %s, %s) RETURNING id",
                             (kind, file, prompt)).fetchone()
        return row["id"]

    def image(self, image_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM images WHERE id = %s", (image_id,)).fetchone()

    def add_message(self, role: str, content: str, image_id: int | None = None, photo_note: str | None = None) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                "INSERT INTO messages (role, content, image_id, photo_note) VALUES (%s, %s, %s, %s) RETURNING id",
                (role, content, image_id, photo_note)).fetchone()
            return db.execute(f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                              "WHERE m.id = %s", (row["id"],)).fetchone()

    def set_photo_note(self, message_id: int, note: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE messages SET photo_note = %s WHERE id = %s", (note, message_id))

    def recent(self, limit: int, before: int | None = None) -> list[dict[str, Any]]:
        """The latest messages, oldest first; ``before`` pages further back."""
        with self.connect() as db:
            rows = db.execute(
                f"SELECT {MESSAGE_COLUMNS} FROM messages m LEFT JOIN images i ON i.id = m.image_id "
                "WHERE (%(before)s::bigint IS NULL OR m.id < %(before)s) ORDER BY m.id DESC LIMIT %(limit)s",
                {"before": before, "limit": limit}).fetchall()
        return list(reversed(rows))

    def message(self, message_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return db.execute("SELECT * FROM messages WHERE id = %s", (message_id,)).fetchone()

    def last_photo(self) -> dict[str, Any] | None:
        """The last photo sent to Luna, which she can edit or step into."""
        with self.connect() as db:
            return db.execute("SELECT * FROM images WHERE kind = 'photo' ORDER BY id DESC LIMIT 1").fetchone()

    def clear(self) -> list[str]:
        """Forget the whole conversation; returns the image files to delete."""
        with self.connect() as db:
            files = [row["file"] for row in db.execute("SELECT file FROM images").fetchall()]
            db.execute("TRUNCATE messages, images RESTART IDENTITY")
        return files
