"""
database/agent/conversations.py — ConversationsRepository.

Defines the conversations table schema (at minimum: id, chat_id, title,
created_at, updated_at). This provides a canonical home for conversation
metadata separate from chat history messages.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from database.base import BaseRepository


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationsRepository(BaseRepository):
    """SQLite-backed store for conversation metadata."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT    PRIMARY KEY,
                    chat_id     TEXT    NOT NULL DEFAULT '',
                    title       TEXT    NOT NULL DEFAULT '',
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_chat ON conversations(chat_id);
            """)
            conn.commit()

    # ── public API ───────────────────────────────────────────────────────────

    def upsert(self, conversation_id: str, chat_id: str = "",
               title: str = "") -> None:
        """Create or update a conversation record."""
        now = _utc_now()
        with self.connection() as conn:
            conn.execute("""
                INSERT INTO conversations (id, chat_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    chat_id    = excluded.chat_id,
                    title      = excluded.title,
                    updated_at = excluded.updated_at
            """, (conversation_id, chat_id, title, now, now))
            conn.commit()

    def get(self, conversation_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_by_chat(self, chat_id: str, limit: int = 50) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE chat_id = ? ORDER BY updated_at DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
