"""
database/memory/chat_history.py — ChatHistoryRepository.

Owns sessions + messages + attachments schema from memory.py.
Exposes: save_turn(), get_history(), get_recent_user_messages(), clear_session(),
         record_attachment(), recent_attachments().
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database.base import BaseRepository

BOT_NAME = "kernel-evolving"


class ChatHistoryRepository(BaseRepository):
    """SQLite-backed repository for chat history (sessions, messages, attachments)."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id                  TEXT PRIMARY KEY,
                    chat_id             TEXT    NOT NULL DEFAULT '',
                    created_at          TEXT    NOT NULL,
                    updated_at          TEXT    NOT NULL,
                    message_count       INTEGER NOT NULL DEFAULT 0,
                    status              TEXT    NOT NULL DEFAULT 'active',
                    session_number      INTEGER,
                    previous_session_id TEXT,
                    summary             TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot        TEXT    NOT NULL DEFAULT 'kernel-evolving',
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
                    content    TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_bot_session ON messages(bot, session_id, id);
                CREATE INDEX IF NOT EXISTS idx_bot_created ON messages(bot, created_at);

                CREATE TABLE IF NOT EXISTS attachments (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot           TEXT    NOT NULL DEFAULT 'kernel-evolving',
                    session_id    TEXT    NOT NULL,
                    chat_id       TEXT    NOT NULL DEFAULT '',
                    kind          TEXT    NOT NULL CHECK(kind IN ('document','photo','voice','audio','video','other')),
                    local_path    TEXT    NOT NULL,
                    original_name TEXT    NOT NULL DEFAULT '',
                    mime_type     TEXT    NOT NULL DEFAULT '',
                    caption       TEXT    NOT NULL DEFAULT '',
                    created_at    TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_att_bot_session ON attachments(bot, session_id, id);
                CREATE INDEX IF NOT EXISTS idx_att_bot_created ON attachments(bot, created_at);
                CREATE INDEX IF NOT EXISTS idx_att_chat ON attachments(chat_id, created_at);
            """)
            # Add new columns to existing DBs (idempotent — ignore if already present)
            for col, definition in [
                ("status",              "TEXT NOT NULL DEFAULT 'active'"),
                ("session_number",      "INTEGER"),
                ("previous_session_id", "TEXT"),
                ("summary",             "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {definition}")
                except Exception:
                    pass  # column already exists
            conn.commit()


    # ── session helpers ───────────────────────────────────────────────────────

    def touch_session(self, session_id: str, chat_id: str) -> None:
        """Create session row if missing, then update updated_at."""
        ts = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, chat_id, created_at, updated_at, message_count) "
                "VALUES (?, ?, ?, ?, 0)",
                (session_id, chat_id or "", ts, ts),
            )
            conn.execute(
                "UPDATE sessions SET chat_id=?, updated_at=? WHERE id=?",
                (chat_id or "", ts, session_id),
            )
            conn.commit()

    def get_active_session(self, chat_id: str) -> Optional[dict]:
        """Return the active session row for a chat_id, or None if none exists."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT id, session_number, message_count, previous_session_id, summary "
                "FROM sessions WHERE chat_id=? AND (status='active' OR status IS NULL) "
                "ORDER BY COALESCE(session_number, 0) DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if row:
                return dict(row)
            # Legacy fallback: session whose id = chat_id (old single-session schema)
            row = conn.execute(
                "SELECT id, session_number, message_count, previous_session_id, summary "
                "FROM sessions WHERE id=?",
                (chat_id,),
            ).fetchone()
            return dict(row) if row else None

    def rotate_session(self, chat_id: str, new_session_id: str,
                       summary: str = "", bot: str = BOT_NAME) -> dict:
        """Close the current active session and open a new one.

        Returns a dict with prev_session_id and new session_number.
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            # Find current active session
            prev = conn.execute(
                "SELECT id, COALESCE(session_number, 1) AS snum FROM sessions "
                "WHERE chat_id=? AND (status='active' OR status IS NULL) "
                "ORDER BY COALESCE(session_number, 0) DESC LIMIT 1",
                (chat_id,),
            ).fetchone()
            if not prev:
                # Legacy: session id = chat_id
                prev = conn.execute(
                    "SELECT id, 1 AS snum FROM sessions WHERE id=?", (chat_id,)
                ).fetchone()
            prev_id = prev["id"] if prev else None
            prev_num = prev["snum"] if prev else 0
            new_num = prev_num + 1

            # Close previous session
            if prev_id:
                conn.execute(
                    "UPDATE sessions SET status='closed', updated_at=? "
                    + (", summary=?" if summary else "") + " WHERE id=?",
                    ([ts, summary, prev_id] if summary else [ts, prev_id]),
                )
            # Open new session
            conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(id, chat_id, created_at, updated_at, message_count, status, session_number, previous_session_id) "
                "VALUES (?, ?, ?, ?, 0, 'active', ?, ?)",
                (new_session_id, chat_id, ts, ts, new_num, prev_id),
            )
            conn.commit()
        return {"prev_session_id": prev_id, "new_session_id": new_session_id, "session_number": new_num}

    def get_history_by_session(self, session_id: str, bot: str = BOT_NAME) -> list[dict]:
        """Return messages for a specific session UUID (for recall_memory cross-session queries)."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE bot=? AND session_id=? ORDER BY id ASC",
                (bot, session_id),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ── message helpers ───────────────────────────────────────────────────────

    def append_messages(self, messages: list[dict], session_id: str,
                        bot: str = BOT_NAME) -> None:
        """Append only NEW messages (idempotent — compares against existing count)."""
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE bot=? AND session_id=?",
                (bot, session_id),
            ).fetchone()[0]
            new_msgs = messages[existing:]
            ts = datetime.now(timezone.utc).isoformat()
            for m in new_msgs:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                conn.execute(
                    "INSERT INTO messages (bot, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
                    (bot, session_id, m["role"], content, ts),
                )
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE bot=? AND session_id=?",
                (bot, session_id),
            ).fetchone()[0]
            conn.execute(
                "UPDATE sessions SET message_count=?, updated_at=? WHERE id=?",
                (count, ts, session_id),
            )
            conn.commit()

    def get_history(self, session_id: str, bot: str = BOT_NAME) -> list[dict]:
        """Return all messages for a session ordered by id ASC."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE bot=? AND session_id=? ORDER BY id ASC",
                (bot, session_id),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_history_by_chat_id(self, chat_id: str, bot: str = BOT_NAME,
                                 legacy_session_hint: str = "") -> list[dict]:
        """Return all messages for a Telegram chat across ALL sessions (current + legacy).

        Chat IDs can span multiple session rows over time — e.g. when the session
        UUID changes or the session file is lost.  This method queries all sessions
        that share the given chat_id, plus a legacy session hint (old UUID from the
        session file) for pre-migration turns.

        Returns messages ordered by id ASC, deduplicated by (role, content).
        """
        with self.connection() as conn:
            session_ids = set()

            # Direct match: sessions with this chat_id
            for row in conn.execute(
                "SELECT id FROM sessions WHERE chat_id=?",
                (chat_id,),
            ).fetchall():
                session_ids.add(row["id"])

            # Legacy session: the pre-migration UUID session (empty chat_id only).
            # Guard: only merge if the session row has no chat_id — prevents the hint
            # from leaking one chat's history into an unrelated chat_id query.
            if legacy_session_hint:
                for row in conn.execute(
                    "SELECT id FROM sessions WHERE id=? AND (chat_id IS NULL OR chat_id='')",
                    (legacy_session_hint,),
                ).fetchall():
                    session_ids.add(row["id"])

            if not session_ids:
                return []

            # Query messages from all matching sessions, ordered by id
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT role, content FROM messages WHERE bot=? AND session_id IN ({placeholders}) ORDER BY id ASC",
                [bot] + list(session_ids),
            ).fetchall()

            # Deduplicate by (role, content substr) — overlapping sessions can produce
            # duplicate turns from the same conversation
            seen = set()
            deduped = []
            for r in rows:
                key = (r["role"], str(r["content"])[:80])
                if key not in seen:
                    seen.add(key)
                    deduped.append({"role": r["role"], "content": r["content"]})
            return deduped

    def get_recent_user_messages(self, bot: str = BOT_NAME,
                                  chat_id: str = "", limit: int = 100) -> list[dict]:
        """Return most recent messages, optionally filtered by chat_id/session_id."""
        with self.connection() as conn:
            if chat_id:
                rows = conn.execute(
                    "SELECT role, content, session_id, created_at FROM messages "
                    "WHERE bot=? AND session_id=? ORDER BY id DESC LIMIT ?",
                    (bot, chat_id, limit * 2),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, session_id, created_at FROM messages "
                    "WHERE bot=? ORDER BY id DESC LIMIT ?",
                    (bot, limit * 2),
                ).fetchall()
        return [
            {"role": r["role"], "content": r["content"],
             "session_id": r["session_id"], "created_at": r["created_at"]}
            for r in reversed(rows)
        ]

    def clear_session(self, session_id: str, bot: str = BOT_NAME) -> None:
        """Delete all messages and session row for a given session."""
        with self.connection() as conn:
            conn.execute("DELETE FROM messages WHERE bot=? AND session_id=?", (bot, session_id))
            conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()

    def clear_all_for_chat(self, chat_id: str, bot: str = BOT_NAME) -> int:
        """Delete ALL messages and session rows for a chat_id (across all session UUIDs).

        Returns the number of messages deleted. Safe to call even if no rows exist.
        Required for /fresh and for session-splitting's clear path.
        """
        with self.connection() as conn:
            # Find all session IDs that belong to this chat
            rows = conn.execute(
                "SELECT id FROM sessions WHERE chat_id=?", (chat_id,)
            ).fetchall()
            session_ids = [r["id"] for r in rows]
            # Also include legacy: session row where id = chat_id (current single-session schema)
            if chat_id and chat_id not in session_ids:
                if conn.execute("SELECT 1 FROM sessions WHERE id=?", (chat_id,)).fetchone():
                    session_ids.append(chat_id)
            if not session_ids:
                return 0
            placeholders = ",".join("?" for _ in session_ids)
            cur = conn.execute(
                f"DELETE FROM messages WHERE bot=? AND session_id IN ({placeholders})",
                [bot] + session_ids,
            )
            deleted = cur.rowcount
            conn.execute(
                f"DELETE FROM sessions WHERE id IN ({placeholders})",
                session_ids,
            )
            conn.commit()
            return deleted

    def clear_attachments_by_chat(self, chat_id: str, bot: str = BOT_NAME) -> int:
        """Delete attachment records for a specific chat_id. Returns rows deleted."""
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM attachments WHERE bot=? AND chat_id=?", (bot, chat_id)
            )
            conn.commit()
            return cur.rowcount

    def clear_all(self, bot: str = BOT_NAME) -> None:
        """Wipe all messages and sessions for this bot (destructive!)."""
        with self.connection() as conn:
            conn.execute("DELETE FROM messages WHERE bot=?", (bot,))
            conn.execute("DELETE FROM sessions")
            conn.commit()

    # ── attachment helpers ────────────────────────────────────────────────────

    def record_attachment(self, *, kind: str, local_path: str,
                           session_id: str, chat_id: str = "",
                           original_name: str = "", mime_type: str = "",
                           caption: str = "", bot: str = BOT_NAME) -> Optional[int]:
        """Persist one attachment record. Returns new row id or None on error."""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as conn:
                cur = conn.execute(
                    "INSERT INTO attachments "
                    "(bot, session_id, chat_id, kind, local_path, original_name, mime_type, caption, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (bot, session_id, chat_id, kind, local_path,
                     original_name, mime_type, caption, ts),
                )
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            return None

    def recent_attachments(self, bot: str = BOT_NAME, chat_id: str = "",
                            limit: int = 5) -> list[dict]:
        """Return most recent attachment records."""
        with self.connection() as conn:
            if chat_id:
                rows = conn.execute(
                    "SELECT id, kind, local_path, original_name, mime_type, caption, chat_id, created_at "
                    "FROM attachments WHERE bot=? AND chat_id=? ORDER BY id DESC LIMIT ?",
                    (bot, chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kind, local_path, original_name, mime_type, caption, chat_id, created_at "
                    "FROM attachments WHERE bot=? ORDER BY id DESC LIMIT ?",
                    (bot, limit),
                ).fetchall()
        return [dict(r) for r in rows]
