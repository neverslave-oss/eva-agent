"""
database/memory/long_term_memory.py — LongTermMemoryRepository.

Wraps memory_entries + memory_session_files schema from long_term_memory.py.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database.base import BaseRepository


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_session_id(session_id: str) -> str:
    if not session_id:
        return "global"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return safe[:64] or "global"


def _message_hash(session_id: str, role: str, content: str) -> str:
    return hashlib.sha256(f"{session_id}|{role}|{content}".encode("utf-8")).hexdigest()


class LongTermMemoryRepository(BaseRepository):
    """SQLite repository for long-term memory entries and session files."""

    def __init__(self, db_path: Path | str, memory_dir: Path) -> None:
        super().__init__(db_path)
        self.memory_dir = Path(memory_dir)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts            TEXT NOT NULL,
                    chat_id       TEXT NOT NULL DEFAULT '',
                    session_id    TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    source        TEXT NOT NULL DEFAULT 'chat',
                    file_name     TEXT NOT NULL,
                    message_hash  TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_entries_ts      ON memory_entries(ts);
                CREATE INDEX IF NOT EXISTS idx_memory_entries_chat     ON memory_entries(chat_id, ts);
                CREATE INDEX IF NOT EXISTS idx_memory_entries_session  ON memory_entries(session_id, ts);

                CREATE TABLE IF NOT EXISTS memory_session_files (
                    session_id    TEXT PRIMARY KEY,
                    file_name     TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                );
            """)
            conn.commit()

    # ── index file helpers ────────────────────────────────────────────────────

    @property
    def memory_index_file(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def _ensure_index_file(self) -> None:
        if self.memory_index_file.exists():
            return
        self.memory_index_file.write_text(
            "# MEMORY\n\n"
            "Long-term memory index for kernel-evolving.\n"
            "Session memory files are listed below as they are created.\n\n"
            "## Sessions\n\n",
            encoding="utf-8",
        )

    def _append_index_entry(self, file_name: str, session_id: str) -> None:
        self._ensure_index_file()
        line = f"- {file_name} (session: {session_id})\n"
        try:
            content = self.memory_index_file.read_text(encoding="utf-8")
        except Exception:
            content = ""
        if line in content:
            return
        with self.memory_index_file.open("a", encoding="utf-8") as f:
            f.write(line)

    # ── session file helpers ──────────────────────────────────────────────────

    def _get_or_create_session_file(self, conn: sqlite3.Connection,
                                     session_id: str) -> Path:
        row = conn.execute(
            "SELECT file_name FROM memory_session_files WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row:
            return self.memory_dir / row["file_name"]

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_name = f"memory-{ts}-{_safe_session_id(session_id)}.md"
        conn.execute(
            "INSERT INTO memory_session_files (session_id, file_name, created_at) VALUES (?, ?, ?)",
            (session_id, file_name, _utc_now_iso()),
        )
        path = self.memory_dir / file_name
        path.write_text(
            f"# Session Memory\n\nsession_id: {session_id}\ncreated_at: {_utc_now_iso()}\n\n## Entries\n\n",
            encoding="utf-8",
        )
        self._append_index_entry(file_name, session_id)
        return path

    # ── public API ────────────────────────────────────────────────────────────

    def persist_messages(self, messages: list[dict], chat_id: str,
                          session_id: str) -> int:
        """Persist messages to SQLite + session markdown file. Idempotent. Returns new count."""
        if not messages:
            return 0
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        new_count = 0
        with self.connection() as conn:
            session_file = self._get_or_create_session_file(conn, session_id)
            for message in messages:
                role = str(message.get("role", ""))
                if role not in {"user", "assistant", "system"}:
                    continue
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                ts = _utc_now_iso()
                msg_hash = _message_hash(session_id, role, content)
                try:
                    conn.execute(
                        "INSERT INTO memory_entries "
                        "(ts, chat_id, session_id, role, content, source, file_name, message_hash) "
                        "VALUES (?, ?, ?, ?, ?, 'chat', ?, ?)",
                        (ts, chat_id or "", session_id, role, content,
                         session_file.name, msg_hash),
                    )
                except sqlite3.IntegrityError:
                    continue
                with session_file.open("a", encoding="utf-8") as f:
                    f.write(f"- {ts} [{role}] {content[:500]}\n")
                new_count += 1
            conn.commit()
        return new_count

    def recent_memory_lines(self, limit: int = 20, max_chars: int = 260) -> list[str]:
        """Return recent memory rows formatted for prompt context."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT ts, role, content FROM memory_entries ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [f"[{r['ts']}] ({r['role']}) {r['content'][:max_chars]}" for r in rows]

    def recent_memory_files(self, limit: int = 3, max_chars: int = 300) -> list[str]:
        """Return recent session file snippets."""
        files = sorted(self.memory_dir.glob("memory-*.md"), reverse=True)[:limit]
        notes = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if content:
                notes.append(f"[{file_path.name}] {content[:max_chars]}")
        return notes
