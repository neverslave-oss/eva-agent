"""
database/agent/prompt_log.py — PromptLogRepository.

Extracted from src/prompt_logger.py. The original module becomes a thin
shim delegating to this class.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from database.base import BaseRepository

logger = logging.getLogger(__name__)


class PromptLogRepository(BaseRepository):
    """SQLite-backed repository for raw prompt/response debug logs."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts           TEXT    NOT NULL,
                    chat_id      TEXT    NOT NULL DEFAULT '',
                    prompt_len   INTEGER NOT NULL,
                    prompt       TEXT    NOT NULL,
                    history      TEXT    NOT NULL DEFAULT '[]',
                    user_message TEXT    NOT NULL DEFAULT '',
                    provider     TEXT    NOT NULL DEFAULT '',
                    model        TEXT    NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_prompt_log_ts      ON prompt_log (ts);
                CREATE INDEX IF NOT EXISTS idx_prompt_log_chat_id ON prompt_log (chat_id);
            """)
            conn.commit()
        self._run_column_migrations()

    def _run_column_migrations(self) -> None:
        """Idempotent column additions — same logic as original prompt_logger.py."""
        migrate_columns = [
            ("history",      "TEXT NOT NULL DEFAULT '[]'"),
            ("user_message", "TEXT NOT NULL DEFAULT ''"),
        ]
        with self.connection() as conn:
            existing = {row[1] for row in conn.execute("PRAGMA table_info(prompt_log)").fetchall()}
            for col_name, col_def in migrate_columns:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE prompt_log ADD COLUMN {col_name} {col_def}")
                    conn.commit()

    # ── public API ───────────────────────────────────────────────────────────

    def log_prompt(self, prompt: str, chat_id: str = "",
                   history: list | None = None, user_message: str = "",
                   provider: str = "", model: str = "") -> None:
        """Persist the fully assembled system prompt + conversation context."""
        try:
            ts = datetime.now(tz=timezone.utc).isoformat()
            history_json = json.dumps(history or [], ensure_ascii=False)
            with self.connection() as conn:
                conn.execute(
                    "INSERT INTO prompt_log (ts, chat_id, prompt_len, prompt, history, user_message, provider, model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, chat_id or "", len(prompt), prompt, history_json,
                     user_message or "", provider or "", model or ""),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("[PromptLogRepository] failed to log: %s", exc)

    def query_recent(self, limit: int = 10, chat_id: str = "") -> list[dict]:
        """Return the most recent `limit` entries (no large text fields)."""
        try:
            with self.connection() as conn:
                if chat_id:
                    rows = conn.execute(
                        "SELECT id, ts, chat_id, prompt_len, user_message, provider, model "
                        "FROM prompt_log WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                        (chat_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, ts, chat_id, prompt_len, user_message, provider, model "
                        "FROM prompt_log ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_entry(self, entry_id: int) -> dict:
        """Return one prompt log entry with full prompt and parsed history."""
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT id, ts, chat_id, prompt_len, prompt, history, user_message, provider, model "
                    "FROM prompt_log WHERE id=?",
                    (entry_id,),
                ).fetchone()
            if not row:
                return {}
            d = dict(row)
            try:
                d["history"] = json.loads(d.get("history") or "[]")
            except Exception:
                d["history"] = []
            return d
        except Exception:
            return {}

    def get_prompt_by_id(self, entry_id: int) -> str:
        """Retrieve the full prompt text for a given log entry id."""
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT prompt FROM prompt_log WHERE id=?", (entry_id,)
                ).fetchone()
            return row[0] if row else ""
        except Exception:
            return ""

    def clear_chat_logs(self, chat_id: str) -> int:
        """Delete prompt-log rows for one chat_id. Returns deleted row count."""
        try:
            with self.connection() as conn:
                cur = conn.execute(
                    "DELETE FROM prompt_log WHERE chat_id=?", (chat_id or "",)
                )
                conn.commit()
                return cur.rowcount if cur.rowcount is not None else 0
        except Exception:
            return 0
