"""
database/agent/promoted_signals.py — PromotedSignalsRepository.

Consolidates the DDL from src/goal_discovery.py AND src/init.py
(both define the same promoted_signals table).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from database.base import BaseRepository


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PromotedSignalsRepository(BaseRepository):
    """SQLite-backed store for promoted evolution signals."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS promoted_signals (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    text       TEXT    NOT NULL,
                    weight     INTEGER NOT NULL DEFAULT 1,
                    category   TEXT,
                    score      REAL,
                    created_at TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_promoted_created ON promoted_signals(created_at);
            """)
            conn.commit()

    # ── public API ───────────────────────────────────────────────────────────

    def insert(self, text: str, weight: int = 1, category: str = None,
               score: float = None) -> int:
        """Insert a promoted signal. Returns new row id."""
        now = _utc_now()
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT INTO promoted_signals (text, weight, category, score, created_at) VALUES (?,?,?,?,?)",
                (text, weight, category, score, now),
            )
            conn.commit()
            return cur.lastrowid

    def get_all(self, limit: int = 100) -> list[dict]:
        """Return most recently created signals."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT text, weight FROM promoted_signals ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_by_keyword(self, keyword: str) -> int:
        """Delete all signals containing keyword. Returns deleted count."""
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM promoted_signals WHERE text LIKE ?",
                (f"%{keyword}%",),
            )
            conn.commit()
            return cur.rowcount

    def count(self) -> int:
        with self.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM promoted_signals").fetchone()
            return row[0] if row else 0
