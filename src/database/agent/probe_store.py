"""
database/agent/probe_store.py — ProbeRepository(BaseRepository).

Extracted from src/probe_store.py. The original module keeps ProbeStore
as a thin shim instantiating this class.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from database.base import BaseRepository


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProbeRepository(BaseRepository):
    """SQLite-backed store for parked speculative probes (ADR-019 Phase 2)."""

    def __init__(self, db_path: Path | str, ttl_days: int = 7) -> None:
        super().__init__(db_path)
        self._ttl_days = ttl_days
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS probes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject      TEXT    NOT NULL,
                    thought_text TEXT    NOT NULL,
                    category     TEXT    NOT NULL,
                    score        REAL    NOT NULL,
                    created_at   TEXT    NOT NULL,
                    expires_at   TEXT    NOT NULL,
                    triggered_at TEXT,
                    triggered_by TEXT,
                    resolved     INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()

    # ── public API ───────────────────────────────────────────────────────────

    def add(self, thought: dict, subject: str) -> int:
        """Insert a new probe record. Returns the new row id."""
        now = _now_utc()
        expires = (datetime.now(timezone.utc) + timedelta(days=self._ttl_days)).isoformat()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO probes
                    (subject, thought_text, category, score, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    subject,
                    thought.get("thought", ""),
                    thought.get("category", "unknown"),
                    float(thought.get("score", 0.0)),
                    now,
                    expires,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def match(self, text: str) -> list[dict]:
        """Return active probes whose subject keywords appear in *text*."""
        now = _now_utc()
        lower_text = text.lower()
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM probes WHERE resolved = 0 AND expires_at > ?",
                (now,),
            ).fetchall()

        matched = []
        for row in rows:
            probe = dict(row)
            keywords = re.split(r"[\s,]+", probe["subject"].lower())
            keywords = [k for k in keywords if k]
            if any(kw in lower_text for kw in keywords):
                matched.append(probe)
        return matched

    def trigger(self, probe_id: int, triggered_by: str) -> None:
        """Set triggered_at / triggered_by on a probe."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE probes SET triggered_at = ?, triggered_by = ? WHERE id = ?",
                (_now_utc(), triggered_by, probe_id),
            )
            conn.commit()

    def resolve(self, probe_id: int) -> None:
        """Mark a probe resolved (resolved=1)."""
        with self.connection() as conn:
            conn.execute("UPDATE probes SET resolved = 1 WHERE id = ?", (probe_id,))
            conn.commit()

    def expire_old(self) -> int:
        """Hard-delete probes past their expires_at. Returns count deleted."""
        now = _now_utc()
        with self.connection() as conn:
            cur = conn.execute("DELETE FROM probes WHERE expires_at <= ?", (now,))
            conn.commit()
            return cur.rowcount

    def list_active(self) -> list[dict]:
        """Return all non-resolved, non-expired probes."""
        now = _now_utc()
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM probes WHERE resolved = 0 AND expires_at > ? ORDER BY created_at DESC",
                (now,),
            ).fetchall()
        return [dict(r) for r in rows]
