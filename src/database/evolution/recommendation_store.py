"""
database/evolution/recommendation_store.py — RecommendationStore(BaseRepository).

Extracted from src/evolver.py::RecommendationStore.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from database.base import BaseRepository


class RecommendationStore(BaseRepository):
    """SQLite store for persisting recommender near-miss results."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recommendation_hits (
                    skill_name TEXT PRIMARY KEY,
                    hit_count  INTEGER NOT NULL DEFAULT 0,
                    last_seen  TEXT
                )
            """)
            conn.commit()

    def record(self, skill_names: list[str]) -> None:
        """Increment hit counter for each recommended skill name."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as conn:
            for name in skill_names:
                conn.execute("""
                    INSERT INTO recommendation_hits (skill_name, hit_count, last_seen)
                    VALUES (?, 1, ?)
                    ON CONFLICT(skill_name) DO UPDATE SET
                        hit_count = hit_count + 1,
                        last_seen = excluded.last_seen
                """, (name, now))
            conn.commit()

    def get_hits(self) -> dict[str, int]:
        """Return mapping of skill_name → hit_count for all recorded skills."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT skill_name, hit_count FROM recommendation_hits"
            ).fetchall()
        return {r["skill_name"]: r["hit_count"] for r in rows}
