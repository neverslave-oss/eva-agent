"""
database/evolution/trajectory_store.py — TrajectoryStore.

Shares evolution.db file; receives EvolutionLogRepository as a constructor dep
(no second sqlite3.connect to the same file).
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from database.evolution.evolution_log import EvolutionLogRepository

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    task            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model_name      TEXT,
    call_type       TEXT NOT NULL,
    tool_calls      TEXT NOT NULL,
    final_reply     TEXT NOT NULL,
    artifacts       TEXT,
    critic_score    REAL,
    critic_verdict  TEXT,
    token_count     INTEGER,
    elapsed_s       REAL
);
"""


class TrajectoryStore:
    """Stores task trajectories using an existing EvolutionLogRepository connection."""

    def __init__(self, repo: EvolutionLogRepository) -> None:
        self._repo = repo
        # Ensure task_trajectories table exists
        with self._repo.connection() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = self._repo.connection()
        try:
            yield conn
        finally:
            conn.close()

    def record(self, task: str, provider: str, call_type: str,
               tool_calls: list, final_reply: str, artifacts: list = None,
               critic_score: float = None, critic_verdict: str = None,
               token_count: int = None, elapsed_s: float = None,
               model_name: str = None, ts: str = None) -> int:
        """Store a trajectory entry. Returns new row id."""
        from datetime import datetime, timezone
        ts = ts or datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO task_trajectories
                   (ts, task, provider, model_name, call_type, tool_calls, final_reply,
                    artifacts, critic_score, critic_verdict, token_count, elapsed_s)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts, task, provider, model_name, call_type,
                    json.dumps(tool_calls or []),
                    final_reply,
                    json.dumps(artifacts or []),
                    critic_score, critic_verdict, token_count, elapsed_s,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_recent(self, limit: int = 50, min_critic_score: float = 0.0) -> list[dict]:
        """Return recent task trajectories, optionally filtered by critic score."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM task_trajectories "
                "WHERE (critic_score IS NULL OR critic_score >= ?) "
                "ORDER BY ts DESC LIMIT ?",
                (min_critic_score, limit),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ("tool_calls", "artifacts"):
                if isinstance(d.get(field), str):
                    try:
                        d[field] = json.loads(d[field])
                    except Exception:
                        pass
            result.append(d)
        return result
