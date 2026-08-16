"""
database/agent/failed_requests.py — FailedRequestsRepository.

Extracted from src/failed_requests.py. The original module becomes a thin
shim that instantiates this class and delegates all public functions.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database.base import BaseRepository

logger = logging.getLogger(__name__)


class FailedRequestsRepository(BaseRepository):
    """SQLite-backed store for failed/partial agent request records (ADR-020)."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_requests (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id          TEXT    NOT NULL,
                    ts               TEXT    NOT NULL,
                    user_message     TEXT    NOT NULL,
                    agent_response   TEXT    NOT NULL,
                    failure_type     TEXT    NOT NULL,
                    resolved         INTEGER NOT NULL DEFAULT 0,
                    resolved_at      TEXT,
                    reward           INTEGER,
                    retry_count      INTEGER NOT NULL DEFAULT 0,
                    pending_confirm  INTEGER NOT NULL DEFAULT 0,
                    confirm_skill    TEXT,
                    confirm_output   TEXT
                )
            """)
            conn.commit()

    # ── public API ───────────────────────────────────────────────────────────

    def record(self, chat_id: str, user_message: str, agent_response: str,
               failure_type: str) -> int:
        """Insert a failed request record. Returns the new row id."""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as conn:
                cur = conn.execute(
                    """INSERT INTO failed_requests
                       (chat_id, ts, user_message, agent_response, failure_type)
                       VALUES (?, ?, ?, ?, ?)""",
                    (str(chat_id), ts, user_message[:2000], agent_response[:2000], failure_type),
                )
                conn.commit()
                row_id = cur.lastrowid
            logger.info(f"[FailedRequests] recorded id={row_id} type={failure_type} chat={chat_id}")
            return row_id
        except Exception as e:
            logger.error(f"[FailedRequests] record error: {e}")
            return -1

    def get_unresolved(self, limit: int = 5) -> list[dict]:
        """Return most recent unresolved failed requests."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM failed_requests WHERE resolved=0 ORDER BY ts DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"[FailedRequests] get_unresolved error: {e}")
            return []

    def mark_pending_confirm(self, request_id: int, skill_name: str, skill_output: str) -> None:
        """Mark a request as awaiting human confirmation."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE failed_requests SET pending_confirm=1, confirm_skill=?, confirm_output=? WHERE id=?",
                    (skill_name[:200], skill_output[:2000], request_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[FailedRequests] mark_pending_confirm error: {e}")

    def mark_resolved(self, request_id: int, reward: int) -> None:
        """Mark a request as confirmed resolved. reward=1 positive, reward=0 negative."""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE failed_requests SET resolved=1, resolved_at=?, reward=?, pending_confirm=0 WHERE id=?",
                    (ts, reward, request_id),
                )
                conn.commit()
            logger.info(f"[FailedRequests] id={request_id} resolved with reward={reward}")
        except Exception as e:
            logger.error(f"[FailedRequests] mark_resolved error: {e}")

    def increment_retry(self, request_id: int) -> int:
        """Bump retry_count and return new value."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE failed_requests SET retry_count = retry_count + 1 WHERE id=?",
                    (request_id,),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT retry_count FROM failed_requests WHERE id=?", (request_id,)
                ).fetchone()
            return row["retry_count"] if row else 0
        except Exception as e:
            logger.error(f"[FailedRequests] increment_retry error: {e}")
            return 0


# ── Module-level shim API (backward compatibility) ───────────────────────────
# These functions mirror the original failed_requests.py shim so existing
# callers that do `import database.agent.failed_requests as fr; fr.detect_failure(...)`
# continue to work after the shim was removed.

import re
from typing import Optional

_FAILURE_PATTERNS = [
    r"\bI (can'?t|cannot|don'?t know how to|am not able to|don'?t have (access|the ability))\b",
    r"\b(not (available|supported|implemented|possible))\b",
    r"\b(no (skill|tool) (found|available|for|matched))\b",
    r"\b(skill not found|no matching skill|couldn'?t find a skill)\b",
    r"\b(I'?m not sure how to|I'?m unable to)\b",
    r"\b(sorry[,.]? (I|but))\b",
    r"\[model_server|model_client|model error\]",
    r"\b(error:|exception:|traceback)\b",
]
_FAILURE_RE = re.compile("|".join(_FAILURE_PATTERNS), re.IGNORECASE)
_MIN_SUBSTANTIVE_REPLY = 80

# Keep _DB_PATH at module level so tests can patch it
try:
    from runtime_paths import FAILED_REQUESTS_DB
    _DB_PATH: str = str(FAILED_REQUESTS_DB)
except Exception:
    _DB_PATH = ""

_repo_instance: Optional["FailedRequestsRepository"] = None
_repo_instance_path: str = ""


def _get_repo() -> "FailedRequestsRepository":
    global _repo_instance, _repo_instance_path
    if _repo_instance is None or _repo_instance_path != _DB_PATH:
        _repo_instance = FailedRequestsRepository(_DB_PATH)
        _repo_instance_path = _DB_PATH
    return _repo_instance


def detect_failure(reply: str, user_message: str) -> Optional[str]:
    """Return failure_type string or None."""
    if not reply or not reply.strip():
        return "skill_error"
    stripped = reply.strip()
    if re.search(r"\[model_server|model_client|model error\]|\btraceback\b|\bexception:", stripped, re.IGNORECASE):
        return "skill_error"
    if re.search(
        r"\b(no (skill|tool) (found|available|for|matched))\b"
        r"|\b(skill not found|no matching skill|couldn'?t find a skill)\b"
        r"|\bI (can'?t|cannot|don'?t have (access|the ability))\b",
        stripped, re.IGNORECASE
    ):
        return "no_skill"
    if re.search(r"\bI'?m (not sure how to|unable to)\b|\bsorry[,.]? (I|but)\b", stripped, re.IGNORECASE):
        return "refusal"
    if _FAILURE_RE.search(stripped):
        return "no_skill"
    if (len(stripped) < _MIN_SUBSTANTIVE_REPLY
            and len(user_message.strip().split()) > 4
            and not user_message.strip().startswith("/")):
        return "partial"
    return None


def record(chat_id: str, user_message: str, agent_response: str, failure_type: str) -> int:
    return _get_repo().record(chat_id, user_message, agent_response, failure_type)


def get_unresolved(limit: int = 5) -> list:
    return _get_repo().get_unresolved(limit)


def mark_pending_confirm(request_id: int, skill_name: str, skill_output: str) -> None:
    _get_repo().mark_pending_confirm(request_id, skill_name, skill_output)


def mark_resolved(request_id: int, reward: int) -> None:
    _get_repo().mark_resolved(request_id, reward)


def increment_retry(request_id: int) -> int:
    return _get_repo().increment_retry(request_id)
