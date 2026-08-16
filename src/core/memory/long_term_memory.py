"""Long-term memory persistence under workspace/memory.

This module is a thin shim over database.memory.LongTermMemoryRepository.
All SQL logic lives in database/memory/long_term_memory.py.
Public function signatures are preserved for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from runtime_paths import MEMORY_DIR, LONG_TERM_MEMORY_DB
from database.memory import LongTermMemoryRepository

# Module-level repo instance
_repo: Optional[LongTermMemoryRepository] = None


def _get_repo() -> LongTermMemoryRepository:
    global _repo
    if _repo is None:
        _repo = LongTermMemoryRepository(LONG_TERM_MEMORY_DB, MEMORY_DIR)
    return _repo


def persist_messages(messages: list[dict], chat_id: str, session_id: str) -> int:
    """Persist messages to SQLite and session markdown file. Returns new count."""
    return _get_repo().persist_messages(messages, chat_id, session_id)


def recent_memory_lines(limit: int = 20, max_chars: int = 260) -> list[str]:
    """Return recent memory rows formatted for prompt context."""
    return _get_repo().recent_memory_lines(limit, max_chars)


def recent_memory_files(limit: int = 3, max_chars: int = 300) -> list[str]:
    """Return recent memory session file snippets."""
    return _get_repo().recent_memory_files(limit, max_chars)
