"""
probe_store.py — ADR-019 Phase 2: Persistent probe record store with TTL expiry.

This module is a thin shim over database.agent.ProbeRepository.
All SQL logic lives in database/agent/probe_store.py.
The ProbeStore class interface is preserved for backward compatibility.
"""
from __future__ import annotations

from typing import Optional
from runtime_paths import PROBES_DB
from database.agent import ProbeRepository


_DEFAULT_DB_PATH = str(PROBES_DB)


class ProbeStore:
    """SQLite-backed store for parked speculative probes. Delegates to ProbeRepository."""

    def __init__(self, db_path: Optional[str] = None, ttl_days: int = 7):
        self._repo = ProbeRepository(db_path or _DEFAULT_DB_PATH, ttl_days=ttl_days)

    def add(self, thought: dict, subject: str) -> int:
        return self._repo.add(thought, subject)

    def match(self, text: str) -> list[dict]:
        return self._repo.match(text)

    def trigger(self, probe_id: int, triggered_by: str) -> None:
        self._repo.trigger(probe_id, triggered_by)

    def resolve(self, probe_id: int) -> None:
        self._repo.resolve(probe_id)

    def expire_old(self) -> int:
        return self._repo.expire_old()

    def list_active(self) -> list[dict]:
        return self._repo.list_active()
