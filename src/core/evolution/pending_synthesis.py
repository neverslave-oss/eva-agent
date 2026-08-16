"""
pending_synthesis.py — ADR-021: Tier 2 approval gate persistence.

Stores pending synthesis requests in ~/.kernel-evolving/pending_synthesis.json
while the user decides whether to approve or reject them.
"""
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

_STORE_PATH = Path(os.path.expanduser("~/.kernel-evolving/pending_synthesis.json"))
_lock = threading.Lock()


def _load() -> list:
    if not _STORE_PATH.exists():
        return []
    try:
        return json.loads(_STORE_PATH.read_text()) or []
    except Exception:
        return []


def _save(records: list):
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(records, indent=2))


def enqueue(task: str, gap: str, provider: str, chat_id: str) -> str:
    """Store a pending synthesis request. Returns the unique synthesis_id."""
    synthesis_id = str(uuid.uuid4())[:12]
    record = {
        "id": synthesis_id,
        "task": task,
        "gap": gap,
        "provider": provider,
        "chat_id": chat_id,
        "created_at": time.time(),
    }
    with _lock:
        records = _load()
        records.append(record)
        _save(records)
    return synthesis_id


def get(synthesis_id: str) -> Optional[dict]:
    """Return the pending record for synthesis_id, or None if not found."""
    with _lock:
        for r in _load():
            if r.get("id") == synthesis_id:
                return dict(r)
    return None


def remove(synthesis_id: str) -> Optional[dict]:
    """Remove and return the record, or None if not found."""
    with _lock:
        records = _load()
        found = None
        remaining = []
        for r in records:
            if r.get("id") == synthesis_id:
                found = dict(r)
            else:
                remaining.append(r)
        if found:
            _save(remaining)
        return found


def list_pending() -> list:
    """Return all pending synthesis records."""
    with _lock:
        return list(_load())


def purge_stale(max_age_seconds: int = 86400):
    """Remove requests older than max_age_seconds (default 24h)."""
    cutoff = time.time() - max_age_seconds
    with _lock:
        records = _load()
        fresh = [r for r in records if r.get("created_at", 0) >= cutoff]
        if len(fresh) != len(records):
            _save(fresh)
