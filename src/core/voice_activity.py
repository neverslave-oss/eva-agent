"""voice_activity.py

Lightweight cross-module voice activity guard used by provider routing.
Tracks active STT/TTS/voice-clone operations with expiring tokens in a JSON file.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_LOCK = threading.Lock()


def _state_path() -> Path:
    raw = os.environ.get(
        "KERNEL_VOICE_ACTIVITY_FILE",
        "~/.kernel-evolving/workspace/tmp/voice_activity.json",
    )
    path = Path(os.path.expanduser(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {"entries": {}}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except Exception:
        pass
    return {"entries": {}}


def _save_state(state: dict) -> None:
    p = _state_path()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def _prune_entries(state: dict) -> dict:
    now = time.time()
    entries = state.get("entries", {})
    alive = {
        k: v for k, v in entries.items()
        if isinstance(v, dict) and float(v.get("expires_at", 0)) > now
    }
    state["entries"] = alive
    return state


def begin_voice_activity(kind: str, ttl_s: int = 300) -> str:
    """Register an active voice operation and return its token."""
    token = f"{kind}:{uuid.uuid4().hex}"
    now = time.time()
    with _LOCK:
        state = _prune_entries(_load_state())
        state["entries"][token] = {
            "kind": kind,
            "started_at": now,
            "expires_at": now + max(1, int(ttl_s)),
        }
        _save_state(state)
    return token


def end_voice_activity(token: str) -> None:
    """Clear one active voice token (best effort)."""
    if not token:
        return
    with _LOCK:
        state = _prune_entries(_load_state())
        state.get("entries", {}).pop(token, None)
        _save_state(state)


def is_voice_active() -> bool:
    """True when one or more unexpired voice operations are active."""
    with _LOCK:
        state = _prune_entries(_load_state())
        _save_state(state)
        return bool(state.get("entries"))


@contextmanager
def voice_activity(kind: str, ttl_s: int = 300):
    """Context manager helper for begin/end voice activity."""
    token = begin_voice_activity(kind=kind, ttl_s=ttl_s)
    try:
        yield token
    finally:
        end_voice_activity(token)
