"""
evolution_state.py — ADR-004 evolution state machine and supervision controls.

Provides:
- EvolutionState: running | paused | stopped
- Configurable iteration cap (default 10)
- Thread-safe increment/check/reset
- Persistent state — survives restarts (stored in ~/.kernel-evolving/workspace/evolution_state.json)
- Used by evolution_hook.py, goal_discovery.py, and api.py endpoints

Design: evolution can only proceed when state == RUNNING and iterations < cap.
Any component checks should_evolve() before calling maybe_evolve().
Pause/stop state persists across restarts — API control is durable.
"""
import json
import os
import threading
import time
from enum import Enum
from runtime_paths import EVOLUTION_STATE_FILE


class EvolutionState(str, Enum):
    RUNNING = "running"
    PAUSED  = "paused"
    STOPPED = "stopped"


# ── Persistence ───────────────────────────────────────────────────────────────

_STATE_FILE = str(EVOLUTION_STATE_FILE)


def _load_persisted() -> dict | None:
    """Load persisted state from disk. Returns None if file missing or corrupt."""
    try:
        if os.path.exists(_STATE_FILE):
            with open(_STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _save_state():
    """Write current state to disk. Must be called with _lock held."""
    try:
        os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump({
                "state":      _state.value,
                "iterations": _iterations,
                "cap":        _cap,
                "paused_at":  _paused_at,
                "stopped_at": _stopped_at,
            }, f)
    except Exception as e:
        print(f"[evolution_state] persist error: {e}")


# ── Singleton state (module-level, shared across all imports) ─────────────────

_lock = threading.Lock()

# Initialise from persisted state if available; otherwise fall back to env
_persisted = _load_persisted()
if _persisted:
    _state      = EvolutionState(_persisted.get("state", "stopped"))
    _iterations = int(_persisted.get("iterations", 0))
    _cap        = int(_persisted.get("cap", os.environ.get("EVOLUTION_MAX_ITERATIONS", "10")))
    _paused_at  = _persisted.get("paused_at")
    _stopped_at = _persisted.get("stopped_at")
    print(f"[evolution_state] loaded persisted state={_state.value} iterations={_iterations}/{_cap}")
else:
    _state      = EvolutionState.RUNNING if os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true" else EvolutionState.STOPPED
    _iterations = 0
    _cap        = int(os.environ.get("EVOLUTION_MAX_ITERATIONS", "10"))
    _paused_at  = None
    _stopped_at = None

_history: list[dict] = []  # control events log (in-memory only, not persisted)


def get_status() -> dict:
    with _lock:
        return {
            "state":        _state.value,
            "iterations":   _iterations,
            "cap":          _cap,
            "remaining":    max(0, _cap - _iterations),
            "paused_at":    _paused_at,
            "stopped_at":   _stopped_at,
            "control_log":  list(_history[-20:]),  # last 20 control events
        }


def should_evolve() -> bool:
    """Returns True only when state is RUNNING and cap not reached.
    Cap tracks *unique skills acquired*, not total calls.
    """
    with _lock:
        if _state != EvolutionState.RUNNING:
            return False
        if _iterations >= _cap:
            _do_pause("auto-paused: iteration cap reached")
            return False
        return True


def increment() -> int:
    """Called only after a NEW skill is acquired or synthesised.
    Not called on cache hits or repeated tasks.
    """
    global _iterations
    with _lock:
        _iterations += 1
        _save_state()
        return _iterations


def start(cap: int | None = None) -> dict:
    """Start or resume evolution. Always resets iteration counter when a new cap is provided."""
    global _state, _paused_at, _stopped_at, _cap, _iterations
    with _lock:
        if cap is not None:
            _cap = max(1, cap)
            _iterations = 0  # new cap = fresh start
        elif _state == EvolutionState.STOPPED:
            _iterations = 0  # full reset on start from stopped
        # Resume from paused/stopped regardless
        _state      = EvolutionState.RUNNING
        _paused_at  = None
        _stopped_at = None
        _history.append({"event": "start", "cap": _cap, "iterations_reset": _iterations, "ts": _now()})
        _save_state()
    return get_status()


def pause(reason: str = "manual") -> dict:
    """Pause evolution. Idempotent if already paused or stopped."""
    global _state, _paused_at
    with _lock:
        if _state == EvolutionState.RUNNING:
            _do_pause(reason)
            _save_state()
    return get_status()


def resume() -> dict:
    """Resume from paused state. If cap was reached, bumps cap by 10 to allow more cycles."""
    global _state, _paused_at, _cap
    with _lock:
        if _state == EvolutionState.PAUSED:
            # If we paused because cap was reached, auto-bump to allow continuing
            if _iterations >= _cap:
                _cap = _iterations + 20
                _history.append({"event": "cap-bump", "new_cap": _cap, "ts": _now()})
            _state     = EvolutionState.RUNNING
            _paused_at = None
            _history.append({"event": "resume", "iterations": _iterations, "cap": _cap, "ts": _now()})
            _save_state()
    return get_status()


def stop() -> dict:
    global _state, _stopped_at
    with _lock:
        _state      = EvolutionState.STOPPED
        _stopped_at = _now()
        _history.append({"event": "stop", "iterations": _iterations, "ts": _now()})
        _save_state()
    return get_status()


def reset(cap: int | None = None) -> dict:
    """Stop and reset iteration counter. Optionally change cap."""
    global _state, _iterations, _cap, _paused_at, _stopped_at
    with _lock:
        _state      = EvolutionState.STOPPED
        _iterations = 0
        _paused_at  = None
        _stopped_at = _now()
        if cap is not None:
            _cap = max(1, cap)
        _history.append({"event": "reset", "new_cap": _cap, "ts": _now()})
        _save_state()
    return get_status()


# ── Internals ─────────────────────────────────────────────────────────────────

def _do_pause(reason: str = "manual"):
    """Must be called with _lock held."""
    global _state, _paused_at
    _state     = EvolutionState.PAUSED
    _paused_at = _now()
    _history.append({"event": "pause", "reason": reason, "iterations": _iterations, "ts": _paused_at})
    _save_state()  # persist immediately so restart won't reset the pause


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
