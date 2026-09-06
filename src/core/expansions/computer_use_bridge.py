"""computer_use_bridge — kernel-side bridge to computer-use sidecar.

No-op safe by default: if sidecar is missing/broken, callers receive harmless
fallback values so kernel boot and normal tool loops are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KERNEL_ROOT = Path(__file__).resolve().parents[3]
_SIDECAR_SRC = _KERNEL_ROOT / "expansions" / "computer-use" / "src"

_sidecar = None


def _import_sidecar():
    if not _SIDECAR_SRC.exists():
        return None
    sys.path.insert(0, str(_SIDECAR_SRC))
    try:
        import computer_use  # type: ignore
        return computer_use
    except Exception:
        return None


def _ensure_loaded() -> bool:
    global _sidecar
    if _sidecar is not None:
        return True
    mod = _import_sidecar()
    if mod is None:
        return False
    _sidecar = mod
    return True


def available() -> bool:
    return _ensure_loaded()


def inject_computer_use_context(chat_id: str = "", text: str = "") -> str:
    if not _ensure_loaded():
        return ""
    if not text.strip():
        return ""
    return "Computer-use expansion available (scaffold). Prefer tool-first safe actions."


def helper_plan_hint(chat_id: str = "", text: str = "") -> str:
    if not _ensure_loaded() or not text.strip():
        return ""
    return "Computer-use helper: plan atomic steps and verify each step."


def run_computer_task(
    chat_id: str = "",
    goal: str = "",
    target: dict | None = None,
    dry_run: bool = True,
) -> dict:
    if not _ensure_loaded():
        return {"ok": False, "reason": "computer-use sidecar unavailable"}
    return {
        "ok": True,
        "mode": "scaffold",
        "chat_id": chat_id,
        "goal": goal,
        "target": target or {},
        "dry_run": dry_run,
        "result": "execution stub",
    }


def debug_snapshot(chat_id: str = "", query: str = "") -> dict:
    if not _ensure_loaded():
        return {
            "available": False,
            "reason": "computer-use sidecar not loaded",
            "chat_id": chat_id,
            "query": query,
        }
    return {
        "available": True,
        "chat_id": chat_id,
        "query": query,
        "sidecar_src": str(_SIDECAR_SRC),
        "mode": "scaffold",
    }
