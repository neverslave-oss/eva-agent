"""computer_use_bridge — kernel-side bridge to computer-use sidecar.

No-op safe by default: if sidecar is missing/broken, callers receive harmless
fallback values so kernel boot and normal tool loops are unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KERNEL_ROOT = Path(__file__).resolve().parents[3]
_SIDECAR_ROOT = _KERNEL_ROOT / "expansions" / "computer-use"
_SIDECAR_SRC = _SIDECAR_ROOT / "src"
_STATE_FILE = _SIDECAR_ROOT / "tmp" / "chat_state.json"

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
    return (
        "Computer-use expansion active: desktop-first, browser-interoperable, "
        "per-chat state retention, dry-run default."
    )


def helper_plan_hint(chat_id: str = "", text: str = "") -> str:
    if not _ensure_loaded() or not text.strip():
        return ""
    return "Plan atomic actions, validate by policy, verify each post-condition."


def run_computer_task(
    chat_id: str = "",
    goal: str = "",
    target: dict | None = None,
    dry_run: bool = True,
) -> dict:
    if not _ensure_loaded():
        return {"ok": False, "reason": "computer-use sidecar unavailable"}

    from computer_use.orchestrator import Orchestrator  # type: ignore
    from computer_use.planner import Planner  # type: ignore
    from computer_use.safety import PolicyEngine  # type: ignore
    from computer_use.state_store import StateStore  # type: ignore
    from computer_use.telemetry import TraceCollector  # type: ignore
    from computer_use.router import choose_driver  # type: ignore
    from computer_use.drivers.playwright_driver import PlaywrightDriver  # type: ignore
    from computer_use.drivers.pyautogui_driver import PyAutoGUIDriver  # type: ignore

    target = target or {"kind": "desktop"}
    kind = (target or {}).get("kind", "desktop")

    # Route to a real driver. Browser targets use Playwright (real Chromium);
    # desktop targets use pyautogui (only meaningful on a host with a display).
    driver_choice = choose_driver(target)
    if kind == "browser" or driver_choice == "playwright":
        driver = PlaywrightDriver()
    else:
        driver = PyAutoGUIDriver()

    planner = Planner()
    policy = PolicyEngine(
        {
            "allow_actions": [
                "observe",
                "click",
                "double_click",
                "type",
                "hotkey",
                "navigate",
                "scroll",
                "wait",
                "assert_text",
                "assert_url",
                "upload",
                "submit",
                "done",
                "abort",
            ],
            "deny_actions": ["delete", "purchase", "send_money"],
        }
    )
    store = StateStore(_STATE_FILE)
    tracer = TraceCollector(_SIDECAR_ROOT)
    orch = Orchestrator(planner=planner, driver=driver, policy=policy, state_store=store, tracer=tracer)
    result = orch.run_once(goal=goal, target=target, chat_id=(chat_id or "default"), dry_run=dry_run)

    # Capture the trace for this run (goal, planned actions, per-action results).
    trace = tracer.events(result.data.get("run_id", "")) if result.data else []
    try:
        driver.close()
    except Exception:
        pass

    return {
        "ok": result.status in {"ok", "done"},
        "status": result.status,
        "message": result.message,
        "completed": result.completed,
        "chat_id": chat_id,
        "goal": goal,
        "target": target,
        "dry_run": dry_run,
        "run_id": result.data.get("run_id"),
        "trace": trace,
    }


def debug_snapshot(chat_id: str = "", query: str = "") -> dict:
    if not _ensure_loaded():
        return {
            "available": False,
            "reason": "computer-use sidecar not loaded",
            "chat_id": chat_id,
            "query": query,
        }

    from computer_use.debug import snapshot  # type: ignore

    snap = snapshot(_SIDECAR_ROOT)
    snap.update({"chat_id": chat_id, "query": query, "sidecar_src": str(_SIDECAR_SRC)})
    return snap
