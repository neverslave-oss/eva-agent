"""computer_use_bridge — kernel-side bridge to computer-use sidecar.

No-op safe by default: if sidecar is missing/broken, callers receive harmless
fallback values so kernel boot and normal tool loops are unaffected.
"""

from __future__ import annotations

import os
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


def _default_watch_callback(chat_id: str):
    """Return a watch callback that streams screenshots to Telegram.

    Returns None (no-op) when there's no chat_id or the telegram channel can't
    be imported, so the LLM planner degrades gracefully outside a chat context.
    The screenshot may be a base64 data-URI (from the driver) or a local path;
    we write it to a temp PNG and send it as a document.
    """
    if not chat_id:
        return None
    try:
        from services.channels import telegram_bot as _tb
    except Exception:
        return None

    # Track the last screenshot message_id so each new screenshot replaces the
    # previous one (delete-then-send), preventing the chat from filling up with
    # screenshots during a long computer-use run.
    _last_shot_msg_id = {"id": None}

    def _watch(screenshot, caption):
        # Always stream a text update so the user sees live progress even when
        # no screenshot is available (e.g. screenshot null / driver has no
        # capture). This turns the silent typing-indicator wait into a
        # readable step-by-step feed of what Eva is doing.
        try:
            if caption:
                _tb.send_message(chat_id, f"🖥️ {caption}")
        except Exception as e:
            print(f"[bridge] watch text send failed: {e}", flush=True)
        if not screenshot:
            return
        try:
            import base64 as _b64
            import tempfile
            if screenshot.startswith("data:image"):
                b64 = screenshot.split(",", 1)[1]
                data = _b64.b64decode(b64)
                fd, path = tempfile.mkstemp(suffix=".png")
                with open(fd, "wb") as f:
                    f.write(data)
                try:
                    # Replace the previous screenshot (delete-then-send) so the
                    # chat only ever holds the latest frame.
                    if _last_shot_msg_id["id"] is not None:
                        try:
                            _tb.delete_message(chat_id, _last_shot_msg_id["id"])
                        except Exception:
                            pass
                    _mid = _tb.send_file(chat_id, path, caption=caption or "")
                    if _mid:
                        _last_shot_msg_id["id"] = _mid
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            elif Path(screenshot).exists():
                if _last_shot_msg_id["id"] is not None:
                    try:
                        _tb.delete_message(chat_id, _last_shot_msg_id["id"])
                    except Exception:
                        pass
                _mid = _tb.send_file(chat_id, screenshot, caption=caption or "")
                if _mid:
                    _last_shot_msg_id["id"] = _mid
        except Exception as e:
            print(f"[bridge] watch send failed: {e}", flush=True)

    return _watch


def _default_confirm_callback(chat_id: str):
    """Return an interactive confirm callback for risky actions.

    Sends an Allow/Deny inline-button request to Telegram and blocks until the
    user responds (or times out, which denies fail-closed). Mirrors the
    auth_gate pattern. Returns None when there's no chat_id or the bot can't
    be reached, so the planner degrades to auto-deny outside a chat context.
    """
    if not chat_id:
        return None

    def _confirm(action):
        try:
            from core.computer_confirm_gate import request_confirm
            desc = action.text or action.selector or action.url or ""
            return request_confirm(chat_id, action.kind, desc)
        except Exception as e:
            print(f"[bridge] confirm gate error: {e}", flush=True)
            return False

    return _confirm


def _default_stuck_callback(chat_id: str):
    """Return an interactive callback for the 'stuck' guard.

    After several consecutive rejected actions the model is looping on the
    same screen. This asks the user to Continue or Stop via Telegram inline
    buttons and blocks until they respond. Returns None when there's no chat
    id or the bot can't be reached, so the planner aborts fail-closed.
    """
    if not chat_id:
        return None

    def _stuck(rejection):
        try:
            from core.computer_confirm_gate import request_confirm
            # Reuse the confirm gate's Allow/Deny prompt: Allow = continue,
            # Deny = stop. The description carries the rejection reason.
            return request_confirm(chat_id, "continue", rejection)
        except Exception as e:
            print(f"[bridge] stuck gate error: {e}", flush=True)
            return False

    return _stuck


def run_computer_task(
    chat_id: str = "",
    goal: str = "",
    target: dict | None = None,
    dry_run: bool = True,
    llm: bool = False,
    step_cap: int = 10,
    confirm_callback=None,
    watch_callback=None,
    stuck_callback=None,
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
                "launch",
                "done",
                "abort",
            ],
            "deny_actions": ["delete", "purchase", "send_money"],
        }
    )
    store = StateStore(_STATE_FILE)
    tracer = TraceCollector(_SIDECAR_ROOT)

    if llm:
        # LLM-driven perceive->decide->act planner (vision brain). Runs its own
        # loop and returns a batch ending in `done`, so the orchestrator is
        # unchanged. Falls back to the deterministic Planner if the vision brain
        # is unavailable (no provider / inference failure).
        from computer_use.llm_planner import LLMPlanner  # type: ignore
        from computer_use.vision_brain import VisionBrain  # type: ignore

        # Default watch mode: stream each step's screenshot to Telegram (only
        # when a chat_id is present and the telegram channel is importable).
        if watch_callback is None:
            watch_callback = _default_watch_callback(chat_id)
        # Default confirm gate: prompt on Telegram and auto-deny (fail-closed).
        # A caller can supply a real interactive confirm_callback to allow risky
        # actions; without one, risky actions are blocked.
        if confirm_callback is None:
            confirm_callback = _default_confirm_callback(chat_id)
        # Default stuck gate: after repeated rejections, ask the user to
        # Continue or Stop instead of looping forever.
        if stuck_callback is None:
            stuck_callback = _default_stuck_callback(chat_id)

        planner = LLMPlanner(
            brain=VisionBrain(),
            driver=driver,
            policy=policy,
            step_cap=step_cap,
            dry_run=dry_run,
            confirm_callback=confirm_callback,
            watch_callback=watch_callback,
            stuck_callback=stuck_callback,
        )
    else:
        planner = Planner()

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
