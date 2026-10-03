"""telegram_maintenance.py — bot maintenance/update/restart controllers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains
_restart_via_start_sh, _check_for_update, _update_check_loop, _handle_stop,
_handle_restart, _handle_update, _handle_rollback, _ensure_agent,
_sync_bot_commands. Kept behavior-identical; telegram_bot.py re-imports these
names so handle_message / start_bot_thread call sites keep working.
"""
import os
import subprocess
from telegram_config import CONFIG_PATH, REPO_DIR, API_BASE
from infra.updater import (
    get_current_version as _get_current_version,
    fetch_latest_version as _fetch_latest_version,
    do_update as _do_update,
)



def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot.send_message apply (same seam as telegram_providers).
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _sm(*a, **k):
    return _bot_module().send_message(*a, **k)


def _restart_via_start_sh():
    """Kill existing model_server, launch start.sh, then exit so the process restarts cleanly."""
    import subprocess as _sp
    # Explicitly kill any running model_server so start.sh picks up code changes
    _sp.call(["pkill", "-f", "src/model_server.py"], stderr=_sp.DEVNULL)
    import os as _os
    _sock = "/tmp/kernel_evolving_model.sock"
    if _os.path.exists(_sock):
        try:
            _os.remove(_sock)
        except Exception:
            pass
    subprocess.Popen(
        ["bash", os.path.join(REPO_DIR, "start.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import threading
    threading.Timer(2.0, lambda: os._exit(0)).start()


def _check_for_update(notify_chat_id: str = ""):
    """Check GitHub for a newer release. Optionally notify via Telegram."""
    current = _get_current_version()
    latest = _fetch_latest_version()
    if not latest:
        return
    _bot_module()._latest_version = latest
    if latest != current and notify_chat_id:
        _sm(
            notify_chat_id,
            f"🆕 Kernel v{latest} available (current: v{current}). /update to apply.",
        )
        print(f"[bot] Update available: {latest} (current: {current})")


def _update_check_loop(chat_id: str):
    """Background thread: check for updates every 6 hours."""
    time.sleep(30)
    while True:
        _check_for_update(notify_chat_id=chat_id)
        time.sleep(6 * 3600)


def _handle_stop(chat_id: str):
    """Emergency stop: halts the running agent/tool loop and frees the GPU,
    but keeps the bot alive so you can /restart or send further commands."""
    # Signal the running tool loop (/stop must stop the agent's tools loop).
    # Cross-process safe: the model_server process observes the same marker.
    try:
        from core.auth_gate import request_stop
        request_stop(str(chat_id))
    except Exception:
        pass
    _sm(chat_id, "🛑 Model stopped — GPU freed. I'm still here, say /restart to bring it back.")
    import subprocess as _sp
    _sp.call(["pkill", "-f", "src/model_server.py"], stderr=_sp.DEVNULL)
    _sp.call(["pkill", "-f", "kernel_evolving_model_server"], stderr=_sp.DEVNULL)
    import os as _os
    _sock = "/tmp/kernel_evolving_model.sock"
    if _os.path.exists(_sock):
        try:
            _os.remove(_sock)
        except Exception:
            pass


def _handle_restart(chat_id: str):
    """Restart the process without pulling. Called when user types /restart."""
    _sm(chat_id, "🔄 Restarting Kernel... back in ~30s.")
    try:
        _restart_via_start_sh()
    except Exception as e:
        _sm(chat_id, f"❌ Restart failed: {str(e)[:200]}")


def _handle_update(chat_id: str):
    """Execute git pull + restart via shared updater. Called when user types /update."""
    _do_update(
        notify=lambda msg: _sm(chat_id, msg),
        restart_fn=_restart_via_start_sh,
    )


def _handle_rollback(chat_id: str):
    """Revert one commit + restart. Called when user types /rollback."""
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", "HEAD~1"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"[bot] git reset: {result.stdout} {result.stderr}")
        _sm(chat_id, "⏪ Rolled back to previous version. Restarting...")
        _restart_via_start_sh()
    except Exception as e:
        _sm(chat_id, f"❌ Rollback failed: {str(e)[:200]}")


def _ensure_agent():
    if not _bot_module()._agent_ready:
        import core.inference.model as _m

        # Check if model server is running OR model is loaded in-process
        from core.inference.model_client import is_server_running as _is_srv
        if _m._model is None and not _is_srv():
            _m.load(CONFIG_PATH)
        import core.agent as agent
        agent.init(CONFIG_PATH)
        _bot_module()._agent_ready = True


def _sync_bot_commands():
    """Register all commands in the Telegram command picker.

    Layout:
      Core system commands  (no prefix)
      Installed skills      /skill_<name>  — prefix avoids collision with system cmds
      Installed routines    /run_<name>    — prefix avoids collision with /run <name> syntax
    Telegram limits: 100 commands max, command name ≤ 32 chars (a-z0-9_).
    """
    # ── Core system commands ──────────────────────────────────────────────────
    cmds = [
        {"command": "help",     "description": "Show command menu"},
        {"command": "init",     "description": "Initialize workspace/databases"},
        {"command": "status",   "description": "System + model status"},
        {"command": "new",      "description": "Start a fresh conversation"},
        {"command": "skills",   "description": "List installed skills"},
        {"command": "routines", "description": "List installed routines"},
        {"command": "models",   "description": "Switch / download models"},
        {"command": "thoughts", "description": "Recent internal thoughts"},
        {"command": "packages", "description": "Ecosystem packages (install/search)"},
        {"command": "replica",  "description": "Manage agent replicas"},
        {"command": "evolve",   "description": "Trigger evolution / inspect state"},
        {"command": "update",   "description": "Check for updates"},
        {"command": "restart",  "description": "Restart kernel-evolving"},
        {"command": "stop",     "description": "🛑 Emergency stop — kills everything"},
        {"command": "verbose",  "description": "Toggle verbose step output"},
        {"command": "version",  "description": "Show current version"},
    ]

    # ── Installed skills  →  /skill_<slug> ───────────────────────────────────
    try:
        import core.agent as _ag
        import re as _re
        def _slug(name: str, prefix_len: int) -> str:
            """a-z0-9_ only, max (32 - prefix_len) chars — Telegram limit 32 total."""
            return _re.sub(r"[^a-z0-9_]", "_", name.lower())[:32 - prefix_len]

        # Budget: 100 total - len(system cmds already added)
        _budget = 100 - len(cmds)
        _skill_budget = max(0, _budget - len(_ag._routines or []))
        _routine_budget = _budget - _skill_budget

        for s in (_ag._skills or [])[:_skill_budget]:
            slug = _slug(s.get("name", ""), prefix_len=6)  # "skill_" = 6
            if not slug:
                continue
            desc = (s.get("description") or "").strip()[:50] or f"Run skill: {s.get('name','?')[:30]}"
            cmds.append({"command": f"skill_{slug}", "description": desc})

        # ── Installed routines  →  /run_<slug> ───────────────────────────────
        for r in (_ag._routines or [])[:_routine_budget]:
            slug = _slug(r.get("name", ""), prefix_len=4)  # "run_" = 4
            if not slug:
                continue
            desc = (r.get("description") or "").strip()[:50] or f"Run routine: {r.get('name','?')[:30]}"
            cmds.append({"command": f"run_{slug}", "description": desc})

    except Exception as _e:
        print(f"[bot] setMyCommands: could not load skills/routines: {_e}", flush=True)

    # Telegram hard limit: 100 commands (already budgeted above, guard anyway)
    cmds = cmds[:100]

    try:
        r = requests.post(f"{API_BASE}/setMyCommands", json={"commands": cmds}, timeout=10)
        ok = False
        try:
            ok = r.json().get("ok", False)
        except Exception:
            pass
        print(f"[bot] setMyCommands: {'ok' if ok else 'failed'} ({len(cmds)} commands)", flush=True)
    except Exception as e:
        print(f"[bot] setMyCommands error: {e}", flush=True)

