"""api/helpers.py — shared helpers for the Kernel Evolving API.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Self-contained
helpers: config-path resolution, interaction logging, builtin-command dispatch,
conversations repo, workspace path safety. Re-exported from api.__init__ so
existing call sites (api._resolve_config_path etc.) keep working.
"""
import json
import os
import time
import uuid

from runtime_paths import LOGS_DIR, WORKSPACE_ROOT


def _resolve_config_path() -> str:
    """Resolve the active config file.

    Honors KERNEL_EVO_CONFIG (set by start.sh --config=... so an alternate
    config can be used on demand). Falls back to the repo's config.yaml.
    """
    override = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    # src/api/helpers.py -> src/api -> src -> repo root
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(_root, "config.yaml")


_LOG_FILE = os.path.join(str(LOGS_DIR), "kernel_calls.jsonl")


def _log_interaction(prompt: str, reply: str, source: str = "api", meta: dict | None = None) -> None:
    """Append one JSONL record to the interaction log (non-blocking best-effort)."""
    try:
        record = {
            "id":        str(uuid.uuid4()),
            "timestamp": time.time(),
            "source":    source,
            "prompt":    prompt,
            "reply":     reply,
            "meta":      meta or {},
        }
        with open(_LOG_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[logger] warning: could not write interaction log: {exc}")


def _run_builtin_command(text: str, chat_id: str) -> dict:
    """Execute a Telegram-style slash command and capture text/button output."""
    import services.channels.telegram_bot as _tb

    captured_texts: list[str] = []
    captured_buttons: list = []
    _orig_send = _tb.send_message
    _orig_send_buttons = _tb.send_buttons

    def _capture(_chat_id, msg, **kwargs):
        captured_texts.append(str(msg))

    def _capture_buttons(_chat_id, msg, buttons, **kwargs):
        captured_texts.append(str(msg))
        captured_buttons.append(buttons)

    _tb.send_message = _capture
    _tb.send_buttons = _capture_buttons
    try:
        _tb.handle_message(chat_id or _tb.ALLOWED_CHAT_ID or "api", text)
    finally:
        _tb.send_message = _orig_send
        _tb.send_buttons = _orig_send_buttons

    reply = "\n".join(captured_texts) if captured_texts else ""
    buttons = captured_buttons[-1] if captured_buttons else []
    return {"reply": reply, "buttons": buttons}


def _conversations_repo():
    """Return a ConversationsRepository bound to the running agent's conversations DB."""
    from runtime_paths import CONVERSATIONS_DB
    from database.agent import ConversationsRepository
    return ConversationsRepository(db_path=CONVERSATIONS_DB)


def _safe_workspace_path(relative: str):
    """Resolve a relative path inside WORKSPACE_ROOT. Returns None if unsafe."""
    import pathlib
    rel = relative.strip().lstrip("/")
    if not rel or ".." in pathlib.PurePosixPath(rel).parts:
        return None
    try:
        resolved = (WORKSPACE_ROOT / rel).resolve()
        if not str(resolved).startswith(str(WORKSPACE_ROOT.resolve())):
            return None
        return resolved
    except Exception:
        return None
