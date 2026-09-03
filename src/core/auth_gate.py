"""
auth_gate.py — Shell command authorization gate.

Before exec_shell runs a command, this module sends an inline-button
authorization request to the Telegram chat and blocks until the user
approves or denies (or times out).

Usage from tools.py:
    from core.auth_gate import request_auth
    result = request_auth(chat_id, command, timeout=60)
    if result != "allow":
        return "(authorization denied)"
"""

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime

# ── Global registry: request_id → {event, result} ───────────────────
_pending: dict[str, dict] = {}

# ── Cross-process pending-auth persistence ───────────────────────────
# request_auth() runs in the model_server process (the tool loop), while
# resolve_auth() — invoked by the Telegram bot's callback handler — runs in the
# API process. Each process has its OWN in-memory `_pending` dict, so a button
# tap in the API process could never resolve the blocking request in the
# model_server process (it would always time out, and the 'stop on denials'
# rule would then abort the tool loop and lose the conversation). We bridge the
# two processes by persisting each pending request to a tmp file keyed by
# request_id. request_auth() writes the request; resolve_auth() writes the
# decision to the same file (and also sets the in-process event so same-process
# callers — e.g. unit tests — keep working). request_auth() waits on its event
# AND polls the file for the cross-process decision.
_AUTH_DIR = os.path.join(tempfile.gettempdir(), "kernel_evolving_auth")
_POLL_INTERVAL = 0.25  # seconds between cross-process file polls

# ── Configurable ──────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 60  # seconds

# Commands that are always allowed (no auth needed)
SAFE_PREFIXES = (
    "echo ", "cat ", "ls ", "pwd", "whoami", "date", "hostname",
    "uname ", "git status", "git log", "git diff", "git branch",
    "python3 -c \"print", "which ", "type ", "head ", "tail ",
    "wc ", "grep ", "du ", "df ", "free", "uptime", "ip addr",
    "curl -sS http://localhost",  # health checks
    "ps ", "systemctl status", "nvidia-smi",
)

# Commands that are always blocked (never allowed even with auth)
BLOCKED_PATTERNS = (
    "rm -rf /", "mkfs", "dd if=", ":(){:|:&};:",  # fork bomb
)


def is_safe_command(cmd: str) -> bool:
    """Return True if the command is in the safe list (no auth needed)."""
    cmd_stripped = cmd.strip()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_stripped:
            return False
    for prefix in SAFE_PREFIXES:
        if cmd_stripped.startswith(prefix):
            return True
    return False


# ── Cross-process pending-auth persistence helpers ───────────────────

def _auth_path(request_id: str) -> str:
    """Return the tmp file path backing a pending auth request."""
    return os.path.join(_AUTH_DIR, f"{request_id}.json")


def _persist_pending(request_id: str, chat_id: str) -> None:
    """Write a pending auth request to disk so the bot process can resolve it."""
    try:
        os.makedirs(_AUTH_DIR, exist_ok=True)
        with open(_auth_path(request_id), "w") as _f:
            json.dump({"chat_id": chat_id, "result": None}, _f)
    except Exception:
        pass


def _read_persisted_result(request_id: str):
    """Return the persisted decision for a request, or None if not yet decided.

    Written by request_auth() (model_server process) and updated by resolve_auth()
    (API/bot process). Returns None while still pending.
    """
    try:
        with open(_auth_path(request_id)) as _f:
            data = json.load(_f)
        return data.get("result")
    except Exception:
        return None


def _persist_decision(request_id: str, result: str) -> None:
    """Write a decision for a pending request so the model_server can observe it."""
    try:
        with open(_auth_path(request_id), "w") as _f:
            json.dump({"chat_id": None, "result": result}, _f)
    except Exception:
        pass


def _drop_pending(request_id: str) -> None:
    """Remove the persisted pending-request file (after resolution/timeout)."""
    try:
        if os.path.exists(_auth_path(request_id)):
            os.remove(_auth_path(request_id))
    except Exception:
        pass


def request_auth(chat_id: str, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Send an authorization request to Telegram and block until response.

    Returns:
        "allow"  — user approved
        "deny"   — user denied
        "timeout" — no response within timeout
    """
    if is_safe_command(command):
        return "allow"

    request_id = str(uuid.uuid4())[:8]
    event = threading.Event()
    _pending[request_id] = {"event": event, "result": None, "chat_id": chat_id}
    # Persist so the bot process (different process) can resolve this request.
    _persist_pending(request_id, chat_id)

    # Import here to avoid circular imports at module level.
    # If the bot is not available (e.g. during tests), deny dangerous commands.
    try:
        from services.channels.telegram_bot import send_buttons, edit_message
        _bot_available = True
    except ImportError:
        _bot_available = False
    except Exception:
        _bot_available = False

    if not _bot_available:
        _pending.pop(request_id, None)
        _drop_pending(request_id)
        # No bot available — deny dangerous commands by default
        return "deny"

    # Use HTML parse mode (not Markdown) because the command is arbitrary shell
    # text that frequently contains `_`, `*`, backticks, etc. — Telegram's legacy
    # Markdown parser rejects those with "can't parse entities", silently dropping
    # the message and its buttons.
    cmd_display = command.replace("`", "'")[:200]
    if len(command) > 200:
        cmd_display += "…"
    cmd_html = (
        cmd_display.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    text = (
        "⚠️ <b>Shell Authorization Required</b>\n\n"
        f"<pre>{cmd_html}</pre>\n\n"
        "Approve this command?"
    )
    buttons = [
        [
            {"text": "✅ Allow", "callback_data": f"auth_allow_{request_id}"},
            {"text": "❌ Deny", "callback_data": f"auth_deny_{request_id}"},
        ]
    ]

    try:
        send_buttons(chat_id, text, buttons, parse_mode="HTML")
    except Exception as e:
        _pending.pop(request_id, None)
        _drop_pending(request_id)
        return f"deny (send error: {e})"

    # Block until callback or timeout. The event is set by resolve_auth() in THIS
    # process (same-process callers / tests); the persisted file is written by
    # resolve_auth() in the bot process (cross-process). Poll both so a button tap
    # in the separate API process resolves this request.
    deadline = time.monotonic() + timeout
    result = None
    while True:
        if event.is_set():
            break
        persisted = _read_persisted_result(request_id)
        if persisted is not None:
            result = persisted
            break
        if time.monotonic() >= deadline:
            break
        event.wait(timeout=min(_POLL_INTERVAL, max(0.0, deadline - time.monotonic())))

    _pending.pop(request_id, None)
    _drop_pending(request_id)
    if result is None:
        result = "timeout"
    return result


def resolve_auth(request_id: str, approved: bool) -> None:
    """
    Called by the Telegram callback handler when the user taps Allow/Deny.

    Writes the decision to the persisted request file so the model_server
    process (which runs request_auth in a separate process) can observe it, in
    addition to setting the local in-process event for same-process callers.
    """
    result = "allow" if approved else "deny"
    # Persist the decision for the model_server process to observe (cross-process).
    _persist_decision(request_id, result)
    entry = _pending.get(request_id)
    if entry:
        entry["result"] = result
        entry["event"].set()


def get_pending_count() -> int:
    """Return number of pending auth requests (for monitoring)."""
    return len(_pending)