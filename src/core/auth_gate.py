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

import threading
import time
import uuid
from datetime import datetime

# ── Global registry: request_id → {event, result} ───────────────────
_pending: dict[str, dict] = {}

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
    _pending[request_id] = {"event": event, "result": None}

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
        # No bot available — deny dangerous commands by default
        return "deny"

    # Escape for Markdown
    cmd_display = command.replace("`", "'")[:200]
    if len(command) > 200:
        cmd_display += "…"

    text = (
        f"⚠️ *Shell Authorization Required*\n\n"
        f"```\n{cmd_display}\n```\n\n"
        f"Approve this command?"
    )
    buttons = [
        [
            {"text": "✅ Allow", "callback_data": f"auth_allow_{request_id}"},
            {"text": "❌ Deny", "callback_data": f"auth_deny_{request_id}"},
        ]
    ]

    try:
        send_buttons(chat_id, text, buttons)
    except Exception as e:
        _pending.pop(request_id, None)
        return f"deny (send error: {e})"

    # Block until callback or timeout
    event.wait(timeout=timeout)

    entry = _pending.pop(request_id, None)
    result = "timeout"
    if entry and entry.get("result") is not None:
        result = entry["result"]

    return result


def resolve_auth(request_id: str, approved: bool) -> None:
    """
    Called by the Telegram callback handler when the user taps Allow/Deny.
    """
    entry = _pending.get(request_id)
    if entry:
        entry["result"] = "allow" if approved else "deny"
        entry["event"].set()


def get_pending_count() -> int:
    """Return number of pending auth requests (for monitoring)."""
    return len(_pending)