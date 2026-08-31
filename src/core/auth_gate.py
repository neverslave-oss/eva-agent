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

import os
import tempfile
import threading
import time
import uuid
from datetime import datetime

# ── Global registry: request_id → {event, result} ───────────────────
_pending: dict[str, dict] = {}

# ── Auto-approve ("Approve all") grant ───────────────────────────────
# chat_id -> monotonically-increasing timestamp when the grant was set.
# Transient: lives only for the currently-executing tool loop and is
# cleared when the loop ends or expires (see AUTO_ALLOW_TTL). Never
# persisted to disk.
_auto_allow: dict[str, float] = {}

# How long (seconds) an "Approve all" grant stays valid. This is the
# safety net that guarantees a grant can never linger if a caller path
# forgets to clear it explicitly.
AUTO_ALLOW_TTL = 600  # 10 minutes

# ── Consecutive-deny counter ─────────────────────────────────────────
# chat_id -> number of consecutive non-approvals (Deny or timeout) for the
# current task. Reset to 0 on any approval. When it reaches MAX_CONSECUTIVE_DENIES
# the tool loop is signalled to stop (see request_stop).
MAX_CONSECUTIVE_DENIES = 3
_deny_count: dict[str, int] = {}

# ── Stop signal (per chat) ───────────────────────────────────────────
# Used by the /stop command and the 3-deny rule to halt a running tool loop.
# The bot and model_server are separate processes, so the marker is persisted
# to a tmp file keyed by chat_id in addition to an in-process event. The tool
# loop checks is_stop_requested() between steps and aborts.
_STOP_DIR = os.path.join(tempfile.gettempdir(), "kernel_evolving_stops")
_stop_events: dict[str, threading.Event] = {}
_STOP_LOCK = threading.Lock()

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


def is_blocked_command(cmd: str) -> bool:
    """Return True if the command matches a BLOCKED_PATTERNS entry.

    Dangerous commands in BLOCKED_PATTERNS are never allowed — not even under
    an active "Approve all" grant. This is checked separately from
    is_safe_command() so a blocked pattern short-circuits to deny even when
    the auto-approve grant would otherwise let the command through.
    """
    cmd_stripped = cmd.strip()
    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_stripped:
            return True
    return False


def request_auth(chat_id: str, command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Send an authorization request to Telegram and block until response.

    Returns:
        "allow"     — user approved (current command only)
        "allow_all" — current command and every subsequent exec_shell
                      in the current task loop runs without prompting
        "deny"      — user denied
        "timeout"   — no response within timeout
    """
    # Dangerous blocked patterns are never auto-approved, even under an
    # active "Approve all" grant.
    if is_blocked_command(command):
        return "deny"

    if is_safe_command(command):
        return "allow"

    # ── Auto-approve grant ("Approve all") ────────────────────────────
    # Checked AFTER the blocked/safe checks above so that dangerous
    # commands (BLOCKED_PATTERNS) still short-circuit to deny even under
    # an active grant. An expired grant is dropped immediately.
    if _auto_allow_expired(chat_id):
        clear_auto_allow(chat_id)
    elif _auto_allow.get(chat_id) is not None:
        return "allow"

    request_id = str(uuid.uuid4())[:8]
    event = threading.Event()
    _pending[request_id] = {"event": event, "result": None, "chat_id": chat_id}

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
        return _record_denial(chat_id)

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
            {"text": "✅ Approve all", "callback_data": f"auth_allow_all_{request_id}"},
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

    # Track consecutive denials — 3 in a row stops the tool loop.
    if result in ("deny", "timeout"):
        return _record_denial(chat_id)
    # Any approval resets the streak.
    _deny_count.pop(chat_id, None)
    return result


def resolve_auth(request_id: str, approved: bool, allow_all: bool = False) -> None:
    """
    Called by the Telegram callback handler when the user taps Allow/Deny.

    Args:
        request_id: the pending request to resolve.
        approved: True for Allow, False for Deny.
        allow_all: when True (and approved), grants auto-approval for the
            current task loop in addition to approving the current command.
    """
    entry = _pending.get(request_id)
    if entry:
        if allow_all and approved:
            entry["result"] = "allow_all"
            chat_id = entry.get("chat_id")
            if chat_id:
                _auto_allow[chat_id] = time.monotonic()
        else:
            entry["result"] = "allow" if approved else "deny"
        entry["event"].set()


def clear_auto_allow(chat_id: str) -> None:
    """Clear any active auto-approve grant for the given chat."""
    _auto_allow.pop(chat_id, None)


def _auto_allow_expired(chat_id: str) -> bool:
    """Return True if an active grant for chat_id has exceeded AUTO_ALLOW_TTL."""
    ts = _auto_allow.get(chat_id)
    if ts is None:
        return False
    return (time.monotonic() - ts) > AUTO_ALLOW_TTL


def _record_denial(chat_id: str) -> str:
    """Increment the consecutive-deny counter and stop the loop at the cap.

    Returns the "deny" result string unchanged so callers behave as before;
    the side effect is signalling the tool loop to stop after
    MAX_CONSECUTIVE_DENIES consecutive non-approvals.
    """
    if chat_id:
        n = _deny_count.get(chat_id, 0) + 1
        _deny_count[chat_id] = n
        if n >= MAX_CONSECUTIVE_DENIES:
            request_stop(chat_id)
    return "deny"


def clear_deny_count(chat_id: str) -> None:
    """Reset the consecutive-deny counter for a chat (called at task start)."""
    _deny_count.pop(chat_id, None)


def _stop_path(chat_id: str) -> str:
    return os.path.join(_STOP_DIR, f"stop_{chat_id}")


def request_stop(chat_id: str) -> None:
    """Signal the agent and its tool loop to stop for the given chat.

    Cross-process safe: sets an in-process event AND writes a tmp marker file
    so the model_server process (which runs the tool loop) can observe it.
    """
    if not chat_id:
        return
    try:
        os.makedirs(_STOP_DIR, exist_ok=True)
        with open(_stop_path(chat_id), "w") as _f:
            _f.write("1")
    except Exception:
        pass
    with _STOP_LOCK:
        _stop_events.setdefault(chat_id, threading.Event()).set()


def is_stop_requested(chat_id: str) -> bool:
    """Return True if a stop has been requested for the chat (either process)."""
    if not chat_id:
        return False
    with _STOP_LOCK:
        _ev = _stop_events.get(chat_id)
        if _ev is not None and _ev.is_set():
            return True
    try:
        return os.path.exists(_stop_path(chat_id))
    except Exception:
        return False


def clear_stop(chat_id: str) -> None:
    """Clear any stop signal for a chat (called when a new task starts)."""
    if not chat_id:
        return
    try:
        if os.path.exists(_stop_path(chat_id)):
            os.remove(_stop_path(chat_id))
    except Exception:
        pass
    with _STOP_LOCK:
        _ev = _stop_events.pop(chat_id, None)
        if _ev is not None:
            _ev.clear()


def clear_task_state(chat_id: str) -> None:
    """Reset all transient per-task auth state for a chat.

    Called at the start of each task's tool loop so nothing (approve-all grant,
    deny counter, or a stale stop signal) leaks into an unrelated later task.
    """
    clear_auto_allow(chat_id)
    clear_deny_count(chat_id)
    clear_stop(chat_id)


def get_pending_count() -> int:
    """Return number of pending auth requests (for monitoring)."""
    return len(_pending)