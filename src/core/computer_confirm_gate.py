"""
computer_confirm_gate.py — Interactive confirmation gate for risky computer-use actions.

Before the LLM planner runs a risky action (delete, purchase, send_money,
submit, non-allowlisted hotkey), this module sends an inline-button
Allow/Deny request to the Telegram chat and blocks until the user responds
(or times out).

Mirrors the auth_gate pattern (request -> inline buttons -> block -> resolve).

Usage from computer_use_bridge:
    from core.computer_confirm_gate import request_confirm
    approved = request_confirm(chat_id, action_kind, description, timeout=60)
"""

import threading
import time
import uuid

# ── Global registry: request_id → {event, result} ───────────────────
_pending: dict[str, dict] = {}

# ── Configurable ──────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 60  # seconds

# Callback data prefixes (handled in telegram_bot.handle_callback).
ALLOW_PREFIX = "cu_allow_"
DENY_PREFIX = "cu_deny_"


def request_confirm(chat_id: str, action_kind: str, description: str = "",
                    timeout: int = DEFAULT_TIMEOUT) -> bool:
    """
    Send an Allow/Deny request to Telegram and block until the user responds.

    Returns:
        True  — user approved
        False — user denied, timed out, or the bot is unavailable (fail-closed)
    """
    request_id = str(uuid.uuid4())[:8]
    event = threading.Event()
    _pending[request_id] = {"event": event, "result": None}

    try:
        from services.channels.telegram_bot import send_buttons
        _bot_available = True
    except Exception:
        _bot_available = False

    if not _bot_available:
        _pending.pop(request_id, None)
        return False  # fail-closed: no bot -> deny

    desc = (description or action_kind).replace("`", "'")[:200]
    text = (
        f"⚠️ *Risky computer-use action*\n\n"
        f"`{action_kind}` — {desc}\n\n"
        f"Allow Eva to run this?"
    )
    buttons = [
        [
            {"text": "✅ Allow", "callback_data": f"{ALLOW_PREFIX}{request_id}"},
            {"text": "❌ Deny", "callback_data": f"{DENY_PREFIX}{request_id}"},
        ]
    ]

    try:
        send_buttons(chat_id, text, buttons)
    except Exception as e:
        _pending.pop(request_id, None)
        return False

    event.wait(timeout=timeout)

    entry = _pending.pop(request_id, None)
    if entry and entry.get("result") is not None:
        return bool(entry["result"])
    return False  # timeout -> deny


def resolve_confirm(request_id: str, approved: bool) -> None:
    """Called by the Telegram callback handler when the user taps Allow/Deny."""
    entry = _pending.get(request_id)
    if entry:
        entry["result"] = bool(approved)
        entry["event"].set()


def get_pending_count() -> int:
    """Return number of pending confirm requests (for monitoring)."""
    return len(_pending)
