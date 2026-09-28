"""
ask_questions_gate.py — Interactive user-question gate for Eva.

Lets Eva ask the user a question, present up to N inline-button options in
Telegram, and block until the user taps one (or times out). Returns the chosen
option label/idx so Eva's tool loop can continue with real user direction.

Mirrors the auth_gate / computer_confirm_gate pattern:
  request -> send_buttons -> block (event) -> resolve on callback -> return.

Usage from tools.py:
    from core.ask_questions_gate import ask_question
    answer = ask_question(chat_id, question, options, timeout=120)
    # answer -> {"selected_idx": int, "selected": str} or {"timeout": True}

Callback routing (telegram_bot.handle_callback):
    "aq_X_<request_id>_<option_idx>"  -> resolve_question(request_id, option_idx)
"""

import threading
import uuid

# ── Global registry: request_id -> {event, result} ───────────────────
_pending: dict[str, dict] = {}

# ── Configurable ──────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 120  # seconds

# Callback data prefix (handled in telegram_bot.handle_callback).
CALLBACK_PREFIX = "aq_"


def ask_question(chat_id: str, question: str, options: list[str],
                 timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Send a question with inline-button options to Telegram and block until the
    user taps one (or times out).

    Returns:
        {"selected_idx": int, "selected": str}  — user picked options[selected_idx]
        {"timeout": True}                       — no response within timeout
        {"error": str}                          — bot unavailable / send failure
    """
    options = [str(o).strip() for o in (options or []) if str(o).strip()]
    if not options:
        return {"error": "ask_questions requires at least one option"}

    request_id = str(uuid.uuid4())[:8]
    event = threading.Event()
    _pending[request_id] = {"event": event, "result": None}

    # Bot module may not be importable in tests / non-bot contexts.
    try:
        from services.channels.telegram_bot import send_buttons
        _bot_available = True
    except Exception:
        _bot_available = False

    if not _bot_available:
        _pending.pop(request_id, None)
        return {"error": "Telegram bot not available"}

    question_disp = str(question).replace("`", "'")[:400]

    # One option per row so long labels stay tappable. callback_data capped at
    # 64 bytes: "aq_<rid>_<idx>" is short; the label is rendered, not encoded.
    buttons = [
        [{"text": opt, "callback_data": f"{CALLBACK_PREFIX}_{request_id}_{i}"}]
        for i, opt in enumerate(options)
    ]

    try:
        send_buttons(chat_id, f"🤔 *Question*\n\n{question_disp}", buttons)
    except Exception as e:
        _pending.pop(request_id, None)
        return {"error": f"send failed: {e}"}

    # Block until callback or timeout.
    event.wait(timeout=timeout)

    entry = _pending.pop(request_id, None)
    if entry and entry.get("result") is not None:
        idx = int(entry["result"])
        return {"selected_idx": idx, "selected": options[idx]}
    return {"timeout": True}


def resolve_question(request_id: str, option_idx: int) -> None:
    """Called by the Telegram callback handler when the user taps an option."""
    entry = _pending.get(request_id)
    if entry:
        entry["result"] = option_idx
        entry["event"].set()


def get_pending_count() -> int:
    """Return the number of pending questions (for monitoring)."""
    return len(_pending)
