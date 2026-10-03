"""telegram_poll.py — Telegram long-poll loop + resilience helpers for the bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains
start_bot_thread, the poll-backoff helpers (_is_dns_resolution_error,
_compute_poll_backoff_seconds), and the poll() loop. Mutable poll telemetry
counters stay on the bot module (tests assert on them); poll() resolves them
through the bot-module seam. Behavior-identical; telegram_bot.py re-imports
these names.
"""
import json
import os
import random
import threading
import time

import requests
from telegram_config import BOT_TOKEN, ALLOWED_CHAT_ID, API_BASE


def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot.random/bot.time/bot._POLL_* apply (same seam as the other modules).
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def start_bot_thread():
    """Start the Telegram polling loop as a daemon thread.
    Assumes the model is already loaded (called from API startup event).
    """
    if not BOT_TOKEN:
        print("[bot] KERNEL_EVO_TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")
        return
    import core.inference.model as _m
    import core.agent as _agent_mod

    from core.inference.model_client import is_server_running as _is_srv
    if _m._model is not None or _is_srv():
        _bot_module()._agent_ready = True

    _bot_module()._sync_bot_commands()

    t = threading.Thread(target=poll, daemon=True, name="kernel-telegram-bot")
    t.start()
    print(
        f"[bot] Telegram bot thread started (allowed chat: {ALLOWED_CHAT_ID or 'all'})"
    )
    if ALLOWED_CHAT_ID:
        u = threading.Thread(
            target=_bot_module()._update_check_loop,
            args=(ALLOWED_CHAT_ID,),
            daemon=True,
            name="kernel-update-checker",
        )
        u.start()
        print("[bot] Update checker started (6h interval)")


OFFSET_FILE = "/tmp/kernel_evolving_telegram_offset"

# Poll loop resilience and lightweight telemetry
_POLL_BACKOFF_BASE_SECONDS = float(os.environ.get("KERNEL_EVO_POLL_BACKOFF_BASE_SECONDS", "5"))
_POLL_BACKOFF_MAX_SECONDS = float(os.environ.get("KERNEL_EVO_POLL_BACKOFF_MAX_SECONDS", "60"))
_POLL_BACKOFF_DNS_MULTIPLIER = float(os.environ.get("KERNEL_EVO_POLL_BACKOFF_DNS_MULTIPLIER", "2"))
_POLL_BACKOFF_JITTER_MAX_SECONDS = float(os.environ.get("KERNEL_EVO_POLL_BACKOFF_JITTER_MAX_SECONDS", "1"))


def _is_dns_resolution_error(exc: Exception) -> bool:
    """Best-effort classifier for DNS resolution failures from requests/urllib."""
    text = str(exc)
    return (
        "NameResolutionError" in text
        or "Failed to resolve" in text
        or "Temporary failure in name resolution" in text
    )


def _compute_poll_backoff_seconds(failure_streak: int, dns_failure: bool = False, jitter_seconds: float | None = None) -> float:
    """Exponential backoff with optional DNS multiplier and bounded jitter."""
    if failure_streak <= 0:
        return 0.0
    delay = _POLL_BACKOFF_BASE_SECONDS * (2 ** (failure_streak - 1))
    if dns_failure:
        delay *= _POLL_BACKOFF_DNS_MULTIPLIER
    delay = min(_POLL_BACKOFF_MAX_SECONDS, delay)
    if jitter_seconds is None:
        jitter_seconds = random.uniform(0, _POLL_BACKOFF_JITTER_MAX_SECONDS)
    return min(_POLL_BACKOFF_MAX_SECONDS, delay + max(0.0, jitter_seconds))


def poll():
    """Long-poll Telegram for updates."""
    _bm = _bot_module()

    # Restore offset from last run so we don't replay already-seen updates
    offset = None
    try:
        with open(OFFSET_FILE) as _f:
            offset = int(_f.read().strip())
        print(f"[bot] Restored Telegram offset: {offset}")
    except Exception:
        pass
    print(f"[bot] Kernel Telegram bot starting...")

    failure_streak = 0

    while True:
        poll_started = time.monotonic()
        try:
            # Telegram expects allowed_updates as a JSON-encoded array, not repeated query params.
            # If encoded incorrectly, normal messages may work while callback_query updates never arrive.
            params = {"timeout": 30, "allowed_updates": json.dumps(["message", "callback_query"])}
            if offset:
                params["offset"] = offset

            resp = requests.get(f"{API_BASE}/getUpdates", params=params, timeout=35)
            data = resp.json()
            updates = data.get("result", [])
            latency_ms = int((time.monotonic() - poll_started) * 1000)

            _bm._POLL_LAST_SUCCESS_EPOCH = time.time()
            if failure_streak > 0:
                print(
                    f"[bot] Poll recovered after {failure_streak} failure(s): "
                    f"latency_ms={latency_ms} updates={len(updates)}"
                )
            failure_streak = 0
            print(f"[bot] Poll ok: latency_ms={latency_ms} updates={len(updates)}", flush=True)

            for update in updates:
                offset = update["update_id"] + 1
                # Persist offset so restarts don't replay seen updates
                try:
                    with open(OFFSET_FILE, "w") as _f:
                        _f.write(str(offset))
                except Exception:
                    pass
                # Handle callback queries (button taps)
                cb = update.get("callback_query", {})
                if cb:
                    cb_chat = cb.get("message", {}).get("chat", {}).get("id")
                    cb_data = cb.get("data", "")
                    cb_msg_id = cb.get("message", {}).get("message_id")
                    try:
                        resp = requests.post(
                            f"{API_BASE}/answerCallbackQuery",
                            json={"callback_query_id": cb.get("id", ""), "text": "Running…", "show_alert": False},
                            timeout=5,
                        )
                        try:
                            ok = resp.json().get("ok", False)
                            print(f"[bot] callback ack: data={cb_data} ok={ok}", flush=True)
                        except Exception:
                            print(f"[bot] callback ack: data={cb_data} non-json response", flush=True)
                    except Exception as e:
                        print(f"[bot] callback ack error for {cb_data}: {e}", flush=True)
                    if cb_chat and cb_data:
                        threading.Thread(
                            target=_bot_module().handle_callback,
                            args=(str(cb_chat), cb_data, cb_msg_id),
                            daemon=True,
                        ).start()
                    continue

                msg = update.get("message", {})
                chat_id = msg.get("chat", {}).get("id")
                text = msg.get("text", "")
                sender_name = msg.get("from", {}).get("first_name", "") or msg.get("from", {}).get("username", "")
                # Extract media
                photos = msg.get("photo", [])
                photo_file_id = photos[-1]["file_id"] if photos else ""  # largest size
                voice = msg.get("voice", {}) or msg.get("audio", {})
                voice_file_id = voice.get("file_id", "")
                # Extract document (PDF, text files, etc.)
                document = msg.get("document", {})
                document_file_id = document.get("file_id", "")
                document_name = document.get("file_name", "document")
                document_mime = document.get("mime_type", "")
                caption = msg.get("caption", "")
                # Use caption as text for media messages
                effective_text = text or caption

                if chat_id and (effective_text or photo_file_id or voice_file_id or document_file_id):
                    # Send typing indicator immediately (before thread starts)
                    _bot_module().send_typing(str(chat_id))
                    # Handle in thread so polling doesn't block
                    threading.Thread(
                        target=_bot_module().handle_message,
                        args=(str(chat_id), effective_text, sender_name, photo_file_id, voice_file_id, document_file_id, document_name, document_mime),
                        daemon=True
                    ).start()

        except KeyboardInterrupt:
            print("[bot] Stopped.")
            break
        except Exception as e:
            failure_streak += 1
            _bm._POLL_FAILURES_TOTAL += 1
            dns_failure = _is_dns_resolution_error(e)
            if dns_failure:
                _bm._POLL_DNS_FAILURES_TOTAL += 1
            backoff = _compute_poll_backoff_seconds(failure_streak, dns_failure=dns_failure)
            print(
                f"[bot] Poll error: type={type(e).__name__} dns={dns_failure} "
                f"streak={failure_streak} backoff_s={backoff:.1f} "
                f"dns_failures_total={_bm._POLL_DNS_FAILURES_TOTAL} err={e}",
                flush=True,
            )
            time.sleep(backoff)


