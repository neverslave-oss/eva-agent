"""telegram_messaging.py — Telegram send/API primitives for the bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains the
Telegram send/API helpers: _call_api, send_message, delete_message,
edit_message, send_buttons, send_typing, send_file, send_voice.
Kept behavior-identical; telegram_bot.py re-imports these names so all call
sites and tests keep working unchanged.
"""
import os
import requests

API_BASE = f"https://api.telegram.org/bot{os.environ.get('KERNEL_EVO_TELEGRAM_BOT_TOKEN')}"

def _call_api(method: str, path: str, body: dict = None):
    """Internal helper to call Kernel's own API."""
    try:
        url = f"http://localhost:8779{path}"
        if method == "GET":
            r = requests.get(url, timeout=5)
        elif method == "DELETE":
            r = requests.delete(url, timeout=5)
        elif method == "POST":
            r = requests.post(url, json=body, timeout=5)
        else:
            return None
        return r.json() if r.ok else None
    except Exception:
        return None


def send_message(chat_id: str, text: str, parse_mode: str = "Markdown") -> int | None:
    """Send a message and return its message_id (or None on failure)."""
    try:
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        # Only include parse_mode when it's a valid string. Telegram rejects
        # `parse_mode: null` with "400 Bad Request: unsupported parse_mode".
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(
            f"{API_BASE}/sendMessage",
            json=payload,
            timeout=10,
        )
        data = r.json()
        if data.get("ok"):
            return data["result"]["message_id"]
        # Telegram rejected the message — log the actual API error for diagnosis.
        print(f"[bot] send_message API error: {data.get('error_code')} {data.get('description', '')[:200]}", flush=True)
    except Exception as e:
        print(f"[bot] send error: {e}")
    return None


def delete_message(chat_id: str, message_id: int) -> bool:
    """Delete a Telegram message (e.g. to replace a stale screenshot).

    Returns True on success. Failures are non-fatal — callers should treat a
    delete failure as "keep the old message" rather than erroring.
    """
    try:
        r = requests.post(
            f"{API_BASE}/deleteMessage",
            json={"chat_id": chat_id, "message_id": message_id},
            timeout=10,
        )
        data = r.json()
        return bool(data.get("ok"))
    except Exception:
        return False


def edit_message(chat_id: str, message_id: int, text: str, parse_mode: str = "Markdown") -> bool:
    """Edit an existing message. Falls back silently if it fails."""
    try:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        # Only include parse_mode when it's a valid string. Telegram rejects
        # `parse_mode: null` with "400 Bad Request: unsupported parse_mode".
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(
            f"{API_BASE}/editMessageText",
            json=payload,
            timeout=10,
        )
        data = r.json()
        if data.get("ok"):
            return True
        # Log the actual Telegram API error for diagnosis.
        print(f"[bot] edit_message API error: {data.get('error_code')} {data.get('description', '')[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"[bot] edit error: {e}")
    return False


def send_buttons(chat_id: str, text: str, buttons: list, parse_mode: str = "Markdown"):
    """Send a message with inline keyboard buttons.

    `parse_mode` defaults to "Markdown" for backward compatibility. Callers
    embedding arbitrary user/model content (e.g. the exec_shell auth prompt)
    should pass parse_mode="HTML" and HTML-escape their content, since Telegram's
    legacy Markdown parser rejects unescaped `_`/`*`/backticks with "can't parse
    entities" — silently dropping the message (and its buttons).
    """
    try:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": {"inline_keyboard": buttons},
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(
            f"{API_BASE}/sendMessage",
            json=payload,
            timeout=10,
        )
        if not r.ok:
            print(f"[bot] send_buttons API error: {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[bot] send_buttons error: {e}")


def send_typing(chat_id: str, action: str = "typing"):
    """Send a Telegram chat action indicator.

    action values (Telegram sendChatAction):
      typing          — ⌨️  text reply coming (default)
      record_voice    — 🎙️  recording/cloning audio
      upload_voice    — ⬆️  uploading audio
      upload_photo    — 🖼️  processing image
      upload_document — 📄  processing document
    """
    try:
        requests.post(
            f"{API_BASE}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
            timeout=3,
        )
    except Exception:
        pass


def send_file(chat_id: str, file_path: str, caption: str = "") -> bool:
    """Send a local file as a Telegram document attachment."""
    try:
        with open(file_path, "rb") as _f:
            r = requests.post(
                f"{API_BASE}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (os.path.basename(file_path), _f)},
                timeout=30,
            )
        return r.ok
    except Exception as e:
        print(f"[bot] send_file error: {e}")
        return False


def send_voice(chat_id: str, wav_path: str, caption: str = "") -> bool:
    """Send a local WAV file as a Telegram voice note."""
    try:
        with open(wav_path, "rb") as _f:
            r = requests.post(
                f"{API_BASE}/sendVoice",
                data={"chat_id": chat_id, "caption": caption},
                files={"voice": (os.path.basename(wav_path), _f, "audio/ogg")},
                timeout=60,
            )
        if not r.ok:
            print(f"[bot] send_voice error: {r.status_code} {r.text[:200]}")
        return r.ok
    except Exception as e:
        print(f"[bot] send_voice error: {e}")
        return False


