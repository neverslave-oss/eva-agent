"""telegram_voice.py — voice/clone/file-download helpers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains the
voice-sample listing, TTS voice-clone synthesis (_clone_voice_reply), the
TypingKeepAlive context manager, and Telegram file download. Kept
behavior-identical; telegram_bot.py re-imports these names.
"""
import os
import tempfile
import threading
import time

import requests
from telegram_config import CONFIG_PATH, API_BASE, BOT_TOKEN
from core.voice_activity import voice_activity

_VOICE_SAMPLES_DIR = os.path.expanduser(os.environ.get(
    "KERNEL_VOICE_SAMPLES_DIR",
    os.path.expanduser("~/.openclaw/media/voice-samples")
))
_DEFAULT_VOICE_SAMPLE = os.environ.get(
    "KERNEL_DEFAULT_VOICE_SAMPLE",
    os.path.join(_VOICE_SAMPLES_DIR, "default-en-phonetic.wav")
)


def _bot_module():
    # Resolve the registered bot module at call time so test patches on
    # bot._active_voice_sample / bot.send_typing apply (same seam as the
    # other split modules).
    import sys
    tb = sys.modules.get("services.channels.telegram_bot")
    if tb is None:  # pragma: no cover - defensive
        import importlib
        tb = importlib.import_module("services.channels.telegram_bot")
    return tb


def _list_voice_samples() -> list[dict]:
    """Return available voice samples as [{name, path}]."""
    if not os.path.isdir(_VOICE_SAMPLES_DIR):
        return []
    entries = []
    for f in sorted(os.listdir(_VOICE_SAMPLES_DIR)):
        if f.endswith((".wav", ".mp3", ".m4a")) and not f.startswith("."):
            entries.append({"name": f.replace(".wav", "").replace(".mp3", "").replace(".m4a", ""), "path": os.path.join(_VOICE_SAMPLES_DIR, f)})
    return entries


def _clone_voice_reply(text: str, voice_sample: str = "") -> str | None:
    """Synthesise text as voice, return path to WAV, or None on failure.
    Uses local olly-voice-server by default (Qwen3TTS).
    If providers.tts is set to a cloud provider in config, routes through
    that provider's TTS API instead (e.g. OpenAI TTS on OpenRouter).
    """
    import tempfile, requests as _req, yaml as _yaml
    if not text or not text.strip():
        print("[voice] clone skipped: empty reply text")
        return None

    # Check if cloud TTS is configured
    _tts_provider = "local"
    _tts_model = "tts-1"
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH) as _f:
                _cfg = _yaml.safe_load(_f)
            _prov = (_cfg or {}).get("providers", {})
            _tts_provider = _prov.get("tts", "local")
            _tts_model = _prov.get("model_overrides", {}).get("tts", _prov.get("models", {}).get(_tts_provider, "tts-1"))
    except Exception:
        pass

    if _tts_provider != "local":
        # Cloud TTS path
        print(f"[voice] cloud TTS ({_tts_provider}/{_tts_model})", flush=True)
        try:
            # OpenAI-compatible TTS (OpenRouter routes this too)
            _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY") or ""
            if _tts_provider == "openrouter":
                _api_key = os.environ.get("OPENROUTER_API_KEY", "")
                _base = "https://openrouter.ai/api/v1"
            else:
                _base = "https://api.openai.com/v1"
            if not _api_key:
                print("[voice] cloud TTS: no API key")
                return None
            r = _req.post(
                f"{_base}/audio/speech",
                headers={"Authorization": f"Bearer {_api_key}"},
                json={"model": _tts_model, "input": text, "voice": "alloy", "response_format": "wav"},
                timeout=120,
            )
            if r.ok and len(r.content) > 1000:
                fd, out = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with open(out, "wb") as f:
                    f.write(r.content)
                return out
            print(f"[voice] cloud TTS failed: {r.status_code} {r.text[:200]}")
            return None
        except Exception as e:
            print(f"[voice] cloud TTS error: {e}")
            return None

    # Local TTS path (olly-voice-server Qwen3TTS)
    sample = voice_sample or _bot_module()._active_voice_sample
    if not os.path.isfile(sample):
        print(f"[voice] sample not found: {sample}")
        return None
    try:
        with voice_activity("clone"):
            with open(sample, "rb") as sf:
                r = _req.post(
                    "http://127.0.0.1:8766/tts/clone",
                    data={"text": text, "model": "1.7", "x_vector_only": "true"},
                    files={"sample": (os.path.basename(sample), sf, "audio/wav")},
                    timeout=300,
                )
        if r.ok and len(r.content) > 1000:
            fd, out = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with open(out, "wb") as f:
                f.write(r.content)
            return out
        print(f"[voice] clone failed: {r.status_code} {r.text[:300]}")
        return None
    except Exception as e:
        print(f"[voice] clone error: {e}")
        return None


class TypingKeepAlive:
    """
    Context manager that pulses a Telegram chat action every 4s
    until the block completes. Telegram's typing indicator expires after ~5s
    so a single send_typing call goes dark on long inference or exec tasks.

    Pass action= to show a richer indicator:
      "typing"       — ⌨️  default, text reply coming
      "record_voice" — 🎙️  recording/cloning audio
      "upload_voice" — ⬆️  uploading audio
      "upload_photo" — 🖼️  processing image

    Usage:
        with TypingKeepAlive(chat_id, action="record_voice"):
            wav = _clone_voice_reply(text)
    """
    def __init__(self, chat_id: str, interval: float = 4.0, action: str = "typing"):
        self._chat_id = chat_id
        self._interval = interval
        self._action = action
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _pulse(self):
        while not self._stop.wait(self._interval):
            _bot_module().send_typing(self._chat_id, self._action)

    def __enter__(self):
        _bot_module().send_typing(self._chat_id, self._action)    # immediate first pulse
        self._stop.clear()
        self._thread = threading.Thread(target=self._pulse, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def download_file(file_id: str) -> "str | None":
    """Download a Telegram file by file_id. Returns local temp path or None."""
    import tempfile, os
    try:
        r = requests.get(f"{API_BASE}/getFile", params={"file_id": file_id}, timeout=10)
        file_path = r.json()["result"]["file_path"]
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        ext = os.path.splitext(file_path)[1] or ".bin"
        tmp = tempfile.mktemp(suffix=ext)
        data = requests.get(url, timeout=30)
        open(tmp, "wb").write(data.content)
        return tmp
    except Exception as e:
        print(f"[bot] download_file error: {e}")
        return None


# (cloud/modal provider pickers moved to telegram_providers.py — see issue #3)

