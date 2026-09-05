"""
telegram_bot.py — Kernel's own Telegram bot.
Runs independently.
Connects directly: Telegram → local model (or cloud provider) → Telegram

Usage: python src/telegram_bot.py
"""

import os
import json
import re
import random
import subprocess
import requests
import threading
import time
import sys
from datetime import date
from pathlib import Path
from runtime_paths import DOCUMENTS_DIR
from core.voice_activity import voice_activity

# Recency window (seconds) for treating an attachment as "recent" context.
# Attachments older than this are stale and must not be injected as if freshly
# uploaded, nor force the reply to reference them (bug: a days-old photo was
# surfaced for an unrelated later turn).
_ATTACHMENT_RECENCY_SECONDS = int(os.environ.get("KERNEL_EVO_ATTACHMENT_RECENCY_SECONDS", "86400"))

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent))

BOT_TOKEN = os.environ.get("KERNEL_EVO_TELEGRAM_BOT_TOKEN")
ALLOWED_CHAT_ID = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
# Honor KERNEL_EVO_CONFIG (set by start.sh --config=...) so an alternate config
# can be used; fall back to the repo's config.yaml.
_CONFIG_OVERRIDE = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
CONFIG_PATH = (
    os.path.abspath(os.path.expanduser(_CONFIG_OVERRIDE))
    if _CONFIG_OVERRIDE
    else str(Path(__file__).parent.parent.parent.parent / "config.yaml")
)
REPO_DIR = str(Path(__file__).parent.parent)


def _esc(text: str) -> str:
    """Escape underscores in dynamic text before wrapping in Telegram Markdown italic _..._"""
    return str(text).replace("_", "\\_")


def _html(text: str) -> str:
    """HTML-escape text for Telegram parse_mode="HTML".

    Telegram's legacy Markdown parser rejects messages containing unescaped
    markdown specials (`_`, `*`, backticks, `[`) with a 400 "can't parse
    entities" error. HTML parse mode only requires escaping `<`, `>`, and `&`,
    so it's far more robust for arbitrary model-generated text (which is full
    of code, links, and punctuation).
    """
    import html as _html_lib
    return _html_lib.escape(str(text), quote=False)


# ── Tool step display (Telegram) ─────────────────────────────────────
# Per-tool emoji + verb so tool-call progress messages are informative
# ("what tool, to do what") instead of a bare "🔧 tool1 → tool2".
_TOOL_EMOJI = {
    "read_file": "📄",
    "write_file": "✍️",
    "exec_shell": "⚙️",
    "http_get": "🌐",
    "web_search": "🔍",
    "browser_use": "🧭",
    "run_skill": "🧩",
    "run_routine": "🔄",
    "send_file": "📤",
    "search_skills": "🔎",
    "list_routines": "📋",
    "recall_memory": "🧠",
}
_TOOL_VERB = {
    "read_file": "read",
    "write_file": "write",
    "exec_shell": "run",
    "http_get": "fetch",
    "web_search": "search",
    "browser_use": "browse",
    "run_skill": "run skill",
    "run_routine": "run routine",
    "send_file": "send file",
    "search_skills": "find skill",
    "list_routines": "list routines",
    "recall_memory": "recall memory",
}
# Arg fields to surface first when summarising a tool call.
_TOOL_ARG_PRIORITY = (
    "path", "file_path", "query", "url", "command", "content",
    "skill_name", "routine_name", "input", "caption",
)


def _tool_args_label(name: str, args) -> str:
    """Return a short human-readable label for a tool call's args."""
    if args is None:
        return ""
    if isinstance(args, dict):
        for k in _TOOL_ARG_PRIORITY:
            v = args.get(k)
            if v:
                s = str(v).strip()
                if s:
                    return s[:80]
        return ""
    s = str(args).strip()
    return s[:80]


def _format_tool_step(n, tool_name, args=None, result=None) -> str:
    """Format one tool-call step as a readable, emoji-tagged Telegram line.

    Uses HTML formatting (Telegram HTML parse mode) — the tool name is in
    <code>, the step number is <b>bold</b>, and dynamic values are HTML-escaped
    so arbitrary model output never breaks the entity parser.
    """
    name = str(tool_name or "?")
    emoji = _TOOL_EMOJI.get(name, "🔧")
    verb = _TOOL_VERB.get(name, name.replace("_", " "))
    # Show the tool name in <code> (precise) with a friendly verb prefix.
    line = f"<b>Step {n}</b> {emoji} <code>{_html(verb)}</code> (<code>{_html(name)}</code>)"
    label = _tool_args_label(name, args)
    if label:
        line += f" — <code>{_html(label)}</code>"
    if result is not None:
        r = str(result).strip()
        if r and not r.lower().startswith("(error") and "error:" not in r.lower()[:60]:
            line += "\n  ✅ ok"
        else:
            line += f"\n  ⚠️ {_html(r[:120])}"
    return line


# GitHub update tracking
_latest_version: str = ""
# GitHub URLs now live in src/updater.py

# Lazy model load
_agent_ready = False

# Verbose mode — stream tool call steps to Telegram when ON
_verbose_mode = False

# In-process memory (list of {role, content} dicts)
_memory: list = []


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


# @TODO: Make configurable. Voice sample path — first user message during first contact sets it if missing.
_VOICE_SAMPLES_DIR = os.path.expanduser(os.environ.get(
    "KERNEL_VOICE_SAMPLES_DIR",
    os.path.expanduser("~/.openclaw/media/voice-samples")
))
_DEFAULT_VOICE_SAMPLE = os.environ.get(
    "KERNEL_DEFAULT_VOICE_SAMPLE",
    os.path.join(_VOICE_SAMPLES_DIR, "default-en-phonetic.wav")
)
# Active voice sample — can be switched via /voices
_active_voice_sample: str = _DEFAULT_VOICE_SAMPLE


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
    sample = voice_sample or _active_voice_sample
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
            send_typing(self._chat_id, self._action)

    def __enter__(self):
        send_typing(self._chat_id, self._action)    # immediate first pulse
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


def _get_current_model_label() -> str:
    """Return human-readable label of the currently loaded model."""
    try:
        from core.inference.model_client import health as _mch
        h = _mch()
        label = h.get("model", "")
        if h.get("nemotron"):
            label = f"{label} [Nemotron]"
        if label:
            return label
    except Exception:
        pass
    # Fallback: read from config.yaml
    try:
        import yaml as _y
        cfg = _y.safe_load(open(CONFIG_PATH))
        return cfg.get("model", {}).get("name", "local model")
    except Exception:
        return "local model"


# ── Local model VRAM-fit helpers (XP7) ──────────────────────────────────────
# Used by the /provider local picker to flag models that likely fit the GPU's
# free VRAM. Estimates are rough (params × bytes-per-param + overhead) — the
# real fit depends on dtype, quantization, and context length.

def _estimate_model_gb(repo_id: str) -> float:
    """Rough model-size estimate in GB from the repo id (params → bytes).

    Heuristic: parse a known param-count marker in the repo id (e.g. 0.8B, 3B,
    7B, 30B). Falls back to 4GB when unknown. Multiply params by ~2 bytes/param
    (bf16) and add ~1GB overhead; quantized 4-bit models use far less, so this
    is intentionally conservative (a "fits" flag is safe).
    """
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)[bB]\b", repo_id)
    if not m:
        return 4.0
    params_b = float(m.group(1))
    gb = params_b * 2.0 + 1.0  # ~2 bytes/param (bf16) + overhead
    return round(gb, 1)


def _vram_fit_mark(est_gb: float, vram_free_mb: int) -> str:
    """Return a short emoji marker showing whether a model likely fits free VRAM."""
    if vram_free_mb <= 0:
        return ""
    free_gb = vram_free_mb / 1024.0
    if est_gb <= free_gb * 0.9:
        return "\U0001f7e2"  # green — fits comfortably
    if est_gb <= free_gb * 1.5:
        return "\U0001f7e1"  # yellow — tight / may need quantization
    return "\U0001f534"      # red — likely too big for free VRAM


def _curated_slot_for_repo(repo_id: str) -> str:
    """Return the default model_slots entry for a curated repo id, or '' if none."""
    try:
        import urllib.request as _ur_c
        with _ur_c.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
            curated = (json.loads(_rc.read()) or {}).get("curated", [])
        for cm in curated:
            if (cm.get("repo_id") or "").lower() == repo_id.lower():
                return cm.get("slot") or ""
    except Exception:
        pass
    return ""


# ── Short callback tokens (XP7) ──────────────────────────────────────────────
# Telegram inline callback_data is limited to 64 bytes, so embedding a full HF
# repo_id (e.g. "google/gemma-4-E2B-it") can exceed the limit and Telegram
# rejects the button with BUTTON_DATA_INVALID. We instead use short opaque
# tokens in callback_data and resolve them to repo ids here, server-side.

_LOCAL_REPO_TOKENS: dict = {}  # token -> repo_id


def _register_repo_token(repo_id: str) -> str:
    """Return a short token for a repo_id, registering it if not already present.

    Tokens are stable per repo_id within this process: 'lm0', 'lm1', ...
    """
    for tok, rid in _LOCAL_REPO_TOKENS.items():
        if rid == repo_id:
            return tok
    tok = f"lm{len(_LOCAL_REPO_TOKENS)}"
    _LOCAL_REPO_TOKENS[tok] = repo_id
    return tok


def _resolve_repo_token(token: str) -> str:
    """Resolve a short token back to a repo_id, or return the token unchanged."""
    return _LOCAL_REPO_TOKENS.get(token, token)


# ── Cloud model catalog for guided /cloud flow ─────────────────────────────
# Models are fetched dynamically from /provider/models API endpoint.
# This gives live model lists from OpenRouter and reasonable defaults
# for OpenAI, Anthropic, HuggingFace, and Copilot.


def _fetch_provider_models(provider: str, capability: str = "text") -> list:
    """Fetch model list for a provider+capability from the API endpoint.
    Falls back to minimal defaults if API unavailable."""
    import requests as _req, yaml as _yaml
    try:
        with open(CONFIG_PATH) as _pf:
            _cfg_p = _yaml.safe_load(_pf)
        _api_port = _cfg_p.get('api', {}).get('port', 8779)
        r = _req.get(f"http://localhost:{_api_port}/provider/models",
                      params={"provider": provider, "capability": capability},
                      timeout=5)
        if r.ok:
            data = r.json()
            return data.get("models", {}).get(provider, [])
    except Exception as e:
        print(f"[cloud] model fetch error for {provider}/{capability}: {e}", flush=True)
    # Fallback: minimal sensible defaults
    _FALLBACK = {
        "openai":     ["gpt-5.4", "gpt-4.1", "gpt-4o"],
        "openrouter": ["deepseek/deepseek-v4-flash", "google/gemma-4-26b-a4b-it", "google/gemini-2.5-pro"],
        "anthropic":  ["claude-sonnet-4-6", "claude-opus-4-5"],
        "hf":         ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-32B"],
        "copilot":    ["claude-sonnet-4.5", "gpt-5.4"],
    }
    return _FALLBACK.get(provider, [])

_CLOUD_CALL_TYPES = ["task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"]
_CALL_TYPE_LABELS = {
    "task_inference": "Chat (task inference)",
    "synthesis": "Skill synthesis",
    "critic": "Critic / verification",
    "planning": "Task planning",
    "trajectory_teacher": "Teacher trajectories",
    "vision": "Vision (images)",
    "stt": "Speech-to-Text (STT)",
    "tts": "Text-to-Speech (TTS / voice clone)",
}


def _show_cloud_provider_picker(chat_id: str):
    """Step 1: pick a cloud provider."""
    buttons = [
        [{"text": "\U0001f916 OpenAI",      "callback_data": "cloud_provider_openai"}],
        [{"text": "\U0001f310 OpenRouter",   "callback_data": "cloud_provider_openrouter"}],
        [{"text": "\U0001f9e0 Anthropic",    "callback_data": "cloud_provider_anthropic"}],
        [{"text": "\U0001f917 HuggingFace",  "callback_data": "cloud_provider_hf"}],
        [{"text": "\u26a1 Copilot",          "callback_data": "cloud_provider_copilot"}],
        [{"text": "\u2190 Back",            "callback_data": "/start"}],
    ]
    send_buttons(chat_id, "\u2601\ufe0f *Cloud mode*\n\nPick a provider for all capabilities. After that, you'll choose a model per call type.", buttons)


def _show_modal_provider_picker(chat_id: str, provider: str, call_type_idx: int):
    """Step for vision/stt/tts: pick a provider (or keep local) before model selection."""
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    buttons = [
        [{"text": "\U0001f916 OpenAI",      "callback_data": f"cmprov_{ct}|openai"}],
        [{"text": "\U0001f310 OpenRouter",   "callback_data": f"cmprov_{ct}|openrouter"}],
        [{"text": "\U0001f9e0 Anthropic",    "callback_data": f"cmprov_{ct}|anthropic"}],
        [{"text": "\U0001f917 HuggingFace",  "callback_data": f"cmprov_{ct}|hf"}],
        [{"text": "\u26a1 Copilot",          "callback_data": f"cmprov_{ct}|copilot"}],
        [{"text": "\U0001f512 Keep local",    "callback_data": f"cmprov_{ct}|local"}],
        [{"text": "\u2190 Back",            "callback_data": "/cloud"}],
    ]
    send_buttons(chat_id, f"\u2601\ufe0f ({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)}) Pick provider for *{label}*:", buttons)


def _fetch_models_for_ct(provider: str, ct: str) -> list:
    """Return model list for a call type, fetching dynamically."""
    cap_map = {"vision": "vision", "stt": "audio", "tts": "text"}  # TTS is text generation
    cap = cap_map.get(ct, "text")
    return _fetch_provider_models(provider, cap)


def _show_modal_model_picker(chat_id: str, modal_provider: str, call_type_idx: int):
    """Show model picker for vision/stt/tts under a specific provider."""
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    models = _fetch_models_for_ct(modal_provider, ct)
    if not models:
        # No dedicated models for this modality+provider — use default for provider
        _apply_modal_model(chat_id, modal_provider, ct, None, call_type_idx)
        return
    buttons = []
    for m in models:
        cb = f"cmm_{modal_provider}|{ct}|{m}"
        if len(cb.encode("utf-8")) > 64:
            cb = cb[:60]
        buttons.append([{"text": f"\U0001f4e1 {m}", "callback_data": cb}])
    buttons.append([{"text": "\u23ed Use default", "callback_data": f"cmm_{modal_provider}|{ct}|default"}])
    buttons.append([{"text": "\u2190 Back", "callback_data": "/cloud"}])
    step_label = f"({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)})"
    send_buttons(chat_id, f"\u2601\ufe0f {step_label} Pick model for *{label}* ({modal_provider}):", buttons)


def _apply_modal_model(chat_id: str, modal_provider: str, call_type: str, model: str | None, call_type_idx: int):
    """Set provider+model for vision/stt/tts, then advance."""
    import yaml as _yaml, json as _json_p, urllib.request as _ur
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    body = {call_type: modal_provider}
    if model:
        body["model_override"] = {call_type: model}
    body["persist"] = True
    payload = _json_p.dumps(body).encode()
    try:
        req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                          data=payload, headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=5).read()
    except Exception as _e:
        print(f"[cloud] modal model apply error: {_e}", flush=True)
    # Advance to next call type (use main provider from first 5 steps for the rest)
    if call_type_idx + 1 >= len(_CLOUD_CALL_TYPES):
        _finish_cloud_setup(chat_id, modal_provider)
    else:
        # Re-read main provider for subsequent steps
        _main_prov = "openrouter"
        try:
            with open(CONFIG_PATH) as _pf:
                _cfg_p2 = _yaml.safe_load(_pf)
            _routing = (_cfg_p2.get("providers", {}) or {})
            for _ct in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
                _v = _routing.get(_ct)
                if _v and _v != "local":
                    _main_prov = _v
                    break
        except Exception:
            pass
        _show_cloud_model_picker(chat_id, _main_prov, call_type_idx + 1)


def _show_cloud_model_picker(chat_id: str, provider: str, call_type_idx: int):
    """Show model picker for a specific call type."""
    if call_type_idx >= len(_CLOUD_CALL_TYPES):
        _finish_cloud_setup(chat_id, provider)
        return
    ct = _CLOUD_CALL_TYPES[call_type_idx]
    label = _CALL_TYPE_LABELS.get(ct, ct)
    # For vision/stt/tts, pick provider first (can differ from main provider)
    if ct in ("vision", "stt", "tts"):
        _show_modal_provider_picker(chat_id, provider, call_type_idx)
        return
    models = _fetch_provider_models(provider, "text")
    if not models:
        _show_cloud_model_picker(chat_id, provider, call_type_idx + 1)
        return
    buttons = []
    for m in models:
        cb = f"cm_{provider}|{ct}|{m}"
        # Cap to 64 bytes to avoid Telegram silently dropping the entire message
        if len(cb.encode("utf-8")) > 64:
            cb = cb[:60]  # truncate if somehow too long
        buttons.append([{"text": f"\U0001f4e1 {m}", "callback_data": cb}])
    buttons.append([{"text": "\u23ed Use default for this", "callback_data": f"cm_{provider}|{ct}|default"}])
    buttons.append([{"text": "\u2190 Back to providers", "callback_data": "/cloud"}])
    step_label = f"({call_type_idx + 1}/{len(_CLOUD_CALL_TYPES)})"
    send_buttons(chat_id, f"\u2601\ufe0f {step_label} Pick model for *{label}*:", buttons)


def _apply_cloud_model(chat_id: str, provider: str, call_type: str, model: str | None):
    """Set model override for one call type, then advance to next."""
    import yaml as _yaml, json as _json_p, urllib.request as _ur
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    body = {call_type: provider}
    if model:
        body["model_override"] = {call_type: model}
    body["persist"] = True
    payload = _json_p.dumps(body).encode()
    try:
        req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                          data=payload, headers={"Content-Type": "application/json"}, method="POST")
        _ur.urlopen(req, timeout=5).read()
    except Exception as _e:
        print(f"[cloud] model override error: {_e}", flush=True)
    try:
        idx = _CLOUD_CALL_TYPES.index(call_type)
    except ValueError:
        idx = 0
    _show_cloud_model_picker(chat_id, provider, idx + 1)


def _finish_cloud_setup(chat_id: str, provider: str):
    """All models selected — show summary."""
    import json as _json_p, urllib.request as _ur
    import yaml as _yaml
    with open(CONFIG_PATH) as _pf:
        _cfg_p = _yaml.safe_load(_pf)
    _api_port = _cfg_p.get('api', {}).get('port', 8779)
    try:
        with _ur.urlopen(f"http://localhost:{_api_port}/provider", timeout=3) as r:
            data = _json_p.loads(r.read())
        routing = data.get("routing", {})
        lines = ["\u2601\ufe0f *Cloud setup complete!*\n"]
        for ct in _CLOUD_CALL_TYPES:
            info = routing.get(ct, {})
            icon = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                    "hf": "\U0001f917", "copilot": "\u26a1", "openrouter": "\U0001f310"}.get(info.get("provider", ""), "\U0001f4e1")
            model_suffix = f" `{info['model']}`" if info.get("model") else ""
            lines.append(f"{icon} `{ct}` \u2192 *{info['provider']}*{model_suffix}")
        lines.append("\nZero local VRAM used. Tap /start to switch modes.")
        send_message(chat_id, "\n".join(lines))
    except Exception as e:
        send_message(chat_id, f"\u2601\ufe0f Cloud configuration saved (provider: {provider}).")


def handle_callback(chat_id: str, data: str, message_id: int):
    """Handle inline button callbacks."""
    try:
        print(f"[bot] callback: chat={chat_id} data={data}", flush=True)
        # ── Shell command authorization ──────────────────────────────────
        if data.startswith("auth_allow_") or data.startswith("auth_deny_"):
            from core.auth_gate import resolve_auth
            approved = data.startswith("auth_allow_")
            request_id = data.split("_", 2)[-1]
            resolve_auth(request_id, approved)
            status = "✅ Approved" if approved else "❌ Denied"
            if message_id:
                edit_message(chat_id, message_id, f"*Shell command* — {status}")
            return
        if data.startswith("install_"):
            parts = data.split("_", 2)
            item_type = parts[1] if len(parts) > 1 else None
            name = parts[2] if len(parts) > 2 else ""
            from bootstrap import install as eco_install
            result = eco_install(name, item_type)
            send_message(chat_id, result["message"])
            return
        elif data.startswith("cloud_provider_"):
            provider = data[len("cloud_provider_"):]
            _show_cloud_model_picker(chat_id, provider, 0)
            return
        elif data.startswith("cmprov_"):
            # cmprov_{call_type}|{provider}
            rest = data[len("cmprov_"):]
            parts = rest.split("|", 1)
            if len(parts) >= 2:
                ct = parts[0]
                modal_provider = parts[1]
                try:
                    idx = _CLOUD_CALL_TYPES.index(ct)
                except ValueError:
                    idx = 0
                if modal_provider == "local":
                    # Keep local — set provider=local, advance
                    _apply_modal_model(chat_id, "local", ct, None, idx)
                else:
                    # Show model picker for this provider
                    _show_modal_model_picker(chat_id, modal_provider, idx)
            return
        elif data.startswith("cmm_"):
            # cmm_{modal_provider}|{call_type}|{model}
            rest = data[len("cmm_"):]
            parts = rest.split("|", 2)
            if len(parts) >= 2:
                modal_provider = parts[0]
                ct = parts[1]
                model = parts[2].replace("~", "/") if len(parts) > 2 and parts[2] != "default" else None
                if ct in _CLOUD_CALL_TYPES:
                    try:
                        idx = _CLOUD_CALL_TYPES.index(ct)
                    except ValueError:
                        idx = 0
                    _apply_modal_model(chat_id, modal_provider, ct, model, idx)
            return
        elif data.startswith("cm_"):
            # cm_{provider}|{call_type}|{model}
            rest = data[len("cm_"):]
            parts = rest.split("|", 2)
            if len(parts) >= 2:
                provider = parts[0]
                ct = parts[1]
                model = parts[2].replace("~", "/") if len(parts) > 2 and parts[2] != "default" else None
                if ct in _CLOUD_CALL_TYPES:
                    _apply_cloud_model(chat_id, provider, ct, model)
                else:
                    send_message(chat_id, f"\u274c Unknown call type: {ct}")
            return
        elif data.startswith("routine_run_"):
            name = data[12:]
            handle_message(chat_id, f"/run {name}")
            return
        elif data.startswith("routine_info_"):
            name = data[13:]
            _ensure_agent()
            import yaml

            with open(CONFIG_PATH) as _f:
                _cfg = yaml.safe_load(_f)
            from core.routines import load_all, find

            routines = load_all(
                str(os.path.expanduser(_cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
            )
            r = find(name, routines)
            if r:
                trigger = r.get("trigger", {})
                t = (
                    trigger.get("also", trigger.get("cron", "manual"))
                    if isinstance(trigger, dict)
                    else "manual"
                )
                send_message(
                    chat_id,
                    f"*{r['name']}*\n_{_esc(r['description'])}_\nTrigger: `{t}`\n\nTap ▶ Run to execute.",
                )
            return
        elif data.startswith("skill_info_"):
            name = data[11:]
            _ensure_agent()
            import yaml

            with open(CONFIG_PATH) as _f:
                _cfg = yaml.safe_load(_f)
            from core.skills import load_all, find

            skills = load_all(
                str(os.path.expanduser(_cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
            )
            s = find(name, skills)
            if s:
                send_message(chat_id, f"*{s['name']}*\n_{_esc(s['description'])}_")
            return
        elif data.startswith("set_voice_"):
            handle_message(chat_id, data)
            return
        elif data == "thought_skip":
            send_message(chat_id, "❌ Skipped.")
            return
        elif data.startswith("thought_do_"):
            # User approved an actionable thought — triage it as a task
            send_message(chat_id, "✅ On it…")
            _ensure_agent()
            import core.agent as _agent_thought
            import core.memory.memory as _mem_thought
            # Find the thought text from the last proactive message in memory
            history = _mem_thought.load(chat_id=str(chat_id))
            thought_text = ""
            for m in reversed(history[-20:]):
                c = str(m.get("content", ""))
                if "Kernel is thinking" in c:
                    # Extract italic text between _ _ markers
                    match = re.search(r'_(.+?)_', c, re.DOTALL)
                    if match:
                        thought_text = match.group(1).strip()
                    break
            if thought_text:
                result = _agent_thought.triage(f"Act on this: {thought_text}", chat_id=str(chat_id))
                send_message(chat_id, f"🐬 {result[:1500]}")
            else:
                send_message(chat_id, "🐬 Could not find the thought to act on — let me know what you want done.")
            return
        elif data.startswith("evo_confirm_resolved:"):
            # ADR-020: user confirmed evolution resolved their request
            try:
                request_id = int(data.split(":", 1)[1])
                import database.agent.failed_requests as _fr
                _fr.mark_resolved(request_id, reward=1)
                # Write positive trajectory for fine-tuning
                try:
                    from core.evolution.evolution_log import EvolutionLog
                    _elog = EvolutionLog()
                    _elog.record_resolution(request_id, reward=1)
                except Exception:
                    pass
                send_message(chat_id, "✅ *Resolved!* I've locked in what I learned. Thanks for confirming — this becomes part of my training.")
                print(f"[bot] ADR-020 evo_confirm_resolved: id={request_id} reward=1", flush=True)
            except Exception as _ce:
                print(f"[bot] evo_confirm_resolved error: {_ce}", flush=True)
                send_message(chat_id, "✅ Marked as resolved.")
            return
        elif data.startswith("evo_confirm_failed:"):
            # ADR-020: user rejected — retry evolution or escalate
            try:
                request_id = int(data.split(":", 1)[1])
                import database.agent.failed_requests as _fr
                retry_count = _fr.increment_retry(request_id)
                # Write negative trajectory
                try:
                    from core.evolution.evolution_log import EvolutionLog
                    _elog = EvolutionLog()
                    _elog.record_resolution(request_id, reward=0)
                except Exception:
                    pass
                if retry_count < 3:
                    send_message(
                        chat_id,
                        f"❌ Got it — still not there. I'll try again during the next idle cycle "
                        f"(attempt {retry_count + 1}/3)."
                    )
                else:
                    _fr.mark_resolved(request_id, reward=0)  # close as unresolvable
                    send_message(
                        chat_id,
                        f"❌ Three attempts, still unresolved. I've flagged this for manual review — "
                        f"you may need to install a specific skill or give me more context."
                    )
                print(f"[bot] ADR-020 evo_confirm_failed: id={request_id} retry={retry_count}", flush=True)
            except Exception as _ce:
                print(f"[bot] evo_confirm_failed error: {_ce}", flush=True)
                send_message(chat_id, "❌ Noted — I'll keep trying.")
            return
        elif data.startswith("tier2_approve:"):
            # ADR-021: user approved Tier 2 skill synthesis
            synthesis_id = data.split(":", 1)[1]
            send_message(chat_id, f"✅ *Approved!* Starting skill synthesis `{synthesis_id}` — I'll let you know when it's done.")
            try:
                import core.evolution.evolution_hook as _evo_hook
                _evo_hook._run_approved_synthesis(synthesis_id)
            except Exception as _ae:
                send_message(chat_id, f"⚠️ Synthesis failed to start: {str(_ae)[:200]}")
            return
        elif data.startswith("tier2_reject:"):
            # ADR-021: user rejected Tier 2 skill synthesis
            synthesis_id = data.split(":", 1)[1]
            send_message(chat_id, "❌ *Skill synthesis cancelled.* No tokens spent, nothing installed.")
            try:
                import core.evolution.evolution_hook as _evo_hook
                _evo_hook._reject_synthesis(synthesis_id)
            except Exception:
                pass
            return

        # Fallback: route unknown callbacks (e.g. /models, /provider) through the message handler
        handle_message(chat_id, data)
        return
    except Exception as e:
        print(f"[bot] callback error for {data}: {e}", flush=True)
        try:
            send_message(chat_id, f"❌ Button error: {str(e)[:200]}")
        except Exception:
            pass   
             
# @TODO: change hardcoded path reference to review.
def _search_collective_memory(query: str) -> str:
    """Run collective memory search via installed context_provider skills.
    Falls back to the direct script path if no skill is found.
    """
    # Prefer skill-based discovery (same path agent.py uses)
    try:
        import core.agent as _ag
        _ctx_skills = [s for s in (_ag._skills or []) if s.get("context_provider") and s.get("context_search_cmd")]
        if _ctx_skills:
            import shlex as _sl, subprocess as _sp
            results = []
            for _cs in _ctx_skills:
                _cmd = _cs["context_search_cmd"].replace("{query}", _sl.quote(query[:200]))
                _res = _sp.run(_cmd, shell=True, capture_output=True, text=True, timeout=6)
                _out = (_res.stdout or "").strip()
                if _out and len(_out) > 20:
                    results.append(_out[:600])
            return "\n\n".join(results)
    except Exception:
        pass
    # Fallback: direct script
    script = os.path.expanduser("~/.openclaw/workspace/collective-memory/scripts/search.py")
    try:
        result = subprocess.run(
            ["python3", script, query, "--top", "2"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""

# @TODO: change hardcoded path reference to review.
def _write_collective_memory(user_message: str, reply: str) -> None:
    """Write a new collective memory entry based on the conversation turn.

    Quality gate: only write entries that are genuinely informative — skip
    greetings, short exchanges, error messages, or anything that adds no signal.
    """
    entries_dir = Path(os.path.expanduser("~/.openclaw/workspace/collective-memory/entries"))
    entries_dir.mkdir(parents=True, exist_ok=True)

    # ── Quality gate ───────────────────────────────────────────────
    # Skip trivial / low-signal turns that pollute the collective memory.
    _skip_patterns = (
        # Greetings
        r'^(hi|hey|hello|ciao|good morning|good evening|yo)[ !.,]*$',
        # One-word or very short messages
        r'^.{0,15}$',
        # Status checks
        r'^(ping|status|ok|yes|no|sure|thanks|thank you|great|nice|cool)[ !.,]*$',
    )
    _msg_lower = user_message.strip().lower()
    for _pat in _skip_patterns:
        if re.match(_pat, _msg_lower, re.I):
            return

    # Skip error replies or short model outputs
    if len(reply) < 300:
        return
    if any(marker in reply for marker in ("[model_server error]", "[model_client error]", "VRAM guard")):
        return

    # Skip replies that are purely conversational without factual content
    # (heuristic: meaningful entries tend to have URLs, file paths, numbers, or code)
    _has_substance = bool(re.search(
        r'(https?://|/home/|/mnt/|```|\$\s|v\d+\.\d+|€|\d{4}-\d{2}-\d{2}|ADR-\d|\bPR\b|\bcommit\b)',
        reply, re.I
    ))
    if not _has_substance:
        return
    # ────────────────────────────────────────────────────────

    today = date.today().isoformat()
    # Infer topic from first 6 words of user message
    words = re.sub(r"[^a-zA-Z0-9\s]", "", user_message).split()[:6]
    topic = " ".join(words) if words else "general"
    slug = re.sub(r"\s+", "-", topic.lower())[:60]
    filename = entries_dir / f"{today}-{slug}.md"

    # Don't overwrite an existing entry with the same slug
    if filename.exists():
        return

    # Simple confidence heuristic: if reply contains hedging words → inferred
    hedging = re.search(
        r"\b(maybe|perhaps|might|could|unsure|unclear|I think|probably)\b", reply, re.I
    )
    confidence = "inferred" if hedging else "confirmed"

    # Bullet-point the reply lines (first 10 non-empty lines)
    lines = [l.strip() for l in reply.splitlines() if l.strip()][:10]
    bullets = "\n".join(f"- {l}" for l in lines)

    content = (
        f"---\n"
        f"date: {today}\n"
        f"agent: kernel\n"
        f"topic: {topic}\n"
        f"tags: [telegram, auto]\n"
        f"confidence: {confidence}\n"
        f"---\n"
        f"# {topic.title()}\n"
        f"{bullets}\n"
    )
    try:
        filename.write_text(content)
        print(f"[bot] collective memory written: {filename.name}", flush=True)
    except Exception as e:
        print(f"[bot] collective memory write error: {e}")


class _MultimodalActivity:
    """Context manager: marks model_activity in_flight during multimodal processing
    so ThinkAtRest idle evolution doesn't fire while vision/audio is running."""
    def __enter__(self):
        try:
            from core.inference.model_client import _mark_activity_start
            _mark_activity_start()
        except Exception:
            pass
        return self
    def __exit__(self, *_):
        try:
            from core.inference.model_client import _mark_activity_end
            _mark_activity_end()
        except Exception:
            pass


def handle_message(chat_id: str, text: str, sender_name: str = "", photo_file_id: str = "", voice_file_id: str = "", document_file_id: str = "", document_name: str = "", document_mime: str = ""):
    global _active_voice_sample
    global _agent_ready

    # Auth check
    if ALLOWED_CHAT_ID and str(chat_id) != str(ALLOWED_CHAT_ID):
        _chat_id_str = str(chat_id)
        if not _chat_id_str.startswith(("api-", "web-", "ui-", "agent-")):
            send_message(chat_id, "❌ Unauthorized.")
            return

    text = text.strip()

    # Send typing indicator immediately
    send_typing(chat_id)

    # Handle image attachments
    if photo_file_id and not text:
        text = "Describe this image."
    if photo_file_id:
        _ensure_agent()
        local_path = download_file(photo_file_id)
        if local_path:
            try:
                working_id = send_message(chat_id, "🔍 _Analysing image\u2026_")
                with _MultimodalActivity(), TypingKeepAlive(chat_id, action="upload_photo"):
                    from core.inference.model import infer_with_image
                    # Force the LOCAL Gemma E2B slot (no cloud credits available)
                    # and cap max_new_tokens to match the proven-working `look`/
                    # describe path. This skips the cloud attempt entirely (which
                    # 402s without credits) and avoids the native-slot hang seen
                    # with large token budgets, so the description returns promptly.
                    reply = infer_with_image(local_path, text or "Describe this image.",
                                             max_new_tokens=1024, force_local=True)
                # Raise on error strings so except block handles them cleanly
                if isinstance(reply, str) and reply.startswith(("[model_server", "[model_client", "[model_server timeout]")):
                    raise RuntimeError(reply)
                if not reply or not reply.strip():
                    raise RuntimeError("Vision returned empty response")
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
                import core.memory.memory as _memory_mod
                _memory_mod.record_attachment(
                    kind="photo", local_path="(temp-deleted)",
                    original_name="photo.jpg", mime_type="image/jpeg",
                    caption=text or "Describe this image.", chat_id=str(chat_id)
                )
                history = _memory_mod.load(chat_id=str(chat_id))
                # Store the user turn with the caption + a brief image summary so the
                # model has full context when the user asks follow-up questions.
                _img_user_content = f"[Image sent]{': ' + text if text else ''}"
                _img_summary = reply[:300] if len(reply) > 300 else reply
                history.append({"role": "user", "content": _img_user_content})
                history.append({"role": "assistant", "content": _img_summary})
                _memory_mod.save(history, chat_id=str(chat_id))
                print(f"[bot] photo: reply len={len(reply)} working_id={working_id}", flush=True)
                if working_id and not edit_message(chat_id, working_id, f"🐬 {reply}", parse_mode=None):
                    print(f"[bot] photo: edit_message failed — falling back to send_message", flush=True)
                    _sent = send_message(chat_id, f"🐬 {reply}", parse_mode=None)
                    print(f"[bot] photo: send_message result={_sent}", flush=True)
                else:
                    print(f"[bot] photo: edit_message ok (or working_id None)", flush=True)
            except Exception as e:
                err = f"🐬 Vision error: {str(e)[:200]}"
                print(f"[bot] photo EXCEPTION: {type(e).__name__}: {e}", flush=True)
                if working_id and not edit_message(chat_id, working_id, err):
                    send_message(chat_id, err)
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
        else:
            send_message(chat_id, "🐬 Could not download image.")
        return

    # Handle voice notes
    if voice_file_id:
        _ensure_agent()
        working_id = send_message(chat_id, "🎙️ _Listening…_")
        local_path = download_file(voice_file_id)
        if local_path:
            try:
                import core.memory.memory as _memory_mod
                import core.agent as _agent_voice
                from core.inference.model import infer_with_audio

                # Stage 1: STT
                if working_id:
                    edit_message(chat_id, working_id, "🎙️ _Transcribing audio…_")
                with _MultimodalActivity(), TypingKeepAlive(chat_id), voice_activity("stt"):
                    transcript = infer_with_audio(
                        local_path,
                        prompt="Transcribe exactly what is said in this audio. Output only the spoken words, nothing else.",
                        max_new_tokens=8192,
                        mode="stt",
                    )
                try:
                    os.unlink(local_path)
                except Exception:
                    pass

                if not transcript or not transcript.strip():
                    raise ValueError("STT returned empty transcript")

                # Prepend any typed text (rare but possible)
                user_text = f"{text} {transcript}".strip() if text else transcript
                print(f"[voice] transcript: {user_text[:120]!r}", flush=True)

                # ── First-contact voice sample collection ────────────────────────
                # If no voice samples are available, save this voice as the first sample
                # so voice cloning works going forward.
                if not _list_voice_samples():
                    _samples_dir = os.path.expanduser(
                        os.environ.get("KERNEL_VOICE_SAMPLES_DIR",
                                        os.path.expanduser("~/.openclaw/media/voice-samples"))
                    )
                    os.makedirs(_samples_dir, exist_ok=True)
                    _sample_path = os.path.join(_samples_dir, "first-contact-voice.ogg")
                    if os.path.isfile(local_path):
                        import shutil
                        shutil.copy2(local_path, _sample_path)
                        print(f"[voice] First-contact sample saved: {_sample_path}", flush=True)
                        # Also set as active voice sample (convert to wav for clone)
                        _active_voice_sample = _sample_path
                    else:
                        print(f"[voice] Cannot save first-contact sample: {local_path} not found", flush=True)
                # ──────────────────────────────────────────────────────────────────

                # Stage 2: Show transcript, start thinking
                if working_id:
                    edit_message(chat_id, working_id, f"🎙️ _{_esc(transcript)}_\n\n🤔 _Thinking…_")

                log_lines = []
                step_num = [0]

                def _voice_step_cb(n, tool_name, args=None, result=None):
                    step_num[0] = n
                    log_lines.append(_format_tool_step(n, tool_name, args, result))
                    snippet = f"🎙️ _{_esc(transcript[:60])}_\n\n🔍 " + "\n\n".join(log_lines[-4:])
                    if working_id:
                        edit_message(chat_id, working_id, snippet)

                # Stage 3: Agent inference
                with TypingKeepAlive(chat_id):
                    reply = _agent_voice.triage(user_text, step_callback=_voice_step_cb, chat_id=str(chat_id))

                if not reply or not reply.strip():
                    raise ValueError("triage returned empty response")

                # Persist voice note + reply into conversation history.
                # Always run after triage so the user turn carries the [Voice note] label,
                # even when infer_with_tools already saved a plain-text user entry.
                _voice_history = _memory_mod.load(chat_id=str(chat_id))
                if (_voice_history
                        and _voice_history[-1].get("role") == "assistant"
                        and len(_voice_history) >= 2
                        and _voice_history[-2].get("role") == "user"):
                    # triage already saved the pair — just relabel the user turn
                    _voice_history[-2]["content"] = f"[Voice note] {user_text}"
                else:
                    _voice_history.append({"role": "user", "content": f"[Voice note] {user_text}"})
                    _voice_history.append({"role": "assistant", "content": reply})
                _memory_mod.save(_voice_history, chat_id=str(chat_id))

                _memory_mod.record_attachment(
                    kind="voice", local_path="(temp-deleted)",
                    original_name="voice.ogg", mime_type="audio/ogg",
                    caption=f"[Voice note] {user_text}", chat_id=str(chat_id)
                )

                # Stage 4: Deliver text reply — keep transcript visible above reply
                # so the user can see what EVA heard before the answer.
                reply_text = f"🎙️ _{_esc(transcript)}_\n\n🐬 {reply}"
                if working_id and not edit_message(chat_id, working_id, reply_text):
                    send_message(chat_id, reply_text)

                # Stage 5: Clone voice async — send status msg first, audio arrives after
                import threading as _threading
                _clone_status_id = send_message(chat_id, "🔊 _Cloning voice reply…_")

                def _send_voice_async(text: str, cid: str, status_id):
                    try:
                        with TypingKeepAlive(cid, action="record_voice"), voice_activity("clone"):
                            _wav = _clone_voice_reply(text)
                        if _wav:
                            send_voice(cid, _wav)
                            os.unlink(_wav)
                            if status_id:
                                edit_message(cid, status_id, "🔊 _Voice reply sent_")
                        else:
                            if status_id:
                                edit_message(cid, status_id, "🔇 _Voice clone unavailable_")
                    except Exception as _ve:
                        print(f"[voice] clone async error: {_ve}")
                        if status_id:
                            edit_message(cid, status_id, f"🔇 _Voice error: {_esc(str(_ve)[:80])}_")

                _threading.Thread(
                    target=_send_voice_async,
                    args=(reply, chat_id, _clone_status_id),
                    daemon=True
                ).start()

            except Exception as e:
                err = f"🐬 Audio error: {str(e)[:200]}"
                if working_id and not edit_message(chat_id, working_id, err):
                    send_message(chat_id, err)
        else:
            err = "🐬 Could not download voice note."
            if working_id and not edit_message(chat_id, working_id, err):
                send_message(chat_id, err)
        return

    # Handle document/PDF attachments
    # Download and save to workspace/documents/, then pass path to agent triage.
    # The agent's tool loop selects the right skill (kernel-doc-retrieval for PDFs,
    # translate-document for translation, etc.) — no extraction happens here.
    if document_file_id:
        _ensure_agent()
        local_path = download_file(document_file_id)
        if local_path:
            try:
                import time as _time, shutil as _shutil
                fname = document_name or os.path.basename(local_path)
                ws_docs = str(DOCUMENTS_DIR)
                os.makedirs(ws_docs, exist_ok=True)
                ts = _time.strftime("%Y%m%d-%H%M%S")
                saved_name = f"{ts}_{fname}"
                saved_path = os.path.join(ws_docs, saved_name)
                _shutil.copy2(local_path, saved_path)
                os.unlink(local_path)
                print(f"[bot] Document saved: {saved_path}", flush=True)

                # Hand path + caption to agent — skill selection happens in triage/tool loop
                user_caption = text.strip() if text.strip() else "Process this document."
                full_prompt = (
                    f"{user_caption}\n\n"
                    f"[File saved to: {saved_path} | name: {fname} | type: {document_mime or 'unknown'}]"
                )

                import core.memory.memory as _memory_mod
                # Record attachment durably before triage
                _memory_mod.record_attachment(
                    kind="document",
                    local_path=saved_path,
                    original_name=fname,
                    mime_type=document_mime or "",
                    caption=user_caption,
                    chat_id=str(chat_id),
                )
                # Inject recent-attachment context into the prompt
                att_ctx = _memory_mod.attachment_context_block(chat_id=str(chat_id), limit=3, max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                if att_ctx:
                    full_prompt = f"{att_ctx}\n\n{full_prompt}"

                # For PDFs: invoke extract_document native tool directly —
                # no skill lookup, no ecosystem dependency, always available.
                is_pdf = fname.lower().endswith(".pdf") or (document_mime or "").lower() == "application/pdf"
                if is_pdf:
                    try:
                        import os as _os
                        from core.skills import load_all as _load_skills, find as _find_skill, run as _run_skill
                        import core.agent as _agent_mod
                        _skills_dir = _agent_mod._config.get("skills_dir", "./skills") if _agent_mod._config else "./skills"
                        _skills_dir = _os.path.expanduser(_os.environ.get("SKILLS_DIR") or _skills_dir)
                        _core = _agent_mod._config.get("core_skills", []) if _agent_mod._config else []
                        _all_skills = _load_skills(_skills_dir, core_skills=_core)
                        _skill = _find_skill("kernel-doc-retrieval", _all_skills)
                        working_id = send_message(chat_id, "🐬 Processing PDF…")
                        with _MultimodalActivity():
                            if _skill:
                                skill_result = _run_skill(_skill, f"/markdown {saved_path}", lambda *a: "")
                                reply = skill_result if skill_result and skill_result.strip() else (
                                    f"✅ PDF processed: `{fname}` — markdown sent as Telegram attachment."
                                )
                            else:
                                reply = f"⚠️ kernel-doc-retrieval skill not found — PDF saved to: `{saved_path}`"
                        if working_id and not edit_message(chat_id, working_id, f"🐬 {reply}"):
                            send_message(chat_id, f"🐬 {reply}")
                    except Exception as _pdf_err:
                        send_message(chat_id, f"🐬 PDF error: {str(_pdf_err)[:200]}")
                    history = _memory_mod.load(chat_id=str(chat_id))
                    history.append({"role": "user", "content": f"[Document: {fname}] {text}"})
                    history.append({"role": "assistant", "content": reply})
                    _memory_mod.save(history, chat_id=str(chat_id))
                    return

                from core.agent import triage
                reply = triage(full_prompt, chat_id=str(chat_id))

                # Completion guard: retry once if reply ignored the attached file
                guard = _memory_mod.attachment_guard(user_caption, reply, chat_id=str(chat_id))
                if not guard["ok"]:
                    print(f"[bot] attachment_guard FAIL (retrying): {guard['reason']}", flush=True)
                    retry_prompt = (
                        f"{att_ctx}\n\n"
                        f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                        f"{full_prompt}"
                    )
                    reply = triage(retry_prompt, chat_id=str(chat_id))
                    guard2 = _memory_mod.attachment_guard(user_caption, reply, chat_id=str(chat_id))
                    if not guard2["ok"]:
                        reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

                history = _memory_mod.load(chat_id=str(chat_id))

                history.append({"role": "user", "content": f"[Document: {fname}] {text}"})
                history.append({"role": "assistant", "content": reply})
                _memory_mod.save(history, chat_id=str(chat_id))

                send_message(chat_id, f"\U0001f42c {reply}")
            except Exception as e:
                send_message(chat_id, f"\U0001f42c Document error: {str(e)[:200]}")
        else:
            send_message(chat_id, "\U0001f42c Could not download document.")
        return

    # Slash commands

    # ── /skill_<slug> — shortcut dispatched from Telegram command picker ──────
    if text.startswith("/skill_"):
        _skill_parts = text[7:].split(None, 1)
        skill_slug = _skill_parts[0].strip()
        skill_user_input = _skill_parts[1].strip() if len(_skill_parts) > 1 else ""
        try:
            import core.agent as _ag_sk
            import re as _re_sk
            def _slug_match(name):
                return _re_sk.sub(r"[^a-z0-9_]", "_", name.lower())[:26]  # 32-6 for "skill_"
            matched = next((s for s in (_ag_sk._skills or []) if _slug_match(s["name"]) == skill_slug), None)
            if matched:
                print(f"[bot] /skill_ dispatch: {matched['name']} input={skill_user_input!r}", flush=True)
                _ensure_agent()
                # Route through the same /run skill path so step-logging and streaming work correctly
                _run_text = f"/run {matched['name']}"
                if skill_user_input:
                    _run_text = f"/run {matched['name']} {skill_user_input}"
                with TypingKeepAlive(chat_id):
                    result = _ag_sk.triage(_run_text, chat_id=str(chat_id))
                send_message(chat_id, f"🐬 {result}" if result else "🐬 Done.")
            else:
                send_message(chat_id, f"🐬 No skill matching `{skill_slug}`. Try /skills.")
        except Exception as _e:
            send_message(chat_id, f"🐬 Skill error: {str(_e)[:200]}")
        return

    # ── /run_<slug> — shortcut dispatched from Telegram command picker ────────
    if text.startswith("/run_"):
        routine_slug = text[5:].split()[0].strip()
        try:
            import core.agent as _ag_rn
            import re as _re_rn
            def _slug_match_r(name):
                return _re_rn.sub(r"[^a-z0-9_]", "_", name.lower())[:28]  # 32-4 for "run_"
            matched = next((r for r in (_ag_rn._routines or []) if _slug_match_r(r["name"]) == routine_slug), None)
            if matched:
                print(f"[bot] /run_ dispatch: {matched['name']}", flush=True)
                _ensure_agent()
                with TypingKeepAlive(chat_id):
                    result = _ag_rn.triage(f"/run {matched['name']}", chat_id=str(chat_id))
                send_message(chat_id, f"🐬 {result}" if result else "🐬 Done.")
            else:
                send_message(chat_id, f"🐬 No routine matching `{routine_slug}`. Try /routines.")
        except Exception as _e:
            send_message(chat_id, f"🐬 Routine error: {str(_e)[:200]}")
        return

    if text == "/new":
        import core.memory.memory as _mem_new
        import prompt_logger as _pl_new
        _mem_new.clear_chat(str(chat_id))
        _pl_new.clear_chat_logs(str(chat_id))
        send_message(chat_id, "🐬 Fresh start — conversation cleared. What's on your mind?")
        return

    if text == "/fresh":
        import core.memory.memory as _mem_fresh
        import prompt_logger as _pl_fresh
        result = _mem_fresh.fresh_chat(str(chat_id))
        _pl_fresh.clear_chat_logs(str(chat_id))
        msgs = result.get("messages_deleted", 0)
        atts = result.get("attachments_deleted", 0)
        send_message(chat_id, f"🐬 Factory reset complete — {msgs} messages and {atts} attachments cleared. Clean slate!")
        return

    if text.startswith("/session"):
        import core.memory.memory as _mem_sess
        sub_sess = text[len("/session"):].strip().lower()
        if sub_sess in ("new", "split"):
            result = _mem_sess.rotate_session(str(chat_id))
            if "error" in result:
                send_message(chat_id, f"❌ Session rotation failed: {result['error']}")
            else:
                prev = result.get("prev_session_id", "?")
                num = result.get("session_number", "?")
                send_message(chat_id,
                    f"🐬 New session #{num} started.\n"
                    f"Previous session: `{prev}`\n"
                    f"Use `recall_memory('{prev}')` to pull context from it."
                )
        else:
            send_message(chat_id,
                "📋 *Session commands*\n\n"
                "`/session new` — start a fresh session (keeps old history accessible via recall_memory)\n"
                "`/fresh` — factory reset (deletes all history for this chat)"
            )
        return

    # /voice-clone with no args → show sample picker + usage hint
    if text == "/voice-clone":
        samples = _list_voice_samples()
        active_label = os.path.basename(_active_voice_sample).replace(".wav","").replace(".mp3","").replace(".m4a","")
        msg = (
            "\U0001f3a4 *Voice Clone*\n"
            "Active: `" + active_label + "`\n\n"
            "Usage:\n"
            "`/voice-clone <text>` \u2014 active sample\n"
            "`/voice-clone ricky <text>` \u2014 ricky sample\n\n"
            "Or tap a sample below to switch:"
        )
        if samples:
            buttons = [[{"text": ("✅ " if s["name"] == active_name else "") + s["name"], "callback_data": f"set_voice_{i}"}] for i, s in enumerate(samples)]
            send_buttons(chat_id, msg, buttons)
        else:
            send_message(chat_id, msg)
        return

    if text == "/voices" or text.startswith("set_voice_"):
        if text.startswith("set_voice_"):
            # callback from inline button: set_voice_<index>
            try:
                idx = int(text.split("_")[-1])
                samples = _list_voice_samples()
                if 0 <= idx < len(samples):
                    _active_voice_sample = samples[idx]["path"]
                    send_message(chat_id, f"🎤 Active voice: *{samples[idx]['name']}*")
                else:
                    send_message(chat_id, "❌ Invalid voice index.")
            except Exception as e:
                send_message(chat_id, f"❌ Error: {e}")
            return
        samples = _list_voice_samples()
        if not samples:
            send_message(chat_id, "🐬 No voice samples found in ~/.openclaw/media/voice-samples/")
            return
        active_name = os.path.basename(_active_voice_sample).replace(".wav", "").replace(".mp3", "").replace(".m4a", "")
        msg = f"🎤 *Available voices*\nActive: `{active_name}`\n\nTap to switch:"
        buttons = [[{"text": ("\u2705 " if s["name"] == active_label else "") + s["name"], "callback_data": f"set_voice_{i}"}] for i, s in enumerate(samples)]
        send_buttons(chat_id, msg, buttons)
        return

    if text in ("/start", "/help"):
        __version__ = _get_current_version()
        intro = (
            f"🐬 *Kernel Evo v{__version__}*\n"
            f"Self-evolving AI agent — pick your mode:\n\n"
            f"🏠 **/local** — load Nemotron-3B locally for inference. "
            f"STT/vision/voice all work. Skill synthesis and planning go to cloud.\n\n"
            f"☁️ **/cloud** — everything runs through cloud providers. "
            f"Zero local VRAM used. No STT/voice/vision (no local model loaded).\n\n"
            f"🎛️ **/models** — pick a provider and its model (local pulls on demand).\n"
            f"🗺️ **/provider** — routing table for each call type."
        )
        buttons = [
            # ─ Mode selection
            [{"text": "🏠 /local — load Nemotron", "callback_data": "/local"},
             {"text": "☁️ /cloud — all cloud", "callback_data": "/cloud"}],
            # ─ Models & Provider
            [{"text": "🎛️ Models", "callback_data": "/models"},
             {"text": "🗺️ Provider", "callback_data": "/provider"}],
            # ─ Chat & Inference
            [{"text": "🧠 Status", "callback_data": "/status"}, {"text": "💭 Thoughts", "callback_data": "/thoughts"}],
            # ─ Skills & Routines
            [{"text": "🔧 Skills", "callback_data": "/skills"}, {"text": "⚙️ Routines", "callback_data": "/routines"}],
            # ─ Replicas
            [{"text": "🤖 List replicas", "callback_data": "/replica list"}, {"text": "➕ Spawn", "callback_data": "/replica spawn"}, {"text": "👥 Clone", "callback_data": "/replica clone"}],
            [{"text": "⏹ Stop replica", "callback_data": "/replica stop"}, {"text": "🗂️ Workspaces", "callback_data": "/workspaces"}],
            # ─ Evolution
            [{"text": "🧬 Evolve", "callback_data": "/evolve"}, {"text": "📊 Evolve status", "callback_data": "/evolve status"}, {"text": "📈 Evolve dash", "callback_data": "/evolve dash"}],
            # ─ System
            [{"text": "📊 Status", "callback_data": "/status"}, {"text": "🔢 Version", "callback_data": "/version"}, {"text": "🖥️ System", "callback_data": "/system"}],
            [{"text": "🚀 Init workspace", "callback_data": "/init"}],
            # ─ Maintenance
            [{"text": "🔄 Update", "callback_data": "/update"}, {"text": "🔁 Restart", "callback_data": "/restart"}, {"text": "⏪ Rollback", "callback_data": "/rollback"}],
            [{"text": "🎤 Voices", "callback_data": "/voices"}, {"text": "🆕 New conversation", "callback_data": "/new"}],
            # ─ Repo access
            [{"text": "🔒 Private repos", "callback_data": "/private_repo show"}, {"text": "🧯 Clone repo", "callback_data": "/clone"}],
        ]
        send_buttons(chat_id, intro, buttons)
        return

    # /init [--force] — alias for /evolve init so all channels use the same path.
    if text == "/init" or text.startswith("/init "):
        text = "/evolve init" + text[len("/init"):]

    if text == "/skills":
        _ensure_agent()
        import yaml

        with open(CONFIG_PATH) as _f:
            _cfg = yaml.safe_load(_f)
        from core.skills import load_all

        skills = load_all(
            str(os.path.expanduser(_cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
        )
        if not skills:
            send_message(chat_id, "🔧 No skills loaded.")
        else:
            # Send as inline buttons (2 per row)
            buttons = []
            row = []
            for s in skills:
                row.append(
                    {"text": s["name"], "callback_data": f"skill_info_{s['name']}"}
                )
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)
            send_buttons(
                chat_id, f"🔧 *Skills ({len(skills)}) — tap to learn more:*", buttons
            )
        return

    if text == "/routines":
        _ensure_agent()
        import yaml

        with open(CONFIG_PATH) as _f:
            _cfg = yaml.safe_load(_f)
        from core.routines import load_all

        routines = load_all(
            str(os.path.expanduser(_cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
        )
        if not routines:
            send_message(chat_id, "⚙️ No routines loaded.")
        else:
            # Send as inline buttons — one per row with Run button
            buttons = []
            for r in routines:
                buttons.append(
                    [
                        {
                            "text": f"⚙️ {r['name']}",
                            "callback_data": f"routine_info_{r['name']}",
                        },
                        {"text": "▶ Run", "callback_data": f"routine_run_{r['name']}"},
                    ]
                )
            send_buttons(chat_id, f"⚙️ *Routines ({len(routines)}):*", buttons)
        return

    if text == "/packages":
        from bootstrap import ECOSYSTEM_ROOT
        import yaml
        skills_list = []
        routines_list = []
        for tier_dir in sorted(ECOSYSTEM_ROOT.iterdir()):
            manifest = tier_dir / "manifest.yaml"
            if not manifest.exists():
                continue
            with open(manifest) as f:
                data = yaml.safe_load(f) or {}
            tier = tier_dir.name
            for s in data.get("skills", []):
                skills_list.append((s["name"], s.get("description", ""), tier))
            for r in data.get("routines", []):
                routines_list.append((r["name"], r.get("description", ""), tier))

        lines_out = [f"📦 *Packages — {len(skills_list)} skills · {len(routines_list)} routines*\n"]
        lines_out.append("\n*Skills:*")
        for name, desc, tier in skills_list:
            lines_out.append(f"  • `{name}` — {_esc(desc)} _[{_esc(tier)}]_")
        lines_out.append("\n*Routines:*")
        for name, desc, tier in routines_list:
            lines_out.append(f"  • `{name}` — {_esc(desc)} _[{_esc(tier)}]_")

        buttons = []
        row = []
        for name, desc, tier in skills_list:
            row.append({"text": f"📥 {name}", "callback_data": f"install_skill_{name}"})
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        row = []
        for name, desc, tier in routines_list:
            row.append({"text": f"📥 {name}", "callback_data": f"install_routine_{name}"})
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        send_buttons(chat_id, "\n".join(lines_out), buttons)
        return

    if text == "/status":
        import torch
        __version__ = _get_current_version()

        free_mb = 0
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_mb = free // (1024 * 1024)

        # Read actual loaded model from model server instead of hardcoding
        try:
            from core.inference.model_client import health as _mc_health
            _mh = _mc_health()
            _model_label = _mh.get("model", "unknown")
            if _mh.get("nemotron"):
                _nmode = _mh.get("nemotron_mode", "ar")
                _model_label = f"{_model_label} [Nemotron/{_nmode}]"
        except Exception:
            _model_label = "unknown (model server unreachable)"

        try:
            from core.inference.provider import get_provider as _gp_s
            _prov_label = _gp_s().get_provider("task_inference")
        except Exception:
            _prov_label = "unknown"

        update_note = f"\n🆕 Update available: {_latest_version}" if _latest_version and _latest_version != __version__ else ""
        send_message(
            chat_id,
            (
                f"🐬 *Kernel Evo Status*\n"
                f"Version: v{__version__}{update_note}\n"
                f"Model: {_model_label}\n"
                f"Provider: {_prov_label}\n"
                f"VRAM free: {free_mb}MB\n"
                f"Ready: {'✅' if _agent_ready else '⏳ loading on first message'}"
            ),
        )
        return

    if text == "/stop":
        _handle_stop(chat_id)
        return

    if text == "/restart":
        _handle_restart(chat_id)
        return

    if text == "/update":
        _handle_update(chat_id)
        return

    if text == "/rollback":
        _handle_rollback(chat_id)
        return

    if text == "/local":
        _ensure_agent()
        import yaml as _yaml, json as _json_p
        with open(CONFIG_PATH) as _pf:
            _cfg_p = _yaml.safe_load(_pf)
        _api_port = _cfg_p.get('api', {}).get('port', 8779)
        try:
            import urllib.request as _ur
            body = {
                "task_inference": "local",
                "synthesis": "openai",
                "critic": "openrouter",
                "planning": "openrouter",
                "trajectory_teacher": "openai",
                "persist": True,
            }
            payload = _json_p.dumps(body).encode()
            req = _ur.Request(f"http://localhost:{_api_port}/provider/set",
                              data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with _ur.urlopen(req, timeout=5) as r:
                result = _json_p.loads(r.read())
            send_message(chat_id, f"🏠 *Local mode activated* — task inference routed to Nemotron (will load on first request).")
        except Exception as e:
            send_message(chat_id, f"\u274c Mode switch error: {e}")
        return

    if text == "/cloud":
        _show_cloud_provider_picker(chat_id)


    # ── /models — local model hot-swap ───────────────────────────────────────
    if text.startswith("/models"):
        parts = text.split(None, 2)
        sub = parts[1].lower() if len(parts) > 1 else ""

        from pathlib import Path as _Path
        from core.hf_cache import resolve_hf_hub_dir, ensure_hf_home_env

        # MS4: shared with model_server.py so the bot's "is this downloaded?"
        # check can't silently disagree with what the server can actually load.
        ensure_hf_home_env()
        _hub_dir = resolve_hf_hub_dir()

        def _cache_prefix(repo_id: str) -> str:
            return f"models--{repo_id.replace('/', '--')}"

        # Build from config.yaml model_catalog; fall back to a hardcoded minimal set
        # so the menu still works if the config key is missing.
        try:
            import yaml as _yaml_mc
            _cfg_models = _yaml_mc.safe_load(open(CONFIG_PATH)) or {}
            _raw_catalog = _cfg_models.get("model_catalog", [])
            KNOWN_MODELS = {
                entry["key"]: {
                    "label": entry["label"],
                    "repo_id": entry["repo_id"],
                    "cache_prefix": _cache_prefix(entry["repo_id"]),
                    "drafter": entry.get("drafter", ""),
                    **({"note": entry["note"]} if entry.get("note") else {}),
                }
                for entry in _raw_catalog if entry.get("key") and entry.get("repo_id")
            }
        except Exception:
            _cfg_models = {}
            KNOWN_MODELS = {}

        def _collect_path_values(obj):
            values = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        values.extend(_collect_path_values(v))
                    elif isinstance(v, str) and ("path" in str(k).lower() or v.startswith("/") or v.startswith("~")):
                        values.append(v)
            elif isinstance(obj, list):
                for item in obj:
                    values.extend(_collect_path_values(item))
            return values

        _configured_paths = []
        for _p in _collect_path_values(_cfg_models):
            try:
                _configured_paths.append(_Path(_p).expanduser())
            except Exception:
                pass

        def _is_downloaded(info: dict) -> bool:
            if any(_hub_dir.glob(f"{info['cache_prefix']}*")):
                return True

            target = info["cache_prefix"]
            for p in _configured_paths:
                try:
                    if p.exists() and target in str(p):
                        return True
                except Exception:
                    continue
            return False

        if sub == "" or sub == "menu":
            # Unified model manager: show active provider/model + provider selector.
            try:
                import core.inference.model_client as _mc
                h = _mc.health()
                current = h.get("model", "unknown")
                vram = h.get("vram_free_mb", 0)
            except Exception:
                current, vram = "unknown", 0

            _prov_data = {}
            try:
                import urllib.request as _ur2
                with _ur2.urlopen(f"http://localhost:{8779}/provider", timeout=3) as _pr:
                    _prov_data = json.loads(_pr.read())
                    _task_provider = _prov_data.get("routing", {}).get("task_inference", {}).get("provider", "?")
                    _task_model = _prov_data.get("routing", {}).get("task_inference", {}).get("model", "")
            except Exception:
                _task_provider, _task_model = "?", ""

            _provider_icons = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                               "hf": "\U0001f917", "copilot": "\u26a1", "openrouter": "\U0001f310"}
            _prov_icon = _provider_icons.get(_task_provider, "\U0001f4e1")

            lines = [
                f"\U0001f916 *Model Manager*\n",
                f"{_prov_icon} Task inference: `{_task_provider}`" + (f" / `{_task_model}`" if _task_model else ""),
                f"\U0001f4be Local model loaded: `{current}`",
                f"\U0001f5a5 VRAM free: `{vram} MB`\n",
                "*Select a provider to pick its model:*",
            ]
            buttons = [
                # Provider selector — each opens that provider's model list
                [{"text": "\U0001f3e0 Local", "callback_data": "/models provider local"},
                 {"text": "\U0001f916 OpenAI", "callback_data": "/models provider openai"}],
                [{"text": "\U0001f9e0 Anthropic", "callback_data": "/models provider anthropic"},
                 {"text": "\U0001f917 HF", "callback_data": "/models provider hf"}],
                [{"text": "\u26a1 Copilot", "callback_data": "/models provider copilot"},
                 {"text": "\U0001f310 OpenRouter", "callback_data": "/models provider openrouter"}],
                # Local management
                [{"text": "\U0001f504 Reload current", "callback_data": "/models reload"},
                 {"text": "\U0001f50d Search Hub", "callback_data": "/models search "}],
            ]
            # If Nemotron is active, add mode-switch buttons
            if h.get("nemotron"):
                cur_mode = h.get("nemotron_mode", "?")
                lines.append(f"\nNemotron mode: `{cur_mode}` (block={h.get('nemotron_block_length',32)})")
                buttons.append([
                    {"text": "AR",          "callback_data": "/models mode ar"},
                    {"text": "Diffusion",   "callback_data": "/models mode diffusion"},
                    {"text": "\u26a1 linear_spec", "callback_data": "/models mode linear_spec"},
                ])
            send_buttons(chat_id, "\n".join(lines), buttons)
            return

        elif sub == "provider" and len(parts) >= 3:
            # /models provider <name> — show that provider's selectable models.
            provider = parts[2].lower()
            if provider == "local":
                # Local picker: actual pulled models (selectable) + curated unpulled (pull).
                try:
                    import urllib.request as _ur_l
                    # Pulled local models from /models
                    local_models = []
                    try:
                        with _ur_l.urlopen(f"http://localhost:{8779}/models", timeout=5) as _rm:
                            local_models = (json.loads(_rm.read()) or {}).get("models", [])
                    except Exception:
                        pass
                    # Curated catalog from /models/curated (marks which are downloaded)
                    curated = []
                    try:
                        with _ur_l.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
                            curated = (json.loads(_rc.read()) or {}).get("curated", [])
                    except Exception:
                        pass
                    downloaded_ids = {m.get("model", "").lower() for m in local_models}

                    lines = ["\U0001f3e0 *Local Models*\n"]
                    buttons = []

                    # 1) Downloaded models — always surfaced, never empty.
                    pulled = [m for m in local_models if m.get("model")]
                    if pulled:
                        lines.append("*✅ Downloaded (tap to use):*")
                        for m in pulled:
                            repo = m.get("model", "")
                            slot = m.get("slot") or ""
                            lines.append(f"\u2705 `{repo}`" + (f" \u2192 slot `{slot}`" if slot else ""))
                            tok = _register_repo_token(repo)
                            buttons.append([{"text": f"\U0001f504 Use {repo}", "callback_data": f"/models use {tok}"}])
                    else:
                        lines.append("*\u2139\ufe0f No models downloaded yet.*\n_Pull one below or search the Hub._")

                    # 2) Curated models not yet pulled → pull.
                    unpulled = [c for c in curated if c.get("repo_id", "").lower() not in downloaded_ids]
                    if unpulled:
                        lines.append("\n*📥 Not downloaded (pull to use):*")
                        for cm in unpulled:
                            repo = cm.get("repo_id", "")
                            label = cm.get("label") or repo
                            lines.append(f"\u23ec `{repo}`")
                            tok = _register_repo_token(repo)
                            buttons.append([{"text": f"\u23ec Pull {label}", "callback_data": f"/models pull {tok}"}])

                    # 3) Search Hub to find + download more locally.
                    buttons.append([
                        {"text": "\U0001f50d Search Hub", "callback_data": "/models search "},
                        {"text": "\U0001f504 Refresh", "callback_data": "/models provider local"},
                        {"text": "\u2190 Back", "callback_data": "/models"},
                    ])
                    send_buttons(chat_id, "\n".join(lines), buttons)
                except Exception as e:
                    send_message(chat_id, f"\u274c Local models error: {e}")
                return

            # Cloud provider: show that provider's available models from /provider/models.
            try:
                import urllib.request as _ur_c
                with _ur_c.urlopen(f"http://localhost:{8779}/provider/models?provider={provider}&capability=text", timeout=8) as _rc:
                    data = json.loads(_rc.read())
                models = (data.get("models", {}) or {}).get(provider, [])
                if not models:
                    send_message(chat_id, f"\u274c No models listed for provider `{provider}` (or it's unavailable).")
                    return
                _provider_icons = {"openai": "\U0001f916", "anthropic": "\U0001f9e0", "hf": "\U0001f917",
                                   "copilot": "\u26a1", "openrouter": "\U0001f310"}
                icon = _provider_icons.get(provider, "\U0001f4e1")
                lines = [f"{icon} *{provider.title()} models*\n", "_Tap a model to route task_inference to it:_"]
                buttons = []
                for m in models[:12]:
                    lines.append(f"\U0001f4e6 `{m}`")
                    buttons.append([{"text": f"\U0001f504 Use {m}", "callback_data": f"/provider set task_inference {provider} persist|model|{m}"}])
                buttons.append([{"text": "\u2190 Back", "callback_data": "/models"}])
                send_buttons(chat_id, "\n".join(lines), buttons)
            except Exception as e:
                send_message(chat_id, f"\u274c Cloud models error: {e}")
            return

        elif sub == "use" and len(parts) >= 3:
            # /models use <token> — set task_inference to local + load the pulled model.
            repo_id = _resolve_repo_token(parts[2].strip())
            slot = _curated_slot_for_repo(repo_id)
            try:
                import urllib.request as _ur_u
                # Assign to slot if we know one, then route task_inference to local.
                if slot:
                    apayload = json.dumps({"repo_id": repo_id, "slot": slot}).encode()
                    areq = _ur_u.Request(f"http://localhost:{8779}/models/assign",
                                         data=apayload, headers={"Content-Type": "application/json"}, method="POST")
                    try:
                        with _ur_u.urlopen(areq, timeout=5):
                            pass
                    except Exception:
                        pass
                body = {"task_inference": "local", "persist": True}
                payload = json.dumps(body).encode()
                req = _ur_u.Request(f"http://localhost:{8779}/provider/set",
                                    data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with _ur_u.urlopen(req, timeout=5) as r:
                    result = json.loads(r.read())
                send_message(chat_id, f"\u2705 Task inference \u2192 local (`{repo_id}`)" + (f" \u2192 slot `{slot}`" if slot else ""))
            except Exception as e:
                send_message(chat_id, f"\u274c Use failed: {e}")
            return

        elif sub == "load" and len(parts) >= 3:
            key = parts[2].lower()
            if key not in KNOWN_MODELS:
                send_message(chat_id, f"\u274c Unknown model key `{key}`. Valid: {', '.join(KNOWN_MODELS)}")
                return
            info = KNOWN_MODELS[key]
            if not _is_downloaded(info):
                send_message(chat_id, f"\u274c Model not downloaded yet in `{_hub_dir}` or configured model paths for `{info['repo_id']}`\nUse `/models download {key}` first.")
                return
            working_id = send_message(chat_id, f"\u23f3 Loading {info['label']}\u2026\n_Step 1/4: starting model server_")
            try:
                import core.inference.model_client as _mc
                import urllib.request as _ur_m
                import subprocess as _sp
                import time as _time

                # Step 1 — ensure model server is running (start it if not)
                if not _mc.is_server_running():
                    edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n_Step 1/4: model server not running — starting..._")
                    _sp.Popen(
                        [sys.executable,
                         str((Path(__file__).resolve().parent / 'model_server.py')),
                         '--config', CONFIG_PATH, '--lazy'],
                        stdout=open('/tmp/kernel_evolving_model_server.log', 'a'),
                        stderr=_sp.STDOUT,
                        start_new_session=True,
                    )
                    # Wait up to 15s for socket
                    for _ in range(15):
                        _time.sleep(1)
                        if _mc.is_server_running():
                            break
                    if not _mc.is_server_running():
                        edit_message(chat_id, working_id, "\u274c Model server failed to start. Check /tmp/kernel_evolving_model_server.log")
                        return
                    edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 1/4: model server started_\n_Step 2/4: unloading current model from VRAM_")
                else:
                    edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 1/4: model server already running_\n_Step 2/4: unloading current model from VRAM_")

                # Step 2 — unload current model to free VRAM
                if _mc.is_server_running():
                    unload_result = _mc.unload()
                    freed = unload_result.get("freed_mb", 0)
                    edit_message(chat_id, working_id, f"\u23f3 Loading {info['label']}\u2026\n\u2705 _Step 2/4: freed ~{freed}MB VRAM_\n_Step 3/4: loading {info['label']} (~30-60s)_")

                # Step 3 — load the new model via swap_model
                result = _mc.swap_model(info["repo_id"], drafter_path=info["drafter"] or "")
                if "error" in result:
                    edit_message(chat_id, working_id, f"\u274c Swap failed: {result['error']}")
                    return

                # Step 4 — switch task_inference to local
                try:
                    req_body = json.dumps({"task_inference": "local"}).encode()
                    req = _ur_m.Request(f"http://localhost:{8779}/provider/set",
                                        data=req_body, headers={"Content-Type": "application/json"}, method="POST")
                    _ur_m.urlopen(req, timeout=5)
                    edit_message(chat_id, working_id,
                        f"\u2705 *{info['label']}* loaded\n"
                        f"Model: `{result.get('model','?')}` | Backend: `{result.get('backend','?')}`\n"
                        f"Task inference: `local` (auto-switched)")
                except Exception as _pe:
                    edit_message(chat_id, working_id,
                        f"\u2705 *{info['label']}* loaded\nModel: `{result.get('model','?')}`\n"
                        f"\u26a0\ufe0f Provider switch failed: {_pe}")
            except Exception as e:
                edit_message(chat_id, working_id, f"\u274c Exception: {e}")
            return

        elif sub == "reload":
            working_id = send_message(chat_id, "\u23f3 Reloading current model\u2026")
            try:
                import yaml as _yaml
                from core.inference import model_client as _mc
                cfg = _yaml.safe_load(open(CONFIG_PATH))
                path = cfg["model"].get("path") or cfg["model"].get("name", "")
                drafter = cfg["model"].get("drafter_path") or ""
                result = _mc.swap_model(path, drafter_path=drafter)
                if "error" in result:
                    edit_message(chat_id, working_id, f"\u274c Reload failed: {result['error']}")
                else:
                    edit_message(chat_id, working_id, f"\u2705 Reloaded: `{result.get('model','?')}`")
            except Exception as e:
                edit_message(chat_id, working_id, f"\u274c Exception: {e}")
            return

        elif sub == "download":
            key = parts[2].lower() if len(parts) >= 3 else ""
            if key not in KNOWN_MODELS:
                send_message(chat_id, f"\u274c Unknown model key. Valid: {', '.join(KNOWN_MODELS)}")
                return
            info = KNOWN_MODELS[key]
            send_message(chat_id, f"\u23ec Downloading {info['label']}\u2026 This runs in background, I'll ping you when done.")
            import threading as _thr
            def _do_download(chat_id=chat_id, info=info, key=key):
                try:
                    from huggingface_hub import snapshot_download
                    from core.hf_cache import resolve_hf_cache_root
                    import os as _os
                    token = _os.environ.get("HF_TOKEN", "")
                    repo_id = info["repo_id"]
                    _hf_home = resolve_hf_cache_root()
                    snapshot_download(repo_id=repo_id, cache_dir=_hf_home, token=token, ignore_patterns=["*.gguf"])
                    send_message(chat_id, f"\u2705 Downloaded {info['label']}\nCache: `{_hf_home}`\nUse `/models load {key}` to switch.")
                except Exception as e:
                    send_message(chat_id, f"\u274c Download failed: {e}")
            _thr.Thread(target=_do_download, daemon=True).start()
            return

        elif sub == "mode" and len(parts) >= 3:
            # /models mode ar|diffusion|linear_spec  — switch Nemotron generation mode at runtime
            new_mode = parts[2].lower()
            valid_modes = ("ar", "diffusion", "linear_spec")
            if new_mode not in valid_modes:
                send_message(chat_id, f"\u274c Invalid mode. Valid: {', '.join(valid_modes)}")
                return
            try:
                import core.inference.model_client as _mc
                h = _mc.health()
                if not h.get("nemotron"):
                    send_message(chat_id, "\u274c Nemotron is not the active model. Load it first with `/models load nemotron3b`.")
                    return
                # Update config live via /provider API trick — write to config.yaml
                import yaml as _yaml
                with open(CONFIG_PATH) as _f:
                    _cfg_live = _yaml.safe_load(_f)
                _cfg_live.setdefault("model", {})["generation_mode"] = new_mode
                with open(CONFIG_PATH, "w") as _f:
                    _yaml.dump(_cfg_live, _f, default_flow_style=False, allow_unicode=True)
                # Reload the model server so it picks up new mode
                working_id = send_message(chat_id, f"\u23f3 Switching Nemotron mode \u2192 `{new_mode}`\u2026 (reloading model ~30s)")
                path = _cfg_live["model"].get("path") or _cfg_live["model"].get("name")
                result = _mc.swap_model(path)
                if "error" in result:
                    edit_message(chat_id, working_id, f"\u274c Mode switch failed: {result['error']}")
                else:
                    edit_message(chat_id, working_id,
                        f"\u2705 Nemotron mode: `{new_mode}`\n"
                        f"block_length={_cfg_live['model'].get('block_length', 32)} "
                        f"threshold={_cfg_live['model'].get('threshold', 0.9)}")
            except Exception as e:
                send_message(chat_id, f"\u274c {e}")
            return

        elif sub == "search":
            # /models search <query> — search HuggingFace Hub via /hub/search
            query = " ".join(parts[2:]).strip()
            try:
                import urllib.request as _ur_s, urllib.parse as _up
                url = f"http://localhost:{8779}/hub/search?q={_up.quote(query)}&limit=10"
                with _ur_s.urlopen(url, timeout=15) as _r:
                    data = json.loads(_r.read())
                results = data.get("models", [])
                if not results:
                    send_message(chat_id, f"\U0001f50d No Hub results for `{query}`")
                    return
                lines = [f"\U0001f50d *Hub Search: {query}*\n"]
                buttons = []
                for m in results[:10]:
                    mid = m.get("id", "")
                    tag = m.get("pipeline_tag") or "unknown"
                    dl = m.get("downloads") or 0
                    lines.append(f"\U0001f4e6 `{mid}`\n   _{tag} \u00b7 {dl} downloads_")
                    # Use a short token to stay under Telegram's 64-byte callback limit.
                    tok = _register_repo_token(mid)
                    buttons.append([{"text": f"\u23ec Pull {mid}", "callback_data": f"/models pull {tok}"}])
                buttons.append([{"text": "\U0001f9e0 Back to models", "callback_data": "/models"}])
                send_buttons(chat_id, "\n".join(lines), buttons)
            except Exception as e:
                send_message(chat_id, f"\u274c Hub search failed: {e}")
            return

        elif sub == "pull":
            # /models pull <repo_id|token> — pull an arbitrary model via /pull (background)
            raw = " ".join(parts[2:]).strip()
            if not raw:
                send_message(chat_id, "Usage: `/models pull <repo_id>` e.g. `/models pull Qwen/Qwen2.5-Omni-3B`")
                return
            # Resolve a short callback token back to the full repo id.
            repo_id = _resolve_repo_token(raw)
            try:
                import urllib.request as _ur_p
                payload = json.dumps({"model": repo_id}).encode()
                req = _ur_p.Request(f"http://localhost:{8779}/pull",
                                    data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with _ur_p.urlopen(req, timeout=5) as _r:
                    data = json.loads(_r.read())
                job_id = data.get("job_id", "")
                send_message(chat_id,
                    f"\u23ec Pulling `{repo_id}` in background (job `{job_id[:8]}\u2026`)\n"
                    f"I'll ping you when it finishes. Then assign to a slot with `/models assign {repo_id} <slot>`.")
                # Poll in a background thread and notify on completion.
                import threading as _thr_p
                def _watch(jid=job_id, rid=repo_id):
                    import time as _tp
                    for _ in range(600):
                        _tp.sleep(3)
                        try:
                            with _ur_p.urlopen(f"http://localhost:{8779}/jobs/{jid}", timeout=5) as _rj:
                                j = json.loads(_rj.read())
                            st = j.get("status", "running")
                            if st == "succeeded":
                                send_message(chat_id, f"\u2705 Pulled `{rid}`\nUse `/models assign {rid} <slot>` to assign.")
                                return
                            if st == "failed":
                                send_message(chat_id, f"\u274c Pull of `{rid}` failed: {j.get('error')}")
                                return
                        except Exception:
                            continue
                    send_message(chat_id, f"\u23f3 Pull of `{rid}` still running \u2014 check later.")
                _thr_p.Thread(target=_watch, daemon=True).start()
            except Exception as e:
                send_message(chat_id, f"\u274c Pull request failed: {e}")
            return

        elif sub == "assign":
            # /models assign <repo_id> <slot> — assign a pulled model to a named slot
            parts_a = text.split()
            if len(parts_a) < 4:
                send_message(chat_id, "Usage: `/models assign <repo_id> <slot>` e.g. `/models assign Qwen/Qwen2.5-Omni-3B audio`")
                return
            repo_id = parts_a[2].strip()
            slot = parts_a[3].strip()
            try:
                import urllib.request as _ur_a
                payload = json.dumps({"repo_id": repo_id, "slot": slot}).encode()
                req = _ur_a.Request(f"http://localhost:{8779}/models/assign",
                                    data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with _ur_a.urlopen(req, timeout=5) as _r:
                    data = json.loads(_r.read())
                if data.get("error"):
                    send_message(chat_id, f"\u274c Assign failed: {data['error']}")
                    return
                send_message(chat_id, f"\u2705 Assigned `{repo_id}` \u2192 slot `{slot}`\nIt will lazy-load on first use.")
            except Exception as e:
                send_message(chat_id, f"\u274c Assign request failed: {e}")
            return

        else:
            send_message(chat_id, "Usage:\n"
                "`/models` — show menu\n"
                "`/models load e2b|e4b|nemotron3b` — hot-swap model (unloads current first)\n"
                "`/models mode ar|diffusion|linear_spec` — switch Nemotron generation mode\n"
                "`/models reload` — reload current\n"
                "`/models download nemotron3b` — download from HuggingFace\n"
                "`/models search <query>` — search HuggingFace Hub\n"
                "`/models pull <repo_id>` — pull any model from Hub (background)\n"
                "`/models assign <repo_id> <slot>` — assign pulled model to a model_slots entry")
            return
    # ─────────────────────────────────────────────────────────────────────────

    if text.startswith("/private_repo"):
        arg = text[13:].strip()
        import yaml
        cfg = yaml.safe_load(open(CONFIG_PATH))
        if not arg or arg == "show":
            current = cfg.get("ecosystem", {}).get("private", "(not set)")
            send_message(chat_id, f"🔒 Private ecosystem repo: `{current}`\n\nTo change: `/private_repo owner/repo`")
            return
        # Validate format
        if "/" not in arg or len(arg.split("/")) != 2:
            send_message(chat_id, "❌ Invalid format. Use: `/private_repo owner/repo`\nExample: `/private_repo myorg/my-kernel-skills`")
            return
        # Save to config
        if "ecosystem" not in cfg:
            cfg["ecosystem"] = {}
        cfg["ecosystem"]["private"] = arg
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        send_message(chat_id, f"✅ Private repo set to `{arg}`\nRe-bootstrapping ecosystem...")
        # Trigger re-bootstrap in background
        import threading
        def _rebootstrap():
            try:
                from bootstrap import bootstrap
                result = bootstrap(cfg)
                send_message(chat_id, f"✅ Bootstrap complete. Skills/routines updated.")
            except Exception as e:
                send_message(chat_id, f"⚠️ Bootstrap error: {str(e)[:200]}")
        threading.Thread(target=_rebootstrap, daemon=True).start()
        return

    if text.startswith("/clone"):
        parts = text[6:].strip().split(None, 1)
        if not parts:
            send_message(chat_id, "Usage: /clone <path> [name]\nExample: /clone ~/.openclaw/workspace/skills/mental-map")
            return
        src_path = parts[0]
        name = parts[1] if len(parts) > 1 else None
        from bootstrap import clone_from_agent
        result = clone_from_agent(src_path, name)
        send_message(chat_id, result["message"])
        return

    if text.startswith("/search"):
        query = text[7:].strip()
        if not query:
            send_message(chat_id, "Usage: /search <query>\nExample: /search tracker")
            return
        from bootstrap import search as eco_search
        results = eco_search(query)
        if not results:
            send_message(chat_id, f"🔍 No results for '{query}'.\nTry /search with a different term.")
            return
        lines = [f"🔍 *Results for '{query}':*\n"]
        for r in results[:10]:
            icon = "🔧" if r["type"] == "skill" else "⚙️"
            lines.append(f"{icon} *{r['name']}* ({r['source']})\n   {r['description'] or 'No description'}")
        buttons = [[{"text": f"📥 Install {r['name']}", "callback_data": f"install_{r['type']}_{r['name']}"}] for r in results[:5]]
        send_buttons(chat_id, "\n".join(lines), buttons)
        return

    if text.startswith("/install"):
        parts = text[8:].strip().split()
        if not parts:
            send_message(chat_id, "Usage: /install <name>\nExample: /install open-workspace-tracker")
            return
        name = parts[0]
        item_type = parts[1] if len(parts) > 1 else None
        from bootstrap import install as eco_install
        result = eco_install(name, item_type)
        send_message(chat_id, result["message"])
        return

    if text == "/verbose":
        global _verbose_mode
        _verbose_mode = not _verbose_mode
        if _verbose_mode:
            send_message(chat_id, "🔍 Verbose mode ON — I'll show my reasoning.")
        else:
            send_message(chat_id, "🔇 Verbose mode OFF.")
        return

    if text.startswith("/replica"):
        parts = text.split(None, 2)
        sub = parts[1] if len(parts) > 1 else "list"

        if sub == "clone":
            if len(parts) > 2:
                # /replica clone <agent_name> — spawn from agents dir
                agent_name = parts[2].strip().lower().replace('.md', '')
                agents_dir = Path.home() / '.openclaw' / 'workspace-client' / 'agents'
                brief_path = agents_dir / f"{agent_name}.md"
                if not brief_path.exists():
                    available = [f.stem for f in agents_dir.glob('*.md')] if agents_dir.exists() else []
                    send_message(chat_id, f"❌ Agent `{agent_name}` not found.\nAvailable: {', '.join(available) or 'none'}")
                else:
                    result = _call_api("POST", "/replica/named", {
                        "name": agent_name,
                        "role": "custom",
                        "brief_path": str(brief_path)
                    })
                    if result and result.get("status") == "spawned":
                        send_message(chat_id, f"✅ Replica `{agent_name}` spawned with brief.\nChat: `/replica msg {agent_name} <message>`")
                    else:
                        reason = result.get("reason", "unknown") if result else "unreachable"
                        send_message(chat_id, f"❌ Could not spawn: {reason}")
            else:
                # Show dynamic list of available agents as buttons
                agents_dir = Path.home() / '.openclaw' / 'workspace-client' / 'agents'
                agents = sorted([f.stem for f in agents_dir.glob('*.md')]) if agents_dir.exists() else []
                if not agents:
                    send_message(chat_id, "No agents found in workspace-client/agents/")
                else:
                    buttons = [[{"text": f"🤖 {a}", "callback_data": f"/replica clone {a}"}] for a in agents]
                    send_buttons(chat_id, "*Clone an agent replica:*\nSelect an agent to spawn with their brief loaded:", buttons)

        elif sub == "list":
            active = _call_api("GET", "/replica/active") or []
            if not active:
                send_message(chat_id, "No active replicas.")
            else:
                lines = ["*Active replicas:*"]
                for r in active:
                    status = "💬 persistent" if r.get("persistent") else ("✅ done" if r.get("done") else "⚙️ running")
                    lines.append(f"• `{r['name']}` ({r['role']}) — {status}")
                send_message(chat_id, "\n".join(lines))

        elif sub == "spawn":
            if len(parts) > 2:
                # /replica spawn <name> [prompt]
                spawn_parts = parts[2].split(None, 1)
                name = spawn_parts[0]
                prompt = spawn_parts[1] if len(spawn_parts) > 1 else None
                body = {"name": name}
                if prompt:
                    body["custom_prompt"] = prompt
                result = _call_api("POST", "/replica/named", body)
                if result and result.get("status") == "spawned":
                    send_message(chat_id, f"✅ Replica `{name}` spawned and ready.\nSend messages to it with:\n`/replica msg {name} <your message>`")
                else:
                    reason = result.get("reason", "unknown error") if result else "unreachable"
                    send_message(chat_id, f"❌ Could not spawn replica: {reason}")
            else:
                send_message(chat_id, "Usage:\n`/replica spawn <name>` — spawn with default prompt\n`/replica spawn <name> <custom prompt>` — spawn with custom role\n\nExample:\n`/replica spawn analyst You are a data analyst.`")

        elif sub == "msg" and len(parts) > 2:
            msg_parts = parts[2].split(None, 1)
            name = msg_parts[0]
            user_msg = msg_parts[1] if len(msg_parts) > 1 else ""
            if not user_msg:
                send_message(chat_id, "Usage: `/replica msg <name> <message>`")
            else:
                result = _call_api("POST", f"/replica/{name}/message", {"message": user_msg})
                if result and "reply" in result:
                    send_message(chat_id, f"🤖 *{name}:* {result['reply']}")
                else:
                    send_message(chat_id, f"❌ Replica `{name}` not found or error.")

        elif sub == "stop" and len(parts) > 2:
            name = parts[2].strip()
            result = _call_api("DELETE", f"/replica/{name}")
            if result and result.get("status") == "stopped":
                send_message(chat_id, f"✅ Replica `{name}` stopped.")
            else:
                send_message(chat_id, f"❌ Could not stop `{name}`.")

        else:
            send_message(chat_id, "Usage:\n`/replica list` — show active replicas\n`/replica clone` — spawn from available agents (dynamic list)\n`/replica clone <name>` — spawn a specific agent\n`/replica spawn <name> [prompt]` — spawn with custom prompt\n`/replica msg <name> <message>` — chat with a replica\n`/replica stop <name>` — stop a named replica")
        return

    if text.startswith("/run "):
        run_arg = text[5:].strip()
        # Split into name + optional input (e.g. "/run olly-recovery check status")
        parts = run_arg.split(" ", 1)
        run_name = parts[0]
        run_input = parts[1] if len(parts) > 1 else ""

        _ensure_agent()
        import yaml
        import os as _os

        with open(CONFIG_PATH) as _f:
            _cfg = yaml.safe_load(_f)
        from core.routines import load_all as load_routines, find as find_routine, run as run_routine
        from core.skills import load_all as load_skills, find as find_skill, run as run_skill
        from core.inference.model import infer, infer_with_tools, _model
        from core.tools import TOOLS

        # Try routine first
        routines_dir = str(os.path.expanduser(_os.environ.get("ROUTINES_DIR") or _cfg.get("routines_dir", str(Path(__file__).parent.parent / "routines"))))
        routines = load_routines(routines_dir)
        r = find_routine(run_name, routines)
        if r:
            working_id = send_message(chat_id, f"⚙️ *{r['name']}* — starting\u2026")
            log_lines = [f"⚙️ *{r['name']}*"]

            def _routine_step(n, tool_name, args, result):
                args_str = str(args)[:120]
                result_str = str(result)[:1200]
                log_lines.append(f"  *Step {n}* `{tool_name}`\n  ▸ `{args_str}`\n  ↳ {result_str}")
                snippet = "\n".join(log_lines[-6:])  # last 6 entries to stay within Telegram limit
                edit_message(chat_id, working_id, snippet)

            from evo_routine_executor import execute_routine as _execute_evo_routine
            workspace = _cfg.get("workspace", "~/.openclaw/workspace")
            with TypingKeepAlive(chat_id):
                result = _execute_evo_routine(
                    r,
                    workspace=workspace,
                    step_callback=_routine_step,
                )
            log_lines.append(f"\n✅ *Done*")
            final = "\n".join(log_lines[-8:])
            if not edit_message(chat_id, working_id, final):
                send_message(chat_id, final)
            send_message(chat_id, f"📋 *Result:*\n{result[:3800]}")
            return

        # Try skill
        skills_dir = str(os.path.expanduser(_os.environ.get("SKILLS_DIR") or _cfg.get("skills_dir", str(Path(__file__).parent.parent / "skills"))))
        skills = load_skills(skills_dir)
        s = find_skill(run_name, skills)
        if s:
            working_id = send_message(chat_id, f"🔧 *{run_name}* — starting\u2026")
            log_lines = [f"🔧 *{run_name}*"]

            def _skill_step(n, tool_name, args, result):
                args_str = str(args)[:120]
                result_str = str(result)[:1200]
                log_lines.append(f"  *Step {n}* `{tool_name}`\n  ▸ `{args_str}`\n  ↳ {result_str}")
                snippet = "\n".join(log_lines[-6:])
                edit_message(chat_id, working_id, snippet)

            user_input = run_input or f"Execute the {run_name} skill."
            with TypingKeepAlive(chat_id):
                result = run_skill(s, user_input, infer)
            log_lines.append(f"\n✅ *Done*")
            final = "\n".join(log_lines[-8:])
            if not edit_message(chat_id, working_id, final):
                send_message(chat_id, final)
            send_message(chat_id, f"📋 *Result:*\n{result[:3800]}")
            return

        send_message(chat_id, f"❌ No routine or skill named `{run_name}` found.\nTry /routines or /skills to see available options.")
        return

    # Regular chat — route to agent
    # ── Evolution commands (ADR-004) ─────────────────────────────────────────────

    # ── Provider hot-swap commands (ADR-013) ─────────────────────────────────────
    if text.startswith("/providers"):
        # Backward-compatible alias: normalize to singular command path.
        text = "/provider" + text[len("/providers"):]

    if text.startswith("/provider"):
        import yaml as _yaml, json as _json_p
        with open(CONFIG_PATH) as _pf:
            _cfg_p = _yaml.safe_load(_pf)
        _api_port = _cfg_p.get('api', {}).get('port', 8779)
        parts = text.split()
        sub = parts[1].strip().lower() if len(parts) > 1 else ""

        if sub == "" or sub == "menu":
            # Show routing table + inline swap buttons
            try:
                import urllib.request as _ur
                with _ur.urlopen(f"http://localhost:{8779}/provider", timeout=3) as r:
                    data = json.loads(r.read())
                routing = data.get("routing", {})
                collect = data.get("collect_trajectories", False)
                lines = ["\U0001f500 *Provider Router*\n"]
                for ct, info in routing.items():
                    icon = {"local": "\U0001f3e0", "openai": "\U0001f916", "anthropic": "\U0001f9e0",
                            "hf": "\U0001f917", "copilot": "\u26a1"}.get(info["provider"], "\U0001f4e1")
                    stream = " \U0001f4e1" if info.get("streaming") else ""
                    model = f" `{info['model']}`" if info.get("model") else ""
                    lines.append(f"{icon} `{ct}` \u2192 *{info['provider']}*{model}{stream}")
                collect_icon = "\U0001f3af ON" if collect else "\u23f8 OFF"
                lines.append(f"\nTrajectory collection: {collect_icon}")
                msg = "\n".join(lines)
                buttons = [
                    # Quick set all
                    [{"text": "\U0001f310 All \u2192 OpenRouter", "callback_data": "/provider set all openrouter persist"},
                     {"text": "\U0001f916 All \u2192 OpenAI",    "callback_data": "/provider set all openai persist"},
                     {"text": "\U0001f3e0 All \u2192 Local",     "callback_data": "/provider set all local persist"}],
                    # Swap per call type row 1
                    [{"text": "\U0001f504 task \u2192 swap",      "callback_data": "/provider swap task_inference"},
                     {"text": "\U0001f9ec synth \u2192 swap",     "callback_data": "/provider swap synthesis"},
                     {"text": "\U0001f4cb critic \u2192 swap",    "callback_data": "/provider swap critic"}],
                    # Swap per call type row 2
                    [{"text": "\U0001f5fa plan \u2192 swap",      "callback_data": "/provider swap planning"},
                     {"text": "\U0001f393 teacher \u2192 swap",   "callback_data": "/provider swap trajectory_teacher"}],
                    # Misc
                    [{"text": "\u21ba Reset defaults",           "callback_data": "/provider reset"},
                     {"text": "\U0001f4ca Availability",         "callback_data": "/provider status"}],
                    [{"text": f"\U0001f3af Collect: {'ON \u2192 off' if collect else 'OFF \u2192 on'}",
                      "callback_data": f"/provider collect {'false' if collect else 'true'}"}],
                    [{"text": "\U0001f9e0 Models / Load Nemotron", "callback_data": "/models"}],
                ]
                send_buttons(chat_id, msg, buttons)
            except Exception as e:
                send_message(chat_id, f"\u274c Provider status error: {e}")

        elif sub == "status":
            try:
                import urllib.request as _ur
                with _ur.urlopen(f"http://localhost:{8779}/provider/available", timeout=3) as r:
                    avail = json.loads(r.read())
                lines = ["\U0001f4ca *Provider Availability*\n"]
                for prov, info in avail.items():
                    icon = "\u2705" if info["ready"] else "\u274c"
                    lines.append(f"{icon} `{prov}` \u2014 {info['reason']}")
                send_message(chat_id, "\n".join(lines))
            except Exception as e:
                send_message(chat_id, f"\u274c {e}")

        elif sub == "reset":
            try:
                import urllib.request as _ur
                for ct in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
                    payload = json.dumps({ct: _cfg.get("providers", {}).get(ct, "local"), "persist": True}).encode()
                    req = _ur.Request(f"http://localhost:{8779}/provider/set",
                                      data=payload, headers={"Content-Type": "application/json"}, method="POST")
                    _ur.urlopen(req, timeout=3)
                send_message(chat_id, "\u21ba Provider routing reset to config.yaml defaults")
            except Exception as e:
                send_message(chat_id, f"\u274c {e}")

        elif sub == "set" and len(parts) >= 4:
            # /provider set <calltype> <provider>  OR  /provider set all <provider>
            # Optionally with a trailing token from the model pickers:
            #   "persist|assign|<token>" — local: also assign the chosen model to its slot.
            #   "persist|model|<model>"  — cloud: also set a model_override for the call type.
            call_type_or_all = parts[2].lower()
            provider_name    = parts[3].lower()
            persist_change = any(p.lower() in ("persist", "--persist") for p in parts[4:])
            assign_repo = ""
            model_override = ""
            for p in parts[4:]:
                if "|assign|" in p:
                    _tok = p.split("|assign|", 1)[1].strip()
                    assign_repo = _resolve_repo_token(_tok)  # token → repo_id
                elif "|model|" in p:
                    model_override = p.split("|model|", 1)[1].strip()
            try:
                import urllib.request as _ur
                valid_providers = {"local", "openai", "anthropic", "hf", "copilot", "openrouter"}
                if provider_name not in valid_providers:
                    send_message(chat_id, f"\u274c Unknown provider `{provider_name}`. Valid: {', '.join(sorted(valid_providers))}")
                    return
                if call_type_or_all == "all":
                    body = {ct: provider_name for ct in ("task_inference","synthesis","critic","planning","trajectory_teacher")}
                else:
                    body = {call_type_or_all: provider_name}
                # Apply a cloud model override if provided (e.g. from /models provider <cloud>).
                if model_override:
                    body["model_override"] = {call_type_or_all: model_override}
                body["persist"] = persist_change
                payload = json.dumps(body).encode()
                req = _ur.Request(f"http://localhost:{8779}/provider/set",
                                  data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with _ur.urlopen(req, timeout=3) as r:
                    result = json.loads(r.read())
                persisted_label = " (persisted)" if result.get("persisted") else " (runtime only)"
                reply = f"\u2705 Provider updated{persisted_label}: `{result['changed']}`"
                if model_override:
                    reply += f"\n\U0001f4e6 Model override: `{model_override}`"

                # XP7: if a local model was chosen from the picker, assign it to its slot.
                if assign_repo and provider_name == "local":
                    slot = _curated_slot_for_repo(assign_repo)
                    if slot:
                        try:
                            apayload = json.dumps({"repo_id": assign_repo, "slot": slot}).encode()
                            areq = _ur.Request(f"http://localhost:{8779}/models/assign",
                                               data=apayload, headers={"Content-Type": "application/json"}, method="POST")
                            with _ur.urlopen(areq, timeout=5) as ar:
                                adata = json.loads(ar.read())
                            if adata.get("error"):
                                reply += f"\n\u26a0\ufe0f Assign to slot `{slot}` failed: {adata['error']}"
                            else:
                                reply += f"\n\u2705 Assigned `{assign_repo}` \u2192 slot `{slot}`"
                        except Exception as _ae:
                            reply += f"\n\u26a0\ufe0f Assign request failed: {_ae}"
                    else:
                        reply += f"\n\u26a0\ufe0f No slot mapping for `{assign_repo}` \u2014 use `/models assign {assign_repo} <slot>`"
                send_message(chat_id, reply)
            except Exception as e:
                send_message(chat_id, f"\u274c {e}")

        elif sub == "swap" and len(parts) >= 3:
            # Inline button: show per-provider choice for one call type
            call_type = parts[2]
            buttons = [
                [
                    {"text": "\U0001f3e0 local",       "callback_data": f"/models provider local"},
                    {"text": "\U0001f916 openai",      "callback_data": f"/models provider openai"},
                    {"text": "\U0001f9e0 anthropic",   "callback_data": f"/models provider anthropic"},
                ],[
                    {"text": "\U0001f917 hf",          "callback_data": f"/models provider hf"},
                    {"text": "\u26a1 copilot",         "callback_data": f"/models provider copilot"},
                    {"text": "\U0001f310 openrouter",  "callback_data": f"/models provider openrouter"},
                ],[
                    {"text": "\u2190 Back",            "callback_data": "/provider"},
                ]
            ]
            send_buttons(chat_id, f"Pick a provider for `{call_type}` \u2014 then choose its model:", buttons)

        elif sub == "local" and len(parts) >= 3:
            # XP7: local model picker — curated models shown as inline buttons,
            # with a VRAM-fit indicator, plus Hub search for RAM-compatible models.
            call_type = parts[2]
            try:
                import urllib.request as _ur_l, os as _os_l
                # Curated local models from the new /models/curated endpoint.
                curated = []
                try:
                    with _ur_l.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
                        curated = (json.loads(_rc.read()) or {}).get("curated", [])
                except Exception:
                    pass
                # Available VRAM (MB) from /health.
                vram_free = 0
                try:
                    with _ur_l.urlopen(f"http://localhost:{8779}/health", timeout=5) as _rh:
                        vram_free = int((json.loads(_rh.read()) or {}).get("vram_free_mb", 0) or 0)
                except Exception:
                    pass

                lines = [f"\U0001f3e0 *Local Model Picker* \u2014 `{call_type}`",
                         f"VRAM free: `{vram_free} MB`\n"]
                buttons = []
                for cm in curated:
                    repo = cm.get("repo_id", "")
                    label = cm.get("label") or repo
                    dl = cm.get("downloaded", False)
                    # Rough VRAM fit estimate: assume ~2 bytes/param + overhead;
                    # flag models likely too big for current free VRAM.
                    est_gb = _estimate_model_gb(repo)
                    fit = _vram_fit_mark(est_gb, vram_free)
                    icon = "\u2705" if dl else "\u23ec"
                    lines.append(f"{icon} {label} `{repo}` {fit}")
                    # Use a short token to stay under Telegram's 64-byte callback limit.
                    tok = _register_repo_token(repo)
                    buttons.append([{
                        "text": f"{icon} {label}",
                        "callback_data": f"/provider set {call_type} local persist|assign|{tok}",
                    }])
                buttons.append([
                    {"text": "\U0001f50d Search Hub for more", "callback_data": f"/models search "},
                    {"text": "\u2190 Back", "callback_data": "/provider"},
                ])
                send_buttons(chat_id, "\n".join(lines), buttons)
            except Exception as e:
                send_message(chat_id, f"\u274c Local picker error: {e}")
            return

        elif sub == "collect" and len(parts) >= 3:
            val = parts[2].lower() == "true"
            persist_change = any(p.lower() in ("persist", "--persist") for p in parts[3:])
            try:
                import urllib.request as _ur
                payload = json.dumps({"collect_trajectories": val, "persist": persist_change}).encode()
                req = _ur.Request(f"http://localhost:{8779}/provider/set",
                                  data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with _ur.urlopen(req, timeout=3) as r:
                    result = json.loads(r.read())
                persisted_label = " (persisted)" if result.get("persisted") else " (runtime only)"
                send_message(chat_id, f"\U0001f3af Trajectory collection{persisted_label}: {'\U0001f7e2 ON' if val else '\u26ab OFF'}")
            except Exception as e:
                send_message(chat_id, f"\u274c {e}")
        else:
            send_message(chat_id,
                "*Provider commands:*\n"
                "`/provider` or `/providers` \u2014 show routing + swap menu\n"
                "`/provider set <calltype> <provider>` \u2014 e.g. `/provider set task_inference openai`\n"
                "`/provider set <calltype> <provider> persist` \u2014 save to config.yaml\n"
                "`/provider set all openai` \u2014 flip everything\n"
                "`/provider status` \u2014 check which keys are set\n"
                "`/provider reset` \u2014 revert to config.yaml defaults\n"
                "`/provider collect true|false [persist]` \u2014 toggle trajectory collection"
            )
        return

    if text.startswith("/evolve"):
        parts = text.split(maxsplit=1)
        sub = parts[1].strip() if len(parts) > 1 else ""

        # /evolve — show evolution menu with state overview
        if sub == "" or sub == "menu":
            state = _call_api("GET", "/evolution/state") or {}
            status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(state.get("state", ""), "❓")
            msg = (
                f"🧬 *Kernel Evolution* {status_icon}\n"
                f"State: `{state.get('state', 'unknown')}`  "
                f"Iterations: `{state.get('iterations', 0)}/{state.get('cap', '?')}`\n"
                "Use the buttons below to control the evolution sandbox."
            )
            buttons = [
                [{"text": "📊 Status", "callback_data": "/evolve status"}, {"text": "📈 Dashboard", "callback_data": "/evolve dash"}],
                [{"text": "▶ Start", "callback_data": "/evolve control start"}, {"text": "⏸ Pause", "callback_data": "/evolve control pause"}],
                [{"text": "▶▶ Resume", "callback_data": "/evolve control resume"}, {"text": "⏹ Stop", "callback_data": "/evolve control stop"}],
                [{"text": "🔄 Reset", "callback_data": "/evolve control reset"}],
                [{"text": "🎯 Send Task", "callback_data": "/evolve task "}],
            ]
            send_buttons(chat_id, msg, buttons)
            return

        # /evolve status — full evolution state + history summary
        if sub == "status":
            state = _call_api("GET", "/evolution/state") or {}
            history = _call_api("GET", "/evolution") or {}
            status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(state.get("state", ""), "❓")
            gaps = history.get("gaps", [])
            installed = history.get("installed_skills", [])
            recent = history.get("cycles", [])[-5:] if history.get("cycles") else []
            lines = [
                f"🧬 *Evolution Status* {status_icon}",
                f"State: `{state.get('state', 'unknown')}` · Iter: `{state.get('iterations', 0)}/{state.get('cap', '?')}` · Remaining: `{state.get('remaining', '?')}`",
                f"Skills installed: `{len(installed)}`",
                f"Open gaps: `{len(gaps)}`",
            ]
            if installed:
                lines.append("\n📦 *Last installed:* " + ", ".join(f"`{s}`" for s in installed[-5:]))
            if gaps:
                gap_texts = []
                for g in gaps[-3:]:
                    if isinstance(g, dict):
                        gap_texts.append(g.get('gap', str(g)))
                    else:
                        gap_texts.append(str(g))
                lines.append("\n❓ *Recent gaps:* " + "; ".join(gap_texts))
            if recent:
                lines.append("\n🔁 *Recent cycles:*")
                for c in recent:
                    lines.append(f"  • {c.get('task', '?')} → {c.get('outcome', '?')}")
            send_message(chat_id, "\n".join(lines))
            return

        # /evolve dash — generate + send dashboard HTML as file
        if sub == "dash":
            working_id = send_message(chat_id, "📊 Generating evolution dashboard…")
            dash_url = "http://localhost:8779/evolution/dashboard"
            try:
                import urllib.request
                import tempfile
                resp = urllib.request.urlopen(dash_url, timeout=10)
                html_bytes = resp.read()
                tmp_path = os.path.join(tempfile.gettempdir(), "kernel_evolution_dashboard.html")
                with open(tmp_path, "wb") as _f:
                    _f.write(html_bytes)
                if working_id:
                    edit_message(chat_id, working_id, "📊 Dashboard ready — sending file…")
                send_file(chat_id, tmp_path, caption="🧬 Kernel Evolution Dashboard")
                os.unlink(tmp_path)
            except Exception as e:
                err_msg = f"❌ Dashboard error: {str(e)[:200]}"
                if working_id:
                    edit_message(chat_id, working_id, err_msg)
                else:
                    send_message(chat_id, err_msg)
            return

        # /evolve control <action> [cap=N]
        if sub.startswith("control"):
            ctrl_parts = sub.split()
            action = ctrl_parts[1] if len(ctrl_parts) > 1 else ""
            cap_val = None
            for p in ctrl_parts[2:]:
                if p.startswith("cap="):
                    try:
                        cap_val = int(p.split("=", 1)[1])
                    except ValueError:
                        pass
            valid_actions = {"start", "pause", "resume", "stop", "reset"}
            if action not in valid_actions:
                send_message(chat_id, f"Usage: `/evolve control <action> [cap=N]`\nActions: start, pause, resume, stop, reset")
                return
            body = {"action": action}
            if cap_val is not None:
                body["cap"] = cap_val
            result = _call_api("POST", "/evolution/control", body)
            if result:
                status_icon = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(result.get("state", ""), "❓")
                send_message(
                    chat_id,
                    f"✅ Evolution `{action}` applied\n"
                    f"State: {status_icon} `{result.get('state')}` · "
                    f"Iter: `{result.get('iterations', 0)}/{result.get('cap', '?')}`"
                )
            else:
                send_message(chat_id, f"❌ Evolution control failed for action `{action}`")
            return

        # /evolve task <description> — manually trigger an evolution cycle
        if sub.startswith("task ") or sub.startswith("task	"):
            task_desc = sub[5:].strip()
            if not task_desc:
                send_message(chat_id, "Usage: `/evolve task <description>`\nExample: `/evolve task analyse sentiment of customer feedback`")
                return
            working_id = send_message(chat_id, f"🧬 Triggering evolution for: _{task_desc}_…")
            result = _call_api("POST", "/evolution/trigger", {"task": task_desc})
            if result:
                outcome = result.get("outcome", "unknown")
                skill = result.get("skill_installed")
                confidence = result.get("confidence")
                lines = [f"🧬 *Evolution cycle complete*", f"Task: `{task_desc}`", f"Outcome: `{outcome}`"]
                if skill:
                    lines.append(f"📦 Skill installed: `{skill}`")
                if confidence is not None:
                    lines.append(f"🎯 Confidence: `{confidence}`")
                if working_id:
                    edit_message(chat_id, working_id, "🧬 Evolution cycle done")
                send_message(chat_id, "\n".join(lines))
            else:
                err = "❌ Evolution trigger failed — is the sandbox running and EVOLUTION_ENABLED=true?"
                if working_id:
                    edit_message(chat_id, working_id, err)
                else:
                    send_message(chat_id, err)
            return

        # /evolve backup [description] — create a timestamped backup
        if sub.startswith("backup"):
            desc = sub[6:].strip() if len(sub) > 6 else ""
            working_id = send_message(chat_id, "💾 Creating backup…")
            body = {"description": desc, "full": True}
            result = _call_api("POST", "/evolve/backup", body)
            if result and result.get("status") == "created":
                backup = result.get("backup", {})
                size = backup.get("size_mb", 0)
                sha = backup.get("sha256", "")[:8]
                msg = (
                    f"✅ *Backup created*\n"
                    f"Path: `{backup.get('backup_path', '')}`\n"
                    f"Size: {size} MB • SHA256: `{sha}`\n"
                    f"Timestamp: {backup.get('timestamp', '')}"
                )
                if working_id:
                    edit_message(chat_id, working_id, msg)
                else:
                    send_message(chat_id, msg)
            else:
                err = result.get("error", "Backup failed")
                if working_id:
                    edit_message(chat_id, working_id, f"❌ Backup failed: {err}")
                else:
                    send_message(chat_id, f"❌ Backup failed: {err}")
            return

        # /evolve init [--force] — initialize workspace and databases
        if sub.startswith("init"):
            force = "--force" in sub
            desc = sub[4:].strip() if len(sub) > 4 else ""
            desc = desc.replace("--force", "").strip()
            working_id = send_message(chat_id, "⚙️ Initializing kernel-evolving…")
            body = {"force": force, "description": desc}
            result = _call_api("POST", "/evolve/init", body)
            if result and result.get("status") == "initialized":
                msg = f"✅ Initialization complete\n{json.dumps(result, indent=2)}"
            else:
                msg = f"⚠️ Init not yet implemented\n{json.dumps(result, indent=2)}"
            if working_id:
                edit_message(chat_id, working_id, msg)
            else:
                send_message(chat_id, msg)
            return

        # /evolve fresh [description] — reset to fresh state (requires confirmation)
        if sub.startswith("fresh"):
            desc = sub[5:].strip() if len(sub) > 5 else ""
            # Step 1: create backup and show confirmation button
            working_id = send_message(chat_id, "💾 Creating backup before fresh reset…")
            body = {"description": "Pre-fresh backup: " + desc, "full": True}
            result = _call_api("POST", "/evolve/backup", body)
            if not result or result.get("status") != "created":
                err = result.get("error", "Backup failed")
                if working_id:
                    edit_message(chat_id, working_id, f"❌ Backup failed: {err}")
                else:
                    send_message(chat_id, f"❌ Backup failed: {err}")
                return
            backup = result.get("backup", {})
            backup_path = backup.get("backup_path", "")
            # Dry run to list actual targets
            dry_result = _call_api("POST", "/evolve/fresh", {"dry_run": True, "description": "Dry run before confirmation"})
            dry_summary = ""
            if dry_result and dry_result.get("status") == "dry_run":
                summary = dry_result.get("summary", {})
                db_count = summary.get("databases", 0)
                file_count = summary.get("files", 0)
                dir_count = summary.get("directories_to_clear", 0)
                dry_summary = (
                    f"\n**Dry‑run summary:**\n"
                    f"• Databases: {db_count} (`memory/chat_history_evolving.db`, `data/evolution.db`, `data/promoted_signals.db`)\n"
                    f"• Files: {file_count} (`user.json`, `todos.md`, etc.)\n"
                    f"• Directories to clear: {dir_count} (`thoughts/`, `notes/`, `memory/`, `data/`, `artifacts/`, `logs/`, `tmp/`)\n"
                )
            else:
                dry_summary = "\n⚠️ Dry run failed — proceeding with generic list.\n"
            # Show confirmation with inline button
            msg = (
                f"🔄 *Fresh Reset Confirmation*\n"
                f"A backup has been created at:\n`{backup_path}`\n"
                f"\n**This will:**\n"
                "• Delete workspace databases from `memory/` and `data/`\n"
                "• Keep ecosystem skills/routines (no change)\n"
                "• Reinitialize workspace with empty databases\n"
                f"{dry_summary}"
                "• Restart kernel-evolving API (recommended)\n"
                f"\nDescription: {desc}"
            )
            buttons = [[
                {"text": "✅ Confirm Fresh Reset", "callback_data": f"/evolve fresh confirm {backup_path}"},
                {"text": "❌ Cancel", "callback_data": "/evolve fresh cancel"}
            ]]
            if working_id:
                edit_message(chat_id, working_id, msg)
                send_buttons(chat_id, "Please confirm:", buttons)
            else:
                send_buttons(chat_id, msg, buttons)
            return

        # /evolve fresh confirm <backup_path> — execute fresh reset after confirmation
        if sub.startswith("fresh confirm"):
            parts = sub.split(maxsplit=2)
            if len(parts) < 3:
                send_message(chat_id, "❌ Missing backup path.")
                return
            backup_path = parts[2]
            working_id = send_message(chat_id, f"🔄 Performing fresh reset using backup {backup_path}…")
            # Call fresh API endpoint (dry_run=False)
            result = _call_api("POST", "/evolve/fresh", {"dry_run": False, "description": "Fresh reset after confirmation"})
            if result and result.get("status") in ["fresh_executed", "dry_run"]:
                deleted = result.get("summary", {}).get("deleted_count", 0)
                cleared = result.get("summary", {}).get("cleared_count", 0)
                msg = (
                    f"✅ *Fresh reset completed*\n"
                    f"Backup: `{backup_path}`\n"
                    f"Deleted {deleted} files, cleared {cleared} directories.\n"
                    f"Databases reinitialized.\n"
                )
                errors = result.get("errors", [])
                if errors:
                    msg += f"\n⚠️ Errors: {len(errors)}"
                # Suggest restart of API
                msg += "\n⚠️ *Restart kernel-evolving API* with `bash start.sh` for clean state."
                if working_id:
                    edit_message(chat_id, working_id, msg)
                else:
                    send_message(chat_id, msg)
            else:
                err = result.get("error", "Fresh failed")
                if working_id:
                    edit_message(chat_id, working_id, f"❌ Fresh failed: {err}")
                else:
                    send_message(chat_id, f"❌ Fresh failed: {err}")
            return

        # /evolve fresh cancel — cancel fresh reset
        if sub.startswith("fresh cancel"):
            send_message(chat_id, "❌ Fresh reset cancelled.")
            return

        # Unknown /evolve subcommand
        send_message(
            chat_id,
            "🧬 *Evolution commands:*\n"
            "`/evolve` — menu + current state\n"
            "`/evolve status` — full status + history\n"
            "`/evolve dash` — send live dashboard HTML\n"
            "`/evolve control start|pause|resume|stop|reset [cap=N]` — state machine control\n"
            "`/evolve task <description>` — manually trigger an evolution cycle\n"
            "`/evolve backup [description]` — create timestamped backup\n"
            "`/evolve init [--force]` — initialize workspace and databases\n"
            "`/evolve fresh [description]` — reset to fresh state (with confirmation)"
        )
        return

    if text == "/thoughts":
        try:
            import requests as _req
            r = _req.get("http://localhost:8779/thoughts", timeout=5)
            thoughts = r.json() if r.ok else []
        except Exception:
            thoughts = []
        if not thoughts:
            send_message(chat_id, "🤔 No thoughts recorded today yet.")
        else:
            recent = thoughts[-10:]
            lines = ["🤔 *Kernel's thoughts today:*\n"]
            for t in recent:
                score = t.get("score", 0.0)
                category = t.get("category", "thought")
                thought_text = t.get("thought", "")
                time_str = t.get("time", "")
                lines.append(f"*{time_str}* [{category}] _{_esc(thought_text)}_ (score: {score:.2f})")
            send_message(chat_id, "\n".join(lines))
        return

# For unrecognised slash commands, try agent.triage() first
    # so skills with exec dispatch (e.g. /markdown, /anonymize) run deterministically.
    # Known built-in commands are already handled above — only forward unknowns.
    _BUILTIN_COMMANDS = {
        "/start", "/help", "/skills", "/routines", "/run", "/skill",
        "/status", "/verbose", "/packages", "/search", "/install",
        "/clone", "/private_repo", "/update", "/restart", "/stop", "/rollback",
        "/replica", "/workspaces", "/system", "/version", "/models",
        "/evolve", "/thoughts", "/new", "/voices", "/provider", "/init",
    }
    if text.startswith("/") or text.startswith("set_voice_"):
        cmd_word = text.split()[0].lower()
        if cmd_word not in _BUILTIN_COMMANDS and not text.startswith("set_voice_"):
            _ensure_agent()
            # Immediate ack + keep typing indicator alive during long exec (e.g. /markdown)
            working_id = send_message(chat_id, f"🐬 Running `{cmd_word}`\u2026")
            import core.agent as _a
            with TypingKeepAlive(chat_id):
                result = _a.triage(text, chat_id=str(chat_id))
            if result:
                reply_text = f"🐬 {result[:3800]}"
                if working_id:
                    edit_message(chat_id, working_id, f"🐬 `{cmd_word}` complete ✅")
                send_message(chat_id, reply_text, parse_mode="")
            return

    if not _agent_ready:
        send_message(chat_id, "⏳ Loading model (~60s)...")

    _ensure_agent()
    from core.inference.model import infer
    import core.memory.memory as _memory_mod

    # Build rich system prompt with live context
    import core.agent as _agent_mod_ctx
    import yaml as _yaml_ctx
    import os as _os_ctx
    from core.memory.context import build_system_prompt as _build_prompt
    from core.skills import load_all as _load_skills
    from core.routines import load_all as _load_routines
    from core.inference.model import vram_free_mb as _vram_free
    _cfg_path = CONFIG_PATH
    _cfg = _yaml_ctx.safe_load(open(_cfg_path))
    _skills = _load_skills(_os_ctx.path.expanduser(_cfg.get("skills_dir", "./skills")))
    _routines = _load_routines(_os_ctx.path.expanduser(_cfg.get("routines_dir", "./routines")))
    system_prompt = _build_prompt(
        _cfg, _skills, _routines,
        vram_free_fn=_vram_free,
        channel="telegram",
        sender_name=sender_name,
        agent_ready=_agent_ready,
    )

    # Prepend relevant collective memory results if found.
    # NOTE: triage() also injects collective memory via agent.py's PD1 path
    # (for all callers, not just Telegram). The Telegram path builds its own
    # system_prompt here and then passes the user message into triage() — both
    # prompts coexist. The legacy subprocess-based search below is now skipped
    # since the HTTP client in core.collective_memory_client handles this more
    # reliably for all callers. Kept as a comment to explain the removed call.

    # No placeholder — streaming commences immediately on first chunk/step
    working_id = [None]  # mutable so closures can assign

    try:
        with TypingKeepAlive(chat_id):
            import core.agent as _agent_mod
            log_lines = []
            step_num = [0]
            # Streaming state — accumulate chunks and edit message live
            _stream_buf = [""]
            _stream_char_count = [0]
            _STREAM_EDIT_EVERY = 40  # edit message every N chars to avoid flood

            def _chunk_cb(chunk: str):
                """Called with each streamed text chunk from the model."""
                _stream_buf[0] += chunk
                _stream_char_count[0] += len(chunk)
                if _stream_char_count[0] >= _STREAM_EDIT_EVERY:
                    _stream_char_count[0] = 0
                    preview = _stream_buf[0]
                    snippet = ("🔍 " + "\n\n".join(log_lines[-3:]) + f"\n\n✍️ {_html(preview)}"
                               if log_lines else f"✍️ {_html(preview)}")
                    if working_id[0]:
                        edit_message(chat_id, working_id[0], snippet[:4000], parse_mode="HTML")
                    else:
                        # First output — create the message now, no prior placeholder
                        working_id[0] = send_message(chat_id, snippet[:4000], parse_mode="HTML")

            def _step_cb(n, tool_name, args=None, result=None):
                step_num[0] = n
                # Clear stream buffer — step output takes over display
                _stream_buf[0] = ""
                _stream_char_count[0] = 0
                # Show an informative, emoji-tagged step line (what tool, to do what).
                log_lines.append(_format_tool_step(n, tool_name, args, result))
                # Keep the last few steps so the message stays readable.
                snippet = "🔍 " + "\n\n".join(log_lines[-4:])
                if working_id[0]:
                    edit_message(chat_id, working_id[0], snippet[:4000], parse_mode="HTML")
                else:
                    working_id[0] = send_message(chat_id, snippet[:4000], parse_mode="HTML")

            # Inject recent attachment context only when message references a file
            _triage_text = text
            _att_ctx = None
            _text_lower = text.lower()
            _ATT_REF_KEYWORDS = (
                "pdf", "document", "attachment", "uploaded", "sent you",
                "the file", "this file", "that file", "the doc", "the pdf",
                "image", "photo", "picture", "voice", "audio", "recording",
                "the one i sent", "what i sent", "that i sent",
            )
            if any(kw in _text_lower for kw in _ATT_REF_KEYWORDS):
                _att_ctx = _memory_mod.attachment_context_block(chat_id=str(chat_id), limit=3, max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
            if _att_ctx:
                _triage_text = f"{_att_ctx}\n\n{text}"

            reply = _agent_mod.triage(_triage_text, step_callback=_step_cb, chat_id=str(chat_id), chunk_callback=_chunk_cb)

            # --- Completion gates ---
            # 1. Attachment guard: if user referenced a file, did reply use it?
            if _att_ctx:
                _guard = _memory_mod.attachment_guard(text, reply, chat_id=str(chat_id), max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                if not _guard["ok"]:
                    print(f"[bot] chat guard FAIL (retrying): {_guard['reason']}", flush=True)
                    _retry_text = (
                        f"{_att_ctx}\n\n"
                        f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                        f"{text}"
                    )
                    reply = _agent_mod.triage(_retry_text, step_callback=_step_cb, chat_id=str(chat_id), chunk_callback=_chunk_cb)
                    _guard2 = _memory_mod.attachment_guard(text, reply, chat_id=str(chat_id), max_age_seconds=_ATTACHMENT_RECENCY_SECONDS)
                    if not _guard2["ok"]:
                        reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

            # 2. Artifact check: if user asked to create/save something, was a file produced?
            _art = _memory_mod.artifact_check(text, reply)
            if not _art["ok"]:
                print(f"[bot] artifact_check FAIL: {_art['reason']}", flush=True)
                reply = reply + "\n\n⚠️ (Note: I was asked to produce a file but couldn't verify it was saved — please check your workspace or re-ask.)"

            # 3. ADR-020: Failure detection — log unresolved requests as evolution anchors
            try:
                import database.agent.failed_requests as _fr
                _failure_type = _fr.detect_failure(reply, text)
                if _failure_type:
                    _fr.record(str(chat_id), text, reply, _failure_type)
                    print(f"[bot] ADR-020 failure logged: type={_failure_type}", flush=True)
            except Exception as _fre:
                print(f"[bot] failed_requests record error: {_fre}", flush=True)

            # Memory is already persisted inside agent.triage().
            # Do NOT save here again, otherwise each plain-chat turn is duplicated
            # and history inflates/noises future responses.
            if step_num[0] == 0 and len(reply) > 200:
                _write_collective_memory(text, reply)

        # Final reply — never send an empty body ("🐬 " only) if model returns blank.
        _final_reply = (reply or "").strip()
        if not _final_reply:
            _streamed = (_stream_buf[0] or "").strip()
            if _streamed:
                _final_reply = _streamed
            elif log_lines:
                _last_steps = "\n\n".join(log_lines[-2:])
                _final_reply = (
                    "I couldn't produce a final answer.\n\n"
                    "Last tool activity:\n"
                    f"{_last_steps}"
                )
            else:
                _final_reply = (
                    "I couldn't produce a final answer and no tool step was emitted. "
                    "Check model/tool logs for this request and retry."
                )

        # Build final message: preserve all step logs then append final answer.
        # Escape dynamic content for Telegram HTML parse mode — legacy Markdown
        # rejects unescaped `_`/`*`/backticks with a 400 "can't parse entities",
        # which silently drops the final answer. HTML only needs < > & escaped.
        if log_lines:
            _steps_section = "🔍 " + "\n\n".join(_html(l) for l in log_lines)
            reply_text = f"{_steps_section}\n\n🐬 {_html(_final_reply)}"
        else:
            reply_text = f"🐬 {_html(_final_reply)}"
        # Telegram hard limit is 4096 chars; truncate from beginning to keep final answer
        if len(reply_text) > 4000:
            reply_text = "…" + reply_text[-3998:]
        if working_id[0] and not edit_message(chat_id, working_id[0], reply_text, parse_mode="HTML"):
            send_message(chat_id, reply_text, parse_mode="HTML")
        elif not working_id[0]:
            send_message(chat_id, reply_text, parse_mode="HTML")
    except Exception as e:
        print(f"[bot] ERROR in infer: {e}", flush=True)
        err_text = f"🐬 Error: {str(e)[:200]}"
        if working_id[0] and not edit_message(chat_id, working_id[0], err_text):
            send_message(chat_id, err_text)

# Self-update helpers — delegated to src/updater.py
# ---------------------------------------------------------------------------

from infra.updater import (
    get_current_version as _get_current_version,
    fetch_latest_version as _fetch_latest_version,
    check_update_available as _check_update_available,
    do_update as _do_update,
)


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
    global _latest_version
    current = _get_current_version()
    latest = _fetch_latest_version()
    if not latest:
        return
    _latest_version = latest
    if latest != current and notify_chat_id:
        send_message(
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
    """Emergency stop: kills model server (freeing GPU) but keeps the bot alive
    so you can /restart or send further commands."""
    send_message(chat_id, "🛑 Model stopped — GPU freed. I'm still here, say /restart to bring it back.")
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
    send_message(chat_id, "🔄 Restarting Kernel... back in ~30s.")
    try:
        _restart_via_start_sh()
    except Exception as e:
        send_message(chat_id, f"❌ Restart failed: {str(e)[:200]}")


def _handle_update(chat_id: str):
    """Execute git pull + restart via shared updater. Called when user types /update."""
    _do_update(
        notify=lambda msg: send_message(chat_id, msg),
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
        send_message(chat_id, "⏪ Rolled back to previous version. Restarting...")
        _restart_via_start_sh()
    except Exception as e:
        send_message(chat_id, f"❌ Rollback failed: {str(e)[:200]}")


def _ensure_agent():
    global _agent_ready
    if not _agent_ready:
        import core.inference.model as _m

        # Check if model server is running OR model is loaded in-process
        from core.inference.model_client import is_server_running as _is_srv
        if _m._model is None and not _is_srv():
            _m.load(CONFIG_PATH)
        import core.agent as agent
        agent.init(CONFIG_PATH)
        _agent_ready = True


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


def start_bot_thread():
    """Start the Telegram polling loop as a daemon thread.
    Assumes the model is already loaded (called from API startup event).
    """
    global _agent_ready
    if not BOT_TOKEN:
        print("[bot] KERNEL_EVO_TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")
        return
    import core.inference.model as _m
    import core.agent as _agent_mod

    from core.inference.model_client import is_server_running as _is_srv
    if _m._model is not None or _is_srv():
        _agent_ready = True

    _sync_bot_commands()

    t = threading.Thread(target=poll, daemon=True, name="kernel-telegram-bot")
    t.start()
    print(
        f"[bot] Telegram bot thread started (allowed chat: {ALLOWED_CHAT_ID or 'all'})"
    )
    if ALLOWED_CHAT_ID:
        u = threading.Thread(
            target=_update_check_loop,
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
_POLL_FAILURES_TOTAL = 0
_POLL_DNS_FAILURES_TOTAL = 0
_POLL_LAST_SUCCESS_EPOCH = 0.0


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
    global _POLL_FAILURES_TOTAL, _POLL_DNS_FAILURES_TOTAL, _POLL_LAST_SUCCESS_EPOCH

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

            _POLL_LAST_SUCCESS_EPOCH = time.time()
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
                            target=handle_callback,
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
                    send_typing(str(chat_id))
                    # Handle in thread so polling doesn't block
                    threading.Thread(
                        target=handle_message,
                        args=(str(chat_id), effective_text, sender_name, photo_file_id, voice_file_id, document_file_id, document_name, document_mime),
                        daemon=True
                    ).start()

        except KeyboardInterrupt:
            print("[bot] Stopped.")
            break
        except Exception as e:
            failure_streak += 1
            _POLL_FAILURES_TOTAL += 1
            dns_failure = _is_dns_resolution_error(e)
            if dns_failure:
                _POLL_DNS_FAILURES_TOTAL += 1
            backoff = _compute_poll_backoff_seconds(failure_streak, dns_failure=dns_failure)
            print(
                f"[bot] Poll error: type={type(e).__name__} dns={dns_failure} "
                f"streak={failure_streak} backoff_s={backoff:.1f} "
                f"dns_failures_total={_POLL_DNS_FAILURES_TOTAL} err={e}",
                flush=True,
            )
            time.sleep(backoff)


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: KERNEL_EVO_TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    # Load .env if not already loaded
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    print("[bot] Starting Kernel Telegram bot...")
    print(f"[bot] Allowed chat: {ALLOWED_CHAT_ID or 'all'}")

    # Load persisted memory on startup
    import core.memory.memory as _memory_startup

    _mem = _memory_startup.load()
    print(f"[bot] Memory loaded: {len(_mem)} message(s) from previous sessions")

    print("[bot] Checking model availability...")
    import core.inference.model as _model_module
    from core.inference.model_client import is_server_running as _is_srv

    if _is_srv():
        print("[bot] Model server detected — skipping in-process load")
    else:
        _model_module.load(CONFIG_PATH)
    print("[bot] Model ready, starting polling...")
    poll()
