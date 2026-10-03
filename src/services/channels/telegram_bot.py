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
from model_helpers import _estimate_model_gb, _vram_fit_mark, _curated_slot_for_repo  # noqa: F401 (re-exported for compatibility)
from telegram_messaging import _call_api, send_message, delete_message, edit_message, send_buttons, send_typing, send_file, send_voice  # noqa: F401 (re-exported for compatibility)
from telegram_formatting import _esc, _html, _TOOL_EMOJI, _TOOL_VERB, _tool_args_label, _format_tool_step  # noqa: F401 (re-exported for compatibility)
from telegram_config import BOT_TOKEN, ALLOWED_CHAT_ID, API_BASE, CONFIG_PATH, REPO_DIR  # noqa: F401 (re-exported for compatibility)
from telegram_providers import _LOCAL_REPO_TOKENS, _register_repo_token, _resolve_repo_token, _fetch_provider_models, _CLOUD_CALL_TYPES, _CALL_TYPE_LABELS, _show_cloud_provider_picker, _show_modal_provider_picker, _fetch_models_for_ct, _show_modal_model_picker, _apply_modal_model, _show_cloud_model_picker, _apply_cloud_model, _finish_cloud_setup  # noqa: F401 (re-exported for compatibility)
from telegram_memory import _search_collective_memory, _write_collective_memory  # noqa: F401 (re-exported for compatibility)
from telegram_maintenance import _restart_via_start_sh, _check_for_update, _update_check_loop, _handle_stop, _handle_restart, _handle_update, _handle_rollback, _ensure_agent, _sync_bot_commands  # noqa: F401 (re-exported for compatibility)
from telegram_voice import _list_voice_samples, _clone_voice_reply, TypingKeepAlive, download_file, _VOICE_SAMPLES_DIR, _DEFAULT_VOICE_SAMPLE  # noqa: F401 (re-exported for compatibility)
from telegram_poll import start_bot_thread, _is_dns_resolution_error, _compute_poll_backoff_seconds, poll  # noqa: F401 (re-exported for compatibility)
from telegram_callbacks import handle_callback, _MultimodalActivity  # noqa: F401 (re-exported for compatibility)
from telegram_media import _handle_photo_message, _handle_voice_message, _handle_document_message  # noqa: F401 (re-exported for compatibility)
from telegram_commands import _handle_skill_command, _handle_run_command, _handle_new, _handle_fresh, _handle_session, _handle_voice_clone, _handle_voices, _handle_start_help  # noqa: F401 (re-exported for compatibility)
from telegram_commands import _handle_skills, _handle_routines, _handle_packages, _handle_status, _handle_local  # noqa: F401 (re-exported for compatibility)
from telegram_commands import _handle_models, _handle_private_repo, _handle_clone, _handle_search, _handle_install, _handle_verbose, _handle_replica, _handle_run  # noqa: F401 (re-exported for compatibility)
from telegram_commands import _handle_provider, _handle_evolve  # noqa: F401 (re-exported for compatibility)
from telegram_handlers import _handle_thoughts, _handle_unknown_command, _handle_core_reply  # noqa: F401 (re-exported for compatibility)



# (text/tool-step formatting moved to telegram_formatting.py — see issue #3)
# GitHub update tracking
_latest_version: str = ""
# GitHub URLs now live in src/updater.py

# Lazy model load
_agent_ready = False

# Verbose mode — stream tool call steps to Telegram when ON
_verbose_mode = False

# In-process memory (list of {role, content} dicts)
_memory: list = []


# (Telegram send/API helpers moved to telegram_messaging.py — see issue #3)
# @TODO: Make configurable. Voice sample path — first user message during first contact sets it if missing.
# Active voice sample — can be switched via /voices. Voice helpers live in telegram_voice.py.
_active_voice_sample: str = _DEFAULT_VOICE_SAMPLE

# (inline-button callbacks + _MultimodalActivity moved to telegram_callbacks.py — see issue #3)
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
    # (photo/voice/document handlers moved to telegram_media.py — see issue #3)
    if photo_file_id:
        return _handle_photo_message(chat_id, text, photo_file_id)
    if voice_file_id:
        return _handle_voice_message(chat_id, text, voice_file_id)
    if document_file_id:
        return _handle_document_message(chat_id, text, document_file_id, document_name, document_mime)

    # Slash commands

    # ── /skill_<slug> — shortcut dispatched from Telegram command picker ──────
    # (slash-command handlers moved to telegram_commands.py — see issue #3)
    if text.startswith("/skill_"):
        return _handle_skill_command(chat_id, text)
    if text.startswith("/run_"):
        return _handle_run_command(chat_id, text)
    if text == "/new":
        return _handle_new(chat_id)
    if text == "/fresh":
        return _handle_fresh(chat_id)
    if text.startswith("/session"):
        return _handle_session(chat_id, text)
    if text == "/voice-clone":
        return _handle_voice_clone(chat_id, text)
    if text == "/voices" or text.startswith("set_voice_"):
        return _handle_voices(chat_id, text)
    if text in ("/start", "/help"):
        return _handle_start_help(chat_id)

    if text == "/init" or text.startswith("/init "):
        text = "/evolve init" + text[len("/init"):]

    if text == "/skills":
        return _handle_skills(chat_id)
    if text == "/routines":
        return _handle_routines(chat_id)
    if text == "/packages":
        return _handle_packages(chat_id)
    if text == "/status":
        return _handle_status(chat_id)
    if text == "/stop":
        return _handle_stop(chat_id)
    if text == "/restart":
        return _handle_restart(chat_id)
    if text == "/update":
        return _handle_update(chat_id)
    if text == "/rollback":
        return _handle_rollback(chat_id)
    if text == "/local":
        return _handle_local(chat_id)
    if text == "/cloud":
        _show_cloud_provider_picker(chat_id)


    # ── /models — local model hot-swap ───────────────────────────────────────
    if text.startswith("/models"):
        return _handle_models(chat_id, text)
    if text.startswith("/private_repo"):
        return _handle_private_repo(chat_id, text)
    if text.startswith("/clone"):
        return _handle_clone(chat_id, text)
    if text.startswith("/search"):
        return _handle_search(chat_id, text)
    if text.startswith("/install"):
        return _handle_install(chat_id, text)
    if text == "/verbose":
        return _handle_verbose(chat_id, text)
    if text.startswith("/replica"):
        return _handle_replica(chat_id, text)
    if text.startswith("/run "):
        return _handle_run(chat_id, text)
    if text.startswith("/providers"):
        # Backward-compatible alias: normalize to singular command path.
        text = "/provider" + text[len("/providers"):]

    if text.startswith("/provider"):
        return _handle_provider(chat_id, text)
    if text.startswith("/evolve"):
        return _handle_evolve(chat_id, text)
    if text == "/thoughts":
        return _handle_thoughts(chat_id)

    # For unrecognised slash commands, try agent.triage() first
    # so skills with exec dispatch (e.g. /markdown, /anonymize) run deterministically.
    # Known built-in commands are handled above - only forward unknowns.
    if _handle_unknown_command(chat_id, text):
        return

    _handle_core_reply(chat_id, text, sender_name)

# Self-update helpers — delegated to src/updater.py
# ---------------------------------------------------------------------------

from infra.updater import (
    get_current_version as _get_current_version,
    fetch_latest_version as _fetch_latest_version,
    check_update_available as _check_update_available,
    do_update as _do_update,
)


# (maintenance/update controllers moved to telegram_maintenance.py — see issue #3)
# (poll loop + backoff moved to telegram_poll.py — see issue #3)
# Poll telemetry counters (tests assert on these via the bot module)
_POLL_FAILURES_TOTAL = 0
_POLL_DNS_FAILURES_TOTAL = 0
_POLL_LAST_SUCCESS_EPOCH = 0.0
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
