"""telegram_config.py — shared config/constants for the Telegram bot modules.

Extracted from telegram_bot.py (issue #3 — decompose monolith) so the split
modules (telegram_messaging, telegram_providers, etc.) can share BOT_TOKEN,
API_BASE, CONFIG_PATH, REPO_DIR without circular imports.
"""
import os
from pathlib import Path

BOT_TOKEN = os.environ.get("KERNEL_EVO_TELEGRAM_BOT_TOKEN")
ALLOWED_CHAT_ID = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID")
API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"
_CONFIG_OVERRIDE = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
CONFIG_PATH = (
    os.path.abspath(os.path.expanduser(_CONFIG_OVERRIDE))
    if _CONFIG_OVERRIDE
    else str(Path(__file__).parent.parent.parent.parent / "config.yaml")
)
REPO_DIR = str(Path(__file__).parent.parent)
