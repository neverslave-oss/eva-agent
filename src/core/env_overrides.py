"""
env_overrides.py — structured runtime environment overrides.

A small, dedicated JSON store for environment-variable overrides (e.g. provider
API keys) pushed by external clients such as the desktop app via /config/env.

Why JSON instead of editing .env / .env.local:
  - Structured and safe: no shell parsing (keys can contain #, spaces, $, ...).
  - App-owned: not coupled to start.sh sourcing order.
  - Clean separation: the base .env stays untouched; overrides live here and
    take precedence at runtime.

Design:
  - load_env_overrides() reads the JSON store and injects values into
    os.environ (called once at API startup, before serving requests).
  - save_env_overrides(overrides) writes the store (used by /config/env).
  - Because values are injected into os.environ, all existing
    os.environ.get(...) reads across the codebase pick them up unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from runtime_paths import DATA_DIR

# Canonical location of the override store (inside the agent workspace data dir).
ENV_OVERRIDES_PATH = DATA_DIR / "env_overrides.json"

# Allowlisted env-var names that may be overridden via this store.
ALLOWED_ENV_KEYS = {
    "OPENAI_API_KEY",
    "TMP_OPEN_AI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
    "KERNEL_EVO_TELEGRAM_BOT_TOKEN",
    "KERNEL_EVO_TELEGRAM_CHAT_ID",
    "KERNEL_USER_NAME",
    "KERNEL_USER_HANDLE",
    # Vision / eye wiring (look tool): eye endpoints + semantic describe brain.
    # Config-driven via config.yaml ${VAR} placeholders; settable from the
    # desktop app settings screen through /config/env.
    "KERNEL_EVO_EYE_LEFT_BASE",
    "KERNEL_EVO_EYE_LEFT_STREAM",
    "KERNEL_EVO_EYE_RIGHT_BASE",
    "KERNEL_EVO_EYE_RIGHT_STREAM",
    "KERNEL_EVO_DESCRIBE_BASE",
    "KERNEL_EVO_DESCRIBE_MODEL",
}


def _read_store() -> dict:
    """Read the JSON override store, returning {} if missing/corrupt."""
    try:
        if ENV_OVERRIDES_PATH.exists():
            data = json.loads(ENV_OVERRIDES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def load_env_overrides() -> dict:
    """Inject persisted overrides into os.environ and return the applied map.

    Only allowlisted keys are applied, and only non-empty values. Existing
    os.environ values are overwritten by the override (override wins).
    """
    applied = {}
    for name, value in _read_store().items():
        name = str(name).strip()
        if name not in ALLOWED_ENV_KEYS:
            continue
        value = "" if value is None else str(value)
        if value.strip() == "":
            continue
        os.environ[name] = value
        applied[name] = value
    if applied:
        print(f"[env_overrides] applied {len(applied)} overrides from {ENV_OVERRIDES_PATH.name}", flush=True)
    return applied


def save_env_overrides(overrides: dict) -> dict:
    """Merge allowlisted, non-empty values into the store and persist to disk.

    Returns the map of keys actually persisted. Empty values are ignored so
    they never clobber an existing override.
    """
    store = _read_store()
    applied = {}
    for name, value in overrides.items():
        name = str(name).strip()
        if name not in ALLOWED_ENV_KEYS:
            continue
        value = "" if value is None else str(value)
        if value.strip() == "":
            continue
        store[name] = value
        applied[name] = value
    try:
        ENV_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENV_OVERRIDES_PATH.write_text(
            json.dumps(store, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[env_overrides] WARNING: could not write {ENV_OVERRIDES_PATH}: {e}", flush=True)
    return applied
