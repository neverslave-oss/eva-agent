"""
test_env_overrides.py — Validates the structured env-override store used by the
desktop app settings screen via `POST /config/env`.

Covers:
  - save_env_overrides: allowlisted keys (incl. vision/eye wiring) persist,
    unknown keys are rejected, empty values never clobble an existing key
  - load_env_overrides: persisted overrides are injected into os.environ
    (the startup path that the eye registry / describe brain read from)

Rules:
  - No real HTTP, no model, no Telegram.
  - The override store path is redirected to a temp file (never the real one).
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import env_overrides as eo


def _fresh_store(tmp_path: Path) -> Path:
    """Point the override store to a temp file and clear os.environ of the
    vision keys so each test starts hermetic."""
    store = tmp_path / "env_overrides.json"
    eo.ENV_OVERRIDES_PATH = store
    for k in eo.ALLOWED_ENV_KEYS:
        os.environ.pop(k, None)
    return store


# ─────────────────────────────────────────────────────────────────────────────
# save_env_overrides
# ─────────────────────────────────────────────────────────────────────────────
class TestSaveEnvOverrides:
    def test_persists_vision_eye_wiring_keys(self, tmp_path):
        store = _fresh_store(tmp_path)
        applied = eo.save_env_overrides({
            "KERNEL_EVO_EYE_LEFT_BASE": "http://192.0.2.50:5010",
            "KERNEL_EVO_EYE_LEFT_STREAM": "http://192.0.2.50:80/stream",
            "KERNEL_EVO_EYE_RIGHT_BASE": "http://192.0.2.60:5000",
            "KERNEL_EVO_EYE_RIGHT_STREAM": "http://192.0.2.60:5000/video_feed",
            "KERNEL_EVO_DESCRIBE_BASE": "http://localhost:8005",
            "KERNEL_EVO_DESCRIBE_MODEL": "google/gemma-4-26b-a4b-it",
        })
        assert sorted(applied) == sorted([
            "KERNEL_EVO_EYE_LEFT_BASE",
            "KERNEL_EVO_EYE_LEFT_STREAM",
            "KERNEL_EVO_EYE_RIGHT_BASE",
            "KERNEL_EVO_EYE_RIGHT_STREAM",
            "KERNEL_EVO_DESCRIBE_BASE",
            "KERNEL_EVO_DESCRIBE_MODEL",
        ])
        # Persisted to the JSON store.
        stored = json.loads(store.read_text())
        assert stored["KERNEL_EVO_EYE_LEFT_BASE"] == "http://192.0.2.50:5010"
        assert stored["KERNEL_EVO_DESCRIBE_MODEL"] == "google/gemma-4-26b-a4b-it"

    def test_rejects_unknown_key(self, tmp_path):
        store = _fresh_store(tmp_path)
        applied = eo.save_env_overrides({"NOT_A_REAL_VAR": "x"})
        assert applied == {}
        assert store.exists() is False or json.loads(store.read_text()) == {}

    def test_ignores_empty_value(self, tmp_path):
        store = _fresh_store(tmp_path)
        eo.save_env_overrides({"KERNEL_EVO_EYE_LEFT_BASE": "http://hub:5010"})
        # Empty value must not clobber the existing override.
        applied = eo.save_env_overrides({"KERNEL_EVO_EYE_LEFT_BASE": ""})
        assert "KERNEL_EVO_EYE_LEFT_BASE" not in applied
        stored = json.loads(store.read_text())
        assert stored["KERNEL_EVO_EYE_LEFT_BASE"] == "http://hub:5010"

    def test_existing_api_key_still_persists(self, tmp_path):
        store = _fresh_store(tmp_path)
        applied = eo.save_env_overrides({"OPENAI_API_KEY": "sk-test"})
        assert "OPENAI_API_KEY" in applied
        assert json.loads(store.read_text())["OPENAI_API_KEY"] == "sk-test"


# ─────────────────────────────────────────────────────────────────────────────
# load_env_overrides
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadEnvOverrides:
    def test_injects_vision_keys_into_os_environ(self, tmp_path):
        store = _fresh_store(tmp_path)
        eo.save_env_overrides({
            "KERNEL_EVO_EYE_RIGHT_BASE": "http://192.0.2.60:5000",
            "KERNEL_EVO_DESCRIBE_BASE": "http://localhost:8005",
        })
        loaded = eo.load_env_overrides()
        assert "KERNEL_EVO_EYE_RIGHT_BASE" in loaded
        assert os.environ.get("KERNEL_EVO_EYE_RIGHT_BASE") == "http://192.0.2.60:5000"
        assert os.environ.get("KERNEL_EVO_DESCRIBE_BASE") == "http://localhost:8005"

    def test_returns_empty_when_no_store(self, tmp_path):
        store = _fresh_store(tmp_path)
        assert eo.load_env_overrides() == {}
