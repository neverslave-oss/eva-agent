"""
tests/integration/conftest.py

Bootstraps the agent module with a minimal config so triage() can be called
in tests without a full runtime init (no GPU, no Telegram, no config.yaml).
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)

MINIMAL_CONFIG = {
    "kernel_workspace": os.path.expanduser("~/.kernel-evolving/workspace"),
    "olly_workspace": "",
    "model": {"name": "test-model", "device": "cpu"},
    "api": {"openclaw_endpoint": "http://localhost:18789"},
    "paths": {"TRACKER": "scripts/tracker.py"},
    "self_identity": {},
    "thinking": {"journal_dir": os.path.expanduser("~/.kernel-evolving/workspace/thoughts")},
}


@pytest.fixture(autouse=True)
def bootstrap_agent():
    """Inject minimal config into agent module globals before every test."""
    import core.agent as agent
    original_config   = getattr(agent, "_config", None)
    original_skills   = getattr(agent, "_skills", None)
    original_routines = getattr(agent, "_routines", None)

    agent._config   = MINIMAL_CONFIG
    agent._skills   = []
    agent._routines = []

    yield

    agent._config   = original_config
    agent._skills   = original_skills
    agent._routines = original_routines
