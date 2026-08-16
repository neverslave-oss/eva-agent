"""Canonical runtime paths for kernel-evolving workspace layout.

This module is the single source of truth for runtime filesystem paths.
Layout:
  ~/.kernel-evolving/workspace/
    data/
    memory/
    artifacts/
    thoughts/
    logs/
"""

from __future__ import annotations

import os
from pathlib import Path


KERNEL_HOME = Path.home() / ".kernel-evolving"
WORKSPACE_ROOT = KERNEL_HOME / "workspace"

DATA_DIR = WORKSPACE_ROOT / "data"
MEMORY_DIR = WORKSPACE_ROOT / "memory"
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
THOUGHTS_DIR = WORKSPACE_ROOT / "thoughts"
LOGS_DIR = WORKSPACE_ROOT / "logs"
TMP_DIR = WORKSPACE_ROOT / "tmp"
RUNTIME_DIR = WORKSPACE_ROOT / "runtime"
NOTES_DIR = WORKSPACE_ROOT / "notes"
SCRIPTS_DIR = WORKSPACE_ROOT / "scripts"
DOCUMENTS_DIR = WORKSPACE_ROOT / "documents"

# Nested folders used by multiple subsystems.
THOUGHTS_IDEAS_DIR = THOUGHTS_DIR / "ideas"

ARTIFACTS_EVAL_RUNS_DIR = ARTIFACTS_DIR / "eval_runs"
ARTIFACTS_EVAL_RESULTS_DIR = ARTIFACTS_DIR / "eval_results"
ARTIFACTS_TRAJECTORIES_DIR = ARTIFACTS_DIR / "trajectories"
ARTIFACTS_FINETUNE_DIR = ARTIFACTS_DIR / "finetune"
ARTIFACTS_ADAPTER_ACTIVE_DIR = ARTIFACTS_DIR / "adapter_active"

# ---------------------------------------------------------------------------
# DB path overrides via environment variables.
# Test suites set these before importing any module to redirect all DB I/O
# to temp directories, preventing contamination of production databases.
#
# Convention: KERNEL_<NAME>_DB  e.g. KERNEL_EVOLUTION_DB=/tmp/test/evolution.db
# ---------------------------------------------------------------------------
def _db(env_key: str, default: Path) -> Path:
    override = os.environ.get(env_key)
    return Path(override) if override else default


# Databases under memory/ (user memory domain)
CHAT_HISTORY_DB    = _db("KERNEL_MEMORY_DB",        MEMORY_DIR / "chat_history_evolving.db")
LONG_TERM_MEMORY_DB = _db("KERNEL_LONG_TERM_DB",    MEMORY_DIR / "memory_store.db")

# Operational databases under data/
EVOLUTION_DB       = _db("KERNEL_EVOLUTION_DB",     DATA_DIR / "evolution.db")
PROMOTED_SIGNALS_DB = _db("KERNEL_SIGNALS_DB",      DATA_DIR / "promoted_signals.db")
FAILED_REQUESTS_DB  = _db("KERNEL_FAILED_REQ_DB",   DATA_DIR / "failed_requests.db")
PROBES_DB           = _db("KERNEL_PROBES_DB",       DATA_DIR / "probes.db")
SYSTEM_DEBUG_LOG_DB = _db("KERNEL_SYSLOG_DB",       DATA_DIR / "system_debug_log.db")
CONVERSATIONS_DB    = _db("KERNEL_CONVERSATIONS_DB", DATA_DIR / "conversations.db")
PROMPT_LOG_DB       = _db("KERNEL_PROMPT_LOG_DB", DATA_DIR / "prompt_log.db")


FINETUNE_GATE_STATE_FILE = DATA_DIR / "finetune_gate_state.json"
EVOLUTION_STATE_FILE = DATA_DIR / "evolution_state.json"
MODEL_ACTIVITY_FILE = RUNTIME_DIR / "model_activity.json"

MANAGED_DIRS = (
    WORKSPACE_ROOT,
    DATA_DIR,
    MEMORY_DIR,
    ARTIFACTS_DIR,
    THOUGHTS_DIR,
    LOGS_DIR,
    TMP_DIR,
    RUNTIME_DIR,
    NOTES_DIR,
    SCRIPTS_DIR,
    DOCUMENTS_DIR,
    THOUGHTS_IDEAS_DIR,
    ARTIFACTS_EVAL_RUNS_DIR,
    ARTIFACTS_EVAL_RESULTS_DIR,
    ARTIFACTS_TRAJECTORIES_DIR,
    ARTIFACTS_FINETUNE_DIR,
    ARTIFACTS_ADAPTER_ACTIVE_DIR,
)


def ensure_runtime_dirs() -> None:
    for path in MANAGED_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def expand_user_path(value: str | None, fallback: Path) -> str:
    """Expand a path from config, falling back to canonical runtime location."""
    if not value:
        return str(fallback)
    return os.path.expanduser(value)


# ---------------------------------------------------------------------------
# Config loader with environment-variable expansion.
#
# `config.yaml` may reference environment variables with `${VAR}` syntax (e.g.
# `${KERNEL_EVO_MODEL_DIR}/models/...`). This keeps machine-specific paths out
# of the committed config while still resolving correctly at runtime, because
# `start.sh` sources `.env` before launching the API/model server.
#
# Values are read from the environment first; if the env var is unset or empty
# the raw `${VAR}` text is left in place (so a missing var degrades gracefully
# instead of crashing).
# ---------------------------------------------------------------------------
_CONFIG_ENV_RE = None  # lazy import to avoid hard dependency at module load


def _expand_env(text: str) -> str:
    """Expand ${VAR} references in a string using the current environment."""
    global _CONFIG_ENV_RE
    if _CONFIG_ENV_RE is None:
        import re
        _CONFIG_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    def _sub(m):
        val = os.environ.get(m.group(1), "")
        return val if val else m.group(0)
    return _CONFIG_ENV_RE.sub(_sub, text)


def _expand_node(node):
    """Recursively expand ${VAR} references throughout a parsed config dict."""
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, list):
        return [_expand_node(item) for item in node]
    if isinstance(node, dict):
        return {k: _expand_node(v) for k, v in node.items()}
    return node


def load_config(path: str = "config.yaml") -> dict:
    """Load a YAML config file, expanding ${VAR} env references in all values.

    Falls back to an empty dict if the file is missing or unparseable, so
    callers can rely on `.get(...)` defaults without extra error handling.
    """
    import yaml
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return {}
    return _expand_node(raw)
