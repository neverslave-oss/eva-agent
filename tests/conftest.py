"""
tests/conftest.py — Shared fixtures and collection hooks for kernel-evolving tests.

DB ISOLATION GUARANTEE
======================
All kernel DB paths are redirected to per-session temp files BEFORE any
application module is imported. This prevents test runs from reading or
writing production databases and ensures tests are hermetic.

Environment variables set here (all honoured by runtime_paths._db()):
  KERNEL_MEMORY_DB        — chat_history_evolving.db  (memory.py)
  KERNEL_LONG_TERM_DB     — memory_store.db           (long_term_memory.py)
  KERNEL_EVOLUTION_DB     — evolution.db              (evolver, evolution_log, trajectory_collector)
  KERNEL_SIGNALS_DB       — promoted_signals.db       (goal_discovery)
  KERNEL_FAILED_REQ_DB    — failed_requests.db        (database.agent.failed_requests)
  KERNEL_PROBES_DB        — probes.db                 (probe_store)
  KERNEL_SYSLOG_DB        — system_debug_log.db
  KERNEL_CONVERSATIONS_DB — conversations.db

HTTP / EXTERNAL CALLS
=====================
test_api.py         — integration, skips if server not running (require_server fixture)
test_docker_sandbox — integration, skips if Docker not available
test_e2e_flow.py    — voice-clone test skips on ConnectionError

No unit test should perform real HTTP requests or write to production DBs.
"""
import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Redirect ALL kernel DB paths to a shared temp directory for this session.
# Must happen before any src/ module is imported.
# ---------------------------------------------------------------------------

_tmp_db_dir = tempfile.mkdtemp(prefix="ke_test_dbs_")

_DB_ENV_MAP = {
    "KERNEL_MEMORY_DB":        "chat_history_test.db",
    "KERNEL_LONG_TERM_DB":     "long_term_memory_test.db",
    "KERNEL_EVOLUTION_DB":     "evolution_test.db",
    "KERNEL_SIGNALS_DB":       "promoted_signals_test.db",
    "KERNEL_FAILED_REQ_DB":    "failed_requests_test.db",
    "KERNEL_PROBES_DB":        "probes_test.db",
    "KERNEL_SYSLOG_DB":        "system_debug_log_test.db",
    "KERNEL_CONVERSATIONS_DB": "conversations_test.db",
}

for _env_key, _filename in _DB_ENV_MAP.items():
    os.environ.setdefault(_env_key, os.path.join(_tmp_db_dir, _filename))


def pytest_sessionfinish(session, exitstatus):
    """Clean up all temp DBs after the full test session finishes."""
    global _tmp_db_dir
    if _tmp_db_dir and os.path.isdir(_tmp_db_dir):
        shutil.rmtree(_tmp_db_dir, ignore_errors=True)
        _tmp_db_dir = None


# ---------------------------------------------------------------------------
# Skip yaml-dependent tests when pyyaml is not installed
# ---------------------------------------------------------------------------

def pytest_collect_file(parent, file_path):
    """Skip files that require yaml when yaml is not installed."""
    _yaml_dependent = {
        "test_agent_triage.py",
        "test_exec_dispatch.py",
        "test_routines.py",
        "test_skills.py",
    }
    if file_path.name in _yaml_dependent:
        try:
            import yaml  # noqa: F401
        except ImportError:
            return None
    return None  # defer to default collector
