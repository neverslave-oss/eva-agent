"""test_eval_tasks.py — Eval test cases for skill dispatch, multi-turn reference resolution,
and check: pattern extraction.
"""
import sys
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_memory_module(db_path: str):
    """Load memory.py with DB_FILE patched to an isolated temp path."""
    import importlib, importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "memory_test_eval_tasks",
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "memory", "memory.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    from pathlib import Path
    mod.DB_FILE = Path(db_path)  # type: ignore[attr-defined]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _make_mock_provider(return_value="mock response"):
    """Build a mock provider object that satisfies triage()'s _prov calls."""
    prov = MagicMock()
    prov.infer_with_tools.return_value = return_value
    prov.get_provider.return_value = "local"
    prov.get_model.return_value = "mock-model"
    return prov


# ── Test 1: Skill dispatch routing ───────────────────────────────────────────

def test_skill_dispatch_routing():
    """ADR-022: triage() routes free-text to _prov.infer_with_tools, not skill dispatch.
    When asked to 'search collective memory for kernel', the provider's infer_with_tools
    should be called (not run_skill pre-emption)."""
    import core.agent as _agent
    import core.memory.memory as _mem
    import core.memory.context as _ctx

    mock_prov = _make_mock_provider("search results")

    with patch.object(_agent, '_config', {"api": {"openclaw_endpoint": "http://localhost:18789"}}), \
         patch.object(_agent, '_skills', []), \
         patch.object(_agent, '_routines', []), \
         patch.object(_agent, '_embedding_client', None), \
         patch.object(_mem, 'load', return_value=[]), \
         patch.object(_ctx, 'build_system_prompt', return_value="sys"), \
         patch('core.inference.provider.get_provider', return_value=mock_prov):
        result = _agent.triage("search collective memory for kernel", chat_id="test_chat_1")

    # ADR-022: the provider's infer_with_tools must be called
    assert mock_prov.infer_with_tools.called, \
        "Expected _prov.infer_with_tools to be called for free-text request"
    assert result == "search results"


# ── Test 2: Multi-turn reference resolution ───────────────────────────────────

def test_multi_turn_reference_resolution():
    """When history contains a prior reference to a project path, triage() should
    pass that context to infer_with_tools (ADR-022 — no interception)."""
    import core.agent as _agent
    import core.memory.memory as _mem
    import core.memory.context as _ctx

    # Build prior history with a project path reference
    prior_msgs = [
        {"role": "user", "content": "my project is at ~/projects/lunar-mapper"},
        {"role": "assistant", "content": "Got it, I'll remember that."},
    ]

    mock_prov = _make_mock_provider("tests ran")

    with patch.object(_agent, '_config', {"api": {"openclaw_endpoint": "http://localhost:18789"}}), \
         patch.object(_agent, '_skills', []), \
         patch.object(_agent, '_routines', []), \
         patch.object(_agent, '_embedding_client', None), \
         patch.object(_mem, 'load', return_value=prior_msgs), \
         patch.object(_ctx, 'build_system_prompt', return_value="sys"), \
         patch('core.inference.provider.get_provider', return_value=mock_prov):
        result = _agent.triage("run the tests for it", chat_id="test_multi_turn_ref")

    # infer_with_tools must have been called
    assert mock_prov.infer_with_tools.called, \
        "Expected _prov.infer_with_tools to be called"
    # The messages passed to infer_with_tools should include the history context
    call_args = mock_prov.infer_with_tools.call_args
    messages_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("messages", [])
    all_text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in messages_arg
    )
    assert "lunar-mapper" in all_text, \
        f"Expected 'lunar-mapper' in context passed to infer, got: {all_text[:500]}"


# ── Test 3: check: pattern extraction ────────────────────────────────────────

def test_check_pattern_known_service():
    """'- check: fantasia' → curl to localhost:8765/health with FANTASIA label."""
    from core.routines import _extract_check_patterns
    body = "- check: fantasia"
    result = _extract_check_patterns(body)
    assert len(result) == 1
    assert "http://localhost:8765/health" in result[0]
    assert "FANTASIA UP" in result[0]
    assert "FANTASIA DOWN" in result[0]


def test_check_pattern_direct_url():
    """'- check: http://localhost:9999/status' → curl that URL directly."""
    from core.routines import _extract_check_patterns
    body = "- check: http://localhost:9999/status"
    result = _extract_check_patterns(body)
    assert len(result) == 1
    assert "http://localhost:9999/status" in result[0]
    assert "UP" in result[0]
    assert "DOWN" in result[0]


def test_check_pattern_prose_skipped():
    """'- If routine check: send note' is prose and should not be extracted."""
    from core.routines import _extract_check_patterns
    body = "- If routine check: send note"
    result = _extract_check_patterns(body)
    assert result == [], f"Expected [], got {result}"


def test_check_pattern_all_known_services():
    """All known service names should be mapped correctly."""
    from core.routines import _extract_check_patterns, _CHECK_SERVICE_MAP
    for service, url in _CHECK_SERVICE_MAP.items():
        body = f"- check: {service}"
        result = _extract_check_patterns(body)
        assert len(result) == 1, f"Expected 1 result for {service}, got {result}"
        assert url in result[0], f"Expected {url} in result for {service}"
        assert service.upper() in result[0]
