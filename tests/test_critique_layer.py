"""
tests/test_critique_layer.py — ADR-019 Phase 3

Tests for CritiqueLayer, routing hint writer, and goal_discovery.resolve_pattern.
All tests use mocks — no real model inference.
"""

import os
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure src/ is on path ────────────────────────────────────────────────────
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_layer(max_iterations=3, timeout=5):
    from core.layers.critique_layer import CritiqueLayer
    cfg = {"critique": {"enabled": True, "max_iterations": max_iterations, "run_timeout_s": timeout}}
    return CritiqueLayer(config=cfg)


def _make_evo_result(gap="capability gap in text", installed=None):
    evo = MagicMock()
    evo.gap = gap
    evo.found = True
    evo.installed = installed or ["test_skill"]
    return evo


def _make_thought(text="What is the gap?"):
    return {"thought": text, "_observer_evidence": "some evidence"}


# ── CritiqueLayer.assess ──────────────────────────────────────────────────────

class TestCritiqueLayerAssess:
    def test_resolved_when_infer_returns_resolved(self):
        """ADR-020: when critique says RESOLVED and no _source_request_id, verdict is PENDING_CONFIRM
        (confirmation buttons sent to user). With _source_request_id present, same outcome."""
        layer = _make_layer()
        infer_fn = MagicMock(return_value="RESOLVED output addresses the gap directly")

        # Patch sandboxed run to return non-empty output
        # Patch send_confirmation_request so no real Telegram call is made
        with patch.object(layer, "_run_skill_sandboxed", return_value="skill output text"), \
             patch.object(layer, "_send_confirmation_request") as mock_confirm:
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=1,
            )

        # ADR-020: RESOLVED transitions to PENDING_CONFIRM — awaiting human confirmation
        assert verdict.verdict == "PENDING_CONFIRM"
        assert verdict.should_retry is False
        assert verdict.attempt == 1
        mock_confirm.assert_called_once()

    def test_failed_when_skill_run_returns_none(self):
        layer = _make_layer()
        infer_fn = MagicMock()

        with patch.object(layer, "_run_skill_sandboxed", return_value=None):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=1,
            )

        assert verdict.verdict == "FAILED"
        assert verdict.should_retry is True  # attempt 1 < max 3
        infer_fn.assert_not_called()  # no point calling model if no output

    def test_partial_should_retry_true_when_attempt_less_than_max(self):
        layer = _make_layer(max_iterations=3)
        infer_fn = MagicMock(return_value="PARTIAL related but incomplete output")

        with patch.object(layer, "_run_skill_sandboxed", return_value="some partial output"):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=1,
            )

        assert verdict.verdict == "PARTIAL"
        assert verdict.should_retry is True

    def test_partial_should_retry_false_when_attempt_equals_max(self):
        layer = _make_layer(max_iterations=3)
        infer_fn = MagicMock(return_value="PARTIAL related but incomplete output")

        with patch.object(layer, "_run_skill_sandboxed", return_value="some partial output"):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=3,
            )

        assert verdict.verdict == "PARTIAL"
        assert verdict.should_retry is False

    def test_failed_should_retry_false_at_max_attempt(self):
        layer = _make_layer(max_iterations=3)
        infer_fn = MagicMock(return_value="FAILED unrelated output")

        with patch.object(layer, "_run_skill_sandboxed", return_value="bad output"):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=3,
            )

        assert verdict.verdict == "FAILED"
        assert verdict.should_retry is False

    def test_partial_when_no_infer_fn(self):
        """If infer_fn is None, we return PARTIAL with an explanation note."""
        layer = _make_layer()
        with patch.object(layer, "_run_skill_sandboxed", return_value="some output"):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=None,
                attempt=1,
            )

        assert verdict.verdict == "PARTIAL"
        assert "infer_fn" in verdict.notes.lower()

    def test_parse_fallback_to_failed_on_garbage_response(self):
        layer = _make_layer()
        infer_fn = MagicMock(return_value="I cannot determine the outcome from the output")

        with patch.object(layer, "_run_skill_sandboxed", return_value="output text"):
            verdict = layer.assess(
                thought=_make_thought(),
                evolution_result=_make_evo_result(),
                infer_fn=infer_fn,
                attempt=1,
            )

        assert verdict.verdict == "FAILED"


# ── _run_skill_sandboxed timeout ──────────────────────────────────────────────

class TestRunSkillSandboxed:
    def test_returns_none_on_timeout(self):
        """Simulate a timeout by using a very short timeout and a slow agent."""
        import time
        from core.layers.critique_layer import CritiqueLayer

        cfg = {"critique": {"enabled": True, "max_iterations": 3, "run_timeout_s": 0}}
        layer = CritiqueLayer(config=cfg)
        # timeout=0 → should time out immediately

        def slow_triage(*args, **kwargs):
            time.sleep(5)
            return "output"

        mock_agent = types.ModuleType("agent")
        mock_agent.triage = slow_triage

        with patch.dict(sys.modules, {"agent": mock_agent}):
            result = layer._run_skill_sandboxed("some task", infer_fn=None)

        assert result is None

    def test_returns_none_on_exception(self):
        """If agent.triage raises, sandboxed runner returns None."""
        layer = _make_layer()

        import core.agent as _real_agent
        with patch.object(_real_agent, "triage", side_effect=RuntimeError("boom")):
            result = layer._run_skill_sandboxed("task", infer_fn=None)

        assert result is None


# ── _write_routing_hint ───────────────────────────────────────────────────────

class TestWriteRoutingHint:
    def test_writes_trigger_context_section(self, tmp_path):
        # Create fake skill dir
        skill_dir = tmp_path / "test_skill_abc"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("# TestSkillAbc\n\nA test skill.\n", encoding="utf-8")

        layer = _make_layer()

        with patch.object(layer, "_write_routing_hint") as mock_write:
            # Call the real method with patched skills_base
            pass

        # Call real method with skills_base overridden
        original = os.path.expanduser

        def fake_expanduser(path):
            if "~/.kernel-evolving/ecosystem" in path:
                return str(tmp_path)
            return original(path)

        with patch("os.path.expanduser", side_effect=fake_expanduser):
            layer._write_routing_hint(
                skill_name="test_skill_abc",
                thought={"thought": "How do I do X?"},
                evidence_summary="3 requests about X",
                resolved_at="2026-01-01T00:00:00+00:00",
            )

        content = skill_md.read_text(encoding="utf-8")
        assert "## Trigger Context" in content
        assert "How do I do X?" in content
        assert "3 requests about X" in content
        assert "2026-01-01T00:00:00+00:00" in content

    def test_idempotent_does_not_write_twice(self, tmp_path):
        skill_dir = tmp_path / "skill_idem"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "# SkillIdem\n\n## Trigger Context\n<!-- already there -->\n",
            encoding="utf-8",
        )

        original_content = skill_md.read_text(encoding="utf-8")
        layer = _make_layer()
        original = os.path.expanduser

        def fake_expanduser(path):
            if "~/.kernel-evolving/ecosystem" in path:
                return str(tmp_path)
            return original(path)

        with patch("os.path.expanduser", side_effect=fake_expanduser):
            layer._write_routing_hint(
                skill_name="skill_idem",
                thought={"thought": "something"},
                evidence_summary="evidence",
                resolved_at="2026-01-01T00:00:00+00:00",
            )

        # Content should be unchanged
        assert skill_md.read_text(encoding="utf-8") == original_content


# ── goal_discovery.resolve_pattern ───────────────────────────────────────────

class TestResolvePattern:
    def _setup_db(self, tmp_path):
        """Patch _DB_FILE in goal_discovery to use a temp path."""
        import core.pipelines.goal_discovery as goal_discovery
        tmp_db = tmp_path / "test_promoted.db"
        return goal_discovery, tmp_db

    def test_removes_matching_rows_from_db(self, tmp_path):
        import core.pipelines.goal_discovery as goal_discovery
        original_db = goal_discovery._DB_FILE

        tmp_db = tmp_path / "promoted.db"
        goal_discovery._DB_FILE = tmp_db

        try:
            # Insert some rows
            conn = goal_discovery._db_conn()
            conn.execute("INSERT INTO promoted_signals (text, weight, created_at) VALUES (?,?,?)",
                         ("I need help with machine learning tasks", 1, "2026-01-01"))
            conn.execute("INSERT INTO promoted_signals (text, weight, created_at) VALUES (?,?,?)",
                         ("unrelated content about cooking", 1, "2026-01-01"))
            conn.commit()
            conn.close()

            goal_discovery.resolve_pattern("machine learning")

            conn = goal_discovery._db_conn()
            rows = conn.execute("SELECT text FROM promoted_signals").fetchall()
            conn.close()

            texts = [r[0] for r in rows]
            assert all("machine" not in t.lower() for t in texts), f"Expected deleted but got: {texts}"
            assert any("cooking" in t for t in texts), "unrelated row should survive"
        finally:
            goal_discovery._DB_FILE = original_db

    def test_removes_from_in_memory_log(self, tmp_path):
        import core.pipelines.goal_discovery as goal_discovery
        original_db = goal_discovery._DB_FILE
        tmp_db = tmp_path / "promoted2.db"
        goal_discovery._DB_FILE = tmp_db

        try:
            # Seed in-memory log
            with goal_discovery._log_lock:
                goal_discovery._interaction_log = [
                    "I need help with machine learning",
                    "unrelated cooking query",
                    "machine learning model question",
                ]

            goal_discovery.resolve_pattern("machine learning")

            with goal_discovery._log_lock:
                remaining = list(goal_discovery._interaction_log)

            assert len(remaining) == 1
            assert "cooking" in remaining[0]
        finally:
            goal_discovery._DB_FILE = original_db
            # Restore log to empty
            with goal_discovery._log_lock:
                goal_discovery._interaction_log = []

    def test_no_op_on_empty_text(self):
        import core.pipelines.goal_discovery as goal_discovery
        # Should not raise
        goal_discovery.resolve_pattern("")
        goal_discovery.resolve_pattern("   ")
