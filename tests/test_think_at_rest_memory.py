"""
test_think_at_rest_memory.py — Tests for ThinkAtRest memory/identity consolidation.

Covers:
- _run_memory_seeker() writes extracted facts to USER.md (not user.json)
- _run_memory_seeker() deduplicates facts already in USER.md
- _run_memory_seeker() keeps user.json name field in sync
- _run_identity_consolidator() appends session learnings to AGENTS.md
- _run_identity_consolidator() is rate-limited (max once per 20h)
- _run_identity_consolidator() skips when not enough history
"""
import sys
import os
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_engine(tmp_path):
    """Create a ThinkAtRest with workspace redirected to tmp_path."""
    with patch("services.thought_engine.Path.home", return_value=tmp_path):
        from services.thought_engine import ThinkAtRest
        cfg = {
            "idle_threshold_s": 300,
            "thought_interval_s": 1800,
            "min_score": 0.4,
            "journal_dir": str(tmp_path / ".kernel-evolving/workspace/thoughts"),
            "ideas_dir": str(tmp_path / ".kernel-evolving/workspace/ideas"),
        }
        engine = ThinkAtRest(cfg)
    return engine


def _ws(tmp_path) -> Path:
    """Workspace path shorthand."""
    return tmp_path / ".kernel-evolving" / "workspace"


# ── _run_memory_seeker tests ─────────────────────────────────────────────────

class TestMemorySeeker:

    def test_facts_written_to_user_md(self, tmp_path):
        """Extracted facts must go to USER.md, not just user.json."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        user_md = ws / "USER.md"
        user_md.write_text("# USER.md\n\n## Known facts\n", encoding="utf-8")

        fake_turns = [{"role": "user", "content": "I have 3 dogs and live in Rome"}]
        fake_facts = '{"pets": "3 dogs", "location": "Rome"}'

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value=fake_facts):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_memory_seeker()

        content = user_md.read_text(encoding="utf-8")
        assert "pets: 3 dogs" in content, "pets fact should be in USER.md"
        assert "location: Rome" in content, "location fact should be in USER.md"

    def test_no_duplicate_facts_written(self, tmp_path):
        """Facts already in USER.md must not be appended again."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        user_md = ws / "USER.md"
        user_md.write_text("# USER.md\n\n## Known facts\n- pets: 3 dogs\n", encoding="utf-8")

        fake_turns = [{"role": "user", "content": "I have 3 dogs"}]
        fake_facts = '{"pets": "3 dogs"}'

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value=fake_facts):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_memory_seeker()

        content = user_md.read_text(encoding="utf-8")
        assert content.count("- pets: 3 dogs") == 1, "Duplicate fact must not be appended"

    def test_user_json_name_kept_in_sync(self, tmp_path):
        """user.json name field must be updated when memory seeker extracts a name."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        user_md = ws / "USER.md"
        user_md.write_text("# USER.md\n\n## Known facts\n", encoding="utf-8")
        user_json = ws / "user.json"
        user_json.write_text('{"name": "unknown"}', encoding="utf-8")

        fake_turns = [{"role": "user", "content": "My name is Fabio"}]
        fake_facts = '{"name": "Fabio"}'

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value=fake_facts):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_memory_seeker()

        data = json.loads(user_json.read_text())
        assert data["name"] == "Fabio", "user.json name must be updated from memory seeker"

    def test_no_write_when_model_not_running(self, tmp_path):
        """Memory seeker must not write anything when model server is down."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        user_md = ws / "USER.md"
        user_md.write_text("# USER.md\n\n## Known facts\n", encoding="utf-8")
        original = user_md.read_text()

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.inference.model_client.is_server_running", return_value=False):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_memory_seeker()

        assert user_md.read_text() == original, "USER.md must not change when model is down"

    def test_empty_facts_no_write(self, tmp_path):
        """Empty facts response must not modify USER.md."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        user_md = ws / "USER.md"
        user_md.write_text("# USER.md\n\n## Known facts\n", encoding="utf-8")
        original = user_md.read_text()

        fake_turns = [{"role": "user", "content": "hello"}]

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value="{}"):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_memory_seeker()

        assert user_md.read_text() == original, "USER.md must not change when no facts extracted"


# ── _run_identity_consolidator tests ────────────────────────────────────────

class TestIdentityConsolidator:

    def test_appends_learnings_to_agents_md(self, tmp_path):
        """Session learnings must be appended to AGENTS.md."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "runtime").mkdir(exist_ok=True)
        agents_md = ws / "AGENTS.md"
        agents_md.write_text("# AGENTS.md\n\n## Who you are\nKernel-Evo.\n", encoding="utf-8")

        fake_turns = [
            item
            for i in range(5)
            for item in [
                {"role": "user", "content": f"msg {i}"},
                {"role": "assistant", "content": f"resp {i}"},
            ]
        ]

        fake_notes = '["Always confirm before running destructive commands", "User prefers concise replies"]'

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value=fake_notes):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_identity_consolidator()

        content = agents_md.read_text(encoding="utf-8")
        assert "Always confirm before running destructive commands" in content
        assert "User prefers concise replies" in content
        assert "Session learnings" in content

    def test_rate_limited_20h(self, tmp_path):
        """Consolidator must not run again within 20 hours."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        runtime = ws / "runtime"
        runtime.mkdir(exist_ok=True)
        agents_md = ws / "AGENTS.md"
        agents_md.write_text("# AGENTS.md\n", encoding="utf-8")

        # Stamp recent run
        stamp = runtime / "last_identity_consolidation.txt"
        stamp.write_text(str(time.time() - 100))  # 100s ago — within 20h window

        infer_mock = MagicMock(return_value='["some learning"]')

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", infer_mock):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_identity_consolidator()

        infer_mock.assert_not_called(), "Should not call infer within 20h rate limit"
        assert "Session learnings" not in agents_md.read_text()

    def test_skips_with_insufficient_history(self, tmp_path):
        """Consolidator must skip when there are fewer than 6 turns."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "runtime").mkdir(exist_ok=True)
        agents_md = ws / "AGENTS.md"
        agents_md.write_text("# AGENTS.md\n", encoding="utf-8")

        short_turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

        infer_mock = MagicMock(return_value='["learning"]')

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=short_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", infer_mock):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_identity_consolidator()

        infer_mock.assert_not_called(), "Should not call infer with < 6 turns"

    def test_no_duplicate_learnings(self, tmp_path):
        """Notes already in AGENTS.md must not be appended again."""
        ws = _ws(tmp_path)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "runtime").mkdir(exist_ok=True)
        existing_note = "Always confirm before running destructive commands"
        agents_md = ws / "AGENTS.md"
        agents_md.write_text(f"# AGENTS.md\n\n- {existing_note}\n", encoding="utf-8")

        fake_turns = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        fake_notes = f'["{existing_note}"]'

        with patch("services.thought_engine.Path.home", return_value=tmp_path), \
             patch("core.memory.memory.load", return_value=fake_turns), \
             patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer", return_value=fake_notes):
            from services.thought_engine import ThinkAtRest
            engine = ThinkAtRest({})
            engine._run_identity_consolidator()

        content = agents_md.read_text(encoding="utf-8")
        assert content.count(existing_note) == 1, "Duplicate note must not be appended to AGENTS.md"
