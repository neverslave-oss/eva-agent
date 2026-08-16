"""
tests/test_failed_requests.py — ADR-020 unit tests for failed_requests.py
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tmp_db(tmp_path):
    """Return a temp DB path and patch the module to use it."""
    return str(tmp_path / "database.agent.failed_requests.db")


# ── detect_failure ────────────────────────────────────────────────────────────

class TestDetectFailure:
    def setup_method(self):
        import database.agent.failed_requests as failed_requests
        self.fr = failed_requests

    def test_none_on_normal_reply(self):
        result = self.fr.detect_failure("Here are the top 3 AI news stories today: ...", "what is the news?")
        assert result is None

    def test_no_skill_on_cant(self):
        result = self.fr.detect_failure("I can't help with that.", "do something")
        assert result == "no_skill"

    def test_no_skill_on_no_skill_found(self):
        result = self.fr.detect_failure("No skill found for this request.", "some task")
        assert result == "no_skill"

    def test_skill_error_on_model_server_prefix(self):
        result = self.fr.detect_failure("[model_server error] inference failed", "anything")
        assert result == "skill_error"

    def test_skill_error_on_empty(self):
        result = self.fr.detect_failure("", "user asked something")
        assert result == "skill_error"

    def test_refusal_on_sorry(self):
        result = self.fr.detect_failure("Sorry, I am unable to do that.", "do something")
        assert result == "refusal"

    def test_partial_on_very_short_reply(self):
        # 5-word query, reply < 80 chars — should be partial
        result = self.fr.detect_failure("OK.", "tell me everything about quantum computing")
        assert result == "partial"

    def test_no_partial_on_short_command(self):
        # Single-word command — short reply is fine
        result = self.fr.detect_failure("Done.", "/help")
        assert result is None

    def test_no_partial_when_reply_long_enough(self):
        long_reply = "a" * 100
        result = self.fr.detect_failure(long_reply, "tell me everything about quantum computing")
        assert result is None


# ── record / get_unresolved / mark_resolved / increment_retry ─────────────────

class TestFailedRequestsCRUD:
    def test_record_and_retrieve(self, tmp_path):
        import database.agent.failed_requests as fr
        with patch.object(fr, "_DB_PATH", _tmp_db(tmp_path)):
            row_id = fr.record("chat1", "what is the news today", "I can't find any news", "no_skill")
            assert row_id > 0
            unresolved = fr.get_unresolved(limit=10)
            assert len(unresolved) == 1
            assert unresolved[0]["user_message"] == "what is the news today"
            assert unresolved[0]["failure_type"] == "no_skill"
            assert unresolved[0]["resolved"] == 0

    def test_mark_resolved_removes_from_unresolved(self, tmp_path):
        import database.agent.failed_requests as fr
        with patch.object(fr, "_DB_PATH", _tmp_db(tmp_path)):
            row_id = fr.record("chat1", "search the web", "no skill found", "no_skill")
            fr.mark_resolved(row_id, reward=1)
            unresolved = fr.get_unresolved(limit=10)
            assert len(unresolved) == 0

    def test_mark_resolved_reward_0(self, tmp_path):
        import database.agent.failed_requests as fr
        with patch.object(fr, "_DB_PATH", _tmp_db(tmp_path)):
            row_id = fr.record("chat1", "search web", "failed", "no_skill")
            fr.mark_resolved(row_id, reward=0)
            unresolved = fr.get_unresolved(limit=10)
            assert len(unresolved) == 0

    def test_increment_retry(self, tmp_path):
        import database.agent.failed_requests as fr
        with patch.object(fr, "_DB_PATH", _tmp_db(tmp_path)):
            row_id = fr.record("chat1", "search web", "failed", "no_skill")
            count1 = fr.increment_retry(row_id)
            count2 = fr.increment_retry(row_id)
            assert count1 == 1
            assert count2 == 2

    def test_mark_pending_confirm(self, tmp_path):
        import database.agent.failed_requests as fr
        with patch.object(fr, "_DB_PATH", _tmp_db(tmp_path)):
            row_id = fr.record("chat1", "search web", "failed", "no_skill")
            fr.mark_pending_confirm(row_id, "web-search-v2", "Here are 3 results...")
            # Still unresolved (pending = awaiting user confirmation)
            unresolved = fr.get_unresolved(limit=10)
            assert len(unresolved) == 1
            assert unresolved[0]["pending_confirm"] == 1
            assert unresolved[0]["confirm_skill"] == "web-search-v2"
