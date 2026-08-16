"""
test_memory.py — Unit tests for memory.py

All tests are isolated: no production DB or JSON files are touched.
Uses tmp_path/monkeypatch to redirect MEMORY_FILE, DB_FILE, DB_DIR.
"""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.memory.memory as mem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_memory(tmp_path, monkeypatch):
    """Redirect all memory paths to tmp_path so tests never touch production files."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    db_file = db_dir / "test_chat.db"
    mem_file = tmp_path / ".kernel_evolving_memory.json"

    monkeypatch.setattr(mem, "MEMORY_FILE", mem_file)
    monkeypatch.setattr(mem, "DB_DIR", db_dir)
    monkeypatch.setattr(mem, "DB_FILE", db_file)
    # Reset session ID so tests don't share a cached one
    monkeypatch.setattr(mem, "_SESSION_ID", None)
    yield


# ---------------------------------------------------------------------------
# _sanitise() tests
# ---------------------------------------------------------------------------

class TestSanitise:

    def test_strips_model_server_error_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "[model_server error] CUDA OOM"},
        ]
        result = mem._sanitise(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_strips_vram_guard_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "tell me"},
            {"role": "assistant", "content": "VRAM guard: skipping inference"},
        ]
        result = mem._sanitise(msgs)
        assert all(m["role"] != "assistant" or "VRAM guard:" not in m["content"] for m in result)
        # The broken assistant turn should be removed
        assert not any("VRAM guard:" in m.get("content", "") for m in result)

    def test_strips_hf_generate_error_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "some prefix (HF generate error: timeout)"},
        ]
        result = mem._sanitise(msgs)
        assert not any("(HF generate error:" in m.get("content", "") for m in result)

    def test_strips_model_client_error_assistant_turns(self):
        msgs = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "[model_client error] connection refused"},
        ]
        result = mem._sanitise(msgs)
        assert not any("[model_client error]" in m.get("content", "") for m in result)

    def test_collapses_consecutive_user_turns_keeps_last(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = mem._sanitise(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "third"

    def test_collapses_multiple_runs_of_consecutive_user_turns(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "c"},
            {"role": "user", "content": "d"},
        ]
        result = mem._sanitise(msgs)
        # Should keep "b", "reply", "d"
        assert len(result) == 3
        assert result[0]["content"] == "b"
        assert result[1]["role"] == "assistant"
        assert result[2]["content"] == "d"

    def test_leaves_well_formed_pairs_untouched(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ]
        result = mem._sanitise(msgs)
        assert result == msgs

    def test_empty_list_returns_empty(self):
        assert mem._sanitise([]) == []


# ---------------------------------------------------------------------------
# load() + save() round-trip with chat_id isolation
# ---------------------------------------------------------------------------

class TestLoadSaveRoundTrip:

    def test_round_trip_basic(self):
        msgs = [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ]
        mem.save(msgs, chat_id="chat_a")
        loaded = mem.load(chat_id="chat_a")
        assert loaded == msgs

    def test_two_chat_ids_do_not_bleed(self):
        msgs_a = [{"role": "user", "content": "chat A message"}]
        msgs_b = [{"role": "user", "content": "chat B message"}]

        mem.save(msgs_a, chat_id="chat_a")
        mem.save(msgs_b, chat_id="chat_b")

        loaded_a = mem.load(chat_id="chat_a")
        loaded_b = mem.load(chat_id="chat_b")

        assert loaded_a == msgs_a
        assert loaded_b == msgs_b
        # Ensure no cross-contamination
        assert not any("chat B" in m["content"] for m in loaded_a)
        assert not any("chat A" in m["content"] for m in loaded_b)

    def test_load_empty_when_no_data(self):
        result = mem.load(chat_id="nonexistent_chat")
        assert result == []


# ---------------------------------------------------------------------------
# clear_chat() — wipes only the target chat_id
# ---------------------------------------------------------------------------

class TestClearChat:

    def test_clear_chat_wipes_json_window(self, tmp_path):
        msgs = [{"role": "user", "content": "test"}]
        mem.save(msgs, chat_id="chat_x")

        # Verify file exists
        mem_file = mem._chat_memory_file("chat_x")
        assert mem_file.exists()

        mem.clear_chat("chat_x")
        assert not mem_file.exists()

    def test_clear_chat_wipes_sqlite_rows_for_target(self):
        msgs = [{"role": "user", "content": "for chat_x"}]
        mem.save(msgs, chat_id="chat_x")

        mem.clear_chat("chat_x")

        # SQLite rows for chat_x should be gone
        import sqlite3
        conn = sqlite3.connect(mem.DB_FILE)
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE bot=? AND session_id=?",
            (mem.BOT_NAME, "chat_x")
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_clear_chat_leaves_other_chat_intact(self):
        msgs_x = [{"role": "user", "content": "x msg"}]
        msgs_y = [{"role": "user", "content": "y msg"}]

        mem.save(msgs_x, chat_id="chat_x")
        mem.save(msgs_y, chat_id="chat_y")

        mem.clear_chat("chat_x")

        # chat_y JSON window should still exist and be loadable
        loaded_y = mem.load(chat_id="chat_y")
        assert loaded_y == msgs_y

        # chat_y SQLite rows should still exist
        import sqlite3
        conn = sqlite3.connect(mem.DB_FILE)
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE bot=? AND session_id=?",
            (mem.BOT_NAME, "chat_y")
        ).fetchone()[0]
        conn.close()
        assert count > 0


# ---------------------------------------------------------------------------
# clear_all() — wipes everything
# ---------------------------------------------------------------------------

class TestClearAll:

    def test_clear_all_wipes_all_json_windows(self):
        for cid in ("c1", "c2", "c3"):
            mem.save([{"role": "user", "content": f"msg {cid}"}], chat_id=cid)
            assert mem._chat_memory_file(cid).exists()

        mem.clear_all()

        for cid in ("c1", "c2", "c3"):
            assert not mem._chat_memory_file(cid).exists()

    def test_clear_all_wipes_all_sqlite_rows(self):
        for cid in ("c1", "c2"):
            mem.save([{"role": "user", "content": f"msg {cid}"}], chat_id=cid)

        mem.clear_all()

        import sqlite3
        conn = sqlite3.connect(mem.DB_FILE)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE bot=?", (mem.BOT_NAME,)
            ).fetchone()[0]
        except Exception:
            count = 0  # table may not exist after wipe — that's fine
        conn.close()
        assert count == 0


# ---------------------------------------------------------------------------
# _sanitise() — user-turn pollution stripping (2026-05-26 fix)
# ---------------------------------------------------------------------------

class TestSanitiseUserPollution:

    def test_user_turn_truncated_at_step_output_marker(self):
        """User content after '[Previous step output]:' is stripped."""
        msgs = [
            {"role": "user", "content": "Save to /tmp/x.txt\n[Previous step output]: (no results)"},
            {"role": "assistant", "content": "Done."},
        ]
        result = mem._sanitise(msgs)
        assert result[0]["content"] == "Save to /tmp/x.txt"

    def test_fully_polluted_user_turn_is_dropped(self):
        """User turn that consists only of pollution markers is dropped entirely."""
        msgs = [
            {"role": "user", "content": "[Previous attempt failed. Critic feedback: try again"},
            {"role": "assistant", "content": "OK"},
        ]
        result = mem._sanitise(msgs)
        # The user turn should be gone; only the assistant turn remains (or empty list)
        user_turns = [m for m in result if m.get("role") == "user"]
        assert len(user_turns) == 0

    def test_polluted_user_followed_by_clean_user_keeps_clean(self):
        """Polluted user turn dropped → consecutive-user-collapse keeps subsequent clean turn."""
        msgs = [
            {"role": "user", "content": "[Previous step output]: failed"},
            {"role": "user", "content": "Please list files"},
            {"role": "assistant", "content": "file1.txt"},
        ]
        result = mem._sanitise(msgs)
        user_turns = [m for m in result if m.get("role") == "user"]
        assert len(user_turns) == 1
        assert user_turns[0]["content"] == "Please list files"

    def test_assistant_error_filter_still_works(self):
        """Existing assistant error filtering is not regressed."""
        from core.memory.memory import _ERROR_MARKERS
        err_marker = _ERROR_MARKERS[0]
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": f"{err_marker} something crashed"},
            {"role": "user", "content": "retry"},
            {"role": "assistant", "content": "all good"},
        ]
        result = mem._sanitise(msgs)
        assistant_turns = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_turns) == 1
        assert assistant_turns[0]["content"] == "all good"
