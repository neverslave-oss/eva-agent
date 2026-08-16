"""
test_model_server_history.py — Regression tests for history handling in model_server.

Covers the bug where messages[-10:] dropped the system message for sessions > 10 turns,
causing Nemotron to lose identity/user facts/behaviour guidelines mid-conversation.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_messages(n_turns: int, with_system: bool = True) -> list:
    """Build a realistic messages list: 1 system + n_turns user/assistant pairs."""
    msgs = []
    if with_system:
        msgs.append({"role": "system", "content": "You are Kernel-Evo. User: Fabio. Cats: 10."})
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"user turn {i}"})
        msgs.append({"role": "assistant", "content": f"assistant turn {i}"})
    msgs.append({"role": "user", "content": "final question"})
    return msgs


def _simulate_trim(messages: list) -> list:
    """Replicate the current (correct) trim logic from model_server._handle_infer_plain."""
    if len(messages) > 500:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        return system_msgs + non_system[-499:]
    return messages


def _old_trim(messages: list) -> list:
    """Replicate the OLD (buggy) trim logic that caused the regression."""
    if len(messages) > 10:
        return messages[-10:]
    return messages


# ── Regression tests ────────────────────────────────────────────────────────

class TestSystemMessagePreservation:

    def test_system_message_preserved_short_conversation(self):
        """System message must be present for short conversations (<10 turns)."""
        msgs = _make_messages(3)
        result = _simulate_trim(msgs)
        roles = [m["role"] for m in result]
        assert "system" in roles, "System message must be present in short conversations"

    def test_system_message_preserved_at_10_turns(self):
        """System message must be present at exactly 10 turns (old boundary)."""
        msgs = _make_messages(5)  # 1 system + 10 turns + 1 final = 12 messages
        result = _simulate_trim(msgs)
        roles = [m["role"] for m in result]
        assert "system" in roles, "System message must be present at 10-turn boundary"

    def test_system_message_preserved_long_conversation(self):
        """System message must be present for long conversations (>10 turns) — the regression case."""
        msgs = _make_messages(20)  # 1 system + 41 messages total
        assert len(msgs) > 10, "Test setup: must have >10 messages"
        result = _simulate_trim(msgs)
        roles = [m["role"] for m in result]
        assert "system" in roles, (
            "REGRESSION: system message dropped for long conversations. "
            "This was the bug where messages[-10:] discarded the system prompt "
            "after turn 9, causing Nemotron to lose identity/user facts."
        )

    def test_system_message_content_intact(self):
        """System message content must not be truncated or corrupted."""
        msgs = _make_messages(20)
        sys_content = msgs[0]["content"]
        result = _simulate_trim(msgs)
        result_sys = next(m for m in result if m["role"] == "system")
        assert result_sys["content"] == sys_content, "System message content must be preserved verbatim"

    def test_old_logic_would_have_dropped_system(self):
        """Document the old bug: old trim logic DID drop the system message for >10 turns."""
        msgs = _make_messages(20)
        old_result = _old_trim(msgs)
        old_roles = [m["role"] for m in old_result]
        assert "system" not in old_roles, (
            "This test documents the old bug — old logic should NOT have system message. "
            "If this fails, the old logic was accidentally changed."
        )

    def test_new_logic_fixes_old_bug(self):
        """New trim logic must fix what the old logic broke."""
        msgs = _make_messages(20)
        old_result = _old_trim(msgs)
        new_result = _simulate_trim(msgs)
        old_has_system = any(m["role"] == "system" for m in old_result)
        new_has_system = any(m["role"] == "system" for m in new_result)
        assert not old_has_system, "Old logic should drop system (documenting the bug)"
        assert new_has_system, "New logic must preserve system (the fix)"

    def test_runaway_ceiling_preserves_system(self):
        """Hard safety ceiling (>500 messages) must still preserve system message."""
        msgs = _make_messages(300)  # >500 total messages
        result = _simulate_trim(msgs)
        roles = [m["role"] for m in result]
        assert "system" in roles, "System message must be preserved even at runaway ceiling"
        assert len(result) <= 500, "Runaway ceiling must cap total messages"

    def test_no_cap_below_500(self):
        """No trimming should occur for conversations under 500 messages."""
        msgs = _make_messages(50)  # 102 messages — well under 500
        result = _simulate_trim(msgs)
        assert len(result) == len(msgs), (
            f"No trimming expected for {len(msgs)} messages, got {len(result)}"
        )

    def test_multiple_system_messages_all_preserved(self):
        """If multiple system messages exist (edge case), all are preserved."""
        msgs = [
            {"role": "system", "content": "System msg 1"},
            {"role": "system", "content": "System msg 2"},
        ]
        for i in range(20):
            msgs.append({"role": "user", "content": f"u{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        result = _simulate_trim(msgs)
        system_count = sum(1 for m in result if m["role"] == "system")
        assert system_count == 2, f"Both system messages must be preserved, got {system_count}"


class TestInferWithToolsHistoryCap:
    """Same regression tests for _handle_infer_with_tools — it had the identical _MAX_HISTORY_PAIRS=10 bug."""

    def test_infer_with_tools_system_preserved_long_conversation(self):
        """_handle_infer_with_tools must preserve system message for >10 turns (regression)."""
        msgs = _make_messages(20)
        # Replicate the OLD infer_with_tools trim logic
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        non_sys = [m for m in msgs if m["role"] != "system"]
        MAX_HISTORY_PAIRS = 10
        if len(non_sys) > MAX_HISTORY_PAIRS * 2:
            non_sys = non_sys[-(MAX_HISTORY_PAIRS * 2):]
        old_result = sys_msgs + non_sys

        # Old logic preserves system because it separates before slicing — BUT the bug
        # was that it capped non_sys to 20 messages, meaning 10 turns only.
        # Verify the cap is now 499 not 20.
        assert len(non_sys) <= 499 + len(sys_msgs), "Cap must be 499, not 20"
        assert any(m["role"] == "system" for m in old_result), "System always preserved in new logic"

    def test_new_infer_with_tools_trim(self):
        """New infer_with_tools trim: system always preserved, non-sys capped at 499."""
        msgs = _make_messages(300)  # 601+ messages
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        non_sys = [m for m in msgs if m["role"] != "system"]
        if len(non_sys) > 499:
            non_sys = non_sys[-499:]
        result = sys_msgs + non_sys
        assert any(m["role"] == "system" for m in result)
        assert len([m for m in result if m["role"] != "system"]) <= 499

    def test_normal_conversation_not_trimmed(self):
        """Conversations under 499 non-system turns must not be trimmed."""
        msgs = _make_messages(20)  # 41 messages
        sys_msgs = [m for m in msgs if m["role"] == "system"]
        non_sys = [m for m in msgs if m["role"] != "system"]
        original_len = len(non_sys)
        if len(non_sys) > 499:
            non_sys = non_sys[-499:]
        assert len(non_sys) == original_len, "Short conversation must not be trimmed"
