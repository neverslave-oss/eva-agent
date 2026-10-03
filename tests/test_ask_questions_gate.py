"""
test_ask_questions_gate.py — Tests for the ask_questions gate.

Tests the ask_questions_gate module in isolation (no Telegram bot needed):
  - option validation / error handling
  - resolve_question callback mechanism
  - timeout behavior
  - tools.py integration (TOOLS schema contains ask_questions; dispatcher returns
    a clean error when no chat_id context is available)
"""

import os
import sys
import threading
import pytest

# Path setup (mirror test_auth_gate.py)
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)

from core.ask_questions_gate import (  # noqa: E402
    ask_question,
    resolve_question,
    _pending,
    CALLBACK_PREFIX,
)


class TestAskQuestionNoOptions:
    def test_no_options_returns_error(self):
        result = ask_question("12345", "Pick one:", [])
        assert "error" in result

    def test_empty_strings_filtered(self):
        result = ask_question("12345", "Pick one:", ["", "   "])
        assert "error" in result


class TestAskQuestionGatewayUnavailable:
    def test_returns_error_when_no_bot(self):
        # In the test env the telegram_bot import path may not resolve/start; the
        # gate must catch it and return an error rather than hang.
        result = ask_question("12345", "Question", ["A", "B"], timeout=1)
        assert "error" in result or "timeout" in result


class TestResolve:
    def test_resolve_sets_result(self):
        event = threading.Event()
        _pending["rid1"] = {"event": event, "result": None}
        resolve_question("rid1", 2)
        assert _pending["rid1"]["result"] == 2
        assert event.is_set()
        _pending.pop("rid1", None)

    def test_resolve_unknown_id_noop(self):
        _pending.clear()
        resolve_question("nonexistent", 0)  # should not raise


class TestToolsIntegration:
    def test_ask_questions_in_tools_schema(self):
        import core.tools as tools_mod
        names = [t["function"]["name"] for t in tools_mod.TOOLS]
        assert "ask_questions" in names
        schema = next(t["function"] for t in tools_mod.TOOLS
                      if t["function"]["name"] == "ask_questions")
        props = schema["parameters"]["properties"]
        assert "question" in props
        assert "options" in props
        assert set(schema["parameters"]["required"]) == {"question", "options"}

    def test_ask_questions_dispatch_requires_chat_id(self):
        import core.tools as tools_mod
        tools_mod._current_chat_id = ""
        result = tools_mod.execute_tool(
            "ask_questions",
            {"question": "Q", "options": ["A", "B"]},
        )
        # No chat context -> clean error string (no hang, no crash)
        assert "error" in result.lower() or "chat_id" in result

    def test_callback_prefix(self):
        assert CALLBACK_PREFIX == "aq_"
