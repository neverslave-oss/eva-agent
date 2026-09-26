"""
test_tool_arg_utils.py — Tests for tool-arg schema validation (T1 in the critique design).

validate_tool_args checks args against the REAL tool schema: every required key
present, non-empty, and type-correct. Used by the harness to decide whether a
tool call can proceed or needs a repair-critique (T2).

Rules (per AGENTS.md):
  - Unit test, no model, no socket, no HTTP, no DB writes.
"""

import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from core.tool_arg_utils import get_tool_schema, validate_tool_args


class TestGetToolSchema:
    def test_known_tool_returns_schema(self):
        s = get_tool_schema("write_file")
        assert s and s.get("required") == ["path", "content"]

    def test_exec_shell_required_command(self):
        s = get_tool_schema("exec_shell")
        assert s and s.get("required") == ["command"]

    def test_unknown_tool_returns_none(self):
        assert get_tool_schema("no_such_tool") is None


class TestValidateToolArgs:
    def test_write_file_valid(self):
        assert validate_tool_args("write_file", {"path": "/tmp/a", "content": "x"}) == []

    def test_write_file_missing_required(self):
        problems = validate_tool_args("write_file", {"path": "/tmp/a"})
        assert any("content" in p for p in problems)

    def test_write_file_missing_all(self):
        problems = validate_tool_args("write_file", {})
        assert any("path" in p for p in problems)
        assert any("content" in p for p in problems)

    def test_write_file_empty_value_is_missing(self):
        problems = validate_tool_args("write_file", {"path": "", "content": ""})
        assert any("path" in p for p in problems)

    def test_exec_shell_valid(self):
        assert validate_tool_args("exec_shell", {"command": "ls -la", "timeout": 30}) == []

    def test_exec_shell_missing_command(self):
        problems = validate_tool_args("exec_shell", {"timeout": 30})
        assert any("command" in p for p in problems)

    def test_missing_args_is_problem(self):
        problems = validate_tool_args("exec_shell", None)
        assert problems  # non-empty = not valid

    def test_unknown_tool_fails_open(self):
        # Unknown tool -> can't validate -> no problems (fail open per design)
        assert validate_tool_args("no_such_tool", {}) == []

    def test_optional_keys_not_required(self):
        # timeout is optional for exec_shell; missing it is fine
        assert validate_tool_args("exec_shell", {"command": "ls"}) == []

    def test_required_empty_for_schema_valid(self):
        # list_routines has no required keys
        assert validate_tool_args("list_routines", {}) == []

    def test_ask_questions_options_must_be_array(self):
        problems = validate_tool_args("ask_questions", {"question": "q?", "options": "not-a-list"})
        assert any("options" in p for p in problems)
        assert validate_tool_args("ask_questions", {"question": "q?", "options": ["a", "b"]}) == []

    def test_integer_type_check(self):
        # browser_use max_steps is integer; a string should be flagged
        assert validate_tool_args("browser_use", {"task": "do x", "max_steps": "15"}) != []
        assert validate_tool_args("browser_use", {"task": "do x", "max_steps": 15}) == []
