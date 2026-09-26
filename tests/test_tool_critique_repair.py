"""
test_tool_critique_repair.py — Tests for the tool-call critique/repair layer (T1/T2).

Covers:
  - _build_schema_repair_hint: produces a schema-aware hint naming the missing
    key and its schema description, plus the original query for value derivation.
  - The T1/T2 integration is exercised via validate_tool_args (see
    test_tool_arg_utils.py); this file tests the hint-builder on top of it.

Rules (per AGENTS.md):
  - Unit test, no model, no socket, no HTTP, no DB writes.
"""

import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from core.inference.model_server import _build_schema_repair_hint, _parse_function_eq_xml
from core.tool_arg_utils import validate_tool_args


class TestBuildSchemaRepairHint:
    def test_names_missing_key_with_schema_desc(self):
        problems = validate_tool_args("write_file", {"path": "/tmp/a"})
        hint = _build_schema_repair_hint("write_file", {"path": "/tmp/a"}, problems, "write hello to /tmp/a")
        assert "write_file" in hint
        # names the missing key
        assert "content" in hint
        assert "missing required key 'content'" in hint

    def test_includes_original_query_for_value_derivation(self):
        problems = validate_tool_args("write_file", {"path": ""})
        hint = _build_schema_repair_hint("write_file", {"path": ""}, problems, "write hello world to out.txt")
        assert "out.txt" in hint  # query echoed so the model can derive the value

    def test_lists_each_problem(self):
        problems = validate_tool_args("write_file", {})
        hint = _build_schema_repair_hint("write_file", {}, problems, "task")
        assert "path" in hint
        assert "content" in hint

    def test_type_problem_reported(self):
        # browser_use max_steps="15" (string for integer) -> type problem
        problems = validate_tool_args("browser_use", {"task": "go", "max_steps": "15"})
        assert any("max_steps" in p for p in problems)
        hint = _build_schema_repair_hint("browser_use", {"task": "go", "max_steps": "15"}, problems, "do it")
        assert "max_steps" in hint

    def test_ends_with_retry_instruction(self):
        problems = validate_tool_args("exec_shell", {})
        hint = _build_schema_repair_hint("exec_shell", {}, problems, "run a command")
        assert "Retry the tool" in hint


class TestT1GateParsesThenValidates:
    """End-to-end of the T1 flow: parse native XML -> validate -> catches missing."""

    def test_parsed_valid_call_is_clean(self):
        raw = '<function=write_file><parameter name="path">/tmp/a</parameter><parameter name="content">hi</parameter></function>'
        calls = _parse_function_eq_xml(raw)
        args = calls[0]["function"]["arguments"]
        assert validate_tool_args("write_file", args) == []

    def test_parsed_partial_call_is_flagged(self):
        raw = '<function=write_file><parameter name="path">/tmp/a</parameter></function>'
        calls = _parse_function_eq_xml(raw)
        args = calls[0]["function"]["arguments"]
        problems = validate_tool_args("write_file", args)
        assert any("content" in p for p in problems)
