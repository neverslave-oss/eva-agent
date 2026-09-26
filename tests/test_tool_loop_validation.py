"""
test_tool_loop_validation.py — Unit tests for the tool-loop validation stop.

Fabio's fix: the tool loop previously stopped on a blind 3-strike repetition
counter, so T2 (read_file) and T4 (ps) kept re-calling the same tool. The new
`_tool_result_answers_query` check decides "did this successful tool result
already answer the single look-up request?" and, if so, stops the loop.

Pure unit tests — no sockets, no real tool execution. Only exercises the
pure helper function in core/inference/model_server.py.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.inference.model_server as ms


# ---------------------------------------------------------------------------
# _tool_result_answers_query
# ---------------------------------------------------------------------------

class TestToolResultAnswersQuery:

    def test_read_file_result_answers_lookup(self):
        # T2 shape: reading USER.md returns the content the user asked about.
        assert ms._tool_result_answers_query(
            "read the user file and tell me who you are",
            "read_file",
            "Name: Fabio. Cats: 10. Diet: vegan.",
        ) is True

    def test_ps_result_answers_process_lookup(self):
        # T4 shape: ps result gives PIDs the user asked about.
        assert ms._tool_result_answers_query(
            "show me running model_server processes and their pids",
            "exec_shell",
            "1234 model_server  0.1 1.2   ...\n5678 model_server ...",
        ) is True

    def test_df_result_answers_disk_lookup(self):
        assert ms._tool_result_answers_query(
            "how full is the disk at /",
            "exec_shell",
            "Filesystem  Size  Used  Avail\n/dev/sda1  1.0T  148G  853G  15% /",
        ) is True

    def test_error_result_does_not_stop(self):
        # Error banner must never be treated as an answer.
        assert ms._tool_result_answers_query(
            "read the user file",
            "read_file",
            "(error: file not found: /nope)",
        ) is False

    def test_empty_result_does_not_stop(self):
        assert ms._tool_result_answers_query(
            "list files", "ls", ""
        ) is False
        assert ms._tool_result_answers_query(
            "list files", "ls", "   "
        ) is False

    def test_non_info_tool_does_not_stop(self):
        # Write-style tool with a big result must not trigger validation-stop.
        assert ms._tool_result_answers_query(
            "write this file for me", "write_file", "very long content " * 50
        ) is False

    def test_query_without_lookup_marker_does_not_stop(self):
        # A complex multi-step task that happens to run read_file first must not
        # be truncated (conservative by design). 'analyze' is not a marker.
        assert ms._tool_result_answers_query(
            "analyze performance and optimize it",
            "read_file",
            "some file contents here that are not short",
        ) is False

    def test_tool_name_case_and_short_result_boundary(self):
        # Very short substantives (>=4 chars) can still count for e.g. a PID.
        assert ms._tool_result_answers_query(
            "what is the pid", "exec_shell", "1234"
        ) is True
        # <4 chars never counts.
        assert ms._tool_result_answers_query(
            "what is the pid", "exec_shell", "12"
        ) is False
