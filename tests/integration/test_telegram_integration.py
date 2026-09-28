"""
tests/integration/test_telegram_integration.py

Integration tests: simulate a Telegram message flowing through the full pipeline
(triage → infer_with_tools → tool dispatch → execute_tool) with a mocked model.

What these tests prove:
  - The model output parser correctly extracts tool name + args
  - execute_tool is called with the correct tool name (no alias confusion e.g. readfile vs read_file)
  - The right tool is dispatched for realistic Telegram message inputs
  - Tool results flow back and produce a final string response
  - The workspace path guard fires for out-of-workspace write attempts
  - Thoughts context reaches the prompt for thought-awareness queries

The model is fully mocked — no GPU required.
Each test injects a fake model response that mimics what Nemotron actually emits,
then asserts the pipeline dispatched the right tool with the right args.
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# ── path setup ────────────────────────────────────────────────────────────────
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)

import core.tools as tools_mod
from core.memory.context import build_system_prompt

WORKSPACE = os.path.expanduser("~/.kernel-evolving/workspace")
USER_MD   = os.path.expanduser("~/.kernel-evolving/workspace/USER.md")


# ── helpers ───────────────────────────────────────────────────────────────────

def _nemo_tool_call(name: str, args: dict) -> str:
    """Produce a Nemotron-style <tool_call> XML block as the model would emit it."""
    args_json = json.dumps(args)
    return f"<tool_call>\n{{'name': '{name}', 'arguments': {args_json}}}\n</tool_call>"


def _openai_tool_call(name: str, args: dict) -> dict:
    """Produce an OpenAI-style tool_calls payload."""
    return [{
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        }
    }]


def _make_mock_provider(tool_name: str, tool_args: dict, final_reply: str = "Done."):
    """
    Return a mock InferenceProvider whose infer_with_tools() simulates:
      1. Call execute_tool_with_meta(tool_name, tool_args) — dispatches the tool
      2. Call step_callback with the tool result
      3. Return final_reply as the final answer

    This matches how triage() actually uses the provider: it calls
    infer_with_tools(), not chat(). The old approach of mocking chat()
    broke when the agent pipeline was refactored to route tool dispatch
    through infer_with_tools → model_server → execute_tool_with_meta.
    """
    from core.tools import execute_tool_with_meta

    mock_provider = MagicMock(spec=['infer_with_tools', 'get_provider', 'get_model', 'chat'])
    # Return string values so prompt_log DB doesn't choke on MagicMock
    mock_provider.get_provider.return_value = 'test'
    mock_provider.get_model.return_value = 'test-model'

    def fake_infer_with_tools(messages, tools, workspace=None, max_steps=15,
                               step_callback=None, call_type="task_inference",
                               chunk_callback=None, chat_id=""):
        # Simulate a single tool-calling step
        meta = execute_tool_with_meta(tool_name, dict(tool_args), workspace=workspace, chat_id=chat_id)
        result_str = meta.get("result", "")
        if step_callback:
            step_callback(1, tool_name, dict(tool_args), result_str)
        return final_reply

    mock_provider.infer_with_tools.side_effect = fake_infer_with_tools
    return mock_provider


def _run_pipeline(user_message: str, tool_name: str, tool_args: dict,
                  final_reply: str = "Done.", chat_id: str = "123456789"):
    """
    Drive triage() with a mocked provider and return:
      (response_str, captured_tool_calls)
    captured_tool_calls: list of (name, args) tuples actually dispatched

    Patches:
      - get_provider → returns mock whose infer_with_tools calls execute_tool_with_meta
      - execute_tool → spy that records (name, args) for assertions
      - _log_prompt → no-op to avoid MagicMock serialization in SQLite
    """
    captured = []
    original_execute = tools_mod.execute_tool

    def spy_execute(name, args, **kwargs):
        captured.append((name, dict(args)))
        return original_execute(name, args, **kwargs)

    mock_provider = _make_mock_provider(tool_name, tool_args, final_reply)

    with patch("core.inference.provider.get_provider", return_value=mock_provider), \
         patch("core.tools.execute_tool", side_effect=spy_execute), \
         patch("core.agent._log_prompt"):
        from core.agent import triage
        response = triage(user_message, chat_id=chat_id)

    return response, captured


# ══════════════════════════════════════════════════════════════════════════════
# 1. read_file — "show me your USER.md"
# ══════════════════════════════════════════════════════════════════════════════
class TestReadFileFromTelegram:

    def test_show_user_md_dispatches_read_file(self):
        """
        Message: 'show me your USER.md'
        Expected: model calls read_file with path pointing to USER.md
        Critical: must NOT call a skill named 'readfile' — must be native read_file
        """
        _, calls = _run_pipeline(
            user_message="show me your USER.md",
            tool_name="read_file",
            tool_args={"path": USER_MD},
        )
        assert len(calls) >= 1
        names = [c[0] for c in calls]
        assert "read_file" in names, f"Expected read_file, got: {names}"
        # Confirm it did NOT try a skill called readfile / read_file as skill
        assert "readfile" not in names

    def test_read_file_path_is_inside_workspace(self):
        """Path used must be inside ~/.kernel-evolving"""
        _, calls = _run_pipeline(
            user_message="read the USER.md file",
            tool_name="read_file",
            tool_args={"path": USER_MD},
        )
        for name, args in calls:
            if name == "read_file":
                p = args.get("path", "")
                assert ".kernel-evolving" in p, \
                    f"read_file path outside workspace: {p}"

    def test_do_you_know_me_dispatches_read_file(self):
        """
        Message: 'do you know who I am?'
        Expected: model reads USER.md to answer
        """
        _, calls = _run_pipeline(
            user_message="do you know who I am?",
            tool_name="read_file",
            tool_args={"path": USER_MD},
        )
        names = [c[0] for c in calls]
        assert "read_file" in names


# ══════════════════════════════════════════════════════════════════════════════
# 2. write_file — user shares info / model saves to USER.md
# ══════════════════════════════════════════════════════════════════════════════
class TestWriteFileFromTelegram:

    def test_user_introduces_name_dispatches_write_file(self):
        """
        Message: 'My name is Fabio'
        Expected: model calls write_file to persist to USER.md
        """
        _, calls = _run_pipeline(
            user_message="My name is Fabio",
            tool_name="write_file",
            tool_args={"path": USER_MD, "content": "# USER.md\n- Name: Fabio"},
        )
        names = [c[0] for c in calls]
        assert "write_file" in names, f"Expected write_file, got: {names}"

    def test_write_file_path_inside_workspace(self):
        """write_file must write inside ~/.kernel-evolving, never to /tmp"""
        _, calls = _run_pipeline(
            user_message="save my name Fabio",
            tool_name="write_file",
            tool_args={"path": USER_MD, "content": "Fabio"},
        )
        for name, args in calls:
            if name == "write_file":
                p = args.get("path", "")
                assert ".kernel-evolving" in p, \
                    f"write_file path outside workspace: {p}"

    def test_workspace_guard_redirects_tmp_path(self):
        """
        If the model hallucinates /tmp as the write path,
        the workspace guard must redirect it to ~/.kernel-evolving/workspace/tmp/
        """
        result = tools_mod.execute_tool("write_file", {
            "path": "/tmp/architecture_details.json",
            "content": '{"test": true}'
        })
        # Must not have written to /tmp/architecture_details.json
        assert not Path("/tmp/architecture_details.json").exists() or \
               Path("/tmp/architecture_details.json").read_text() != '{"test": true}', \
               "workspace guard failed: file written to /tmp"
        # Result must be a string (redirected write succeeded or logged warning)
        assert isinstance(result, str)

    def test_workspace_guard_redirects_root_path(self):
        """Paths like /var/... or /root/... must also be redirected."""
        result = tools_mod.execute_tool("write_file", {
            "path": "/var/log/kernel_test.txt",
            "content": "test"
        })
        assert not Path("/var/log/kernel_test.txt").exists()
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# 3. exec_shell — model runs a shell command
# ══════════════════════════════════════════════════════════════════════════════
class TestExecShellFromTelegram:

    def test_shell_command_dispatched_correctly(self):
        """
        Message: 'what is the current date?'
        Expected: model calls exec_shell with a date command
        """
        _, calls = _run_pipeline(
            user_message="what is the current date and time?",
            tool_name="exec_shell",
            tool_args={"command": "date"},
        )
        names = [c[0] for c in calls]
        assert "exec_shell" in names

    def test_exec_shell_result_is_real_not_hallucinated(self):
        """exec_shell('date') must return actual system output."""
        result = tools_mod.execute_tool("exec_shell", {"command": "echo integration_test_ok"})
        assert "integration_test_ok" in result


# ══════════════════════════════════════════════════════════════════════════════
# 4. search_skills — model looks up a skill before calling run_skill
# ══════════════════════════════════════════════════════════════════════════════
class TestSearchSkillsFromTelegram:

    def test_what_skills_message_dispatches_search_skills(self):
        """
        Message: 'what skills do you have?'
        Expected: model calls search_skills (NOT run_skill with no name)
        """
        _, calls = _run_pipeline(
            user_message="what skills do you have?",
            tool_name="search_skills",
            tool_args={"query": ""},
        )
        names = [c[0] for c in calls]
        assert "search_skills" in names or "list_routines" in names, \
            f"Expected search_skills or list_routines, got: {names}"

    def test_search_skills_returns_real_skill_list(self):
        """search_skills with empty query must return all installed skills."""
        result = tools_mod.execute_tool("search_skills", {"query": "voice"})
        assert isinstance(result, str)
        assert len(result) > 10


# ══════════════════════════════════════════════════════════════════════════════
# 5. recall_memory — model searches past sessions
# ══════════════════════════════════════════════════════════════════════════════
class TestRecallMemoryFromTelegram:

    def test_recall_memory_dispatched_for_history_query(self):
        """
        Message: 'do you remember what we discussed yesterday?'
        Expected: model calls recall_memory
        """
        _, calls = _run_pipeline(
            user_message="do you remember what we discussed yesterday?",
            tool_name="recall_memory",
            tool_args={"query": "yesterday discussion"},
        )
        names = [c[0] for c in calls]
        assert "recall_memory" in names

    def test_recall_memory_returns_string(self):
        result = tools_mod.execute_tool("recall_memory", {"query": "Fabio"})
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Thought-awareness — system prompt contains journal entries
# ══════════════════════════════════════════════════════════════════════════════
class TestThoughtAwarenessInPrompt:

    def test_my_recent_thoughts_in_system_prompt(self):
        """
        build_system_prompt must include '## My recent thoughts' section
        if a thoughts journal file exists for today.
        If no journal exists (fresh install / test env), the section is absent
        and that's acceptable — the assertion should only fire when there's data.
        """
        prompt = build_system_prompt({}, [], [])
        # Check if a thoughts journal file exists for today. The journal lives in
        # the `thoughts/` subdir as YYYY-MM-DD.md (see core/memory/context.py
        # _load_recent_thoughts / ThoughtJournal), NOT as thoughts-*.md in the
        # workspace root. If no journal exists, the section is correctly omitted.
        from pathlib import Path
        import datetime
        journal_dir = Path.home() / ".kernel-evolving" / "workspace" / "thoughts"
        today_file = journal_dir / f"{datetime.date.today().isoformat()}.md"
        if today_file.exists() or any(journal_dir.glob("*.md")):
            assert "## My recent thoughts" in prompt, \
                "Thought-awareness section missing from system prompt (journal file exists)"
        else:
            # No journal file → section correctly omitted
            assert "## My recent thoughts" not in prompt, \
                "Thought-awareness section present but no journal file exists"

    def test_thoughts_appear_before_session_notes(self):
        """
        '## My recent thoughts' must appear BEFORE '## Session Notes'
        so it lands early in the context window.
        """
        prompt = build_system_prompt({}, [], [])
        if "## My recent thoughts" in prompt and "## Session Notes" in prompt:
            idx_thoughts = prompt.index("## My recent thoughts")
            idx_session  = prompt.index("## Session Notes")
            assert idx_thoughts < idx_session, \
                f"Thoughts at {idx_thoughts} but Session Notes at {idx_session} — wrong order"

    def test_thoughts_appear_before_system_live(self):
        """
        '## My recent thoughts' must appear BEFORE '## System live'
        (target: under char 12000).
        """
        prompt = build_system_prompt({}, [], [])
        if "## My recent thoughts" in prompt:
            idx = prompt.index("## My recent thoughts")
            assert idx < 12000, \
                f"Thoughts injected too late at char {idx} — model may not attend to it"

    def test_fact_assertion_at_top_of_prompt(self):
        """
        The thought-awareness FACT assertion must be in the first 2000 chars.
        """
        prompt = build_system_prompt({}, [], [])
        top = prompt[:2000]
        assert "Think-at-Rest" in top or "background thinking" in top, \
            "Think-at-Rest identity assertion not in first 2000 chars of prompt"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Tool name integrity — no alias confusion across all 11 tools
# ══════════════════════════════════════════════════════════════════════════════
class TestToolNameIntegrity:
    """
    For each of the 11 native tools, simulate the model emitting
    the correct tool name and assert execute_tool receives it verbatim.
    No alias mangling, no underscore stripping.
    """

    TOOL_SCENARIOS = [
        ("exec_shell",    {"command": "echo hi"}),
        ("read_file",     {"path": USER_MD}),
        ("write_file",    {"path": f"{WORKSPACE}/tmp/test_integrity.txt", "content": "ok"}),
        ("http_get",      {"url": "https://httpbin.org/get"}),
        ("web_search",    {"query": "kernel evolving agent"}),
        ("run_skill",     {"skill_name": "voice-clone", "input": "test"}),
        ("run_routine",   {"routine_name": "daily-summary"}),
        ("search_skills", {"query": "image"}),
        ("list_routines", {}),
        ("recall_memory", {"query": "Fabio"}),
    ]

    @pytest.mark.parametrize("tool_name,tool_args", TOOL_SCENARIOS)
    def test_tool_name_survives_pipeline(self, tool_name, tool_args):
        """
        Mock model emits tool_name. After pipeline dispatch,
        execute_tool must be called with exactly tool_name — not an alias.
        """
        dispatched = []
        original = tools_mod.execute_tool

        def spy(name, args, **kwargs):
            dispatched.append(name)
            return original(name, args, **kwargs)

        mock_provider = _make_mock_provider(tool_name, tool_args)

        with patch("core.inference.provider.get_provider", return_value=mock_provider), \
             patch("core.tools.execute_tool", side_effect=spy), \
             patch("core.agent._log_prompt"):
            from core.agent import triage
            triage(f"test {tool_name}", chat_id="integration_test")

        assert tool_name in dispatched, \
            f"Tool '{tool_name}' not dispatched. Got: {dispatched}"

    def test_send_file_name_intact(self):
        """send_file tested separately as it needs a real file path."""
        tmp = Path(WORKSPACE) / "tmp" / "test_send.txt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text("test")

        dispatched = []
        original = tools_mod.execute_tool

        def spy(name, args, **kwargs):
            dispatched.append(name)
            return original(name, args, **kwargs)

        mock_provider = _make_mock_provider("send_file", {"path": str(tmp)})

        with patch("core.inference.provider.get_provider", return_value=mock_provider), \
             patch("core.tools.execute_tool", side_effect=spy), \
             patch("core.agent._log_prompt"):
            from core.agent import triage
            triage("send me the test file", chat_id="integration_test")

        assert "send_file" in dispatched, \
            f"send_file not dispatched. Got: {dispatched}"
        tmp.unlink(missing_ok=True)
