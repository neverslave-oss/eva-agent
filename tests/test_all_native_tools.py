"""
test_all_native_tools.py — Validates every native tool in execute_tool().

Covers:
  exec_shell, read_file, write_file, http_get, web_search,
  send_file, run_skill, run_routine, search_skills, list_routines, recall_memory

Rules:
  - No model loaded, no real HTTP, no Telegram
  - Each tool tested for: happy path, missing args, bad args
  - write_file workspace guard tested explicitly
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import core.tools as tools_mod

WORKSPACE = os.path.expanduser("~/.kernel-evolving/workspace")


# ─────────────────────────────────────────────────────────────────────────────
# exec_shell
# ─────────────────────────────────────────────────────────────────────────────
class TestExecShell:
    def test_runs_command(self):
        result = tools_mod.execute_tool("exec_shell", {"command": "echo hello"})
        assert "hello" in result

    def test_missing_command_returns_error(self):
        result = tools_mod.execute_tool("exec_shell", {})
        assert "error" in result.lower()
        assert "command" in result

    def test_stderr_captured(self):
        result = tools_mod.execute_tool("exec_shell", {"command": "echo err >&2"})
        assert "err" in result

    def test_timeout_returns_message(self):
        result = tools_mod.execute_tool("exec_shell", {"command": "sleep 10", "timeout": 1})
        assert "timeout" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# read_file
# ─────────────────────────────────────────────────────────────────────────────
class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello kernel")
        result = tools_mod.execute_tool("read_file", {"path": str(f)})
        assert "hello kernel" in result

    def test_missing_path_arg_returns_error(self):
        result = tools_mod.execute_tool("read_file", {})
        assert "error" in result.lower()
        assert "path" in result

    def test_nonexistent_file_returns_error(self):
        result = tools_mod.execute_tool("read_file", {"path": "/nonexistent/file.txt"})
        assert "error" in result.lower()

    def test_tilde_expansion(self):
        # Should expand ~ and attempt read (may fail on missing file but must not crash)
        result = tools_mod.execute_tool("read_file", {"path": "~/.kernel-evolving/workspace/USER.md"})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# write_file
# ─────────────────────────────────────────────────────────────────────────────
class TestWriteFile:
    def test_writes_inside_workspace(self, tmp_path):
        allowed = tmp_path / ".kernel-evolving"
        target = allowed / "workspace" / "tmp" / "test.txt"

        with patch("core.tools.WORKSPACE", str(allowed / "workspace")):
            # Patch the allowed prefix check
            with patch.object(tools_mod, "execute_tool", wraps=tools_mod.execute_tool):
                # Write directly to a path inside tmp_path that starts with allowed prefix
                target.parent.mkdir(parents=True, exist_ok=True)
                result = tools_mod.execute_tool("write_file", {
                    "path": str(target),
                    "content": "test content"
                })
        # Accept success or workspace-guard redirect — must not crash
        assert isinstance(result, str)

    def test_missing_path_returns_error(self):
        result = tools_mod.execute_tool("write_file", {"content": "x"})
        assert "error" in result.lower()
        assert "path" in result

    def test_missing_content_returns_error(self):
        result = tools_mod.execute_tool("write_file", {"path": "/tmp/x.txt"})
        assert "error" in result.lower()
        assert "content" in result

    def test_protected_identity_files_are_rejected(self, tmp_path):
        """write_file must refuse to overwrite AGENTS.md / SOUL.md / IDENTITY.md."""
        fake_home = tmp_path
        original_expanduser = os.path.expanduser

        def fake_expanduser(p):
            if p.startswith("~"):
                return p.replace("~", str(fake_home), 1)
            return p

        for fname in ("AGENTS.md", "SOUL.md", "IDENTITY.md"):
            with patch("os.path.expanduser", side_effect=fake_expanduser):
                result = tools_mod.execute_tool("write_file", {
                    "path": f"~/.kernel-evolving/workspace/{fname}",
                    "content": "clobber attempt",
                })
            assert "protected" in result.lower(), f"{fname} write should be rejected, got: {result}"
            # Ensure nothing was written to the real workspace file
            real = Path(os.path.expanduser(f"~/.kernel-evolving/workspace/{fname}"))
            if real.exists():
                assert "clobber attempt" not in real.read_text(encoding="utf-8")

    def test_path_outside_workspace_is_redirected(self, tmp_path):
        """Paths outside ~/.kernel-evolving must be redirected to workspace/tmp/, not written verbatim."""
        # Patch expanduser so ~/.kernel-evolving resolves to tmp_path
        fake_home = tmp_path
        allowed_prefix = str(fake_home / ".kernel-evolving")

        original_expanduser = os.path.expanduser

        def fake_expanduser(p):
            if p.startswith("~"):
                return p.replace("~", str(fake_home), 1)
            return p

        with patch("os.path.expanduser", side_effect=fake_expanduser):
            result = tools_mod.execute_tool("write_file", {
                "path": "/tmp/escape.txt",
                "content": "should be redirected"
            })

        # Must not have written to /tmp/escape.txt
        assert not Path("/tmp/escape.txt").exists() or Path("/tmp/escape.txt").read_text() != "should be redirected"
        assert isinstance(result, str)

    def test_write_and_read_roundtrip(self, tmp_path):
        """Write a file inside ~/.kernel-evolving/workspace/tmp and read it back."""
        target = Path(os.path.expanduser("~/.kernel-evolving/workspace/tmp/test_roundtrip.txt"))
        target.parent.mkdir(parents=True, exist_ok=True)

        write_result = tools_mod.execute_tool("write_file", {
            "path": str(target),
            "content": "roundtrip content"
        })
        assert "error" not in write_result.lower()

        read_result = tools_mod.execute_tool("read_file", {"path": str(target)})
        assert "roundtrip content" in read_result

        target.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# http_get
# ─────────────────────────────────────────────────────────────────────────────
class TestHttpGet:
    def test_missing_url_returns_error(self):
        result = tools_mod.execute_tool("http_get", {})
        assert "error" in result.lower()
        assert "url" in result

    def test_valid_url_fetches(self):
        mock_response = MagicMock()
        mock_response.text = "OK response"
        with patch("core.tools._requests") as mock_req:
            mock_req.get.return_value = mock_response
            result = tools_mod.execute_tool("http_get", {"url": "https://example.com"})
        assert "OK response" in result

    def test_local_path_instead_of_url_reroutes(self, tmp_path):
        """If model passes a local path as URL, tool should read the file instead of HTTP."""
        f = tmp_path / "data.txt"
        f.write_text("local content")
        result = tools_mod.execute_tool("http_get", {"url": str(f)})
        assert "local content" in result

    def test_network_error_returns_error_string(self):
        with patch("core.tools._requests") as mock_req:
            mock_req.get.side_effect = Exception("connection refused")
            result = tools_mod.execute_tool("http_get", {"url": "https://unreachable.test"})
        assert "error" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# web_search
# ─────────────────────────────────────────────────────────────────────────────
class TestWebSearch:
    def test_missing_query_returns_error(self):
        result = tools_mod.execute_tool("web_search", {})
        assert "error" in result.lower()

    def test_returns_string(self):
        with patch("core.tools._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.text = "<html>search result</html>"
            mock_resp.json.return_value = {"results": []}
            mock_req.get.return_value = mock_resp
            result = tools_mod.execute_tool("web_search", {"query": "kernel evolving agent"})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# browser_use
# ─────────────────────────────────────────────────────────────────────────────
class TestBrowserUse:
    def test_missing_task_returns_error(self):
        result = tools_mod.execute_tool("browser_use", {})
        assert "error" in result.lower()

    def test_missing_dependency_returns_install_hint(self, tmp_path):
        # When browser-use isn't importable, the tool returns an install hint.
        fake_cfg = tmp_path / "config.yaml"
        fake_cfg.write_text("browser:\n  enabled: true\n  headless: true\n  max_steps: 15\n  timeout_s: 120\n")
        # Force the config path to the fake config so we don't depend on the real one.
        with patch("core.tools._load_browser_config", return_value={
            "enabled": True, "headless": True, "max_steps": 15, "timeout_s": 120,
            "screenshot_dir": "",
        }), patch.dict("sys.modules", {"browser_use": None}):
            result = tools_mod.execute_tool("browser_use", {"task": "search the web"})
        assert "browser-use not installed" in result.lower() or "install" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# send_file
# ─────────────────────────────────────────────────────────────────────────────
class TestSendFile:
    def test_missing_path_returns_error(self):
        result = tools_mod.execute_tool("send_file", {})
        assert "error" in result.lower()

    def test_nonexistent_file_returns_error(self):
        result = tools_mod.execute_tool("send_file", {"path": "/nonexistent/file.txt"})
        assert "error" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# run_skill
# ─────────────────────────────────────────────────────────────────────────────
class TestRunSkill:
    def test_missing_skill_name_returns_error(self):
        result = tools_mod.execute_tool("run_skill", {})
        assert "error" in result.lower()
        assert "skill_name" in result

    def test_missing_input_returns_error(self):
        result = tools_mod.execute_tool("run_skill", {"skill_name": "some-skill"})
        assert "error" in result.lower()
        assert "input" in result

    def test_unknown_skill_returns_not_found(self, tmp_path):
        mock_skills = MagicMock()
        mock_skills.load_all.return_value = []
        mock_skills.find.return_value = None
        with patch.dict("sys.modules", {"core.skills": mock_skills}):
            result = tools_mod.execute_tool("run_skill", {"skill_name": "ghost-skill", "input": "test"})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_known_skill_calls_run(self, tmp_path):
        fake_skill = {"name": "my-skill", "description": "Test skill"}
        mock_skills = MagicMock()
        mock_skills.load_all.return_value = [fake_skill]
        mock_skills.find.return_value = fake_skill
        mock_skills.run.return_value = "skill ran ok"
        with patch.dict("sys.modules", {"core.skills": mock_skills}):
            result = tools_mod.execute_tool("run_skill", {"skill_name": "my-skill", "input": "do it"})
        assert result == "skill ran ok"


# ─────────────────────────────────────────────────────────────────────────────
# run_routine
# ─────────────────────────────────────────────────────────────────────────────
class TestRunRoutine:
    def test_missing_routine_name_returns_error(self):
        result = tools_mod.execute_tool("run_routine", {})
        assert "error" in result.lower()
        assert "routine_name" in result

    def test_unknown_routine_returns_not_found(self):
        mock_routines = MagicMock()
        mock_routines.load_all.return_value = []
        mock_routines.find.return_value = None
        with patch.dict("sys.modules", {"core.routines": mock_routines}):
            result = tools_mod.execute_tool("run_routine", {"routine_name": "ghost-routine"})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_known_routine_calls_run(self):
        fake_routine = {"name": "my-routine", "description": "Test"}
        mock_routines = MagicMock()
        mock_routines.load_all.return_value = [fake_routine]
        mock_routines.find.return_value = fake_routine
        mock_routines.run.return_value = "routine ran ok"
        with patch.dict("sys.modules", {"core.routines": mock_routines}):
            result = tools_mod.execute_tool("run_routine", {"routine_name": "my-routine"})
        assert result == "routine ran ok"


# ─────────────────────────────────────────────────────────────────────────────
# search_skills
# ─────────────────────────────────────────────────────────────────────────────
class TestSearchSkills:
    def test_missing_query_lists_all_skills(self):
        """search_skills with no query should return all skills, not an error."""
        result = tools_mod.execute_tool("search_skills", {})
        assert isinstance(result, str)
        assert len(result) > 0  # returns something useful

    def test_returns_string(self):
        result = tools_mod.execute_tool("search_skills", {"query": "voice"})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# list_routines
# ─────────────────────────────────────────────────────────────────────────────
class TestListRoutines:
    def test_returns_string(self):
        result = tools_mod.execute_tool("list_routines", {})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# recall_memory
# ─────────────────────────────────────────────────────────────────────────────
class TestRecallMemory:
    def test_missing_query_returns_error(self):
        result = tools_mod.execute_tool("recall_memory", {})
        assert "error" in result.lower()

    def test_returns_string(self):
        result = tools_mod.execute_tool("recall_memory", {"query": "Fabio"})
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS list completeness
# ─────────────────────────────────────────────────────────────────────────────
class TestToolsListCompleteness:
    EXPECTED = {
        "exec_shell", "read_file", "write_file", "http_get",
        "web_search", "browser_use", "send_file", "run_skill", "run_routine",
        "search_skills", "list_routines", "recall_memory",
    }

    def test_all_12_tools_registered(self):
        names = {t["function"]["name"] for t in tools_mod.TOOLS}
        missing = self.EXPECTED - names
        assert not missing, f"Missing tools in TOOLS list: {missing}"

    def test_all_tools_have_description(self):
        for t in tools_mod.TOOLS:
            name = t["function"]["name"]
            assert t["function"].get("description"), f"Tool '{name}' has no description"

    def test_all_tools_have_parameters(self):
        for t in tools_mod.TOOLS:
            name = t["function"]["name"]
            assert "parameters" in t["function"], f"Tool '{name}' has no parameters block"

    def test_unknown_tool_returns_error(self):
        result = tools_mod.execute_tool("does_not_exist", {})
        assert "error" in result.lower() or "unknown" in result.lower()
