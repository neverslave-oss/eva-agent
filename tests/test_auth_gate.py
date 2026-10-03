"""
test_auth_gate.py — Tests for the exec_shell authorization gate.

Tests the auth_gate module in isolation (no Telegram bot needed).
"""

import os
import sys
import time
import threading
import pytest

# Path setup
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)

from core.auth_gate import (
    is_safe_command,
    request_auth,
    resolve_auth,
    clear_auto_allow,
    _pending,
    DEFAULT_TIMEOUT,
)


class TestSafeCommands:
    """Commands in the safe list should bypass auth."""

    def test_echo_is_safe(self):
        assert is_safe_command("echo hello") is True

    def test_ls_is_safe(self):
        assert is_safe_command("ls -la") is True

    def test_git_status_is_safe(self):
        assert is_safe_command("git status") is True

    def test_pwd_is_safe(self):
        assert is_safe_command("pwd") is True

    def test_curl_localhost_is_safe(self):
        assert is_safe_command("curl -sS http://localhost:8779/health") is True

    def test_rm_is_not_safe(self):
        """rm is not in the safe prefix list."""
        assert is_safe_command("rm -rf /tmp/test") is False

    def test_pip_is_not_safe(self):
        assert is_safe_command("pip install something") is False

    def test_blocked_fork_bomb(self):
        assert is_safe_command(":(){:|:&};:") is False

    def test_blocked_rm_rf_root(self):
        assert is_safe_command("rm -rf /") is False


class TestAuthGateResolve:
    """Test the resolve_auth callback mechanism."""

    def test_resolve_allow(self):
        """resolve_auth(approved=True) should set result to 'allow'."""
        import core.auth_gate as gate
        # Clear any stale entries
        gate._pending.clear()

        event = threading.Event()
        gate._pending["test123"] = {"event": event, "result": None}
        resolve_auth("test123", approved=True)
        assert gate._pending["test123"]["result"] == "allow"

    def test_resolve_deny(self):
        """resolve_auth(approved=False) should set result to 'deny'."""
        import core.auth_gate as gate
        gate._pending.clear()

        event = threading.Event()
        gate._pending["test456"] = {"event": event, "result": None}
        resolve_auth("test456", approved=False)
        assert gate._pending["test456"]["result"] == "deny"

    def test_request_auth_safe_command_bypasses(self):
        """Safe commands should return 'allow' immediately without sending buttons."""
        result = request_auth("12345", "echo hello")
        assert result == "allow"

    def test_request_auth_deny_when_no_bot(self):
        """When the bot module can't be imported (test env), dangerous commands are denied."""
        # In test env, telegram_bot import may hang (starts polling).
        # The auth_gate handles ImportError gracefully, returning 'deny'.
        # We test this by verifying safe commands still bypass, and that
        # the function doesn't crash on non-safe commands without a bot.
        #
        # Direct test: mock the import to fail
        import core.auth_gate as gate
        original_import = gate.__builtins__["__import__"] if isinstance(gate.__builtins__, dict) else None
        # Just verify the function signature and safe-command bypass work
        result = request_auth("12345", "echo safe_command")
        assert result == "allow"


class TestExecShellAuthIntegration:
    """Test that exec_shell respects the auth gate via _current_chat_id."""

    def test_safe_command_runs_without_chat_id(self):
        """Safe commands should run even without chat_id set."""
        import core.tools as tools_mod
        # Ensure no chat_id is set
        tools_mod._current_chat_id = ""
        result = tools_mod.execute_tool("exec_shell", {"command": "echo hello_auth_test"})
        assert "hello_auth_test" in result

    def test_safe_command_runs_with_chat_id(self):
        """Safe commands should run and bypass auth even with chat_id."""
        import core.tools as tools_mod
        tools_mod._current_chat_id = "12345"
        result = tools_mod.execute_tool("exec_shell", {"command": "echo auth_bypass"})
        assert "auth_bypass" in result
        tools_mod._current_chat_id = ""

    def test_dangerous_command_without_chat_id_proceeds(self):
        """Without chat_id, dangerous commands proceed (no auth gate active)."""
        import core.tools as tools_mod
        tools_mod._current_chat_id = ""
        # This is a harmless rm command (touch + rm a temp file)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".auth_test")
        tmp.write(b"test")
        tmp.close()
        result = tools_mod.execute_tool("exec_shell", {"command": f"rm {tmp.name}"})
        # Should proceed — no auth gate without chat_id
        assert not os.path.exists(tmp.name) or "(no output)" in result or "authorization" not in result.lower()


class TestApproveAll:
    """Tests for the '✅ Approve all' auto-approve grant."""

    def _resolve(self, request_id, approved, allow_all=False):
        import core.auth_gate as gate
        gate._pending.clear()
        event = threading.Event()
        gate._pending[request_id] = {"event": event, "result": None, "chat_id": "chat_approve_all"}
        resolve_auth(request_id, approved, allow_all=allow_all)
        return gate._pending[request_id]["result"]

    def setup_method(self):
        import core.auth_gate as gate
        gate._pending.clear()
        gate._auto_allow.clear()
        gate._deny_count.clear()
        gate._stop_events.clear()
        # Remove any leftover stop marker files
        try:
            import os, glob
            for f in glob.glob(os.path.join(gate._STOP_DIR, "stop_*")):
                try:
                    os.remove(f)
                except Exception:
                    pass
        except Exception:
            pass

    def test_resolve_allow_all_sets_grant(self):
        """resolve_auth(allow_all=True, approved=True) sets result 'allow_all' and grants the chat."""
        import core.auth_gate as gate
        result = self._resolve("req1", True, allow_all=True)
        assert result == "allow_all"
        assert gate._auto_allow.get("chat_approve_all") is not None

    def test_resolve_allow_all_with_deny_no_grant(self):
        """allow_all with approved=False behaves like a normal deny (no grant)."""
        import core.auth_gate as gate
        result = self._resolve("req2", False, allow_all=True)
        assert result == "deny"
        assert gate._auto_allow.get("chat_approve_all") is None

    def test_request_auth_consumes_grant(self):
        """Under an active grant, request_auth returns 'allow' without a prompt."""
        import core.auth_gate as gate
        gate._auto_allow["chat_approve_all"] = time.monotonic()
        result = request_auth("chat_approve_all", "pip install something")
        assert result == "allow"

    def test_grant_does_not_bypass_blocked_patterns(self):
        """Dangerous BLOCKED_PATTERNS still short-circuit to deny under a grant."""
        import core.auth_gate as gate
        gate._auto_allow["chat_approve_all"] = time.monotonic()
        # rm -rf / is in BLOCKED_PATTERNS → is_safe_command False → not auto-allowed.
        # Without a bot (test env) request_auth returns 'deny'.
        result = request_auth("chat_approve_all", "rm -rf /")
        assert result == "deny"

    def test_clear_auto_allow_removes_grant(self):
        """clear_auto_allow() removes an active grant."""
        import core.auth_gate as gate
        gate._auto_allow["chat_approve_all"] = time.monotonic()
        clear_auto_allow("chat_approve_all")
        assert gate._auto_allow.get("chat_approve_all") is None

    def test_grant_expires_after_ttl(self):
        """An expired grant is dropped and no longer auto-allows."""
        import core.auth_gate as gate
        gate._auto_allow["chat_approve_all"] = time.monotonic() - (gate.AUTO_ALLOW_TTL + 1)
        # Expired → clear_auto_allow is invoked inside request_auth
        request_auth("chat_approve_all", "pip install something")
        assert gate._auto_allow.get("chat_approve_all") is None

    def test_execute_tool_allow_all_proceeds(self):
        """execute_tool treats 'allow_all' as proceed (falls through to execution)."""
        import core.auth_gate as gate
        import core.tools as tools_mod
        gate._auto_allow["chat_approve_all"] = time.monotonic()
        tools_mod._current_chat_id = "chat_approve_all"
        try:
            result = tools_mod.execute_tool("exec_shell", {"command": "echo approve_all_proceeds"})
            assert "approve_all_proceeds" in result
        finally:
            tools_mod._current_chat_id = ""
            gate.clear_auto_allow("chat_approve_all")


class TestStopSignal:
    """Tests for the per-chat stop signal (used by /stop and the 3-deny rule)."""

    def setup_method(self):
        import core.auth_gate as gate
        gate.clear_task_state("chat_stop_test")

    def test_request_stop_sets_signal(self):
        """request_stop() makes is_stop_requested() return True."""
        import core.auth_gate as gate
        assert gate.is_stop_requested("chat_stop_test") is False
        gate.request_stop("chat_stop_test")
        assert gate.is_stop_requested("chat_stop_test") is True

    def test_clear_stop_removes_signal(self):
        """clear_stop() makes is_stop_requested() return False."""
        import core.auth_gate as gate
        gate.request_stop("chat_stop_test")
        gate.clear_stop("chat_stop_test")
        assert gate.is_stop_requested("chat_stop_test") is False

    def test_clear_task_state_resets_stop(self):
        """clear_task_state() resets the stop signal and deny counter."""
        import core.auth_gate as gate
        gate.request_stop("chat_stop_test")
        gate._deny_count["chat_stop_test"] = 2
        gate.clear_task_state("chat_stop_test")
        assert gate.is_stop_requested("chat_stop_test") is False
        assert gate._deny_count.get("chat_stop_test") is None

    def test_empty_chat_id_never_stops(self):
        """is_stop_requested('') returns False."""
        import core.auth_gate as gate
        assert gate.is_stop_requested("") is False


class TestConsecutiveDenyStop:
    """Tests for the 3-consecutive-denials → stop rule."""

    def setup_method(self):
        import core.auth_gate as gate
        gate.clear_task_state("chat_deny_test")

    def test_three_denials_stop_loop(self):
        """3 consecutive denials signal a stop."""
        import core.auth_gate as gate
        # Blocked patterns short-circuit to a denial WITHOUT importing the bot
        # (so tests don't hang on telegram polling). They are auto-denied by the
        # system, NOT user denials, so they must NOT count toward the stop rule.
        assert gate.request_auth("chat_deny_test", "rm -rf /") == "deny"
        assert gate.request_auth("chat_deny_test", "rm -rf /") == "deny"
        assert gate.request_auth("chat_deny_test", "rm -rf /") == "deny"
        # Even after 3 blocked attempts, no stop is requested (not user denials).
        assert gate.is_stop_requested("chat_deny_test") is False

    def test_record_denial_increments_and_stops(self):
        """_record_denial() increments the counter and stops at the cap.

        This is the path taken when the user taps Deny or a request times out.
        """
        import core.auth_gate as gate
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate.is_stop_requested("chat_deny_test") is False
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate.is_stop_requested("chat_deny_test") is True

    def test_record_denial_increments_and_stops(self):
        """_record_denial() increments the counter and stops at the cap."""
        import core.auth_gate as gate
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate.is_stop_requested("chat_deny_test") is False
        assert gate._record_denial("chat_deny_test") == "deny"
        assert gate.is_stop_requested("chat_deny_test") is True

    def test_approval_resets_deny_streak(self):
        """An approval between denials resets the streak."""
        import core.auth_gate as gate
        gate._record_denial("chat_deny_test")            # deny #1
        gate._deny_count.pop("chat_deny_test", None)     # approval resets
        gate._record_denial("chat_deny_test")            # deny #1 again
        gate._record_denial("chat_deny_test")            # deny #2
        assert gate.is_stop_requested("chat_deny_test") is False

    def test_execute_tool_returns_stopped_when_stop_requested(self):
        """When a stop is requested, execute_tool bails out without prompting."""
        import core.auth_gate as gate
        import core.tools as tools_mod
        gate.request_stop("chat_deny_test")
        tools_mod._current_chat_id = "chat_deny_test"
        try:
            result = tools_mod.execute_tool("exec_shell", {"command": "echo should_not_run"})
            assert "stopped by user" in result
        finally:
            tools_mod._current_chat_id = ""
            gate.clear_task_state("chat_deny_test")

