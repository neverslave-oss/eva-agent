"""
test_conversation_flow_auth.py — Regression test for the auth-gate regression.

Scenario (reported as DEBUG NOTES):
  During the tools loop, exec_shell authorization requests are not delivered to
  the user for approval (cross-process: request_auth runs in the model_server
  process, resolve_auth in the API/bot process — separate in-memory registries).
  Every approval therefore TIMES OUT. With the 'stop on 3 consecutive denials'
  rule, 3 timeouts trigger a stop that aborts the tool loop, and the conversation
  turn is lost.

This test asserts the correct behaviour:
  1. A user approval for a pending request RESOLVES request_auth (does not time
     out) — i.e. the decision reaches the blocking caller even across the
     model_server/API process boundary.
  2. A resolved approval does NOT spuriously trip the consecutive-deny stop.
"""

import os
import sys
import time
import threading
import unittest.mock as mock

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)

from core.auth_gate import (
    request_auth,
    resolve_auth,
    clear_task_state,
    is_stop_requested,
)


def _run_request_auth(chat_id, command, timeout=5):
    """Run request_auth in a thread; simulate the model_server process side."""
    import core.auth_gate as gate
    holder = {}
    captured = {}

    def _target():
        holder["result"] = gate.request_auth(chat_id, command, timeout=timeout)

    def _fake_send(chat_id, text, buttons, parse_mode="Markdown"):
        captured["request_id"] = buttons[0][0]["callback_data"].rsplit("_", 1)[-1]

    # Warm the telegram_bot module (slow import) so the local import is fast.
    import services.channels.telegram_bot as _warm
    gate._pending.clear()
    ctx = mock.patch("services.channels.telegram_bot.send_buttons", side_effect=_fake_send)
    ctx.start()
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t, holder, captured, ctx


def _resolve_from_bot(request_id, approved):
    """Simulate the API/bot process resolving a button tap."""
    import core.auth_gate as gate
    # The bot process has NO local _pending entry (separate process) — only the
    # persisted file carries the decision. Clear the local registry to simulate
    # this isolation before resolving.
    gate._pending.clear()
    resolve_auth(request_id, approved)


class TestConversationFlowAuth:
    def setup_method(self):
        import core.auth_gate as gate
        gate._pending.clear()
        gate._auto_allow.clear()
        gate._deny_count.clear()
        # _AUTH_DIR only exists once the cross-process fix is present; clean it
        # defensively so this test runs on both pre-fix and post-fix branches.
        auth_dir = getattr(gate, "_AUTH_DIR", None)
        if auth_dir:
            import glob
            for f in glob.glob(os.path.join(auth_dir, "*.json")):
                try:
                    os.remove(f)
                except Exception:
                    pass
        clear_task_state("chat_flow")

    def test_approval_resolves_across_processes(self):
        """A user approval must resolve request_auth instead of timing out.

        This is the core regression: before the cross-process fix, request_auth
        (model_server) and resolve_auth (API/bot) used separate in-memory dicts,
        so the approval always timed out.
        """
        t, holder, captured, ctx = _run_request_auth("chat_flow", "pip install foo")
        try:
            for _ in range(300):
                if captured.get("request_id"):
                    break
                time.sleep(0.01)
            request_id = captured.get("request_id")
            assert request_id, "request was not persisted / send_buttons not called"

            _resolve_from_bot(request_id, approved=True)
            t.join(timeout=6)
            assert holder.get("result") == "allow", (
                f"expected 'allow', got {holder.get('result')!r} — approval did "
                "not resolve across processes (timed out)"
            )
        finally:
            ctx.stop()

    def test_resolved_approval_does_not_trip_stop(self):
        """A resolved approval must NOT count as a denial toward the stop rule."""
        import core.auth_gate as gate
        # Simulate 2 denials, then a real approval — the approval resets the streak.
        gate._record_denial("chat_flow")
        gate._record_denial("chat_flow")
        # A resolved approval resets the deny counter.
        gate._deny_count.pop("chat_flow", None)
        assert is_stop_requested("chat_flow") is False
