"""
test_computer_tool.py — Validates the Phase 4 `computer` native tool wiring.

Covers:
  - `computer` is registered in the TOOLS registry (so it appears in the
    native tool list and system prompt)
  - execute_tool() dispatches `computer` to the computer_use_bridge
  - happy path returns a JSON envelope with ok/status/run_id
  - missing goal returns a clear error
  - bridge unavailable degrades to a safe error (kernel never breaks)

Rules:
  - No model loaded, no real GUI, no Telegram
  - Uses unittest.mock to stub the sidecar bridge so tests stay hermetic
"""

import os
import sys
import json
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import core.tools as tools_mod


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
class TestComputerRegistry:
    def test_computer_tool_is_registered(self):
        names = [t["function"]["name"] for t in tools_mod.TOOLS]
        assert "computer" in names

    def test_computer_tool_requires_goal(self):
        tool = next(t for t in tools_mod.TOOLS if t["function"]["name"] == "computer")
        required = tool["function"]["parameters"]["required"]
        assert "goal" in required


# ─────────────────────────────────────────────────────────────────────────────
# execute_tool dispatch
# ─────────────────────────────────────────────────────────────────────────────
class TestComputerDispatch:
    def test_missing_goal_returns_error(self):
        result = tools_mod.execute_tool("computer", {})
        assert "error" in result.lower()
        assert "goal" in result

    def test_happy_path_returns_json_envelope(self):
        fake = {
            "ok": True,
            "status": "done",
            "message": "done",
            "completed": True,
            "chat_id": "chat-1",
            "goal": "open https://docs.openclaw.ai",
            "target": {"kind": "browser"},
            "dry_run": True,
            "run_id": "run-123",
        }
        with patch(
            "core.expansions.computer_use_bridge.run_computer_task",
            return_value=fake,
        ) as m:
            result = tools_mod.execute_tool(
                "computer",
                {"goal": "open https://docs.openclaw.ai", "target": {"kind": "browser"}},
                chat_id="chat-1",
            )
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["run_id"] == "run-123"
        # dry_run defaults to True when not supplied
        assert m.call_args.kwargs["dry_run"] is True

    def test_explicit_live_mode_passes_dry_run_false(self):
        fake = {"ok": True, "status": "ok", "completed": False, "dry_run": False, "run_id": "r2"}
        with patch(
            "core.expansions.computer_use_bridge.run_computer_task",
            return_value=fake,
        ) as m:
            tools_mod.execute_tool(
                "computer",
                {"goal": "click the button", "dry_run": False},
                chat_id="chat-1",
            )
        assert m.call_args.kwargs["dry_run"] is False

    def test_bridge_unavailable_degrades_safely(self):
        with patch(
            "core.expansions.computer_use_bridge.run_computer_task",
            side_effect=ImportError("sidecar missing"),
        ):
            result = tools_mod.execute_tool("computer", {"goal": "do a thing"})
        assert "error" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Contract: RL action space ⊆ policy allowlist
# ─────────────────────────────────────────────────────────────────────────────
def test_rl_action_space_is_allowed_by_bridge_policy():
    """Regression: every action kind the RL planner can pick must be permitted
    by the computer-use policy allowlist. The live failure was the RL planner
    emitting `fill` (the core of any form task) and the bridge policy blocking
    it with 'action not allowed: fill', which aborted the run the moment it
    tried to fill a form. Fix: keep the allowlist in sync with ACTION_KINDS."""
    import re as _re

    # RL planner's discrete action space (source of truth for what the agent
    # may choose).
    rl_src = tools_mod.Path(
        os.path.join(
            os.path.dirname(__file__), "..", "expansions", "computer-use", "src",
            "computer_use", "rl_planner.py",
        )
    ).read_text(encoding="utf-8")
    m = _re.search(r"ACTION_KINDS = \[(.*?)\]", rl_src, _re.S)
    assert m, "ACTION_KINDS not found in rl_planner.py"
    rl_actions = _re.findall(r'"([a-z_]+)"', m.group(1))

    # Bridge policy allowlist (the gate run_computer_task applies).
    bridge_src = tools_mod.Path(
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "expansions",
                     "computer_use_bridge.py")
    ).read_text(encoding="utf-8")
    am = _re.search(r'"allow_actions": \[(.*?)\]', bridge_src, _re.S)
    assert am, "allow_actions not found in computer_use_bridge.py"
    allowed = _re.findall(r'"([a-z_]+)"', am.group(1))

    missing = [a for a in rl_actions if a not in allowed]
    assert not missing, (
        f"RL planner can pick actions the bridge policy blocks: {missing}. "
        "Every RL action kind must be in the policy allow_actions."
    )
