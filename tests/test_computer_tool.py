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
