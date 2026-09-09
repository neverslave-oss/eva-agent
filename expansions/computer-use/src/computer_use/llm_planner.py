"""llm_planner.py — LLM-driven computer-use planner.

Replaces the deterministic rule-based planner's decision-making with a
perceive -> decide -> act -> verify loop driven by a vision LLM (default
MiniMax-M3 via HF Router / novita). Each step looks at the *current* screenshot
and decides ONE next action, then executes it, then looks again — the way a
human operates.

Implements the same `plan(goal, observation)` interface as the deterministic
Planner, so the orchestrator needs zero changes. The loop (including policy
validation, confirm gate, watch mode, and trajectory capture) runs inside
plan() and returns an ActionBatch ending in `done`.

Guardrails (all enforced here):
  - action whitelist (schema-valid kinds only)
  - PolicyEngine in front of every action
  - step cap (default 10)
  - dry-run default
  - explicit confirm gate for risky actions
  - watch mode: screenshots streamed to a callback (Telegram) as they arrive
  - trajectory capture to ARTIFACTS_TRAJECTORIES_DIR (JSONL)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from computer_use.schema import Action, ActionBatch, Observation
from computer_use.safety import PolicyEngine, PolicyViolation
from computer_use.vision_brain import VisionBrain

logger = logging.getLogger(__name__)

# Default risky actions that require explicit human confirmation.
_DEFAULT_RISKY = {"delete", "purchase", "send_money", "submit", "hotkey"}

# Hotkeys that are safe to run without human confirmation. Navigation / text
# editing shortcuts are harmless; destructive or system-level ones stay denied.
# Matched case-insensitively against the action's `text` (e.g. "ctrl+l").
_SAFE_HOTKEYS = {
    "enter", "return", "tab", "escape", "esc",
    "ctrl+l", "ctrl+t", "ctrl+w", "ctrl+enter", "ctrl+a", "ctrl+c", "ctrl+v",
    "ctrl+x", "ctrl+z", "ctrl+y", "ctrl+f", "ctrl+shift+t",
    "alt+tab", "alt+left", "alt+right", "super", "win",
}


class LLMPlanner:
    """Perceive->decide->act loop planner backed by a vision LLM."""

    def __init__(
        self,
        brain: VisionBrain | None = None,
        driver=None,
        policy: PolicyEngine | None = None,
        step_cap: int = 10,
        dry_run: bool = True,
        risky_actions: set[str] | None = None,
        confirm_callback=None,   # fn(action) -> bool; None => risky actions auto-denied
        watch_callback=None,     # fn(screenshot_path, caption) -> None (Telegram stream)
        trajectory_dir: str | Path | None = None,
    ):
        self.brain = brain or VisionBrain()
        self.driver = driver
        self.policy = policy or PolicyEngine({})
        self.step_cap = max(1, int(step_cap))
        self.dry_run = dry_run
        self.risky_actions = risky_actions or set(_DEFAULT_RISKY)
        self.confirm_callback = confirm_callback
        self.watch_callback = watch_callback
        self._trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self._history: list[dict] = []

    def _trajectory_path(self) -> Path | None:
        if self._trajectory_dir is not None:
            return self._trajectory_dir
        try:
            from runtime_paths import ARTIFACTS_TRAJECTORIES_DIR
            return Path(ARTIFACTS_TRAJECTORIES_DIR)
        except Exception:
            return None

    def _record_trajectory(self, goal: str, outcome: str, target: dict) -> None:
        path = self._trajectory_path()
        if path is None:
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
            rec = {
                "goal": goal,
                "outcome": outcome,
                "target": target,
                "model": "MiniMaxAI/MiniMax-M3",
                "steps": self._history,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            with open(path / f"computer_use_{ts}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info("[llm_planner] trajectory recorded (%s steps) -> %s", len(self._history), path)
        except Exception as e:
            logger.warning("[llm_planner] trajectory write failed: %s", e)

    def _confirm(self, action: Action) -> bool:
        """Explicit confirm gate for risky actions.

        Safe hotkeys (navigation / text editing) pass without confirmation;
        other risky actions still require the confirm callback and are denied
        (fail-closed) when none is supplied.
        """
        if action.kind not in self.risky_actions:
            return True
        if action.kind == "hotkey" and self._is_safe_hotkey(action.text):
            return True
        if self.confirm_callback is None:
            logger.info("[llm_planner] risky action %s auto-denied (no confirm callback)", action.kind)
            return False
        try:
            return bool(self.confirm_callback(action))
        except Exception as e:
            logger.warning("[llm_planner] confirm callback error: %s", e)
            return False

    @staticmethod
    def _is_safe_hotkey(text: str | None) -> bool:
        """Return True when the hotkey combo is on the safe allowlist."""
        if not text:
            return False
        return text.strip().lower() in _SAFE_HOTKEYS

    def _watch(self, screenshot: str | None, caption: str) -> None:
        if self.watch_callback is None:
            return
        try:
            self.watch_callback(screenshot, caption)
        except Exception as e:
            logger.warning("[llm_planner] watch callback error: %s", e)

    def plan(self, goal: str, observation: Observation | None = None) -> ActionBatch:
        self._history = []
        target = {"kind": "desktop"}

        for step in range(1, self.step_cap + 1):
            # 1. Perceive — capture the current screen.
            screenshot = None
            if self.driver is not None:
                try:
                    screenshot = self.driver.screenshot()
                except Exception as e:
                    logger.warning("[llm_planner] screenshot failed: %s", e)

            # 2. Decide — ask the vision brain for ONE next action.
            action = self.brain.next_action(goal, screenshot, self._history)
            if action is None:
                self._watch(screenshot, f"Step {step}: brain returned no valid action — aborting")
                self._record_trajectory(goal, "error", target)
                return ActionBatch(actions=[Action(kind="abort", text="vision brain unavailable")])

            # 3. Watch mode — stream the screenshot + decision.
            self._watch(screenshot, f"Step {step}: {action.kind} {action.text or action.selector or action.url or ''}")

            # 4. Terminal actions.
            if action.kind == "done":
                self._record_trajectory(goal, "done", target)
                return ActionBatch(actions=[Action(kind="done", text=action.text or "done")])
            if action.kind == "abort":
                self._record_trajectory(goal, "abort", target)
                return ActionBatch(actions=[Action(kind="abort", text=action.text or "aborted")])

            # 5. Policy validation.
            try:
                self.policy.validate_action(action)
            except PolicyViolation as e:
                self._watch(screenshot, f"Step {step}: blocked by policy ({e})")
                self._record_trajectory(goal, "blocked", target)
                return ActionBatch(actions=[Action(kind="abort", text=f"blocked by policy: {e}")])

            # 6. Confirm gate for risky actions.
            if not self._confirm(action):
                self._watch(screenshot, f"Step {step}: risky action {action.kind} not confirmed — aborting")
                self._record_trajectory(goal, "denied", target)
                return ActionBatch(actions=[Action(kind="abort", text=f"{action.kind} not confirmed")])

            # 7. Execute.
            result = {"status": "dry_run"}
            if self.driver is not None and not self.dry_run:
                try:
                    result = self.driver.execute(action, target)
                except Exception as e:
                    result = {"status": "error", "error": str(e)}

            self._history.append({"action": action.model_dump(), "result": result})
            logger.info("[llm_planner] step %d: %s -> %s", step, action.kind, result.get("status"))

            if result.get("status") == "error":
                self._watch(screenshot, f"Step {step}: execution error — aborting")
                self._record_trajectory(goal, "error", target)
                return ActionBatch(actions=[Action(kind="abort", text=result.get("error", "execution error"))])

        # Step cap reached without done.
        self._record_trajectory(goal, "step_cap", target)
        return ActionBatch(actions=[Action(kind="done", text="step cap reached")])
