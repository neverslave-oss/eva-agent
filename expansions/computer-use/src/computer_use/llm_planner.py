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
import time
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
        stuck_callback=None,    # fn(rejection) -> bool; None => abort on repeated rejections
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
        self.stuck_callback = stuck_callback
        self._trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self._history: list[dict] = []
        self._hint: str | None = None
        self._step_rejections = 0
        self._last_screen_hash: str | None = None
        self._frozen_steps = 0

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

    @staticmethod
    def _screen_hash(screenshot) -> str | None:
        """A cheap stable hash of the screenshot bytes, to detect a frozen screen.

        Screenshots are data-URIs (base64 PNG). Two identical screenshots hash
        identically; a changed screen hashes differently. Returns None when there
        is no screenshot to hash (so the frozen-screen guard is skipped).
        """
        if not screenshot:
            return None
        try:
            if isinstance(screenshot, str) and screenshot.startswith("data:"):
                # data:image/png;base64,<payload>
                payload = screenshot.split(",", 1)[1]
                return hashlib.sha256(payload.encode()).hexdigest()
            if isinstance(screenshot, (bytes, bytearray)):
                return hashlib.sha256(bytes(screenshot)).hexdigest()
            # Path to a file — hash the file bytes.
            if isinstance(screenshot, (str, os.PathLike)):
                with open(screenshot, "rb") as fh:
                    return hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            return None
        return None

    @staticmethod
    def _is_blind_click(action) -> bool:
        """A click/double_click with no real x,y selector is a blind click."""
        if action.kind not in ("click", "double_click"):
            return False
        sel = (action.selector or "").strip()
        if not sel or sel.lower() == "auto":
            return True
        # Must look like real coordinates: "x,y"
        try:
            x, y = sel.split(",")
            int(x.strip()); int(y.strip())
            return False
        except Exception:
            return True

    @staticmethod
    def _action_signature(action) -> str:
        """A canonical signature for an action, to detect repeats."""
        if action.kind == "click":
            return f"click:{action.selector}"
        if action.kind == "double_click":
            return f"double_click:{action.selector}"
        if action.kind == "type":
            return f"type:{action.text}"
        if action.kind == "hotkey":
            return f"hotkey:{action.text}"
        if action.kind == "navigate":
            return f"navigate:{action.url}"
        if action.kind == "launch":
            return f"launch:{action.text}"
        return f"{action.kind}:{action.text or action.selector or action.url or ''}"

    def _is_repeat(self, action) -> bool:
        """True if this action's signature already appears in history."""
        sig = self._action_signature(action)
        return any(
            h.get("action", {}).get("kind") == action.kind
            and self._action_signature(Action(**h["action"])) == sig
            for h in self._history
        )

    def plan(self, goal: str, observation: Observation | None = None) -> ActionBatch:
        self._history = []
        target = {"kind": "desktop"}

        for step in range(1, self.step_cap + 1):
            step_start = time.monotonic()

            # 1. Perceive — capture the current screen.
            screenshot = None
            if self.driver is not None:
                try:
                    screenshot = self.driver.screenshot()
                except Exception as e:
                    logger.warning("[llm_planner] screenshot failed: %s", e)

            # 1a. Black/locked-frame guard — abort BEFORE sending to the vision
            # pipeline. A blanked/locked screen (screensaver overlay or login
            # dialog) captures as a near-black frame; feeding it to the cloud
            # vision model just burns API calls on a screen Eva can't act on
            # (and can loop forever). Detect it and stop fail-closed.
            if (self.driver is not None
                    and hasattr(self.driver, "is_frame_black")
                    and self.driver.is_frame_black()):
                logger.error("[llm_planner] step %d: screen is black/locked — aborting", step)
                self._watch(screenshot, f"Step {step}: screen is black/locked — aborting")
                self._record_trajectory(goal, "locked", target)
                return ActionBatch(actions=[Action(kind="abort", text="screen is black/locked; cannot act")])

            # 1a2. Frozen-screen guard — the PRIMARY defense against the churn.
            # The model was emitting *different* valid-looking blind clicks on an
            # unchanged screen, so action-rejection never fired. Instead, detect
            # that the screen itself hasn't changed across consecutive steps and
            # stop-and-ask regardless of what action the model proposes. This runs
            # BEFORE the vision call so we don't burn API credits on a frozen
            # screen. After N frozen steps, ask the user (or abort fail-closed).
            cur_hash = self._screen_hash(screenshot)
            if cur_hash is not None:
                if cur_hash == self._last_screen_hash:
                    self._frozen_steps += 1
                else:
                    self._frozen_steps = 0
                self._last_screen_hash = cur_hash
                if self._frozen_steps >= 3:
                    frozen_msg = (f"screen unchanged for {self._frozen_steps} steps — the model keeps "
                                  "guessing on a frozen screen; stop and ask the user")
                    logger.error("[llm_planner] step %d: %s", step, frozen_msg)
                    self._watch(screenshot, f"Step {step}: {frozen_msg}")
                    if self.stuck_callback is not None:
                        try:
                            proceed = bool(self.stuck_callback(frozen_msg))
                        except Exception as e:
                            logger.warning("[llm_planner] stuck callback error: %s", e)
                            proceed = False
                        if proceed:
                            # User says continue — reset the frozen counter and keep
                            # going (maybe they changed the screen or want a retry).
                            self._frozen_steps = 0
                            self._last_screen_hash = None
                            self._watch(screenshot, f"Step {step}: user said continue — retrying")
                            continue
                    self._record_trajectory(goal, "frozen", target)
                    return ActionBatch(actions=[Action(kind="abort", text=frozen_msg)])

            # 1b. Watch mode — stream the screenshot to Telegram the moment it's
            # captured, BEFORE the (slow) vision inference, so the user sees the
            # live screen while the model is thinking (not after).
            self._watch(screenshot, f"Step {step}: perceiving screen…")

            # 2. Decide — ask the vision brain for ONE next action.
            action = self.brain.next_action(goal, screenshot, self._history, hint=self._hint)
            elapsed_ms = int((time.monotonic() - step_start) * 1000)
            raw_llm = getattr(self.brain, "last_raw", None)
            if action is None:
                reason = getattr(self.brain, "last_error", None) or "no valid action"
                logger.error("[llm_planner] step %d: vision brain failed — %s", step, reason)
                self._watch(screenshot, f"Step {step}: vision brain failed ({reason}) — aborting")
                self._record_trajectory(goal, "error", target)
                return ActionBatch(actions=[Action(kind="abort", text=f"vision brain unavailable: {reason}")])

            # 3. Watch mode — stream the decision.
            self._watch(screenshot, f"Step {step}: {action.kind} {action.text or action.selector or action.url or ''}")

            # 3.5 Guardrails — code-enforced rejection of blind clicks and repeats.
            # These are enforced here (not just prompted) so the model cannot
            # emit a blind click or redo the same action. On rejection we feed
            # corrective feedback back to the vision brain and retry; after 2
            # consecutive rejections we abort fail-closed.
            rejection = None
            if self._is_blind_click(action):
                rejection = ("blind click rejected: click/double_click must include real 'x,y' "
                             "coordinates in selector; if you cannot identify exact coordinates, "
                             "return done or wait instead")
            elif self._is_repeat(action):
                rejection = ("repeated action rejected: you already performed this exact action; "
                             "if the goal is met return done, otherwise take a DIFFERENT action")
            if rejection:
                self._step_rejections += 1
                self._hint = rejection
                self._watch(screenshot, f"Step {step}: {rejection} — retrying")
                # After N consecutive rejections the model is stuck (same screen,
                # same rejected actions). Stop looping: ask the user to continue or
                # stop. Without a stuck_callback, abort fail-closed.
                if self._step_rejections >= 3:
                    if self.stuck_callback is not None:
                        try:
                            proceed = bool(self.stuck_callback(rejection))
                        except Exception as e:
                            logger.warning("[llm_planner] stuck callback error: %s", e)
                            proceed = False
                        if proceed:
                            # User says continue — reset the counter and keep going
                            # with the corrective hint so the model gets a fresh shot.
                            self._step_rejections = 0
                            self._watch(screenshot, f"Step {step}: user said continue — retrying with hint")
                            continue
                    self._record_trajectory(goal, "rejected", target)
                    return ActionBatch(actions=[Action(kind="abort", text=rejection)])
                continue
            self._step_rejections = 0
            self._hint = None

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

            self._history.append({
                "step": step,
                "screenshot": screenshot,      # what the LLM actually saw (data-URI or path)
                "raw_llm": raw_llm,            # exact raw output the model returned
                "action": action.model_dump(), # parsed, schema-valid action
                "result": result,              # execution result
                "elapsed_ms": elapsed_ms,      # wall time for this perceive+decide step
            })
            logger.info("[llm_planner] step %d: %s -> %s (%dms)", step, action.kind, result.get("status"), elapsed_ms)

            if result.get("status") == "error":
                self._watch(screenshot, f"Step {step}: execution error — aborting")
                self._record_trajectory(goal, "error", target)
                return ActionBatch(actions=[Action(kind="abort", text=result.get("error", "execution error"))])

        # Step cap reached without done.
        self._record_trajectory(goal, "step_cap", target)
        return ActionBatch(actions=[Action(kind="done", text="step cap reached")])
