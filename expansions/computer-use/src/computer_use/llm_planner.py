"""llm_planner.py — LLM-driven computer-use planner.

Replaces the deterministic rule-based planner's decision-making with a
perceive -> decide -> act -> verify loop driven by a vision LLM (DeepSeek
Vision via HF Router / deepinfra). Each step looks at the *current* screenshot
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

import hashlib
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
            # Record the actual configured vision model (DeepSeek Vision) rather
            # than a stale hardcoded label, so traces are truthful.
            model = None
            if self.brain is not None and hasattr(self.brain, "resolved_model"):
                try:
                    model = self.brain.resolved_model()
                except Exception:
                    model = None
            rec = {
                "goal": goal,
                "outcome": outcome,
                "target": target,
                "model": model,
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
        """A perceptual hash of the screenshot, to detect a frozen screen.

        The old implementation hashed the raw PNG bytes, which is defeated by
        the cursor position and pixel-level noise: two visually-identical frames
        (a frozen screen) produce different byte-hashes every capture, so the
        frozen-screen guard never fired. This perceptual hash downscales to a
        tiny grayscale grid and hashes the quantized brightness bits, so two
        visually-identical frames hash identically regardless of cursor/noise.
        Returns None when there is no screenshot to hash (guard skipped).
        """
        if not screenshot:
            return None
        try:
            from PIL import Image
            import io
            img = None
            if isinstance(screenshot, str) and screenshot.startswith("data:"):
                import base64
                payload = screenshot.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(payload)))
            elif isinstance(screenshot, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(screenshot)))
            elif isinstance(screenshot, (str, os.PathLike)) and Path(screenshot).exists():
                img = Image.open(screenshot)
            if img is None:
                return None
            # Perceptual hash: downscale -> grayscale -> tiny grid -> quantize.
            # Cursor and noise vanish at this resolution; content does not.
            img = img.convert("L").resize((16, 16), Image.LANCZOS)
            px = list(img.getdata())
            avg = sum(px) / float(len(px)) if px else 0.0
            bits = "".join("1" if v > avg else "0" for v in px)
            return hashlib.sha256(bits.encode()).hexdigest()
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

    def _is_desktop_driver(self) -> bool:
        """True when the active driver is a real desktop (full-screen) driver, not
        a browser driver. The black/white-locked-screen guards must only fire on
        a desktop surface: a browser sitting on `about:blank` is a normal,
        navigable starting state (white), and must not be treated as a dead /
        locked screen — otherwise the vision loop self-aborts before it can emit
        `navigate`. Discriminate by the driver's OWN attributes (browser drivers
        carry a Playwright session, `_pw`/`_page`); desktop drivers never do."""
        d = self.driver
        if d is None:
            return True  # no driver known: keep the old fail-closed behavior
        # HybridDriver delegates to its current medium (desktop|browser).
        mode = getattr(d, "mode", None)
        if mode in ("desktop", "browser"):
            return mode == "desktop"
        # Browser drivers own a Playwright session/page (even before lazy
        # launch, the attribute exists). Desktop drivers never have these.
        if hasattr(d, "_pw") or hasattr(d, "_context") or hasattr(d, "_page"):
            return False
        return True

    def plan(self, goal: str, observation: Observation | None = None) -> ActionBatch:
        self._history = []
        target = {"kind": "desktop"}
        # Plan-first execution: the brain returns an ordered list of actions once
        # (instead of one stateless action per step). We execute each step,
        # validate the result, and re-plan from the current state if a step
        # fails — so a stale plan never gets blindly followed to the end.
        plan: list[Action] = []
        plan_idx = 0

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
            #
            # Scoped to DESKTOP drivers only: a browser sitting on a fresh
            # blank tab is a NORMAL, navigable starting state, not a dead
            # screen — the white/black guards must be allowed to pass so the
            # vision loop can emit `navigate` and actually start the task.
            if (self._is_desktop_driver()
                    and hasattr(self.driver, "is_frame_black")
                    and self.driver.is_frame_black()):
                logger.error("[llm_planner] step %d: screen is black/locked — aborting", step)
                self._watch(screenshot, f"Step {step}: screen is black/locked — aborting")
                self._record_trajectory(goal, "locked", target)
                return ActionBatch(actions=[Action(kind="abort", text="screen is black/locked; cannot act")])

            # 1a. White-blank guard — some blanked screens (a browser that
            # failed to launch and left a white compositor surface, or a white
            # screensaver overlay) capture as a near-white frame. The black-guard
            # above misses these, so a white blank would stream per-step until
            # the frozen-guard caught it after several unchanged frames. Detect
            # it here and abort immediately, same fail-closed behavior.
            #
            # Same desktop-only scope as above: about:blank is white but is a
            # legitimate browser starting point for a navigate task — only fire
            # this on a real desktop surface, never a browser context.
            if (self._is_desktop_driver()
                    and hasattr(self.driver, "is_frame_white")
                    and self.driver.is_frame_white()):
                logger.error("[llm_planner] step %d: screen is blank-white — aborting", step)
                self._watch(screenshot, f"Step {step}: screen is blank-white — aborting")
                self._record_trajectory(goal, "blank", target)
                return ActionBatch(actions=[Action(kind="abort", text="screen is blank-white; cannot act")])

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

            # 2. Ensure we have a plan. When the current plan is exhausted (or
            # was invalidated by a rejection/failure), ask the brain for a fresh
            # ordered plan of actions from the current screen state.
            if plan_idx >= len(plan):
                plan = self.brain.plan_actions(goal, screenshot, self._history, hint=self._hint) or []
                plan_idx = 0
                if not plan:
                    reason = getattr(self.brain, "last_error", None) or "no valid plan"
                    logger.error("[llm_planner] step %d: vision brain failed — %s", step, reason)
                    self._watch(screenshot, f"Step {step}: vision brain failed ({reason}) — aborting")
                    self._record_trajectory(goal, "error", target)
                    return ActionBatch(actions=[Action(kind="abort", text=f"vision brain unavailable: {reason}")])
                self._watch(screenshot, f"Step {step}: plan of {len(plan)} actions")

            # 3. Take the next action from the plan.
            action = plan[plan_idx]
            plan_idx += 1
            elapsed_ms = int((time.monotonic() - step_start) * 1000)
            raw_llm = getattr(self.brain, "last_raw", None)

            # 3. Watch mode — stream the decision.
            self._watch(screenshot, f"Step {step}: {action.kind} {action.text or action.selector or action.url or ''}")

            # 3.5 Guardrails — code-enforced rejection of blind clicks and repeats.
            # These are enforced here (not just prompted) so the model cannot
            # emit a blind click or redo the same action. On rejection we feed
            # corrective feedback back to the vision brain, invalidate the rest
            # of the plan, and re-plan; after 3 consecutive rejections we stop
            # and ask the user (fail-closed if no callback).
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
                self._watch(screenshot, f"Step {step}: {rejection} — re-planning")
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
                            plan = []
                            continue
                    self._record_trajectory(goal, "rejected", target)
                    return ActionBatch(actions=[Action(kind="abort", text=rejection)])
                # Invalidate the rest of the plan; next iteration re-plans with the hint.
                plan = []
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
                # Execution failed — the rest of the plan is stale. Re-plan from
                # the current screen state instead of blindly following it.
                self._watch(screenshot, f"Step {step}: execution error ({result.get('error')}) — re-planning")
                plan = []
                continue

        # Step cap reached without done.
        self._record_trajectory(goal, "step_cap", target)
        return ActionBatch(actions=[Action(kind="done", text="step cap reached")])
