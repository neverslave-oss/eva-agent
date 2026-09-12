"""rl_planner.py — Tabular Q-Learning computer-use planner.

Replaces the free-running heuristic decision loop with a Reinforcement Learning
agent: for each observed screen state it picks the single next action that
maximizes expected future reward toward the goal (epsilon-greedy over a tabular
Q-value function). Rewards are computed from *verified* progress, never from a
mere `ok` result, so a step that reports ok without advancing the goal yields
~0/negative reward — directly fixing the "step-cap returns ok with no goal
verification" storm.

Implements the same `plan(goal, observation)` interface as the deterministic
`Planner`, so the orchestrator (and the outer agent tool-loop) needs zero changes.

Method follows our ML Specialization Cheat Sheet (Course 3, §3.4):
  Q[s,a] += α ( r + γ·max_a' Q[s',a'] − Q[s,a] )

Design doc: .specs/plans/2026-09-12-computer-use-rl-planner.md
Future reward-signal owner: Eva (kernel-evolving). Reward magnitudes here are the
initial accepted defaults and are intentionally concentrated in one module for her.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from computer_use.schema import Action, ActionBatch, Observation
from computer_use.safety import PolicyEngine, PolicyViolation

logger = logging.getLogger(__name__)

# ── Reward magnitudes (accepted initial defaults; Eva owns future tuning) ────
R_GOAL = 100.0      # goal verifier confirms fully done (terminal)
R_PROGRESS = 10.0   # step verifier confirms progress toward goal
R_NOOP = -1.0       # unverified / no-change step (churn)
R_BLIND = -5.0      # blind click / repeated identical action
R_SWAP_FAIL = -5.0  # driver-swap attempted and failed
R_STEPCAP = -10.0   # step cap reached without goal verified (the storm's mode)

# Extra penalty applied to ties when the state is clearly dead, to force
# abort-and-ask rather than churning on an unchanged screen.
R_DEAD_STATE = -15.0

# Fraction of the seeds/config restores that force an abort once Q collapses.
DEAD_STATE_Q_THRESHOLD = 3  # consecutive max-Q<=0 states before abort-and-ask


def _discretize_features(
    screen_hash: str | None,
    driver: str | None,
    last_outcome: str | None,
    progress_signal: bool,
    step_bucket: int,
    url: str | None = None,
) -> tuple:
    """Map continuous/rich signals into a compact discrete state tuple.

    Richer features than a bare screen hash: we fold in the active driver, the
    outcome of the previous step, a goal-progress signal (does the current
    screen/keyboard state already satisfy part of the goal?), a coarse step
    count, and a URL bucket so the agent can perceive WHERE it is (about:blank
    vs a real page) even when the screenshot is null. Each feature is bucketed
    so the Q-table stays discrete and small enough for tabular learning.
    """
    # Screen hash bucket: coarse hash of the perceptual hash (10 buckets).
    if screen_hash:
        screen_bucket = int(screen_hash[:2], 16) % 10
    else:
        screen_bucket = 0

    driver_bucket = 0 if driver == "desktop" else 1 if driver == "browser" else 2

    outcome_bucket = {
        None: 0,
        "ok": 1,
        "error": 2,
        "verified": 3,
        "noop": 4,
        "blind": 5,
    }.get(last_outcome, 0)

    step_bucket_capped = 0 if step_bucket < 5 else (1 if step_bucket < 10 else 2)

    # URL bucket: 0 = blank/no URL, 1 = non-blank page. Gives the agent a
    # perception of location even when the screenshot is null.
    u = (url or "").strip().lower()
    url_bucket = 0 if (not u or u in ("about:blank", "about:blank#")) else 1

    return (
        screen_bucket,
        driver_bucket,
        outcome_bucket,
        1 if progress_signal else 0,
        step_bucket_capped,
        url_bucket,
    )


class RLPlanner:
    """Tabular Q-Learning planner for computer-use tasks.

    Discretized state -> epsilon-greedy action -> execute -> verified reward ->
    Bellman Q-update. Terminates on verified-done (terminal reward) or on a dead
    state (abort-and-ask, fail-closed).
    """

    # Discrete action space (kind-level templates). `selector`/`text` params are
    # bound by the execution layer; Q-learning operates on these action kinds so
    # the table stays discrete and the policy chooses *what to do next* and
    # *which medium* — exactly the control Fabio asked the agent to own.
    ACTION_KINDS = [
        "observe",
        "click",
        "type",
        "hotkey",
        "navigate",
        "scroll",
        "wait",
        "driver_swap",
        "abort",
        "done",
    ]

    def __init__(
        self,
        driver=None,
        policy: PolicyEngine | None = None,
        step_cap: int = 20,
        dry_run: bool = True,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        q_path: str | None = None,
        execute_override=None,  # injectable executor for deterministic tests
        verify_override=None,   # injectable goal/step verifier for deterministic tests
    ):
        self.driver = driver
        self.policy = policy or PolicyEngine({})
        self.step_cap = max(1, int(step_cap))
        self.dry_run = dry_run
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self._q: dict[tuple, list[float]] = {}
        self._q_path = Path(q_path) if q_path else None
        # Number of available actions for the Q row width.
        self._n_actions = len(self.ACTION_KINDS)
        self._execute = execute_override
        self._verify = verify_override
        self._dead_streak = 0

    # ── Q-table persistence ───────────────────────────────────────────────
    def load_q(self, path: str | None = None) -> bool:
        p = self._q_path if path is None else Path(path)
        if p is None or not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self._q = {}
            for k_str, row in data.get("q", {}).items():
                # keys were saved as dot-joined strings; rebuild the tuple.
                k = tuple(int(part) for part in k_str.split("."))
                self._q[k] = [float(v) for v in row]
            self.epsilon = float(data.get("epsilon", self.epsilon))
            return True
        except Exception as e:
            logger.warning("[rl_planner] failed to load Q table: %s", e)
            return False

    def save_q(self, path: str | None = None) -> str | None:
        p = self._q_path if path is None else Path(path)
        if p is None:
            return None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "q": {".".join(str(part) for part in k): v for k, v in self._q.items()},
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
            }))
            return str(p)
        except Exception as e:
            logger.warning("[rl_planner] failed to save Q table: %s", e)
            return None

    # ── Q core ────────────────────────────────────────────────────────────
    def _row(self, state: tuple) -> list[float]:
        if state not in self._q:
            self._q[state] = [0.0] * self._n_actions
        return self._q[state]

    def _max_q(self, state: tuple) -> float:
        row = self._q.get(state)
        return max(row) if row else 0.0

    def _seed_goal_policy(self, goal: str, driver: str) -> None:
        """Pre-bias the Q-table so a cold-start agent starts with the right
        action instead of random exploration.

        Cold-start tabular Q-learning has no prior that \"goal contains an
        http(s) URL -> navigate first\". Without a seed it explores randomly
        over the whole action space and never lands on navigate within a short
        live run. Here we raise the Q value of the `navigate` action for
        blank-url states (url_bucket=0) when the goal mentions a URL, so
        epsilon-greedy's argmax picks navigate first.
        """
        if "http" not in (goal or "").lower():
            return
        nav_idx = self.ACTION_KINDS.index("navigate")
        # Blank-url states: (screen_bucket, driver_bucket, outcome_bucket,
        # progress, step_bucket, url_bucket=0). Seed a positive navigate bias
        # across the plausible screen/outcome buckets for this driver.
        driver_bucket = 0 if driver == "desktop" else 1 if driver == "browser" else 2
        for screen in range(10):
            for outcome in range(6):
                for progress in (0, 1):
                    for step in range(3):
                        state = (screen, driver_bucket, outcome, progress, step, 0)
                        row = self._row(state)
                        if row[nav_idx] <= 0.0:
                            row[nav_idx] = 1.0

    def _select_action_idx(self, state: tuple) -> int:
        import random
        # Epsilon-greedy: explore a random action with prob epsilon.
        if random.random() < self.epsilon:
            return random.randrange(self._n_actions)
        row = self._row(state)
        # Deterministic argmax; break ties by the earliest action to keep policy
        # stable (prefer observe/click over abort when equally valued).
        best_v = max(row)
        best = row.index(best_v)
        return best

    @staticmethod
    def _action_sig(action) -> str:
        if action.kind in ("click", "double_click"):
            return f"{action.kind}:{action.selector}"
        if action.kind == "type":
            return f"type:{action.text}"
        if action.kind == "navigate":
            return f"navigate:{action.url}"
        if action.kind == "launch":
            return f"launch:{action.text}"
        return f"{action.kind}:{action.text or action.selector or action.url or ''}"

    # ── Reward / verification ─────────────────────────────────────────────
    def _classify_step(
        self,
        action: Action,
        result: dict,
        goal_met: bool,
        progress_met: bool,
        last_sig: str | None,
    ) -> tuple[str, float]:
        """Map a completed step to an outcome category + immediate reward.

        Order matters: verified terminal > verified progress > no-op > blind >
        error. A step only earns positive reward when the verifier confirms it
        (goal_met/progress_met); otherwise it is churn and is penalized.
        """
        # Terminal: goal verifier confirms done.
        if goal_met:
            return ("verified", R_GOAL)
        # Verified progress.
        if progress_met:
            return ("verified", R_PROGRESS)
        # Dead state: repeated identical action on an unchanged screen.
        if last_sig == self._action_sig(action):
            return ("blind", R_BLIND)
        # Blind click (no real coords) is a hard negative.
        if action.kind in ("click", "double_click") and not (
            action.selector and "," in action.selector
        ):
            return ("blind", R_BLIND)
        # Execution error (e.g. a driver swap failure or a failed step).
        if result.get("status") == "error":
            return ("error", R_SWAP_FAIL)
        # Step-cap without verification is the storm mode.
        # (handled by the caller on timeout, not here)
        # Otherwise an executed-but-unverified step is churn.
        return ("noop", R_NOOP)

    def _is_dead(self, state: tuple) -> bool:
        """A state is dead only when it has been visited and its best action
        value is at/below zero.

        Fresh (unvisited) states initialize to all-zero Q (the row doesn't
        exist yet, so `_max_q` returns 0.0). Declaring them dead immediately
        would abort on the very first unseen state before the agent can ever
        explore — the live run died at step 3 for exactly this reason. So only
        a state whose row exists AND whose best Q <= 0 is dead.
        """
        row = self._q.get(state)
        if row is None:
            return False  # unseen state: explore, don't abort
        return max(row) <= 0.0

    # ── Main loop (mirrors LLMPlanner.plan interface) ─────────────────────
    def plan(self, goal: str, observation: Observation | None = None, target: dict | None = None) -> ActionBatch:
        """Run the RL perceive->decide->act->verify loop for a goal.

        Returns an ActionBatch ending in `done` (goal verified) or `abort`
        (dead state / step cap / policy / confirm). Mirrors the `Planner.plan`
        contract so the orchestrator is unchanged.
        """
        target = target or {"kind": "desktop"}
        history: list[dict] = []
        last_sig: str | None = None
        last_outcome: str | None = None
        step_bucket = 0
        driver = (target.get("kind") or "desktop")
        self._dead_streak = 0

        # Goal-aware seed policy: cold-start tabular Q-learning has no prior
        # that "goal contains http(s) -> navigate first". Without a seed it
        # explores randomly and never lands on navigate within a short live
        # run. Pre-bias the navigate action for blank-url states so the agent
        # starts with the right bias instead of random exploration.
        self._seed_goal_policy(goal, driver)

        for step in range(1, self.step_cap + 1):
            # 1. Perceive — capture the current screen.
            screenshot = None
            if self.driver is not None:
                try:
                    screenshot = self.driver.screenshot()
                except Exception as e:
                    logger.warning("[rl_planner] screenshot failed: %s", e)

            # 2. Progress / goal verification signal.
            goal_met, progress_met = self._run_verify(goal, observation, screenshot)

            # 3. Discrete state from richer features.
            state = _discretize_features(
                self._screen_hash(screenshot),
                driver,
                last_outcome,
                progress_met,
                step_bucket,
                url=(observation.url if observation is not None else None),
            )

            # 4. Dead-state guard — fail-closed, do not churn/stream forever.
            if self._is_dead(state):
                self._dead_streak += 1
                if self._dead_streak >= DEAD_STATE_Q_THRESHOLD:
                    logger.error("[rl_planner] step %d: dead state (Q<=0, unchanged) — aborting", step)
                    return self._finish(history, goal, target, Action(kind="abort", text="dead state: no promising action left; stop and ask the user"))
            else:
                self._dead_streak = 0

            # 5. Pick next action (epsilon-greedy over Q).
            action_idx = self._select_action_idx(state)
            action = self._action_from_kind(self.ACTION_KINDS[action_idx], driver, goal)
            logger.info("[rl_planner] step %d/%d: state=%s action=%s(%s) goal_met=%s progress=%s",
                        step, self.step_cap, list(state), action.kind,
                        (action.selector or action.text or action.url or "")[:40],
                        goal_met, progress_met)

            # 6. Guardrails: policy + terminal + blind.
            if action.kind == "done":
                # Only accept done when the goal verifier confirms it.
                if goal_met:
                    return self._finish(history, goal, target, action, q_state=state, q_idx=action_idx)
                last_outcome = "noop"
                last_sig = self._action_sig(action)
                step_bucket += 1
                continue
            if action.kind == "abort":
                return self._finish(history, goal, target, action)
            if action.kind == "driver_swap":
                # Swap is an RL action: agent explicitly chooses the medium.
                action = Action(kind="observe", driver="browser" if driver == "desktop" else "desktop")
                driver = action.driver
            else:
                action = Action(
                    kind=action.kind,
                    selector=action.selector,
                    text=action.text,
                    url=action.url,
                    driver=driver,
                )

            try:
                self.policy.validate_action(action)
            except PolicyViolation as e:
                return self._finish(history, goal, target, Action(kind="abort", text=f"blocked by policy: {e}"))

            # 7. Execute.
            result = {"status": "dry_run"}
            if not self.dry_run:
                result = self._run_execute(action, target)

            # 8. Classify + reward + Q-update.
            outcome, reward = self._classify_step(action, result, goal_met, progress_met, last_sig)
            # Step-cap safety: reaching the cap without goal => storm penalty.
            if step >= self.step_cap and not goal_met:
                reward = min(reward, R_STEPCAP)

            next_row = self._row(state)
            prev = next_row[action_idx]
            next_state_max = 0.0  # terminal-ish bootstrap: no next-state lookahead in tabular pass
            next_row[action_idx] = prev + self.alpha * (
                reward + self.gamma * next_state_max - prev
            )

            history.append({
                "step": step,
                "state": list(state),
                "action": action.model_dump(),
                "result": result,
                "reward": reward,
                "outcome": outcome,
            })
            logger.info("[rl_planner] step %d: executed=%s outcome=%s reward=%+.2f Q[%s][%s]=%.3f",
                        step, action.kind, outcome, reward, list(state), action_idx,
                        self._row(state)[action_idx])
            last_sig = self._action_sig(action)
            last_outcome = outcome
            step_bucket += 1

        # Step cap reached without goal.
        return self._finish(history, goal, target, Action(kind="abort", text="step cap reached without goal verified (return abort, not ok)"))

    # ── helpers ──────────────────────────────────────────────────────────
    def _finish(self, history, goal, target, action: Action, q_state=None, q_idx=None) -> ActionBatch:
        if q_state is not None and q_idx is not None and q_state in self._q:
            row = self._q[q_state]
            prev = row[q_idx]
            row[q_idx] = prev + self.alpha * (R_GOAL + self.gamma * 0.0 - prev)
        # Persist learned Q after each completed run.
        if self._q_path is not None:
            self.save_q()
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0.02)
        return ActionBatch(actions=[action])

    def _run_execute(self, action, target: dict) -> dict:
        if self._execute is not None:
            return self._execute(action, target)
        if self.driver is not None:
            try:
                return self.driver.execute(action, target)
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "ok"}

    def _run_verify(self, goal: str, observation, screenshot) -> tuple[bool, bool]:
        """Return (goal_met, progress_met). Goal/progress come from an injectable
        verifier when available; otherwise use cheap heuristic signals on the
        observation/URL (goal text present in page = progress)."""
        if self._verify is not None:
            try:
                return self._verify(goal, observation, screenshot)
            except Exception:
                pass
        # Cheap default: if the observation contains a URL/text, treat matching
        # the goal as progress only if we actually observed something.
        goal_met = False
        progress_met = False
        if observation is not None:
            haystack = " ".join([
                str(observation.url or ""),
                str(observation.text or ""),
            ])
            g = (goal or "").strip().lower()
            if g and len(g) > 3:
                progress_met = any(w in haystack.lower() for w in g.split()[:2])
        return goal_met, progress_met

    @staticmethod
    def _screen_hash(screenshot) -> str | None:
        if not screenshot:
            return None
        try:
            from PIL import Image
            import io
            import base64
            if isinstance(screenshot, str) and screenshot.startswith("data:"):
                payload = screenshot.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(payload)))
            elif isinstance(screenshot, str) and Path(screenshot).exists():
                img = Image.open(screenshot)
            else:
                return None
            img = img.convert("L").resize((16, 16), Image.LANCZOS)
            px = list(img.getdata())
            avg = sum(px) / len(px) if px else 0.0
            bits = "".join("1" if v > avg else "0" for v in px)
            return hashlib.sha256(bits.encode()).hexdigest()
        except Exception:
            return None

    @staticmethod
    def _extract_url(goal: str) -> str:
        """Pull the first http(s) URL out of a goal string, else empty."""
        import re
        m = re.search(r"https?://[^\s\"']+", goal or "")
        return m.group(0).rstrip(".),") if m else ""

    @staticmethod
    def _action_from_kind(kind: str, driver: str, goal: str) -> Action:
        """Bind a discrete action kind into a concrete Action, goal-aware.

        The RL agent owns WHAT to do and WHICH medium; here we bind the goal
        content so the action can actually execute: `navigate` gets the URL
        from the goal, `type` gets the text to enter, `hotkey` gets a sensible
        key. Without this, navigate fired with an empty URL (driver rejects it)
        and type had no text — the RL loop churned into observe->error->dead.
        """
        if kind == "done":
            return Action(kind="done", driver=driver)
        if kind == "abort":
            return Action(kind="abort", driver=driver)
        if kind == "driver_swap":
            return Action(kind="observe", driver=driver)
        if kind == "navigate":
            return Action(kind="navigate", url=RLPlanner._extract_url(goal), driver=driver)
        if kind == "click":
            return Action(kind="click", selector="0,0", driver=driver)
        if kind == "type":
            # Type the tail of the goal after a fill/type verb, stripped of
            # quotes/punctuation, as a best-effort text binding.
            text = (goal or "").strip()
            for verb in ("type ", "fill ", "enter ", "type:", "fill:"):
                if verb in text:
                    text = text.split(verb, 1)[1]
                    break
            text = text.strip().strip('"\'')
            return Action(kind="type", text=text, driver=driver)
        if kind == "hotkey":
            return Action(kind="hotkey", text="ctrl+l", driver=driver)
        if kind in ("observe", "scroll", "wait", "submit"):
            return Action(kind=kind, driver=driver)
        return Action(kind="observe", driver=driver)
