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
        "launch",
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
        watch_callback=None,    # fn(screenshot, caption) -> None (Telegram stream)
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
        self.watch_callback = watch_callback
        self._dead_streak = 0
        # Signature of the last observation, used for the general progress
        # signal ("the world changed since the last step"). Modality-agnostic:
        # works for browser (url/text) and desktop (screen) alike.
        self._obs_sig: str | None = None

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

    def _live_driver_mode(self, fallback: str) -> str:
        """Return the HybridDriver's live runtime mode (desktop | browser).

        The RL state must reflect which medium the agent is actually on after
        the desktop<->browser handoff, not a hardcoded target.kind. When the
        driver exposes a `.mode` attribute (HybridDriver), read it; otherwise
        fall back to the current driver string.
        """
        d = self.driver
        if d is not None:
            mode = getattr(d, "mode", None)
            if mode in ("desktop", "browser"):
                return mode
        return fallback

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
            # Desktop-app goal (no URL): seed `launch` bias so a cold-start
            # agent opens the named app instead of churning observe/scroll.
            app = RLPlanner._extract_app(goal)
            if app:
                launch_idx = self.ACTION_KINDS.index("launch")
                driver_bucket = 0 if driver == "desktop" else 1 if driver == "browser" else 2
                for screen in range(10):
                    for outcome in range(6):
                        for progress in (0, 1):
                            for step in range(3):
                                state = (screen, driver_bucket, outcome, progress, step, 0)
                                row = self._row(state)
                                if row[launch_idx] <= 0.0:
                                    row[launch_idx] = 1.5
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
        # Start from the requested target kind, but the live driver mode is
        # read fresh each step from the HybridDriver so the RL state reflects
        # the real desktop<->browser handoff, not a hardcoded target.
        driver = (target.get("kind") or "desktop")
        self._dead_streak = 0

        # Goal-aware seed policy: cold-start tabular Q-learning has no prior
        # that "goal contains http(s) -> navigate first". Without a seed it
        # explores randomly and never lands on navigate within a short live
        # run. Pre-bias the navigate action for blank-url states so the agent
        # starts with the right bias instead of random exploration.
        self._seed_goal_policy(goal, driver)

        for step in range(1, self.step_cap + 1):
            # 1. Perceive — capture the current screen + location.
            screenshot = None
            # The bridge passes observation=None, so we must read the URL/text
            # from the driver itself — otherwise url_bucket stays 0 and the
            # agent can't perceive that it navigated (the live run showed
            # url stuck at about:blank even after navigate). Refresh the
            # observation EVERY step, not just the first: a stale observation
            # captured at about:blank is reused forever, so url_bucket never
            # updates after navigate even though the page changed.
            if self.driver is not None:
                try:
                    screenshot = self.driver.screenshot()
                except Exception as e:
                    logger.warning("[rl_planner] screenshot failed: %s", e)
                try:
                    observation = self.driver.observe(target)
                except Exception as e:
                    logger.warning("[rl_planner] observe failed: %s", e)

            # Stream this step's screenshot to the user (Telegram) when a
            # watch_callback is wired — mirrors the LLM planner path so RL
            # runs show live progress instead of silence.
            self._watch(screenshot, f"step {step}/{self.step_cap}: {goal}")

            # 1.5 Live driver mode: reflect the HybridDriver's real runtime
            # mode (desktop -> browser handoff), not a hardcoded target.kind.
            # The RL state must know which medium the agent is actually on.
            driver = self._live_driver_mode(driver)

            # 2. Progress / goal verification signal.
            # General progress: the world changed since the last step (screen
            # hash or observed text/URL moved). This is modality-agnostic — it
            # works for browser AND desktop, unlike a URL-specific gate.
            goal_met, progress_met = self._run_verify(
                goal, observation, screenshot, prev_obs_sig=self._obs_sig
            )

            # 2.5 Terminal: goal verified → emit `done` immediately and stop.
            # Without this the ε-greedy policy keeps picking `observe` and
            # churns +100 rewards to step-cap; a verified goal is terminal, so
            # we must end the run with `done` (and the terminal +100) right
            # here rather than letting exploration pick another action.
            if goal_met:
                logger.info("[rl_planner] step %d: goal verified — emitting done", step)
                return self._finish(history, goal, target, Action(kind="done", text="goal verified"))

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
            # Remember this step's observation signature for the general
            # progress signal on the next step ("the world changed").
            if observation is not None:
                self._obs_sig = self._obs_signature(observation, screenshot)

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

    def _run_verify(self, goal: str, observation, screenshot, prev_obs_sig: str | None = None) -> tuple[bool, bool]:
        """Return (goal_met, progress_met).

        goal_met: injectable verifier when available, else False (a general
        goal verifier must inspect whatever modality exists — text, URL, or
        screen — never assume a URL).

        progress_met: GENERAL, modality-agnostic signal. When an injectable
        verifier exists it decides; otherwise progress = "the world changed
        since the last step" (observation signature differs from the previous
        step). This works for browser AND desktop — no URL assumption.
        """
        if self._verify is not None:
            try:
                return self._verify(goal, observation, screenshot)
            except Exception:
                pass
        # General default: progress = the world changed since the last step.
        # Compare the current observation signature to the previous one.
        goal_met = False
        progress_met = False
        if observation is not None:
            sig = self._obs_signature(observation, screenshot)
            progress_met = bool(sig) and sig != prev_obs_sig
            # Default goal check (modality-agnostic): confirm the goal when the
            # observed URL/text actually matches what the goal asks for. This is
            # what lets the agent emit a verified `done` (terminal +100) instead
            # of churning to step-cap and aborting.
            goal_met = self._default_goal_met(goal, observation)
        return goal_met, progress_met

    @staticmethod
    def _default_goal_met(goal: str, observation) -> bool:
        """Modality-agnostic default goal verifier.

        Returns True when the observed URL/text satisfies the goal. Never
        assumes a URL exists (desktop has none): it checks whatever modality
        is present.

        1. If the goal names a URL (http/https), confirm when the observed URL
           matches its host (+ path when the goal path is non-trivial).
        2. Otherwise, confirm when the observed text contains a distinctive
           goal token (a word of >=5 chars from the goal).
        """
        g = (goal or "").strip()
        if not g:
            return False
        obs_url = (observation.url or "") if observation is not None else ""
        obs_text = (observation.text or "") if observation is not None else ""

        # 1. URL goal: match host (+ non-trivial path).
        import re as _re
        m = _re.search(r"https?://([^/\s]+)(/[^\s]*)?", g, _re.IGNORECASE)
        if m:
            goal_host = (m.group(1) or "").lower().rstrip("/")
            goal_path = (m.group(2) or "").rstrip("/")
            if goal_host and obs_url:
                try:
                    from urllib.parse import urlparse
                    obs = urlparse(obs_url)
                    obs_host = (obs.hostname or "").lower()
                    obs_path = (obs.path or "").rstrip("/")
                except Exception:
                    obs_host, obs_path = "", ""
                if obs_host and obs_host == goal_host:
                    # Non-trivial goal path must also match; else host match suffices.
                    if len(goal_path) > 1:
                        return obs_path == goal_path
                    return True
            return False

        # 2. Text goal: a distinctive goal token present in the observed text.
        #    (Desktop observations often carry empty URL+text, so this only
        #    short-circuits on an actual text match — it does NOT gate the
        #    desktop check below behind non-empty observation.)
        tokens = [w for w in _re.split(r"[^A-Za-z0-9]+", g) if len(w) >= 5]
        low_text = obs_text.lower()
        if tokens and any(t.lower() in low_text for t in tokens):
            return True

        # 3. Desktop-app goal: the goal names an app (no URL, no text match) —
        #    confirm when a matching window/process is present on the desktop.
        #    This runs EVEN when the observation is empty (empty URL+text is
        #    normal for a desktop window): the app-open check is independent of
        #    observation content. This is the real done-signal that lets desktop
        #    tasks terminate with a verified `done` instead of "no observation =
        #    no progress -> iterate forever / dead/abort".
        return RLPlanner._desktop_app_open(goal)

    @staticmethod
    def _desktop_app_open(goal: str) -> bool:
        """True when the goal-named desktop app is actually running.

        The window-title heuristic failed for Thunar: its title is the folder
        name ("repositories — File Manager"), not the binary name "thunar", so
        matching on window titles never saw the app the `launch` action opened.
        A running process is the reliable signal — if `launch` succeeded (A),
        the app binary appears in pgrep. Best-effort: any failure returns False
        so verification never false-positives.
        """
        import subprocess
        app = (RLPlanner._extract_app(goal) or "").lower()
        if not app:
            return False
        try:
            out = subprocess.run(
                ["pgrep", "-x", app], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return True
            # Case/alias fallback: match by substring against command lines.
            out = subprocess.run(
                ["pgrep", "-f", app], capture_output=True, text=True, timeout=5,
            )
            return out.returncode == 0 and bool(out.stdout.strip())
        except Exception:
            return False

    @staticmethod
    def _obs_signature(observation, screenshot) -> str | None:
        """A stable signature of what the agent currently perceives, across
        modalities: URL, visible text, and screen hash. Used as the general
        progress signal ("the world changed"). None when nothing is observed.
        """
        parts = []
        if observation is not None:
            if observation.url:
                parts.append("u:" + str(observation.url))
            if observation.text:
                parts.append("t:" + str(observation.text)[:2000])
        if screenshot:
            parts.append("s:" + str(RLPlanner._screen_hash(screenshot)))
        return "|".join(parts) if parts else None

    def _watch(self, screenshot: str | None, caption: str) -> None:
        """Stream a screenshot to the user (Telegram) when a watch_callback is
        wired — mirrors the LLM planner path so RL runs show live progress."""
        if self.watch_callback is None:
            return
        try:
            self.watch_callback(screenshot, caption)
        except Exception as e:
            logger.warning("[rl_planner] watch callback error: %s", e)

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
    def _extract_app(goal: str) -> str:
        """Pull a desktop app name out of a goal string, else empty.

        Heuristic, best-effort: prefer an explicit launch/open verb target
        ("open Thunar" -> "thunar"), else a capitalized standalone token
        that isn't a URL. Only used to bind/seed the `launch` action; the
        desktop driver resolves the actual binary via PATH/xdg-open.
        """
        import re
        g = (goal or "").strip()
        if not g:
            return ""
        skip = {"the", "a", "an", "browser", "desktop", "web", "file", "manager", "application"}
        for m in re.finditer(
            r"\b(?:open|launch|start)\s+(?:the\s+)?([A-Za-z][A-Za-z0-9_.-]{1,40})",
            g, re.IGNORECASE,
        ):
            app = m.group(1).lower()
            if app not in ("browser", "desktop", "web"):
                return app
        for tok in re.findall(r"[A-Z][A-Za-z0-9_.-]{1,40}", g):
            low = tok.lower()
            if not re.search(r"https?://", low) and low not in skip:
                return low
        return ""

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
        if kind == "launch":
            # Desktop app launch: bind the app name from the goal (e.g.
            # "Open the Thunar file manager" -> "thunar"). The desktop driver
            # resolves it via PATH/xdg-open.
            return Action(kind="launch", text=RLPlanner._extract_app(goal), driver=driver)
        if kind in ("observe", "scroll", "wait", "submit"):
            return Action(kind=kind, driver=driver)
        return Action(kind="observe", driver=driver)
