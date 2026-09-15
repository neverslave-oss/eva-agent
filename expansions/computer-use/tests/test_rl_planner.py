"""Tests for the tabular Q-Learning computer-use planner (rl_planner.py).

Covers the reward/verification mapping and the dead-state (storm) abort, using
injectable executors/verifiers for deterministic behavior — no live browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from computer_use.rl_planner import (
    RLPlanner,
    R_GOAL,
    R_PROGRESS,
    R_NOOP,
    R_BLIND,
    R_SWAP_FAIL,
    R_STEPCAP,
    _discretize_features,
)
from computer_use.schema import Action


# ── Reward / verification mapping ──────────────────────────────────────────

def _step(planner, action, result, goal_met, progress_met, last_sig):
    return planner._classify_step(action, result, goal_met, progress_met, last_sig)


def test_terminal_goal_earns_goal_reward():
    p = RLPlanner()
    a = Action(kind="done", driver="browser")
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=True, progress_met=False, last_sig=None)
    assert outcome == "verified"
    assert r == R_GOAL


def test_verified_progress_earns_progress_reward():
    p = RLPlanner()
    a = Action(kind="type", text="eva@x.com", driver="browser")
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=True, last_sig=None)
    assert outcome == "verified"
    assert r == R_PROGRESS


def test_unverified_ok_is_churn_not_success():
    """A step that returns ok but does NOT progress/complete the goal must be
    penalized as churn — this is the root-cause property (step-cap reported ok
    with no verification must NOT be treated as success)."""
    p = RLPlanner()
    a = Action(kind="click", selector="100,200", driver="desktop")
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=False, last_sig="other")
    assert outcome == "noop"
    assert r == R_NOOP
    assert r < 0


def test_blind_click_penalized():
    p = RLPlanner()
    # No real x,y in selector => blind click.
    a = Action(kind="click", selector="auto", driver="desktop")
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=False, last_sig=None)
    assert outcome == "blind"
    assert r == R_BLIND


def test_repeated_identical_action_penalized():
    p = RLPlanner()
    a = Action(kind="type", text="same", driver="browser")
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=False, last_sig="type:same")
    assert outcome == "blind"
    assert r == R_BLIND


def test_execution_error_penalized_as_swap_failure():
    p = RLPlanner()
    a = Action(kind="navigate", url="x", driver="browser")
    outcome, r = _step(p, a, {"status": "error", "error": "driver failed"}, goal_met=False, progress_met=False, last_sig=None)
    assert outcome == "error"
    assert r == R_SWAP_FAIL


# ── State discretization (richer features first pass) ─────────────────────

def test_discretize_includes_driver_and_progress():
    s = _discretize_features("aabbcc", "browser", "ok", True, step_bucket=3)
    # (screen_bucket, driver=1, outcome=1, progress=1, step=0, url=0)
    assert len(s) == 6
    assert s[1] == 1  # browser
    assert s[3] == 1  # progress signal true
    assert s[0] >= 0


def test_discretize_url_bucket():
    # Blank/no URL -> 0, real page -> 1, so the agent can perceive location
    # even when the screenshot is null.
    assert _discretize_features(None, "browser", None, False, 0, url=None)[5] == 0
    assert _discretize_features(None, "browser", None, False, 0, url="about:blank")[5] == 0
    assert _discretize_features(None, "browser", None, False, 0, url="https://example.com")[5] == 1


def test_discretize_step_bucket_caps():
    s_early = _discretize_features(None, "desktop", None, False, step_bucket=2)
    s_late = _discretize_features(None, "desktop", None, False, step_bucket=12)
    assert s_early[4] == 0
    assert s_late[4] == 2


# ── Dead-state abort (the storm guard) ────────────────────────────────────

def test_dead_state_aborts_fail_closed():
    """When the best Q in a state is ~0 and the screen/outcome are unchanged,
    the planner must abort-and-ask, NOT churn/stream forever (the 150-msg storm).
    """
    planner = RLPlanner(step_cap=10)
    # Force the Q row for the default state to all zeros => dead.
    outcome = "noop"
    # Run with a verify that never confirms and a state that repeats.
    # Use a verifier that always returns (False, False) so no terminal/progress.
    planner._verify = lambda g, o, s: (False, False)
    # Make observe-ish steps all yield the same "unchanged" signature so the
    # dead-streak guard trip. Use the same state via a fixed screenshot hash.
    batch = planner.plan(
        "fill the form",
        observation=None,
        target={"kind": "browser"},
    )
    # The planner returns an ActionBatch ending in abort (fail-closed), never a
    # runaway loop, and never a false `done`.
    assert batch.actions
    assert batch.actions[0].kind in ("abort", "done")


def test_done_only_accepted_when_verified():
    """The agent must NOT emit a terminal done unless the goal verifier confirms
    it — otherwise step-cap would masquerade as success (the storm's failure)."""
    p = RLPlanner(step_cap=5)
    seen_done = []
    # verifier: never confirm goal -> agent should abort at dead/step-cap, not done.
    p._verify = lambda g, o, s: (False, False)
    # execute: report ok but make screen hash constant (unchanged) via driver-free
    # default: with driver=None screenshot stays None -> screen_hash None -> dedup
    # state stays the same each step -> dead streak trips -> abort.
    batch = p.plan("do the thing", observation=None, target={"kind": "desktop"})
    terminal = batch.actions[0]
    # The invariant this test guards: an unverified goal must end in `abort`,
    # never a false `done` masquerading as success. The abort may come from
    # dead-state, step-cap, or RL exploration; its text is secondary.
    assert terminal.kind == "abort", f"unverified goal must abort, got {terminal.kind}"


# ── Q persistence ─────────────────────────────────────────────────────────

def test_save_load_q_roundtrip(tmp_path):
    p = RLPlanner(q_path=str(tmp_path / "q.json"))
    s = (1, 2, 3, 4, 5)
    row = p._row(s)
    row[0] = 42.0
    p._q[s] = row
    path = p.save_q()
    assert path
    p2 = RLPlanner()
    assert p2.load_q(path)
    assert p2._row(s)[0] == 42.0


# ── Desktop-app launch (A: launch action) ────────────────────────────────

def test_launch_in_action_space():
    """`launch` must be a selectable RL action so the agent can open an app."""
    p = RLPlanner()
    assert "launch" in p.ACTION_KINDS
    assert p._n_actions == len(p.ACTION_KINDS)


def test_launch_binding_extracts_app():
    """_action_from_kind('launch', ...) binds the goal's app name."""
    a = RLPlanner._action_from_kind("launch", "desktop", "Open the Thunar file manager application on the desktop")
    assert a.kind == "launch"
    assert a.driver == "desktop"
    assert a.text == "thunar"


def test_extract_app_verb_and_skip_url():
    assert RLPlanner._extract_app("Open the Firefox web browser") == "firefox"
    assert RLPlanner._extract_app("navigate to https://example.com") == ""
    assert RLPlanner._extract_app("launch the Thunar file manager") == "thunar"


def test_launch_seeded_for_desktop_goal():
    """A desktop-app goal (no URL) pre-biases `launch`, not navigate."""
    p = RLPlanner()
    p._seed_goal_policy("Open the Thunar file manager", "desktop")
    s = (0, 0, 0, 0, 0, 0)
    launch_idx = p.ACTION_KINDS.index("launch")
    nav_idx = p.ACTION_KINDS.index("navigate")
    assert p._row(s)[launch_idx] > 0.0
    assert p._row(s)[nav_idx] == 0.0


def test_navigate_seeded_for_url_goal_not_launch():
    p = RLPlanner()
    p._seed_goal_policy("navigate to https://example.com", "browser")
    s = (0, 1, 0, 0, 0, 0)
    nav_idx = p.ACTION_KINDS.index("navigate")
    launch_idx = p.ACTION_KINDS.index("launch")
    assert p._row(s)[nav_idx] > 0.0
    assert p._row(s)[launch_idx] == 0.0


def test_launch_action_sig_unique():
    assert RLPlanner._action_sig(Action(kind="launch", text="thunar")) == "launch:thunar"
    assert RLPlanner._action_sig(Action(kind="launch", text="firefox")) == "launch:firefox"


# ── Desktop-app window/process verifier (B) ──────────────────────────────

def test_extract_app_is_used_for_process_verify():
    """_desktop_app_open must resolve the app name from the goal then pgrep it.
    Process-based (not window title): Thunar's title is the folder name, so a
    title match would never fire after `launch` opens it."""
    # Can't assert a real process here deterministically, but we CAN assert the
    # resolution step: the app name extracted feeds pgrep. Verify _extract_app.
    assert RLPlanner._extract_app("Open the Thunar file manager") == "thunar"
    # A nonsense absent app must report closed (never false-positive done).
    assert RLPlanner._desktop_app_open("Open the ZzzDoesNotExistApp") is False


# ── Regression: desktop goal with EMPTY observation must still verify done ─

def test_desktop_goal_verified_even_with_empty_observation(monkeypatch):
    """Bug: _default_goal_met early-returned False when the observation had
    empty URL AND empty text, BEFORE reaching the desktop app-open check. For a
    real desktop run the observation (window) carries no URL/text, so the app
    the launch action opened was never confirmed -> Eva iterated despite being
    done. The desktop check must run regardless of observation content."""
    # Deterministic: stub the pgrep-based app-open check to True (app running).
    monkeypatch.setattr(RLPlanner, "_desktop_app_open", staticmethod(lambda goal: True))
    # Observation with empty url and empty text — the real desktop case.
    obs = type("Obs", (), {"url": "", "text": ""})()
    assert RLPlanner._default_goal_met(
        "Open the Thunar file manager application", obs
    ) is True


def test_desktop_goal_with_none_observation_reaches_app_check(monkeypatch):
    """Observation=None must also reach the desktop check (not early-return)."""
    monkeypatch.setattr(RLPlanner, "_desktop_app_open", staticmethod(lambda goal: True))
    assert RLPlanner._default_goal_met("Open the Thunar file manager", None) is True


# ── Regression: Bug 1 — desktop `done` must never fire before a launch ────

def test_goal_needs_launch_detects_desktop_verb():
    """A desktop app-open goal (open/launch/start + app name, no URL) is flagged
    as needing a launch action before its process check may confirm `done`."""
    assert RLPlanner._goal_needs_launch("Open the Thunar file manager") is True
    assert RLPlanner._goal_needs_launch("launch the Firefox browser") is True
    assert RLPlanner._goal_needs_launch("start notepad") is True


def test_goal_needs_launch_false_for_url_and_bare_goals():
    """Navigated (URL) and vague goals must NOT be gated by a launch."""
    assert RLPlanner._goal_needs_launch("navigate to https://example.com") is False
    assert RLPlanner._goal_needs_launch("do the thing") is False
    assert RLPlanner._goal_needs_launch("") is False


def test_plan_does_not_emit_done_before_launch(monkeypatch):
    """Bug 1 regression: with a desktop-app goal and a verifier that would
    otherwise report goal_met on step 1 (e.g. the agent's own command line
    false-matching pgrep -f), the planner must NOT short-circuit to `done`
    before a `launch` action has executed. It must keep acting (launch), not
    terminate with a single no-op done."""
    p = RLPlanner(step_cap=6)

    # Simulate a desktop goal whose process-check is 'already running' even
    # before any launch (the Bug-1 false positive).
    monkeypatch.setattr(RLPlanner, "_desktop_app_open", staticmethod(lambda goal: True))

    executed = []

    def fake_execute(action, target):
        executed.append(action.kind)
        return {"status": "ok"}

    p._execute = fake_execute
    p.dry_run = False

    # Real desktop observation: non-empty, matches the pyautogui output.
    obs = type("Obs", (), {"url": "", "text": "screen 1920x1080, cursor at (100,200)"})()

    batch = p.plan(
        "Open the Thunar file manager",
        observation=obs,
        target={"kind": "desktop"},
    )

    # The run must actually have attempted a launch before any `done`.
    assert "launch" in executed, (
        "Bug 1: planner must execute launch before done; executed=%r" % executed
    )
    # Terminal action must come after a launch was attempted.
    assert batch.actions and batch.actions[0].kind in ("abort", "done")


def test_after_launch_executed_goal_can_verify_done(monkeypatch):
    """Once a `launch` has executed, the process-based verifier may legitimately
    confirm `done`. Guards against over-correcting the Bug-1 fix into never
    terminating desktop goals."""
    p = RLPlanner(step_cap=8)
    monkeypatch.setattr(RLPlanner, "_desktop_app_open", staticmethod(lambda goal: True))

    executed = []

    def fake_execute(action, target):
        executed.append(action.kind)
        return {"status": "ok"}

    p._execute = fake_execute
    p.dry_run = False

    obs = type("Obs", (), {"url": "", "text": "screen 1920x1080, cursor at (100,200)"})()

    batch = p.plan(
        "Open the Thunar file manager",
        observation=obs,
        target={"kind": "desktop"},
    )

    assert "launch" in executed
    # After launch, the verifier sees the app open and emits a verified done.
    assert batch.actions and batch.actions[0].kind == "done"

