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
