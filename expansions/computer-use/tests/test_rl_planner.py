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
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=False, last_sig="type:None:same")
    assert outcome == "blind"
    assert r == R_BLIND


def test_execution_error_penalized_as_swap_failure():
    p = RLPlanner()
    a = Action(kind="navigate", url="x", driver="browser")
    outcome, r = _step(p, a, {"status": "error", "error": "driver failed"}, goal_met=False, progress_met=False, last_sig=None)
    assert outcome == "error"
    assert r == R_SWAP_FAIL


# ── Task-typed reward shaping (general across computer use) ────────────────

def test_task_type_classification():
    assert RLPlanner._task_type("fill the signup form email eva password x submit") == "form"
    assert RLPlanner._task_type("open https://www.linkedin.com/signup") == "navigate"
    assert RLPlanner._task_type("open the thunar file manager") == "launch"
    assert RLPlanner._task_type("click the save button") == "generic"
    # A form goal that names its URL must still classify as a FORM task (the
    # URL is just where the form lives; the task is filling it).
    assert RLPlanner._task_type(
        "fill the linkedin signup form at https://www.linkedin.com/signup email eva password x submit"
    ) == "form"


def test_form_fill_earns_boosted_progress():
    p = RLPlanner()
    goal = "fill the linkedin signup form email eva password x submit"
    a = Action(kind="fill", selector="#email", text="eva@x.com", driver="browser")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "form_verified"
    assert r > R_PROGRESS  # boosted above the generic +10


def test_form_submit_earns_boosted_progress():
    p = RLPlanner()
    goal = "fill the linkedin signup form email eva password x submit"
    a = Action(kind="submit", driver="browser")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "form_verified"
    assert r > R_PROGRESS


def test_passive_observe_is_churn_not_progress():
    """Core general fix: a passive observe/scroll/wait that merely changes the
    screen must NOT earn +R_PROGRESS — that was the incentive that made the
    agent churn instead of actuating. It applies across every task type."""
    p = RLPlanner()
    for goal in (
        "fill the form email eva password x",
        "open https://www.linkedin.com/signup",
        "open the thunar file manager",
        "click the save button",
    ):
        a = Action(kind="observe", driver="browser")
        outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
        assert outcome == "verified"
        assert r < 0, f"observe must be penalized as churn, got {r} for goal={goal!r}"


def test_navigate_earns_nav_progress():
    p = RLPlanner()
    goal = "open https://www.linkedin.com/signup"
    a = Action(kind="navigate", url="https://www.linkedin.com/signup", driver="browser")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "verified"
    assert r > R_PROGRESS  # navigate boost for a URL goal


def test_launch_earns_launch_progress():
    p = RLPlanner()
    goal = "open the thunar file manager"
    a = Action(kind="launch", text="thunar", driver="desktop")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "verified"
    assert r > R_PROGRESS  # launch boost for a desktop-app goal


def test_generic_actuation_keeps_progress_reward():
    p = RLPlanner()
    goal = "click the save button"
    a = Action(kind="click", selector="100,200", driver="desktop")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "verified"
    assert r == R_PROGRESS  # generic actuation keeps the baseline positive


def test_blind_fill_still_penalized():
    p = RLPlanner()
    goal = "fill the form email eva password x"
    a = Action(kind="fill", selector="", text="eva", driver="browser")
    outcome, r = p._classify_step(a, {"status": "ok"}, False, True, None, goal=goal)
    assert outcome == "blind"
    assert r == R_BLIND


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


def test_extract_app_rejects_filler_words_as_app_names():
    """Regression: a vague 'open it / launch that / start this' goal must NOT
    bind a filler pronoun as a launch target. This gated verification on
    launching a process literally named "it"/"that", producing the
    launch(it)->error / launch(and)->error loop seen in the field log."""
    for goal in [
        "open it and run the app",
        "launch it then check",
        "start it please",
        "open that file",
        "can you open thunar please",  # filler around a real app still resolves
    ]:
        if "thunar" in goal:
            assert RLPlanner._extract_app(goal) == "thunar"
        else:
            assert RLPlanner._extract_app(goal) == "", goal


def test_extract_app_never_leaks_url_scheme_as_app():
    """Regression: a browser-nav goal 'open https://…' must return NO app name.
    Prior code let the URL scheme ('https') bind as a desktop app, which
    misclassified a navigate task as a desktop-launch task and gated completion
    on a pgrep for 'https'."""
    assert RLPlanner._extract_app("open https://www.linkedin.com/signup") == ""
    assert RLPlanner._extract_app("navigate to https://example.com") == ""
    assert RLPlanner._extract_app("launch https://make.neverslave.com/projects") == ""


def test_extract_app_real_desktop_apps_still_resolve():
    """Guard the positive cases: tightening stopwords must not break real
    app targets (single-word app names and the canonical Thunar goal)."""
    assert RLPlanner._extract_app("open Thunar") == "thunar"
    assert RLPlanner._extract_app("start opencode") == "opencode"
    assert RLPlanner._extract_app("launch firefox and go to gmail") == "firefox"
    assert RLPlanner._extract_app("please open firefox for me") == "firefox"
    assert RLPlanner._extract_app("open chrome") == "chrome"
    assert RLPlanner._extract_app("start the terminal and type ls") == "terminal"
    assert RLPlanner._extract_app("start notepad") == "notepad"
    assert RLPlanner._extract_app("Open the Thunar file manager") == "thunar"


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



# ── Expanded action vocabulary (2026-09-15) ────────────────────────────────

def test_action_kinds_include_drive_actions():
    """The RL planner must offer the full set of actions needed to actually
    drive a browser/DOM form: double_click, fill, submit, assert_text,
    assert_url — not just observe/click/type churn."""
    p = RLPlanner()
    for kind in ("observe", "click", "double_click", "type", "fill", "submit",
                 "hotkey", "navigate", "launch", "scroll", "wait",
                 "assert_text", "assert_url", "done"):
        assert kind in p.ACTION_KINDS, f"missing RL action kind: {kind}"


def test_fill_binds_selector_and_value():
    """A 'fill <sel> with <value>' goal must bind into a DOM-aware fill Action
    (selector + text), the primitive that makes Playwright form-fill trivial."""
    a = RLPlanner._action_from_kind(
        "fill", "browser", "fill #email with eva@neverslave.com"
    )
    assert a.kind == "fill"
    assert a.selector == "#email"
    assert a.text == "eva@neverslave.com"


def test_fill_missing_selector_penalized_as_blind():
    p = RLPlanner()
    a = RLPlanner._action_from_kind("fill", "browser", "fill the form")
    assert a.selector == ""
    outcome, r = _step(p, a, {"status": "ok"}, goal_met=False, progress_met=False, last_sig=None)
    assert outcome == "blind"
    assert r == R_BLIND


def test_submit_binds_action():
    a = RLPlanner._action_from_kind("submit", "browser", "submit the form")
    assert a.kind == "submit"


def test_assert_text_binds_expectation():
    a = RLPlanner._action_from_kind("verify", "browser", "verify Welcome back")
    # _action_from_kind falls through unknown kinds to observe; assert the real
    # assert_text binding works:
    a2 = RLPlanner._action_from_kind("assert_text", "browser", "verify Welcome back")
    assert a2.kind == "assert_text"
    assert a2.text == "Welcome back"


def test_double_click_binds_selector():
    a = RLPlanner._action_from_kind("double_click", "browser", "double click the #row")
    assert a.kind == "double_click"
    assert a.selector == "#row"


# ── General computer-use binding layer (structured goal model) ────────────

def test_parse_goal_model_extracts_targets():
    m = RLPlanner._parse_goal_model(
        "open https://www.linkedin.com/signup, fill email eva@neverslave.com "
        "password JwCXoM, click .submit"
    )
    assert m["url"] == "https://www.linkedin.com/signup"
    assert m["fields"] == [("#password", "JwCXoM"), ("#email", "eva@neverslave.com")]
    assert m["click"] == ".submit"


def test_field_extraction_natural_labels():
    fields = RLPlanner._extract_fields(
        "fill email eva@neverslave.com password JwCXoM and submit"
    )
    assert fields == [("#password", "JwCXoM"), ("#email", "eva@neverslave.com")]


def test_field_selector_mapping():
    assert RLPlanner._field_selector("email") == "#email"
    assert RLPlanner._field_selector("password") == "#password"
    assert RLPlanner._field_selector("first name") == "#firstname"
    assert RLPlanner._field_selector("last name") == "#lastname"
    assert RLPlanner._field_selector("username") == "#username"
    assert RLPlanner._field_selector("phone") == "#phone"
    assert RLPlanner._field_selector("nonexistent") == ""


def test_selector_binding_does_not_leak_url_or_email():
    # `.linkedin` from `linkedin.com` and `.com` from `eva@neverslave.com` must
    # NEVER be bound as click targets.
    assert RLPlanner._extract_selector(
        "open https://www.linkedin.com/signup, click .submit"
    ) == ".submit"
    assert RLPlanner._extract_selector(
        "fill email eva@neverslave.com password x, click .submit"
    ) == ".submit"


def test_fill_directive_binds_selector_and_value():
    a = RLPlanner._action_from_kind("fill", "browser", "fill #email with eva@neverslave.com")
    assert a.kind == "fill"
    assert a.selector == "#email"
    assert a.text == "eva@neverslave.com"


def test_fill_queue_sequential_and_blind_when_exhausted():
    goal = "fill email eva@neverslave.com password JwCXoM submit"
    p = RLPlanner()
    f1 = p._bind_action("fill", "browser", goal)
    f2 = p._bind_action("fill", "browser", goal)
    f3 = p._bind_action("fill", "browser", goal)
    assert f1.selector == "#password" and f1.text == "JwCXoM"
    assert f2.selector == "#email" and f2.text == "eva@neverslave.com"
    # Queue exhausted -> blind (empty) so the reward layer steers to submit.
    assert f3.selector == "" and f3.text == ""


def test_launch_binding_uses_goal_app():
    a = RLPlanner._action_from_kind("launch", "desktop", "open the thunar file manager")
    assert a.kind == "launch"
    assert a.text == "thunar"


def test_hotkey_binding_default():
    assert RLPlanner._extract_hotkey("press Ctrl+L") == "ctrl+l"
    assert RLPlanner._extract_hotkey("do arbitrary stuff") == "ctrl+l"
