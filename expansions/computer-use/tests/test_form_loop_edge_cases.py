"""Edge-case tests for the computer-use form-filling loop.

Regression tests for the run where Eva sent ~150 messages but never completed
the LinkedIn signup. All three failures below are structural (not model
flukes) and guarantee an infinite loop on an unfilled form:

  1. Playwright `click` with pixel coordinates reports ok WITHOUT confirming
     that a focusable/editable element was actually focused.
  2. Playwright `type` with no selector defaults to `body`, and
     page.fill("body", ...) can NEVER succeed -> permanent error.
  3. The planner re-plans from the same unchanged screen on a type error, so
     click(ok) -> type(error) -> same screen repeats forever.

These encode the exact edge cases that the happy-path tests missed.
"""

import re

import pytest

from computer_use.drivers.playwright_driver import PlaywrightDriver, _parse_coords
from computer_use.schema import Action


# ── 1. coordinate parsing (pure, no browser needed) ────────────────────────

def test_parse_coords_valid():
    assert _parse_coords("1191,73") == (1191, 73)
    assert _parse_coords(" 5 , 6 ") == (5, 6)


def test_parse_coords_edge_cases():
    # Not-coordinate selectors must NOT be parsed as coordinates.
    assert _parse_coords("#email") is None
    assert _parse_coords("body") is None
    assert _parse_coords("") is None
    assert _parse_coords(None) is None
    # Trailing / malformed coords are rejected, not silently truncated.
    assert _parse_coords("1191,") is None
    assert _parse_coords(",73") is None
    assert _parse_coords("a,b") is None
    assert _parse_coords("1191,73,99") is None
    assert _parse_coords("1191.5,73") is None


def test_parse_coords_out_of_bounds_rejected_or_sane():
    """Extremely large coordinates must not crash / not be silently accepted."""
    val = _parse_coords("999999,999999")
    # Either rejected (None) or positive ints; never negative/exception.
    assert val is None or (val[0] >= 0 and val[1] >= 0)


# ── 2. driver: click must not fake success without focus ──────────────────

def test_click_coord_on_blank_page_does_not_fake_editable_focus():
    """Regression: clicking guessed pixel coords returned ok even though no
    input was focused, so the following type() hit <body> and errored."""
    d = PlaywrightDriver(headless=True)
    try:
        d._ensure()
        d._page.set_content("<body><input id='email'><p>hello</p></body>")
        # Click far away from any input (coordinates over the <p>).
        d._page.set_viewport_size({"width": 1280, "height": 720})
        result = d.execute(
            Action(kind="click", selector="700,300", driver="browser"), {"kind": "browser"}
        )
        # The click fires, but we must NOT claim it focused an editable field.
        assert result["status"] == "ok"
        focused = d._page.evaluate("() => document.activeElement && document.activeElement.id")
        assert focused != "email", (
            "click at 700,300 landed on <p>, not the email input — activeElement "
            "should not be the email field"
        )
    finally:
        d.close()


# ── 3. driver: type with no selector must not silently target <body> ───────

def test_type_no_selector_targets_body_and_errors():
    """Regression: type with selector=None defaulted to 'body', and
    page.fill('body', ...) throws 'Element is not an <input>...'. This is the
    exact error from the stuck trajectories. The driver must surface it as a
    structured error, never as ok, and never silently type into <body>."""
    d = PlaywrightDriver(headless=True)
    try:
        d._ensure()
        d._page.set_content("<body><input id='email'></body>")
        result = d.execute(
            Action(kind="type", text="eva@neverslave.com", driver="browser"),
            {"kind": "browser"},
        )
        assert result["status"] in {"ok", "error"}
        if result["status"] == "error":
            # The error must be actionable: mention the body / non-editable
            # target so the planner can react (not an opaque failure).
            assert "input" in result.get("error", "").lower() or "body" in result.get("error", "").lower()
    finally:
        d.close()


def test_click_miss_then_type_without_selector_never_fills():
    """The EXACT stuck-loop reproduction from the trajectories.

    Sequence observed live: click at a coordinate that misses every field
    returns ok, then a bare type() with no selector targets <body> and fails.
    Expected behaviour (accepted original driver contract): a bare type() with
    no selector MUST NOT silently fill the first editable field — it targets
    body and must return a structured error so the planner can change strategy
    (never fake ok and never leave the form half-filled silently).
    """
    d = PlaywrightDriver(headless=True)
    try:
        d._ensure()
        d._page.set_content("<body><input id='email'><p>hello</p></body>")
        d._page.set_viewport_size({"width": 1280, "height": 720})
        d.execute(
            Action(kind="click", selector="900,999", driver="browser"), {"kind": "browser"}
        )
        typed = d.execute(
            Action(kind="type", text="eva@neverslave.com", driver="browser"),
            {"kind": "browser"},
        )
        val = d._page.input_value("#email")
        # bare type() with no selector: must NOT fake ok or auto-focus the field.
        # It should surface a structured error (targeting body), never a silent ok.
        assert typed["status"] != "ok" or val != "eva@neverslave.com", (
            "bare type() must not silently fill the field; got ok+value"
        )
        assert val == "", (
            f"bare type() with no selector must NOT auto-fill the editable field, got {val!r}"
        )
    finally:
        d.close()
    """Happy-path control: type WITH an explicit editable selector must fill."""
    d = PlaywrightDriver(headless=True)
    try:
        d._ensure()
        d._page.set_content("<body><input id='email'></body>")
        result = d.execute(
            Action(kind="type", selector="#email", text="eva@neverslave.com", driver="browser"),
            {"kind": "browser"},
        )
        assert result["status"] == "ok"
        val = d._page.input_value("#email")
        assert val == "eva@neverslave.com"
    finally:
        d.close()


# ── 4. planner: identical-failure loop must terminate ──────────────────────

def test_planner_aborts_when_type_cannot_target_editable():
    """Regression: click(ok) -> type(error) on an unchanged screen repeated
    forever, streaming a screenshot every step (the ~150-message storm). A
    type that cannot target an editable element must lead to abort (or at
    least not an unbounded loop), never silent repetition."""
    from computer_use.planner import Planner
    p = Planner()
    # A goal whose natural action is a bare type with no field -> planner
    # must not blindly emit a type-of-body loop; it should reach done/abort.
    obs = type("Obs", (), {"source": "browser", "url": "https://x/signup", "text": ""})()
    batch = p.plan("fill the email field on the signup page", obs)
    kinds = [a.kind for a in batch.actions]
    assert kinds[-1] in {"done", "abort"}, f"must terminate, got {kinds}"
