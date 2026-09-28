"""test_playwright_integration.py — real-browser integration test.

Drives actual Chromium via Playwright to prove the driver is real (not a stub):
navigates to a live page, reads the DOM, and executes a real click.

Skipped automatically if Playwright or a Chromium executable is unavailable.
"""

import os
import pytest

from computer_use.drivers.playwright_driver import PlaywrightDriver
from computer_use.schema import Action

pytestmark = pytest.mark.skipif(
    not os.path.exists("/usr/bin/chromium"),
    reason="system Chromium not available",
)


def test_real_browser_navigate_and_observe():
    d = PlaywrightDriver()
    try:
        res = d.execute(Action(kind="navigate", url="https://example.com"), {"kind": "browser"})
        assert res["status"] == "ok", res
        obs = d.observe({"kind": "browser"})
        assert obs.source == "browser"
        assert obs.url and "example.com" in obs.url
        assert obs.text  # real DOM text present
        assert obs.state_hash
    finally:
        d.close()


def test_real_browser_screenshot():
    d = PlaywrightDriver()
    try:
        d.execute(Action(kind="navigate", url="https://example.com"), {"kind": "browser"})
        shot = d.screenshot()
        assert shot and shot.startswith("data:image/png;base64,")
    finally:
        d.close()


def test_real_browser_click_errors_on_missing_selector():
    """A real driver must NOT fake success on a missing element."""
    d = PlaywrightDriver()
    try:
        d.execute(Action(kind="navigate", url="https://example.com"), {"kind": "browser"})
        res = d.execute(Action(kind="click", selector="#does-not-exist"), {"kind": "browser"})
        assert res["status"] == "error"
        assert "error" in res
    finally:
        d.close()
