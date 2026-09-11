"""Hybrid desktop/browser driver with graceful runtime switching.

Starts on the desktop (PyAutoGUI) driver. When the desktop driver fails an
action AND the goal looks browser-ish, it hands off to the Playwright browser
driver to handle the browser phase (navigate / fill forms by selector / read
the DOM). Once the browser phase is done — the model emits a desktop-only
action, or the browser driver can't handle the action — it switches back to
the desktop driver for the rest of the task.

The planner only ever sees `observe` / `screenshot` / `execute`, so this
wrapper needs no planner changes.
"""

from __future__ import annotations

from computer_use.schema import Observation
from computer_use.drivers.base import BaseDriver

# Goal keywords that hint the task involves a browser. Used as the gate for
# the desktop -> browser handoff so we don't yank a pure-desktop task into a
# browser we don't need.
_BROWSER_GOAL_WORDS = (
    "browser", "firefox", "chrome", "edge", "safari", "navigate",
    "form", "signup", "login", "register", "url", "http", "www",
    "search", "google", "linkedin", "facebook", "twitter", "youtube",
    "website", "webpage", "page",
)


class HybridDriver(BaseDriver):
    """Delegates to a desktop or browser driver, switching at runtime."""

    def __init__(self, desktop: BaseDriver, browser: BaseDriver, goal: str = ""):
        self.desktop = desktop
        self.browser = browser
        self.goal = goal or ""
        self.mode = "desktop"  # "desktop" | "browser"
        self._switched_to_browser = False

    # ── helpers ───────────────────────────────────────────────────────────
    def _current(self):
        return self.browser if self.mode == "browser" else self.desktop

    def _goal_is_browser(self) -> bool:
        g = self.goal.lower()
        return any(w in g for w in _BROWSER_GOAL_WORDS)

    def _switch_to_browser(self, reason: str) -> None:
        if self.mode != "browser":
            print(f"[hybrid] desktop -> browser ({reason})", flush=True)
            self.mode = "browser"
            self._switched_to_browser = True

    def _switch_to_desktop(self, reason: str) -> None:
        if self.mode != "desktop":
            print(f"[hybrid] browser -> desktop ({reason})", flush=True)
            self.mode = "desktop"

    # ── BaseDriver interface ─────────────────────────────────────────────
    def observe(self, target: dict) -> Observation:
        return self._current().observe(target)

    def screenshot(self, path: str | None = None) -> str | None:
        d = self._current()
        if hasattr(d, "screenshot"):
            return d.screenshot(path)
        return None

    def is_frame_black(self, threshold: float = 6.0) -> bool:
        d = self._current()
        if hasattr(d, "is_frame_black"):
            return d.is_frame_black(threshold)
        return False

    def close(self) -> None:
        for d in (self.desktop, self.browser):
            try:
                if hasattr(d, "close"):
                    d.close()
            except Exception:
                pass

    def execute(self, action, target: dict) -> dict:
        kind = getattr(action, "kind", None)
        # The LLM decides the medium for each action via the `driver` field
        # ("desktop" | "browser"). When it declares a driver, switch to it
        # explicitly — this is authoritative and replaces the old fragile
        # "switch on desktop failure" heuristic (blind clicks return false-
        # success, so that heuristic never fired).
        declared = getattr(action, "driver", None)
        if declared == "browser":
            self._switch_to_browser("llm declared browser")
            return self.browser.execute(action, target)
        if declared == "desktop":
            self._switch_to_desktop("llm declared desktop")
            return self.desktop.execute(action, target)

        # Fallback when the LLM didn't declare a driver: keep the old
        # graceful behavior so nothing regresses.
        if self.mode == "desktop":
            # A `navigate` action is browser-only — pyautogui can't do it.
            if kind == "navigate":
                self._switch_to_browser("navigate action")
                return self.browser.execute(action, target)

            result = self.desktop.execute(action, target)

            # Desktop failed AND the goal looks browser-ish -> try the browser
            # driver for this action.
            if result.get("status") == "error" and self._goal_is_browser():
                self._switch_to_browser(f"desktop {kind} failed: {result.get('error')}")
                return self.browser.execute(action, target)

            return result

        # ── browser mode (fallback path) ──────────────────────────────
        # A `launch` is a desktop action — the browser phase is over, go back
        # to the desktop to handle the rest of the task.
        if kind == "launch":
            self._switch_to_desktop("launch is a desktop action")
            return self.desktop.execute(action, target)

        result = self.browser.execute(action, target)

        # The browser driver can't handle this action (unsupported) — return
        # to the desktop driver to handle the rest.
        if result.get("status") == "error" and "unsupported" in (result.get("error") or ""):
            self._switch_to_desktop(f"browser unsupported: {result.get('error')}")
            return self.desktop.execute(action, target)

        return result
