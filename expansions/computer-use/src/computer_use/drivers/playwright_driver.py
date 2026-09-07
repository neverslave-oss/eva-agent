"""playwright_driver.py — real browser driver backed by Playwright + Chromium.

Drives a real browser (system Chromium by default) headless. Provides real
perception (DOM snapshot + screenshot) and real action execution
(navigate/click/type/scroll/wait/assert_text/assert_url/submit/done).

The driver is lazy: Playwright is imported only when the driver is first used,
so the expansion never breaks kernel boot when Playwright is absent.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

from computer_use.schema import Observation
from .base import BaseDriver

# Lazy import flag
_playwright = None


def _load_playwright():
    global _playwright
    if _playwright is None:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright
    return _playwright


def _default_chromium() -> str | None:
    """Return a usable chromium executable path, or None."""
    for cand in ("/usr/bin/chromium", "/usr/bin/chromium-browser",
                 "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"):
        if os.path.exists(cand):
            return cand
    return None


class PlaywrightDriver(BaseDriver):
    """Real browser driver. Each instance owns one browser context.

    The browser is launched lazily on first observe/execute and closed on
    close(). This keeps the expansion side-effect-free until actually used.
    """

    def __init__(self, executable_path: str | None = None, headless: bool = True,
                 trace_dir: str | Path | None = None):
        self.executable_path = executable_path or _default_chromium()
        self.headless = headless
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _ensure(self):
        if self._page is not None:
            return
        pw = _load_playwright()
        self._pw = pw().start()
        launch_kwargs = {"headless": self.headless}
        if self.executable_path:
            launch_kwargs["executable_path"] = self.executable_path
        self._browser = self._pw.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context()
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._context.tracing.start(screenshots=True, snapshots=True)
        self._page = self._context.new_page()

    def close(self):
        try:
            if self._context is not None and self.trace_dir:
                self._context.tracing.stop(path=str(self.trace_dir / "trace.zip"))
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self._page = None

    def __enter__(self):
        self._ensure()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── perception ───────────────────────────────────────────────────────
    def observe(self, target: dict) -> Observation:
        self._ensure()
        url = None
        text = ""
        try:
            url = self._page.url
            text = self._page.inner_text("body")[:4000]
        except Exception:
            pass
        return Observation(
            source="browser",
            url=url,
            text=text,
            state_hash=self._state_hash(url, text),
        )

    def screenshot(self, path: str | Path | None = None) -> str | None:
        """Capture a screenshot; returns a data-URI string or None on failure."""
        self._ensure()
        try:
            if path:
                self._page.screenshot(path=str(path))
                return str(path)
            png = self._page.screenshot()
            return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _state_hash(url: str | None, text: str) -> str:
        import hashlib
        return hashlib.sha256(f"{url}|{text[:2000]}".encode()).hexdigest()[:16]

    # ── execution ────────────────────────────────────────────────────────
    def execute(self, action, target: dict) -> dict:
        self._ensure()
        kind = action.kind
        try:
            if kind == "navigate":
                url = action.url or (target or {}).get("url")
                if not url:
                    return {"status": "error", "driver": "playwright", "action": kind,
                            "error": "navigate requires url"}
                self._page.goto(url, wait_until="domcontentloaded", timeout=action.timeout_ms)
                return {"status": "ok", "driver": "playwright", "action": kind, "url": self._page.url}

            if kind == "click":
                sel = action.selector or "body"
                self._page.click(sel, timeout=action.timeout_ms)
                return {"status": "ok", "driver": "playwright", "action": kind, "selector": sel}

            if kind == "double_click":
                sel = action.selector or "body"
                self._page.dblclick(sel, timeout=action.timeout_ms)
                return {"status": "ok", "driver": "playwright", "action": kind, "selector": sel}

            if kind == "type":
                sel = action.selector or "body"
                self._page.fill(sel, action.text or "")
                return {"status": "ok", "driver": "playwright", "action": kind, "selector": sel}

            if kind == "hotkey":
                key = action.text or action.selector or ""
                if not key:
                    return {"status": "error", "driver": "playwright", "action": kind,
                            "error": "hotkey requires key"}
                self._page.keyboard.press(key)
                return {"status": "ok", "driver": "playwright", "action": kind, "key": key}

            if kind == "scroll":
                self._page.mouse.wheel(0, 800)
                return {"status": "ok", "driver": "playwright", "action": kind}

            if kind == "wait":
                time.sleep(action.timeout_ms / 1000.0)
                return {"status": "ok", "driver": "playwright", "action": kind}

            if kind == "assert_text":
                body = self._page.inner_text("body")
                found = (action.text or "") in body
                return {"status": "ok" if found else "error", "driver": "playwright",
                        "action": kind, "found": found}

            if kind == "assert_url":
                found = (action.url or "") in self._page.url
                return {"status": "ok" if found else "error", "driver": "playwright",
                        "action": kind, "found": found}

            if kind == "submit":
                self._page.keyboard.press("Enter")
                return {"status": "ok", "driver": "playwright", "action": kind}

            if kind in ("done", "abort"):
                return {"status": "ok", "driver": "playwright", "action": kind}

            return {"status": "error", "driver": "playwright", "action": kind,
                    "error": f"unsupported action: {kind}"}
        except Exception as e:
            return {"status": "error", "driver": "playwright", "action": kind, "error": str(e)}
