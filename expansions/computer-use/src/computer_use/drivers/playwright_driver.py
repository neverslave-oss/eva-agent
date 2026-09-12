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


def _default_firefox() -> str | None:
    """Return a Firefox executable path, or None to use Playwright's bundled build.

    Prefers Playwright's own Firefox build (firefox-1538) because the system
    Firefox (firefox-esr) is often incompatible with Playwright's CDP protocol
    and fails to launch. Only falls back to a system Firefox when the bundled
    build is absent.
    """
    import glob
    bundled = glob.glob(os.path.expanduser("~/.cache/ms-playwright/firefox-*/firefox/firefox"))
    if bundled:
        return None  # let Playwright use its compatible bundled Firefox
    for cand in ("/usr/bin/firefox", "/usr/bin/firefox-esr",
                 "/usr/local/bin/firefox"):
        if os.path.exists(cand):
            return cand
    return None


def _parse_coords(sel: str) -> tuple[int, int] | None:
    """Parse an 'x,y' screen-coordinate selector into (x, y), else None.

    The vision brain emits click coordinates as "x,y" (e.g. "1191,73").
    Playwright must treat these as mouse coordinates, not CSS selectors.
    """
    if not sel:
        return None
    s = sel.strip()
    if "," not in s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        return (int(parts[0].strip()), int(parts[1].strip()))
    except Exception:
        return None


class PlaywrightDriver(BaseDriver):
    """Real browser driver backed by Firefox. Each instance owns one browser context.

    The browser is launched lazily on first observe/execute and closed on
    close(). This keeps the expansion side-effect-free until actually used.
    """

    def __init__(self, executable_path: str | None = None, headless: bool = True,
                 trace_dir: str | Path | None = None):
        self.executable_path = executable_path or _default_firefox()
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
        self._browser = self._pw.firefox.launch(**launch_kwargs)
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

    def _frame_image(self):
        """Decode the current screenshot into a PIL image, or None on failure."""
        try:
            from PIL import Image
            import io
            png = self._page.screenshot()
            return Image.open(io.BytesIO(png))
        except Exception:
            return None

    def is_frame_black(self, threshold: float = 6.0) -> bool:
        """True when the captured browser frame is effectively black/locked.

        Mirrors the pyautogui driver so the black/locked guard also fires on
        browser frames (a failed launch / blank compositor surface). Returns
        False on any capture error so we never block on a transient failure.
        """
        try:
            img = self._frame_image()
            if img is None:
                return False
            gray = img.convert("L")
            px = list(gray.getdata())
            mean = sum(px) / float(len(px)) if px else 0.0
            return mean < threshold
        except Exception:
            return False

    def is_frame_white(self, threshold: float = 250.0, near_white_ratio: float = 0.95) -> bool:
        """True when the captured browser frame is effectively blank-white.

        Mirrors the pyautogui driver so the white-blank guard also fires on
        browser frames (a browser that failed to launch and left a white
        compositor surface). Returns False on any capture error.
        """
        try:
            img = self._frame_image()
            if img is None:
                return False
            gray = img.convert("L")
            px = list(gray.getdata())
            if not px:
                return False
            mean = sum(px) / float(len(px))
            near_white = sum(1 for v in px if v > threshold) / float(len(px))
            return mean > threshold and near_white > near_white_ratio
        except Exception:
            return False

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
                coord = _parse_coords(sel)
                if coord:
                    self._page.mouse.click(coord[0], coord[1])
                else:
                    self._page.click(sel, timeout=action.timeout_ms)
                return {"status": "ok", "driver": "playwright", "action": kind, "selector": sel}

            if kind == "double_click":
                sel = action.selector or "body"
                coord = _parse_coords(sel)
                if coord:
                    self._page.mouse.dblclick(coord[0], coord[1])
                else:
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
