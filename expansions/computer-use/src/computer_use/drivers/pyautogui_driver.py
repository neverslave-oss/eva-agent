"""pyautogui_driver.py — real desktop driver backed by pyautogui + WSLg/X.

Drives the real desktop session (WSLg X server on :0) via pyautogui: moves the
mouse, clicks, types, scrolls, and captures the screen. Perception uses the
live screen dimensions and (when available) a screenshot.

The driver is lazy: pyautogui is imported only when the driver is first used,
so the expansion never breaks kernel boot when pyautogui or a display is absent.
"""

from __future__ import annotations

import os
import time

from computer_use.schema import Observation
from .base import BaseDriver

# Lazy import flag
_pyautogui = None


def _load_pyautogui():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        # Fail fast on a missing display so we never silently "succeed".
        _ = pyautogui.size()
        _pyautogui = pyautogui
    return _pyautogui


class PyAutoGUIDriver(BaseDriver):
    """Real desktop driver. Requires a live X/WSLg display.

    If no display is reachable, observe()/execute() return a structured error
    rather than faking success — the honest behavior for a headless host.
    """

    def __init__(self, display: str | None = None):
        self.display = display or os.environ.get("DISPLAY") or ":0"

    def _ensure(self):
        if self.display:
            os.environ["DISPLAY"] = self.display
        return _load_pyautogui()

    # ── perception ───────────────────────────────────────────────────────
    def observe(self, target: dict) -> Observation:
        try:
            pg = self._ensure()
            w, h = pg.size()
            x, y = pg.position()
            return Observation(
                source="desktop",
                text=f"screen {w}x{h}, cursor at ({x},{y})",
                state_hash=self._state_hash(w, h, x, y),
            )
        except Exception as e:
            return Observation(source="desktop", text=f"error: {e}", state_hash=None)

    def screenshot(self, path: str | None = None) -> str | None:
        try:
            pg = self._ensure()
            if path:
                pg.screenshot(path)
                return path
            import base64
            import io
            buf = io.BytesIO()
            pg.screenshot().save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _state_hash(w, h, x, y):
        import hashlib
        return hashlib.sha256(f"{w}x{h}@{x},{y}".encode()).hexdigest()[:16]

    # ── execution ────────────────────────────────────────────────────────
    def execute(self, action, target: dict) -> dict:
        try:
            pg = self._ensure()
        except Exception as e:
            return {"status": "error", "driver": "pyautogui", "action": action.kind,
                    "error": f"no display: {e}"}

        kind = action.kind
        try:
            if kind == "click":
                sel = action.selector or "auto"
                if sel != "auto":
                    # Best-effort: treat selector as a screen coordinate "x,y"
                    try:
                        x, y = (int(v) for v in str(sel).split(","))
                        pg.click(x, y)
                    except Exception:
                        pg.click()
                else:
                    pg.click()
                return {"status": "ok", "driver": "pyautogui", "action": kind, "selector": sel}

            if kind == "double_click":
                pg.doubleClick()
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            if kind == "type":
                pg.write(action.text or "")
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            if kind == "hotkey":
                key = action.text or action.selector or ""
                if not key:
                    return {"status": "error", "driver": "pyautogui", "action": kind,
                            "error": "hotkey requires key"}
                pg.hotkey(*key.split("+"))
                return {"status": "ok", "driver": "pyautogui", "action": kind, "key": key}

            if kind == "scroll":
                pg.scroll(-3)
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            if kind == "wait":
                time.sleep(action.timeout_ms / 1000.0)
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            if kind == "submit":
                pg.press("enter")
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            if kind in ("done", "abort"):
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            return {"status": "error", "driver": "pyautogui", "action": kind,
                    "error": f"unsupported action: {kind}"}
        except Exception as e:
            return {"status": "error", "driver": "pyautogui", "action": kind, "error": str(e)}
