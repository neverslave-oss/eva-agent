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


def _detect_display() -> str | None:
    """Auto-detect the active X display by scanning /tmp/.X11-unix sockets.

    WSLg exposes its socket as X1 (display :1), not :0, and the kernel process
    often has no DISPLAY env var. Defaulting to :0 silently breaks desktop
    control, so scan the X socket dir and pick the highest-numbered display.
    """
    xdir = "/tmp/.X11-unix"
    try:
        nums = []
        for e in os.listdir(xdir):
            if e.startswith("X"):
                try:
                    nums.append(int(e[1:]))
                except ValueError:
                    continue
        if nums:
            return f":{max(nums)}"
    except Exception:
        pass
    return None


class PyAutoGUIDriver(BaseDriver):
    """Real desktop driver. Requires a live X/WSLg display.

    If no display is reachable, observe()/execute() return a structured error
    rather than faking success — the honest behavior for a headless host.
    """

    def __init__(self, display: str | None = None):
        self.display = display or os.environ.get("DISPLAY") or _detect_display() or ":0"

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
        """Capture the screen as PNG (base64 data-URI, or to a path).

        pyautogui's screenshot backend requires gnome-screenshot (sudo install),
        which is often absent on WSLg — so we fall back to mss, which grabs the
        X display directly via XCB with no system dependency.

        The image is downscaled (max dimension capped) and re-encoded with
        lossless PNG optimization so the streamed screenshot stays small
        (a raw 1920x1080 frame is ~2MB; the downscale cuts that dramatically).
        """
        try:
            import base64
            import io
            img = self._capture()
            if img is None:
                return None
            # Downscale to keep the payload small while preserving detail.
            img = self._downscale(img)
            if path:
                img.save(path, format="PNG", optimize=True)
                return path
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None

    @staticmethod
    def _downscale(img, max_dim: int = 1280):
        """Losslessly downscale an image so its largest side is <= max_dim.

        Keeps the aspect ratio and uses PNG (lossless) so no detail is lost,
        only resolution — which cuts the streamed payload from ~2MB to a few
        hundred KB.
        """
        try:
            from PIL import Image
            w, h = img.size
            longest = max(w, h)
            if longest <= max_dim:
                return img
            ratio = max_dim / float(longest)
            nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
            return img.resize((nw, nh), Image.LANCZOS)
        except Exception:
            return img

    def is_frame_black(self, threshold: float = 6.0) -> bool:
        """True when the captured frame is effectively black/locked.

        A blanked/locked screen (screensaver overlay or login dialog) captures
        as a near-black frame. Detecting it lets the planner abort BEFORE
        sending the frame to the vision pipeline, so we don't burn cloud calls
        on a screen Eva can't act on. Returns False on any capture error so we
        never block on a transient failure.
        """
        try:
            img = self._capture()
            if img is None:
                return False
            # Mean of the luminance channel; a real desktop is well above this.
            gray = img.convert("L")
            px = list(gray.getdata())
            mean = sum(px) / float(len(px)) if px else 0.0
            return mean < threshold
        except Exception:
            return False

    def is_frame_white(self, threshold: float = 250.0, near_white_ratio: float = 0.95) -> bool:
        """True when the captured frame is effectively blank-white.

        Some blank/blanked screens (e.g. a browser that failed to launch and
        left a white compositor surface, or a white screensaver overlay) capture
        as a near-white frame. The black-guard (`is_frame_black`) misses these,
        so a white blank would stream per-step until the frozen-guard caught it
        after several unchanged frames. This lets the planner abort immediately
        on a white blank too. Returns False on any capture error so we never
        block on a transient failure.
        """
        try:
            img = self._capture()
            if img is None:
                return False
            gray = img.convert("L")
            px = list(gray.getdata())
            if not px:
                return False
            mean = sum(px) / float(len(px))
            near_white = sum(1 for v in px if v > threshold) / float(len(px))
            # Both a high overall luminance AND an overwhelming near-white
            # majority — a normal bright desktop has dark text/icons, so it
            # won't be >95% pure-white.
            return mean > threshold and near_white > near_white_ratio
        except Exception:
            return False


    def _wake_screen(self) -> None:
        """Wake/disable the screensaver so captures aren't blanked to black.

        xfce4-screensaver puts a full-screen blanking window on top of the
        desktop; every X capture (mss, ffmpeg) then reads black. Disabling the
        screensaver and nudging the display before grabbing ensures the real
        desktop pixels are captured. Best-effort: failures are ignored so the
        capture path never breaks.
        """
        try:
            import subprocess
            env = dict(os.environ)
            if self.display:
                env["DISPLAY"] = self.display
            for cmd in (
                ["xset", "s", "off"],
                ["xset", "dpms", "force", "on"],
                ["xdotool", "key", "Return"],
            ):
                try:
                    subprocess.run(cmd, env=env, timeout=3,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        except Exception:
            pass

    def _capture(self):
        """Return a PIL Image of the screen, trying pyautogui then mss."""
        # Wake/disable the screensaver first so the capture isn't a black frame.
        self._wake_screen()
        try:
            pg = self._ensure()
            return pg.screenshot()
        except Exception:
            pass
        # Fallback: mss grabs the X display directly (no gnome-screenshot needed).
        try:
            import mss
            from PIL import Image
            with mss.mss() as sct:
                mon = sct.monitors[1]
                raw = sct.grab(mon)
                return Image.frombytes("RGB", raw.size, raw.rgb)
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
                if sel == "auto" or not sel:
                    # 'auto' = click at the current cursor position. This is
                    # legitimate for the deterministic planner (bare "click"
                    # goal), so allow it here. False-success protection for
                    # the LLM vision planner lives in the brain, which is
                    # instructed to always emit real x,y coordinates for clicks.
                    pg.click()
                    return {"status": "ok", "driver": "pyautogui", "action": kind, "selector": sel}
                try:
                    x, y = (int(v) for v in str(sel).split(","))
                except Exception:
                    return {"status": "error", "driver": "pyautogui", "action": kind,
                            "error": f"invalid click coordinates: {sel!r}"}
                pg.click(x, y)
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

            if kind == "launch":
                import shutil
                import subprocess
                app = (action.text or action.metadata.get("app") or "").strip()
                if not app:
                    return {"status": "error", "driver": "pyautogui", "action": kind,
                            "error": "launch requires an app name"}
                # Resolve the app command; fall back to xdg-open for GUI apps.
                cmd = shutil.which(app)
                if cmd:
                    proc = subprocess.Popen(
                        [cmd],
                        env={**os.environ, "DISPLAY": self.display or ":0"},
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                else:
                    # No binary on PATH — try xdg-open, but capture stderr and
                    # verify it actually succeeded instead of blindly returning ok.
                    proc = subprocess.Popen(
                        ["xdg-open", app],
                        env={**os.environ, "DISPLAY": self.display or ":0"},
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                time.sleep(1.5)
                # A launch is only 'ok' if the process is still alive (didn't
                # exit immediately with an error). xdg-open exits non-zero when
                # the target doesn't exist — surface that as a real error.
                rc = proc.poll()
                if rc is not None:
                    err = (proc.stderr.read().decode(errors="replace").strip()
                           if proc.stderr else "") or f"exit code {rc}"
                    return {"status": "error", "driver": "pyautogui", "action": kind,
                            "error": f"launch failed for {app!r}: {err}"}
                return {"status": "ok", "driver": "pyautogui", "action": kind, "app": app}

            if kind in ("done", "abort"):
                return {"status": "ok", "driver": "pyautogui", "action": kind}

            return {"status": "error", "driver": "pyautogui", "action": kind,
                    "error": f"unsupported action: {kind}"}
        except Exception as e:
            return {"status": "error", "driver": "pyautogui", "action": kind, "error": str(e)}
