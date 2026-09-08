from __future__ import annotations


def choose_driver(target: dict, enabled: dict | None = None) -> str:
    """Select execution driver from target hints.

    target examples:
      {"kind":"browser"}
      {"kind":"desktop","platform":"windows","app":"notepad.exe"}
      {"kind":"remote","runtime":"openclaw"}
    """
    enabled = enabled or {
        "playwright": True,
        "pyautogui": True,
        "pywinauto": True,
        "openclaw": True,
    }

    kind = (target or {}).get("kind", "desktop")
    platform = (target or {}).get("platform", "").lower()

    if kind == "browser" and enabled.get("playwright", False):
        return "playwright"
    if kind == "remote" and enabled.get("openclaw", False):
        return "openclaw"
    if kind == "desktop" and platform == "windows" and enabled.get("pywinauto", False):
        return "pywinauto"
    if enabled.get("pyautogui", False):
        return "pyautogui"
    return "playwright"
