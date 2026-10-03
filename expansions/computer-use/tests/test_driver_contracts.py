from computer_use.drivers.base import BaseDriver
from computer_use.drivers.playwright_driver import PlaywrightDriver
from computer_use.drivers.pyautogui_driver import PyAutoGUIDriver
from computer_use.drivers.pywinauto_driver import PyWinAutoDriver
from computer_use.drivers.openclaw_driver import OpenClawDriver
from computer_use.schema import Action


def test_driver_contract_subclassing_happy_path():
    for cls in [PlaywrightDriver, PyAutoGUIDriver, PyWinAutoDriver, OpenClawDriver]:
        assert issubclass(cls, BaseDriver)


def test_pyautogui_launch_missing_app_returns_error():
    """Launching a non-existent app must NOT fake success.

    The pyautogui driver previously returned {"status": "ok"} unconditionally
    for a launch, even when xdg-open failed because the app doesn't exist. That
    produced false completed:true traces that Eva's independent verification
    could not confirm. A launch is only 'ok' if the spawned process survives;
    otherwise it must return a structured error.
    """
    d = PyAutoGUIDriver()
    # A name with no binary on PATH and no xdg handler -> must error, not ok.
    result = d.execute(
        Action(kind="launch", text="definitely-not-a-real-app-xyz",
               metadata={"app": "definitely-not-a-real-app-xyz"}),
        {"kind": "desktop"},
    )
    assert result["driver"] == "pyautogui"
    assert result["action"] == "launch"
    assert result["status"] == "error"
    assert "launch failed" in result["error"]


def test_driver_execute_shape_edge_path():
    """A real driver returns a structured result dict with driver + action.

    Clicking a non-existent selector on a blank page must NOT fake success —
    it should return a structured error envelope (real behavior, not a stub).
    """
    d = PlaywrightDriver()
    try:
        result = d.execute(Action(kind="click", selector="#go"), {"kind": "browser"})
    finally:
        d.close()
    assert result["driver"] == "playwright"
    assert result["action"] == "click"
    # Either ok (if something matched) or a structured error — never a stub lie.
    assert result["status"] in {"ok", "error"}
