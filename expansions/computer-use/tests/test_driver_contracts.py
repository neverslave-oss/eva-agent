from computer_use.drivers.base import BaseDriver
from computer_use.drivers.playwright_driver import PlaywrightDriver
from computer_use.drivers.pyautogui_driver import PyAutoGUIDriver
from computer_use.drivers.pywinauto_driver import PyWinAutoDriver
from computer_use.drivers.openclaw_driver import OpenClawDriver
from computer_use.schema import Action


def test_driver_contract_subclassing_happy_path():
    for cls in [PlaywrightDriver, PyAutoGUIDriver, PyWinAutoDriver, OpenClawDriver]:
        assert issubclass(cls, BaseDriver)


def test_driver_execute_shape_edge_path():
    d = PlaywrightDriver()
    result = d.execute(Action(kind="click", selector="#go"), {"kind": "browser"})
    assert result["status"] == "ok"
    assert result["driver"] == "playwright"
