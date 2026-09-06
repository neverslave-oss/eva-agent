from computer_use.router import choose_driver


def test_choose_driver_browser_happy_path():
    assert choose_driver({"kind": "browser"}) == "playwright"


def test_choose_driver_windows_edge_path():
    out = choose_driver({"kind": "desktop", "platform": "windows"}, enabled={"pywinauto": True, "pyautogui": True})
    assert out == "pywinauto"


def test_choose_driver_fallback_edge_path():
    out = choose_driver({"kind": "desktop", "platform": "linux"}, enabled={"pyautogui": True})
    assert out == "pyautogui"


def test_choose_driver_desktop_default_happy_path():
    out = choose_driver({}, enabled={"pyautogui": True, "playwright": True})
    assert out == "pyautogui"
