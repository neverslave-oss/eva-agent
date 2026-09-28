from computer_use.planner import Planner
from computer_use.schema import Observation


def test_planner_url_happy_path():
    p = Planner()
    obs = Observation(source="browser", url="https://example.com", text="")
    batch = p.plan('open https://docs.openclaw.ai and click', obs)
    kinds = [a.kind for a in batch.actions]
    assert "navigate" in kinds
    assert "click" in kinds
    assert kinds[-1] == "done"


def test_planner_empty_goal_edge_case():
    p = Planner()
    obs = Observation(source="desktop", text="")
    batch = p.plan(" ", obs)
    assert batch.actions[0].kind == "abort"


def test_planner_launch_app():
    p = Planner()
    obs = Observation(source="desktop", text="")
    batch = p.plan("open firefox", obs)
    kinds = [a.kind for a in batch.actions]
    assert "launch" in kinds
    launch = [a for a in batch.actions if a.kind == "launch"][0]
    assert launch.text == "firefox"
    assert kinds[-1] == "done"


def test_planner_launch_variants():
    p = Planner()
    obs = Observation(source="desktop", text="")
    for goal, app in [("launch calculator", "calculator"), ("start the terminal", "the terminal")]:
        batch = p.plan(goal, obs)
        kinds = [a.kind for a in batch.actions]
        assert "launch" in kinds, goal


# ── Regression: vision parse + drive (2026-09-16) ──────────────────────
def test_parse_action_tolerates_string_metadata():
    """Vision emits metadata sometimes as a string (e.g. 'Navigate to the
    blog...') but Action declares text dict. Must be coerced, not rejected —
    otherwise the vision loop aborts on its FIRST correct action."""
    from computer_use.vision_brain import _parse_action
    raw = ('{"kind": "navigate", "url": "https://fabiopacifici.com/blog", '
           '"driver": "browser", "timeout_ms": 5000, '
           '"metadata": "Navigate to the blog on fabiopacifici.com"}')
    a = _parse_action(raw)
    assert a is not None
    assert a.kind == "navigate" and a.url == "https://fabiopacifici.com/blog"


def test_parse_plan_tolerates_string_metadata():
    """Same coercion must apply in _parse_plan — the parser the LLMPlanner
    actually uses via plan_actions (that path was still aborting)."""
    from computer_use.vision_brain import _parse_plan
    raw = ('[{"kind": "navigate", "url": "https://fabiopacifici.com", "driver": "browser", '
           '"metadata": "Opening Firefox and navigating to the blog post"}]')
    plan = _parse_plan(raw)
    assert plan and plan[0].kind == "navigate"
    assert plan[0].url == "https://fabiopacifici.com"


def test_blank_white_guard_skips_browser_driver():
    """The black/white-blank guards must NOT abort a browser sitting on
    about:blank — that is a normal, navigable start for a navigate task.
    Scoped to desktop only (Playwright=False, PyAutoGUI=True)."""
    from computer_use.llm_planner import LLMPlanner
    from computer_use.drivers.playwright_driver import PlaywrightDriver
    from computer_use.drivers.pyautogui_driver import PyAutoGUIDriver
    assert LLMPlanner(driver=PlaywrightDriver(headless=True))._is_desktop_driver() is False
    assert LLMPlanner(driver=PyAutoGUIDriver())._is_desktop_driver() is True
    assert LLMPlanner()._is_desktop_driver() is True
