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
