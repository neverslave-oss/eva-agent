from computer_use.orchestrator import Orchestrator
from computer_use.schema import Action, ActionBatch, Observation
from computer_use.safety import PolicyEngine
from computer_use.state_store import StateStore


class _PlannerDone:
    def plan(self, goal, observation):
        return ActionBatch(actions=[Action(kind="done", text="complete")])


class _PlannerClick:
    def plan(self, goal, observation):
        return ActionBatch(actions=[Action(kind="click", selector="#x")])


class _Driver:
    def __init__(self):
        self.executed = []

    def observe(self, target):
        return Observation(source="browser", url="https://docs.openclaw.ai", text="ok")

    def execute(self, action, target):
        self.executed.append(action.kind)
        return {"status": "ok"}


def test_orchestrator_done_happy_path(tmp_path):
    d = _Driver()
    o = Orchestrator(
        _PlannerDone(),
        d,
        policy=PolicyEngine({"allow_actions": ["done"]}),
        state_store=StateStore(tmp_path / "state.json"),
    )
    res = o.run_once("finish task", target={"kind": "browser"})
    assert res.status == "done"
    assert res.completed is True
    assert res.data["run_id"]


def test_orchestrator_blocked_edge_case(tmp_path):
    o = Orchestrator(
        _PlannerClick(),
        _Driver(),
        policy=PolicyEngine({"allow_actions": ["done"]}),
        state_store=StateStore(tmp_path / "state.json"),
    )
    res = o.run_once("click", target={"kind": "browser"})
    assert res.status == "blocked"
    assert res.completed is False


def test_orchestrator_dry_run_skips_execution_edge_case(tmp_path):
    d = _Driver()
    o = Orchestrator(
        _PlannerClick(),
        d,
        policy=PolicyEngine({"allow_actions": ["click", "done"]}),
        state_store=StateStore(tmp_path / "state.json"),
    )
    res = o.run_once("click", target={"kind": "desktop"}, dry_run=True)
    assert res.status in {"ok", "done"}
    assert d.executed == []


def test_orchestrator_live_mode_executes_edge_case(tmp_path):
    d = _Driver()
    o = Orchestrator(
        _PlannerClick(),
        d,
        policy=PolicyEngine({"allow_actions": ["click", "done"]}),
        state_store=StateStore(tmp_path / "state.json"),
    )
    _ = o.run_once("click", target={"kind": "desktop"}, dry_run=False)
    assert d.executed == ["click"]
