from computer_use.orchestrator import Orchestrator
from computer_use.schema import Action, ActionBatch, Observation
from computer_use.safety import PolicyEngine


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


def test_orchestrator_done_happy_path():
    o = Orchestrator(_PlannerDone(), _Driver(), policy=PolicyEngine({"allow_actions": ["done"]}))
    res = o.run_once("finish task", target={"kind": "browser"})
    assert res.status == "done"
    assert res.completed is True


def test_orchestrator_blocked_edge_case():
    o = Orchestrator(_PlannerClick(), _Driver(), policy=PolicyEngine({"allow_actions": ["done"]}))
    res = o.run_once("click", target={"kind": "browser"})
    assert res.status == "blocked"
    assert res.completed is False
