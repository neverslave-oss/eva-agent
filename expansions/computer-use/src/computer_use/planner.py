from __future__ import annotations

from .schema import Action, ActionBatch, Observation


class Planner:
    """Minimal planner stub.

    Real implementation will call model/tooling and emit validated ActionBatch.
    """

    def plan(self, goal: str, observation: Observation) -> ActionBatch:
        if not goal.strip():
            return ActionBatch(actions=[Action(kind="abort", text="empty goal")])
        return ActionBatch(actions=[Action(kind="done", text="scaffold planner done")])
