from __future__ import annotations

from .schema import ActionBatch, ExecutionResult, Expectation, Observation
from .safety import PolicyEngine, PolicyViolation
from .verifier import Verifier


class Orchestrator:
    def __init__(self, planner, driver, policy: PolicyEngine | None = None, verifier: Verifier | None = None):
        self.planner = planner
        self.driver = driver
        self.policy = policy or PolicyEngine({})
        self.verifier = verifier or Verifier()

    def run_once(self, goal: str, target: dict | None = None, expectation: Expectation | None = None) -> ExecutionResult:
        observation: Observation = self.driver.observe(target or {})
        batch: ActionBatch = self.planner.plan(goal, observation)

        for action in batch.actions:
            try:
                self.policy.validate_action(action)
            except PolicyViolation as e:
                return ExecutionResult(status="blocked", message=str(e), completed=False)

            if action.kind == "done":
                return ExecutionResult(status="done", message=action.text or "done", completed=True)
            if action.kind == "abort":
                return ExecutionResult(status="error", message=action.text or "aborted", completed=False)

            self.driver.execute(action, target or {})

        post = self.driver.observe(target or {})
        ok, msg = self.verifier.verify(expectation, post)
        return ExecutionResult(status="ok" if ok else "error", message=msg, completed=ok)
