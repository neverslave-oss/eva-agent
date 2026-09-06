from __future__ import annotations

import uuid

from .schema import ActionBatch, ExecutionResult, Expectation, Observation
from .safety import PolicyEngine, PolicyViolation
from .state_store import StateStore
from .verifier import Verifier


class Orchestrator:
    def __init__(
        self,
        planner,
        driver,
        policy: PolicyEngine | None = None,
        verifier: Verifier | None = None,
        state_store: StateStore | None = None,
    ):
        self.planner = planner
        self.driver = driver
        self.policy = policy or PolicyEngine({})
        self.verifier = verifier or Verifier()
        self.state_store = state_store

    def run_once(
        self,
        goal: str,
        target: dict | None = None,
        expectation: Expectation | None = None,
        *,
        chat_id: str = "default",
        run_id: str | None = None,
        dry_run: bool = True,
    ) -> ExecutionResult:
        target = target or {}
        run_id = run_id or str(uuid.uuid4())
        observation: Observation = self.driver.observe(target)
        batch: ActionBatch = self.planner.plan(goal, observation)

        if self.state_store:
            self.state_store.update_run(
                chat_id,
                run_id,
                {
                    "goal": goal,
                    "step": 0,
                    "status": "planned",
                    "dry_run": dry_run,
                    "target": target,
                    "actions": [a.kind for a in batch.actions],
                },
            )

        for idx, action in enumerate(batch.actions, start=1):
            try:
                self.policy.validate_action(action)
            except PolicyViolation as e:
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "blocked", "error": str(e)})
                return ExecutionResult(status="blocked", message=str(e), completed=False, data={"run_id": run_id})

            if action.kind == "done":
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "done"})
                return ExecutionResult(status="done", message=action.text or "done", completed=True, data={"run_id": run_id})

            if action.kind == "abort":
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "error", "error": action.text or "aborted"})
                return ExecutionResult(status="error", message=action.text or "aborted", completed=False, data={"run_id": run_id})

            if not dry_run:
                self.driver.execute(action, target)

            if self.state_store:
                self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "running"})

        post = self.driver.observe(target)
        ok, msg = self.verifier.verify(expectation, post)
        status = "ok" if ok else "error"
        if self.state_store:
            self.state_store.update_run(chat_id, run_id, {"status": status, "completed": bool(ok), "verification": msg})
        return ExecutionResult(status=status, message=msg, completed=ok, data={"run_id": run_id})
