from __future__ import annotations

import uuid

from .schema import ActionBatch, ExecutionResult, Expectation, Observation
from .safety import PolicyEngine, PolicyViolation
from .state_store import StateStore
from .verifier import Verifier
from .telemetry import TraceCollector


class Orchestrator:
    def __init__(
        self,
        planner,
        driver,
        policy: PolicyEngine | None = None,
        verifier: Verifier | None = None,
        state_store: StateStore | None = None,
        tracer: TraceCollector | None = None,
    ):
        self.planner = planner
        self.driver = driver
        self.policy = policy or PolicyEngine({})
        self.verifier = verifier or Verifier()
        self.state_store = state_store
        self.tracer = tracer or TraceCollector()

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

        self.tracer.run_started(run_id, goal=goal, target=target, dry_run=dry_run)

        observation: Observation = self.driver.observe(target)
        self.tracer.observation(
            run_id,
            source=observation.source,
            url=observation.url,
            text=observation.text,
            state_hash=observation.state_hash,
        )

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
            action_dict = action.model_dump() if hasattr(action, "model_dump") else dict(action)
            try:
                self.policy.validate_action(action)
            except PolicyViolation as e:
                self.tracer.action_blocked(run_id, idx, action_dict, reason=str(e))
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "blocked", "error": str(e)})
                return ExecutionResult(status="blocked", message=str(e), completed=False, data={"run_id": run_id})

            if action.kind == "done":
                self.tracer.action_planned(run_id, idx, action_dict)
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "done"})
                self.tracer.run_finished(run_id, "done", True, action.text or "done")
                return ExecutionResult(status="done", message=action.text or "done", completed=True, data={"run_id": run_id})

            if action.kind == "abort":
                self.tracer.action_planned(run_id, idx, action_dict)
                if self.state_store:
                    self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "error", "error": action.text or "aborted"})
                self.tracer.run_finished(run_id, "error", False, action.text or "aborted")
                return ExecutionResult(status="error", message=action.text or "aborted", completed=False, data={"run_id": run_id})

            self.tracer.action_planned(run_id, idx, action_dict)
            if not dry_run:
                result = self.driver.execute(action, target)
                self.tracer.action_executed(run_id, idx, action_dict, result)
            else:
                self.tracer.action_executed(run_id, idx, action_dict, {"status": "dry_run"})

            if self.state_store:
                self.state_store.update_run(chat_id, run_id, {"step": idx, "status": "running"})

        post = self.driver.observe(target)
        self.tracer.observation(
            run_id,
            source=post.source,
            url=post.url,
            text=post.text,
            state_hash=post.state_hash,
        )
        ok, msg = self.verifier.verify(expectation, post)
        self.tracer.verification(run_id, ok=ok, message=msg)
        status = "ok" if ok else "error"
        if self.state_store:
            self.state_store.update_run(chat_id, run_id, {"status": status, "completed": bool(ok), "verification": msg})
        self.tracer.run_finished(run_id, status, bool(ok), msg)
        return ExecutionResult(status=status, message=msg, completed=ok, data={"run_id": run_id})
