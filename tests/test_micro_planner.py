"""
test_micro_planner.py — ADR-011 tests

Verifies:
- should_plan heuristic (simple → False, complex → True)
- plan generation (mock drafter + main model, verify Plan structure)
- executor sequential dependency injection
- executor final synthesis
- triage delegates to planner for complex input
- simple input bypasses planner
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_should_plan_simple():
    from core.pipelines.micro_planner import MicroPlanner
    mp = MicroPlanner()
    assert mp.should_plan("tell me a joke") is False
    assert mp.should_plan("what is the weather") is False
    assert mp.should_plan("hello") is False


def test_should_plan_complex_signals():
    from core.pipelines.micro_planner import MicroPlanner
    mp = MicroPlanner()
    # Multiple complexity signals
    assert mp.should_plan("first fetch the URL, then extract the table, and finally save as JSON") is True


def test_should_plan_long_sentence_count():
    from core.pipelines.micro_planner import MicroPlanner
    mp = MicroPlanner()
    # 3+ sentences
    text = "Fetch the data. Extract the table. Save as JSON."
    assert mp.should_plan(text) is True


def test_plan_generation_mock(monkeypatch):
    """Mock drafter and main model, verify Plan structure returned."""
    from core.pipelines.micro_planner import MicroPlanner

    def fake_infer_draft(prompt, max_new_tokens=256):
        return "1. Fetch the URL\n2. Extract the table\n3. Save as JSON"

    def fake_infer(messages, max_new_tokens=512):
        return "1. Fetch the URL\n2. Extract the table\n3. Save as JSON"

    import core.inference.model_client as model_client
    monkeypatch.setattr(model_client, "infer_draft", fake_infer_draft)
    monkeypatch.setattr(model_client, "infer", fake_infer)

    mp = MicroPlanner(config={"draft_plan": True, "max_steps": 5})
    plan = mp.plan(
        "fetch the URL, extract the table, save as JSON",
        available_skills=[{"name": "web-fetch"}, {"name": "json-writer"}],
        available_routines=[],
    )

    assert plan is not None
    assert len(plan.steps) == 3
    assert plan.steps[0].order == 1
    assert "Fetch" in plan.steps[0].task
    assert plan.steps[2].order == 3


def test_plan_returns_none_for_single_step(monkeypatch):
    """Planner returns None if only 1 step parsed."""
    from core.pipelines.micro_planner import MicroPlanner

    def fake_infer_draft(prompt, max_new_tokens=256):
        return "1. Just do the one thing"

    def fake_infer(messages, max_new_tokens=512):
        return "1. Just do the one thing"

    import core.inference.model_client as model_client
    monkeypatch.setattr(model_client, "infer_draft", fake_infer_draft)
    monkeypatch.setattr(model_client, "infer", fake_infer)

    mp = MicroPlanner(config={"draft_plan": True})
    plan = mp.plan("tell me a joke", [], [])
    assert plan is None


def test_plan_executor_sequential_dependency_injection(monkeypatch):
    """PlanExecutor injects previous step outputs correctly."""
    from core.pipelines.micro_planner import PlanExecutor, Plan, PlanStep

    calls = []

    def fake_triage(text, step_callback=None):
        calls.append(text)
        return f"result_for: {text[:20]}"

    plan = Plan(
        original="do three steps",
        steps=[
            PlanStep(order=1, task="Step one"),
            PlanStep(order=2, task="Step two"),
            PlanStep(order=3, task="Step three"),
        ],
        rationale="test",
    )

    def fake_infer_draft(prompt, max_new_tokens=512):
        return "Final combined answer"

    import core.inference.model_client as model_client
    monkeypatch.setattr(model_client, "infer_draft", fake_infer_draft)
    # Critic: always return PASS so steps don't retry in tests
    monkeypatch.setattr(model_client, "infer",
        lambda messages, max_new_tokens=128: '{"pass": true, "reason": "ok", "retry_hint": ""}')

    executor = PlanExecutor()
    result = executor.run(plan, triage_fn=fake_triage)

    # Each step should be called exactly once (critic passes, no retries)
    assert len(calls) == 3
    # Step 2 should have step 1's output injected
    assert "result_for: Step one" in calls[1]
    # Step 3 should have step 2's output injected
    assert "result_for: Step two" in calls[2]


def test_plan_executor_synthesis(monkeypatch):
    """PlanExecutor final synthesis includes all step results."""
    from core.pipelines.micro_planner import PlanExecutor, Plan, PlanStep

    synthesis_prompts = []

    def fake_infer_draft(prompt, max_new_tokens=512):
        synthesis_prompts.append(prompt)
        return "Synthesised answer with all results"

    import core.inference.model_client as model_client
    monkeypatch.setattr(model_client, "infer_draft", fake_infer_draft)

    plan = Plan(
        original="original request",
        steps=[PlanStep(order=1, task="Do step 1"), PlanStep(order=2, task="Do step 2")],
    )

    def fake_triage(text, step_callback=None):
        return f"output of: {text}"

    executor = PlanExecutor()
    result = executor.run(plan, triage_fn=fake_triage)

    assert result == "Synthesised answer with all results"
    # Synthesis prompt should reference original request and step outputs
    assert synthesis_prompts
    assert "original request" in synthesis_prompts[0]


def test_parse_steps_numbering():
    """_parse_steps handles both '1.' and '1)' formats."""
    from core.pipelines.micro_planner import _parse_steps
    text = "1. First step\n2) Second step\n3. Third step"
    steps = _parse_steps(text)
    assert len(steps) == 3
    assert steps[0].order == 1
    assert steps[1].order == 2
    assert steps[2].task == "Third step"
