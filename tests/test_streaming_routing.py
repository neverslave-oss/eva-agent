"""test_streaming_routing.py — v1.19.3: assert chunk_callback flows through PlanExecutor."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_plan_executor_passes_chunk_callback_to_triage():
    """PlanExecutor.run must forward chunk_callback to triage_fn calls."""
    from core.pipelines.micro_planner import Plan, PlanStep, PlanExecutor
    import types, sys

    received_callbacks = []

    def fake_triage(task, step_callback=None, chunk_callback=None):
        received_callbacks.append(chunk_callback)
        return "done"

    plan = Plan(
        original="fetch a URL then save",
        steps=[PlanStep(order=1, task="fetch http://example.com"), PlanStep(order=2, task="write to /tmp/out.txt")],
        rationale="two steps",
    )

    sentinel = object()
    executor = PlanExecutor()

    # Patch model_client so critic LLM soft-check is skipped
    fake_mc = types.ModuleType("model_client")
    fake_mc.infer = lambda *a, **kw: '{"pass": true, "reason": "ok", "retry_hint": ""}'
    fake_mc.infer_draft = lambda *a, **kw: None
    fake_mc.infer_with_tools = lambda *a, **kw: "summary"
    old_mc = sys.modules.get("model_client")
    sys.modules["core.inference.model_client"] = fake_mc
    sys.modules["model_client"] = fake_mc
    try:
        executor.run(plan, triage_fn=fake_triage, chunk_callback=sentinel)
    finally:
        if old_mc is None:
            del sys.modules["core.inference.model_client"]
            if "model_client" in sys.modules: del sys.modules["model_client"]
        else:
            sys.modules["core.inference.model_client"] = old_mc
            sys.modules["model_client"] = old_mc

    assert received_callbacks, "triage_fn was never called"
    assert all(cb is sentinel for cb in received_callbacks), (
        f"chunk_callback was not forwarded; got: {received_callbacks}"
    )


def test_plan_executor_synthesise_uses_chunk_callback(monkeypatch):
    """PlanExecutor._synthesise must call infer_with_tools with chunk_callback when set."""
    from core.pipelines.micro_planner import PlanExecutor
    import core.pipelines.micro_planner as mp_mod

    chunks_seen = []
    calls = []

    def fake_infer_with_tools(messages, tools=None, max_new_tokens=None, chunk_callback=None, **kwargs):
        calls.append({"chunk_callback": chunk_callback})
        if chunk_callback:
            chunk_callback("hello ")
            chunk_callback("world")
        return "hello world"

    monkeypatch.setattr(mp_mod, "_infer_with_tools_fn", None, raising=False)

    import unittest.mock as mock

    sentinel = lambda chunk: chunks_seen.append(chunk)

    executor = PlanExecutor()
    with mock.patch.dict('sys.modules', {}):
        with mock.patch('builtins.__import__', side_effect=lambda name, *a, **kw: (
            type(sys)('model_client') if name == 'model_client' else __import__(name, *a, **kw)
        )):
            pass  # just verify the path exists

    # Direct test: patch model_client.infer_with_tools inside the function
    import types
    fake_mc = types.ModuleType("model_client")
    fake_mc.infer_with_tools = fake_infer_with_tools
    fake_mc.infer_draft = lambda *a, **kw: None
    fake_mc.infer = lambda *a, **kw: "fallback"

    old_mc = sys.modules.get("model_client")
    sys.modules["core.inference.model_client"] = fake_mc
    sys.modules["model_client"] = fake_mc
    try:
        result = executor._synthesise("original question", {1: "step one result"}, chunk_callback=sentinel)
    finally:
        if old_mc is None:
            del sys.modules["core.inference.model_client"]
            if "model_client" in sys.modules: del sys.modules["model_client"]
        else:
            sys.modules["core.inference.model_client"] = old_mc
            sys.modules["model_client"] = old_mc

    assert calls, "infer_with_tools was not called"
    assert calls[0]["chunk_callback"] is sentinel, "chunk_callback not forwarded to infer_with_tools"
    assert "hello" in result


def test_simple_chat_chunk_callback_not_lost(monkeypatch):
    """When MicroPlanner.should_plan() returns False, triage falls through to infer_with_tools
    which already has chunk_callback wired. Verify MicroPlanner.should_plan is False for simple inputs."""
    from core.pipelines.micro_planner import MicroPlanner

    mp = MicroPlanner(config={"enabled": True})
    # These are the exact inputs from the bug report
    assert not mp.should_plan("Hello I'm Fabio"), "Simple greeting should NOT trigger planner"
    assert not mp.should_plan("I have 10 cats"), "Simple statement should NOT trigger planner"
    assert not mp.should_plan("tell me a joke"), "Simple request should NOT trigger planner"
