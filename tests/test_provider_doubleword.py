"""tests/test_provider_doubleword.py — Doubleword provider routing + call tests.

Doubleword (https://api.doubleword.ai/v1) is an OpenAI-compatible inference
provider added to the ADR-013 provider chain. These tests assert:

  - `doubleword` routes via config like any other cloud provider.
  - The configured model resolves for the `doubleword` provider.
  - `_call_doubleword` POSTs to https://api.doubleword.ai/v1/chat/completions
    with the OpenAI-compatible payload/headers and parses the standard
    choices[0].message.content response.
  - The 401-no-key guard raises RuntimeError (chain skips cleanly).
  - The tool loop wires the doubleword base URL when selected as provider.

No real HTTP, no model server, no Telegram, no GPU. All network calls mocked.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.inference.provider import InferenceProvider


def _make_cfg(providers_block=None):
    cfg = {}
    if providers_block is not None:
        cfg["providers"] = providers_block
    return cfg


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Isolate routing tests from the live shell environment."""
    for key in [k for k in os.environ if k.startswith("PROVIDER_")]:
        monkeypatch.delenv(key, raising=False)


# ── Routing ─────────────────────────────────────────────────────────────

def test_provider_routes_doubleword_for_task_inference():
    cfg = _make_cfg({"task_inference": "doubleword"})
    prov = InferenceProvider(cfg)
    assert prov.get_provider("task_inference") == "doubleword"


def test_provider_doubleword_model_selection():
    cfg = _make_cfg({
        "task_inference": "doubleword",
        "models": {"doubleword": "deepseek-ai/DeepSeek-V4-Flash-0731"},
    })
    prov = InferenceProvider(cfg)
    assert prov.get_model("doubleword") == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert prov.get_provider("task_inference") == "doubleword"


# ── _call_doubleword HTTP path ──────────────────────────────────────────

def test_call_doubleword_success(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-test-key")
    prov = InferenceProvider(_make_cfg())

    fake_response = io_bytes(json.dumps({
        "choices": [{"message": {"content": "otters in space"}}],
    }).encode())
    mock_urlopen = MagicMock(return_value=fake_response)
    captured = {}

    import urllib.request
    with patch("urllib.request.urlopen", mock_urlopen):
        result = prov._call_doubleword(
            [{"role": "user", "content": "hi"}],
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
        )
        captured["req"] = mock_urlopen.call_args[0][0]

    assert result == "otters in space"
    req = captured["req"]
    assert req.full_url == "https://api.doubleword.ai/v1/chat/completions"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer dw-test-key"
    body = json.loads(req.data.decode())
    assert body["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert body["messages"][0]["content"] == "hi"


def test_call_doubleword_requires_key():
    env = {k: v for k, v in os.environ.items() if k != "DOUBLEWORD_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        prov = InferenceProvider(_make_cfg())
        with pytest.raises(RuntimeError):
            prov._call_doubleword([{"role": "user", "content": "hi"}], model="m")


def test_provider_skips_doubleword_without_key(monkeypatch):
    """Chain must not attempt doubleword when no key is set."""
    env = {k: v for k, v in os.environ.items() if k != "DOUBLEWORD_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        prov = InferenceProvider(_make_cfg())
        # Force doubleword as the configured provider with no key; the chain must
        # skip it (no attempt / no keyguard crash) and continue.
        cfg = _make_cfg({"task_inference": "doubleword"})
        prov = InferenceProvider(cfg)
        # monkeypatch local fallback to return a sentinel so we can assert the
        # chain landed there (i.e. doubleword was skipped, not attempted).
        calls = []
        def _mock_local_infer(messages, max_new_tokens=1024):
            calls.append("local")
            return "local-response"
        import core.inference.provider as provider_mod
        monkeypatch.setattr(prov, "_call_openai", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not reach openai")))
        with patch.dict("sys.modules", {"core.inference.model_client": MagicMock(infer=_mock_local_infer)}):
            result = prov.infer(
                [{"role": "user", "content": "hello"}],
                call_type="task_inference",
            )
        assert result == "local-response"
        assert "local" in calls


# ── Tool loop wiring ────────────────────────────────────────────────────

def test_tool_loop_uses_doubleword_base_url(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-test-key")
    prov = InferenceProvider(_make_cfg())

    # Non-streaming final response: urlopen returns an object whose .read() is the
    # full JSON body.
    fake_resp = MagicMock()
    fake_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "hi"}}]
    }).encode()
    captured = {}
    def _fake_urlopen(req, timeout=120):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        return fake_resp
    with patch("urllib.request.urlopen", _fake_urlopen):
        result = prov._openai_tool_loop(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            workspace="/tmp",
            max_steps=1,
            step_callback=None,
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
            provider="doubleword",
        )
    assert result == "hi"
    assert captured["url"] == "https://api.doubleword.ai/v1/chat/completions"
    assert captured["auth"] == "Bearer dw-test-key"



# ── Async flex tier (_call_doubleword_async) ───────────────────────────

def _mock_response_flow(submit_body, poll_bodies):
    """Return a urlopen mock that serves one submit response then poll bodies.

    submit_body: dict — the POST /responses response (must include an id).
    poll_bodies: list of dicts — GET /responses/{id} responses, consumed in order.
    """
    import io
    seq = []
    seq.append(io.BytesIO(json.dumps(submit_body).encode()))
    for b in poll_bodies:
        seq.append(io.BytesIO(json.dumps(b).encode()))
    calls = []
    def _fake_urlopen(req, timeout=None):
        calls.append(req)
        return seq.pop(0)
    return _fake_urlopen, calls


def test_call_doubleword_async_success(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-test-key")
    prov = InferenceProvider(_make_cfg())

    submit = {"id": "resp_123", "status": "queued", "output": []}
    polls = [
        {"id": "resp_123", "status": "in_progress", "output": []},
        {"id": "resp_123", "status": "completed", "output": [
            {"type": "message", "content": [
                {"type": "output_text", "text": "async result"}
            ]}
        ]},
    ]
    fake_urlopen, calls = _mock_response_flow(submit, polls)

    with patch("time.sleep", return_value=None), \
         patch("urllib.request.urlopen", fake_urlopen):
        result = prov._call_doubleword_async(
            [{"role": "user", "content": "hi"}],
            model="deepseek-ai/DeepSeek-V4-Flash-0731",
        )

    assert result == "async result"
    # First call = submit POST; subsequent = GET polls
    assert calls[0].full_url == "https://api.doubleword.ai/v1/responses"
    assert calls[0].get_method() == "POST"
    submit_payload = json.loads(calls[0].data.decode())
    assert submit_payload["service_tier"] == "flex"
    assert submit_payload["background"] is True
    assert submit_payload["input"][0]["content"] == "hi"
    # polls hit the retrieve endpoint
    assert calls[1].full_url == "https://api.doubleword.ai/v1/responses/resp_123"


def test_call_doubleword_async_requires_key():
    env = {k: v for k, v in os.environ.items() if k != "DOUBLEWORD_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        prov = InferenceProvider(_make_cfg())
        with pytest.raises(RuntimeError):
            prov._call_doubleword_async([{"role": "user", "content": "hi"}], model="m")


def test_call_doubleword_async_raises_on_failed_status(monkeypatch):
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-test-key")
    prov = InferenceProvider(_make_cfg())
    submit = {"id": "resp_x", "status": "queued", "output": []}
    polls = [{"id": "resp_x", "status": "failed", "error": {"message": "boom"}, "output": []}]
    fake_urlopen, _ = _mock_response_flow(submit, polls)
    with patch("time.sleep", return_value=None), \
         patch("urllib.request.urlopen", fake_urlopen):
        with pytest.raises(RuntimeError, match="failed"):
            prov._call_doubleword_async([{"role": "user", "content": "hi"}], model="m")


def test_async_gating_routes_background_types_to_flex(monkeypatch):
    """Synthesis routes to async flex; task_inference stays realtime."""
    monkeypatch.setenv("DOUBLEWORD_API_KEY", "dw-test-key")
    prov = InferenceProvider(_make_cfg({
        "task_inference": "doubleword",
        "synthesis": "doubleword",
        "models": {"doubleword": "deepseek-ai/DeepSeek-V4-Flash-0731"},
    }))
    assert "synthesis" in prov._doubleword_async_call_types
    assert "task_inference" not in prov._doubleword_async_call_types

    # monkeypatch both callers; assert dispatch picks async for synthesis
    calls = []
    prov._call_doubleword_async = lambda *a, **k: calls.append("async") or "A"
    prov._call_doubleword = lambda *a, **k: calls.append("realtime") or "R"
    assert prov.infer([{"role": "user", "content": "x"}], call_type="synthesis") == "A"
    assert prov.infer([{"role": "user", "content": "x"}], call_type="task_inference") == "R"
    assert calls == ["async", "realtime"]


def io_bytes(b):
    """Return a file-like object wrapping bytes for urlopen-style reads."""
    import io
    return io.BytesIO(b)
