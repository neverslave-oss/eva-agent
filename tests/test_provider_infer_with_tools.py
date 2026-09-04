"""
test_provider_infer_with_tools.py — Unit tests for provider fallback in infer_with_tools.

No real sockets, no network calls — all external calls are mocked.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.inference.provider as prov_mod
from core.inference.provider import InferenceProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Isolate tests from the live shell environment.

    InferenceProvider.get_provider() reads PROVIDER_<CALL_TYPE> env vars
    (e.g. PROVIDER_TASK_INFERENCE) which, if set in the test runner's shell,
    would override the config under test and route to an unexpected provider.
    Clear them so each test asserts against its own controlled config.
    """
    for key in [k for k in os.environ if k.startswith("PROVIDER_")]:
        monkeypatch.delenv(key, raising=False)


def _make_provider(task_inference="local", fallback="openai"):
    config = {
        "providers": {
            "task_inference": task_inference,
            "fallback_provider": fallback,
            "gpu_temp_critical_c": 0,  # disable thermal fallback in tests
        }
    }
    return InferenceProvider(config)


# ---------------------------------------------------------------------------
# Local provider success path
# ---------------------------------------------------------------------------

class TestLocalProviderSuccess:

    def test_local_success_returns_result(self):
        p = _make_provider(task_inference="local")
        with patch("core.inference.model_client.infer_with_tools", return_value="local result") as mock_iwt:
            result = p.infer_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
            )
        assert result == "local result"
        mock_iwt.assert_called_once()

    def test_local_success_passes_chunk_callback(self):
        p = _make_provider(task_inference="local")
        chunks = []
        cb = lambda t: chunks.append(t)

        with patch("core.inference.model_client.infer_with_tools", return_value="ok") as mock_iwt:
            p.infer_with_tools(
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                chunk_callback=cb,
            )
        # chunk_callback is passed through to model_client
        call_kwargs = mock_iwt.call_args
        assert call_kwargs.kwargs.get("chunk_callback") is cb or cb in call_kwargs.args


# ---------------------------------------------------------------------------
# Local provider fails → fallback to openai tool loop
# ---------------------------------------------------------------------------

class TestLocalFallbackToOpenai:

    def test_local_fail_falls_back_to_openai(self):
        p = _make_provider(task_inference="local", fallback="openai")

        # local raises; openai returns a result. A dummy API key is set so the
        # provider's key-guard does not skip the (mocked) OpenAI path — no real
        # key or network call is used.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("core.inference.model_client.infer_with_tools", side_effect=Exception("socket error")), \
             patch.object(p, "_openai_tool_loop", return_value="openai result") as mock_oa:
            result = p.infer_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        assert result == "openai result"
        mock_oa.assert_called_once()

    def test_both_fail_returns_unavailable(self):
        p = _make_provider(task_inference="local", fallback="openai")

        with patch("core.inference.model_client.infer_with_tools", side_effect=Exception("socket error")), \
             patch.object(p, "_openai_tool_loop", side_effect=Exception("openai error")):
            result = p.infer_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )

        assert result == "(inference unavailable)"


# ---------------------------------------------------------------------------
# chunk_callback forwarded correctly on OpenAI streaming path
# ---------------------------------------------------------------------------

class TestChunkCallbackForwarded:

    def test_chunk_callback_forwarded_to_openai_loop(self):
        p = _make_provider(task_inference="openai", fallback="openai")
        chunks = []
        cb = lambda t: chunks.append(t)

        # Dummy key so the key-guard doesn't skip the (mocked) OpenAI path.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch.object(p, "_openai_tool_loop", return_value="result") as mock_oa:
            p.infer_with_tools(
                messages=[{"role": "user", "content": "go"}],
                tools=[],
                chunk_callback=cb,
            )

        # chunk_callback must appear in the call
        call_args = mock_oa.call_args
        assert call_args.kwargs.get("chunk_callback") is cb or cb in call_args.args

    def test_chunk_callback_forwarded_on_fallback_path(self):
        p = _make_provider(task_inference="local", fallback="openai")
        chunks = []
        cb = lambda t: chunks.append(t)

        # Dummy key so the key-guard doesn't skip the (mocked) OpenAI fallback.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), \
             patch("core.inference.model_client.infer_with_tools", side_effect=Exception("fail")), \
             patch.object(p, "_openai_tool_loop", return_value="fallback result") as mock_oa:
            result = p.infer_with_tools(
                messages=[{"role": "user", "content": "go"}],
                tools=[],
                chunk_callback=cb,
            )

        assert result == "fallback result"
        call_args = mock_oa.call_args
        assert call_args.kwargs.get("chunk_callback") is cb or cb in call_args.args


# ---------------------------------------------------------------------------
# Unknown provider returns "(inference unavailable)"
# ---------------------------------------------------------------------------

class TestUnknownProvider:

    def test_unknown_provider_returns_unavailable(self):
        p = _make_provider(task_inference="unknown_provider", fallback="unknown_provider")
        # Block local AND all cloud tool-loop fallbacks so the chain is fully exhausted
        cloud_exc = Exception("no cloud")
        with patch("core.inference.model_client.infer_with_tools", side_effect=Exception("no local")), \
             patch.object(p, "_openai_tool_loop", side_effect=cloud_exc), \
             patch.object(p, "_anthropic_tool_loop", side_effect=cloud_exc), \
             patch.object(p, "_hf_tool_loop", side_effect=cloud_exc):
            result = p.infer_with_tools(
                messages=[{"role": "user", "content": "test"}],
                tools=[],
            )
        assert result == "(inference unavailable)"
