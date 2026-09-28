"""tests/test_provider.py — ADR-013 provider routing tests."""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.inference.provider import InferenceProvider, get_provider


def _make_cfg(providers_block=None):
    cfg = {}
    if providers_block is not None:
        cfg["providers"] = providers_block
    return cfg


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Isolate routing tests from the live shell environment.

    InferenceProvider.get_provider() reads PROVIDER_<CALL_TYPE> env vars
    (e.g. PROVIDER_TASK_INFERENCE) which, if set in the test runner's shell,
    would override the config under test. Clear them so each test asserts
    against its own controlled config, not the environment.
    """
    for key in [k for k in os.environ if k.startswith("PROVIDER_")]:
        monkeypatch.delenv(key, raising=False)


# ── Routing tests ─────────────────────────────────────────────────────────────

def test_provider_routes_local_by_default():
    prov = InferenceProvider(_make_cfg())
    assert prov.get_provider("task_inference") == "local"
    assert prov.get_provider("synthesis") == "local"


def test_provider_routes_openai_for_synthesis():
    cfg = _make_cfg({"synthesis": "openai"})
    prov = InferenceProvider(cfg)
    assert prov.get_provider("synthesis") == "openai"
    assert prov.get_provider("task_inference") == "local"


def test_provider_env_override(monkeypatch):
    monkeypatch.setenv("PROVIDER_TASK_INFERENCE", "anthropic")
    cfg = _make_cfg({"task_inference": "local"})
    prov = InferenceProvider(cfg)
    assert prov.get_provider("task_inference") == "anthropic"


def test_provider_model_selection():
    cfg = _make_cfg({
        "task_inference": "openai",
        "models": {"openai": "gpt-5.4"},
    })
    prov = InferenceProvider(cfg)
    assert prov.get_model("openai") == "gpt-5.4"


def test_provider_model_override_per_calltype():
    cfg = _make_cfg({
        "synthesis": "openai",
        "models": {"openai": "gpt-5.4"},
        "model_overrides": {"synthesis": "gpt-5.4"},
    })
    prov = InferenceProvider(cfg)
    assert prov.get_model("openai", call_type="synthesis") == "gpt-5.4"


def test_provider_falls_back_to_local_on_error(monkeypatch):
    cfg = _make_cfg({
        "task_inference": "openai",
        "models": {"openai": "gpt-5.4"},
    })
    prov = InferenceProvider(cfg)

    # Mock _call_openai to raise, and local infer to return sentinel
    calls = []

    def _mock_call_openai(messages, model):
        raise RuntimeError("openai down")

    def _mock_local_infer(messages, max_new_tokens=1024):
        calls.append("local")
        return "local-response"

    monkeypatch.setattr(prov, "_call_openai", _mock_call_openai)
    with patch.dict("sys.modules", {"core.inference.model_client": MagicMock(infer=_mock_local_infer)}):
        result = prov.infer([{"role": "user", "content": "hello"}], call_type="task_inference")

    assert result == "local-response"
    assert "local" in calls


def _make_provider_with_temp_cfg(critical=85, resume=70, fallback="openai"):
    cfg = {
        "providers": {
            "task_inference": "local",
            "synthesis": "openai",
            "gpu_temp_critical_c": critical,
            "gpu_temp_resume_c": resume,
            "fallback_provider": fallback,
            "models": {"openai": "gpt-5.4"},
        }
    }
    from core.inference.provider import InferenceProvider
    return InferenceProvider(cfg)


def test_thermal_fallback_triggers_above_critical():
    """When GPU temp >= critical, task_inference should route to fallback provider."""
    from unittest.mock import patch
    p = _make_provider_with_temp_cfg(critical=80)
    with patch("core.inference.provider._gpu_temp_celsius", return_value=85):
        route = p.get_provider("task_inference")
    assert route == "openai"
    assert p._temp_fallback_active is True


def test_thermal_fallback_resumes_below_resume():
    """Once active, fallback should stop when temp drops below resume threshold."""
    from unittest.mock import patch
    p = _make_provider_with_temp_cfg(critical=80, resume=70)
    p._temp_fallback_active = True
    with patch("core.inference.provider._gpu_temp_celsius", return_value=65):
        route = p.get_provider("task_inference")
    assert route == "local"
    assert p._temp_fallback_active is False


def test_thermal_fallback_does_not_affect_synthesis():
    """Thermal fallback only applies to task_inference, not synthesis."""
    from unittest.mock import patch
    p = _make_provider_with_temp_cfg(critical=80)
    p._temp_fallback_active = True
    # synthesis is explicitly openai — should stay openai regardless
    with patch("core.inference.provider._gpu_temp_celsius", return_value=90):
        route = p.get_provider("synthesis")
    assert route == "openai"


def test_thermal_fallback_disabled_when_critical_zero():
    """Setting gpu_temp_critical_c=0 disables thermal fallback entirely."""
    from unittest.mock import patch
    p = _make_provider_with_temp_cfg(critical=0)
    with patch("core.inference.provider._gpu_temp_celsius", return_value=99):
        route = p.get_provider("task_inference")
    assert route == "local"


def test_thermal_fallback_stays_active_between_critical_and_resume():
    """Once triggered, fallback should stay active until temp drops below resume."""
    from unittest.mock import patch
    p = _make_provider_with_temp_cfg(critical=80, resume=70)
    p._temp_fallback_active = True
    # Temp between resume and critical — stays active
    with patch("core.inference.provider._gpu_temp_celsius", return_value=75):
        route = p.get_provider("task_inference")
    assert route == "openai"
    assert p._temp_fallback_active is True
