"""
test_nemotron_native_agentic.py — Regression tests for wiring Nemotron + its FC
adapter as a standalone, self-driving tool-calling model (native-agentic path).

Covers:
  - model.native_agentic=true makes _load_nemotron mark Nemotron native-agentic
    (so the single-model tool loop is used, not the Qwen two-stage detour).
  - The FC adapter (model.adapter_path) is loaded as an activatable PeftModel
    wrapper instead of the linear_spec speed-drafter unwrap.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.inference.model_server as ms


def _set_globals():
    ms._model = None
    ms._processor = None
    ms._config = {
        "model": {
            "name": "nvidia/Nemotron-Labs-Diffusion-3B",
            "path": "/mnt/e/models/.../Nemotron-Labs-Diffusion-3B",
            "dtype": "bfloat16",
            "generation_mode": "linear_spec",
            "block_length": 32,
            "threshold": 0.9,
            "quantize": "none",
        },
        "model_context_lengths": {"nemotron": 65536},
    }
    ms._nemotron_mode = "linear_spec"
    ms._loaded_adapters = {}
    ms._is_nemotron = False
    ms._native_agentic = False
    ms._model_supports_tools = False


@pytest.fixture(autouse=True)
def reset():
    _set_globals()
    yield


def _run_load(native_agentic=False, adapter_path=None, monkeypatch=None, unwrap=True):
    """Drive _load_nemotron with mocked transformers/peft/torch, returning the
    value assigned to ms._model after the call."""
    ms._config["model"]["native_agentic"] = native_agentic
    ms._config["model"]["adapter_path"] = adapter_path

    base = MagicMock()
    base.model = MagicMock()  # PeftModel wrapper exposes `.model`

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(ms, "_adapter_name", lambda p: "fc_adapter")

    # Patch inside-function imports at their source modules.
    import transformers, peft
    monkeypatch.setattr(transformers, "AutoTokenizer", MagicMock())
    transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
    monkeypatch.setattr(transformers, "AutoModel", MagicMock())
    transformers.AutoModel.from_pretrained.return_value = base

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*a, **k):
            cfg = MagicMock()
            cfg.max_position_embeddings = 262144
            return cfg
    monkeypatch.setattr(transformers, "AutoConfig", FakeAutoConfig)

    wrapper = MagicMock()
    wrapper.model = MagicMock()
    fake_peft = MagicMock()
    fake_peft.from_pretrained.return_value = wrapper
    fake_peft.from_pretrained.return_value.eval.return_value = wrapper
    monkeypatch.setattr(peft, "PeftModel", fake_peft)

    ms._load_nemotron(config_path=None, model_path_override="/mnt/e/.../Nemotron")

    return base, wrapper


def test_native_agentic_sets_native_flag(monkeypatch):
    """model.native_agentic=true must surface _is_nemotron=True AND
    _native_agentic=True so the two-stage Qwen detour is skipped."""
    _run_load(native_agentic=True, monkeypatch=monkeypatch)
    assert ms._is_nemotron is True
    assert ms._native_agentic is True


def test_default_nemotron_not_native(monkeypatch):
    """Without the flag, Nemotron stays non-native-agentic (Qwen two-stage path)."""
    _run_load(native_agentic=False, monkeypatch=monkeypatch)
    assert ms._is_nemotron is True
    assert ms._native_agentic is False


def test_native_agentic_keeps_peft_wrapper(monkeypatch):
    """native_agentic=true + adapter_path must keep the PeftModel wrapper
    (so the FC adapter stays activatable), and register the adapter name."""
    base, wrapper = _run_load(native_agentic=True, adapter_path="/tmp/fc", monkeypatch=monkeypatch)
    assert ms._model is wrapper          # wrapped, NOT base.model unwrap
    assert ms._loaded_adapters.get("/tmp/fc") == "fc_adapter"


def test_default_mode_unwraps_drafter(monkeypatch):
    """Default (non-native) linear_spec mode unwraps to base.model (speed drafter),
    preserving the original behaviour — _model becomes base.model, not the wrapper."""
    base, wrapper = _run_load(native_agentic=False, adapter_path="/tmp/drafter", monkeypatch=monkeypatch)
    assert ms._model is wrapper.model     # unwrapped to base.model
    assert ms._loaded_adapters.get("/tmp/drafter") is None
