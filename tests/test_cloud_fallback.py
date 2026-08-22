"""
test_cloud_fallback.py — Regression tests for cloud audio/vision 402 fallback.

Covers the bug where EVA errored out (issues #426 audio / #427 vision) when the
cloud audio/vision provider failed (HTTP 402 out-of-credit, missing key, network
error). When `providers.vision` / `providers.stt` are set to a cloud provider and
that provider fails, the handler must fall back to the native Gemma E2B
multimodal slot instead of returning the cloud error.

Rules:
  - No real HTTP, no model server, no Telegram, no GPU.
  - All network/model calls mocked; no production DBs (conftest redirects them).
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.inference.model_server as ms


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — build mock model/processor objects that satisfy the local HF path
# ─────────────────────────────────────────────────────────────────────────────
def _mock_param(device="cuda:0", dtype=None):
    """A mock parameter exposing .device and .dtype."""
    if dtype is None:
        import torch
        dtype = torch.float16
    p = MagicMock()
    p.device = device
    p.dtype = dtype
    return p


def _mock_processor(model):
    proc = MagicMock()
    # apply_chat_template returns a text prompt string
    proc.apply_chat_template.return_value = "<prompt>"
    # processor(text=..., images=...) returns input tensors
    inputs = MagicMock()
    inputs.to.return_value = inputs
    # input_ids with a shape of [1, 5]
    ids = MagicMock()
    ids.shape = [1, 5]
    inputs.input_ids = ids
    proc.return_value = inputs
    return proc


def _mock_model():
    model = MagicMock()
    # next(model.parameters()) -> a mock param; return a FRESH iterator on every
    # call because the code calls next(model.parameters()) more than once.
    params = [_mock_param()]
    model.parameters.side_effect = lambda: iter(params)
    # generate() returns output with shape [1, 12]
    out = MagicMock()
    out.shape = [1, 12]
    model.generate.return_value = out
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Vision fallback
# ─────────────────────────────────────────────────────────────────────────────
class TestVisionCloudFallback:
    def _setup(self):
        """Configure cloud vision, force cloud failure, and stub the native slot."""
        cfg = {"providers": {"vision": "openrouter", "stt": "local"},
               "model_overrides": {}, "models": {}}
        img = MagicMock()
        img.convert.return_value = img
        model = _mock_model()
        proc = _mock_processor(model)
        return cfg, img, model, proc

    def test_falls_back_to_native_when_cloud_vision_402(self):
        """Cloud vision HTTP 402 must NOT be returned; native Gemma slot is used."""
        cfg, img, model, proc = self._setup()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_cloud_multimodal_infer",
                          return_value={"error": "Cloud vision failed: HTTP 402"}) as m_cloud, \
             patch("PIL.Image.open", return_value=img), \
             patch.object(ms, "_audio_capable", False), \
             patch.object(ms, "_mm_model", model), \
             patch.object(ms, "_mm_processor", proc), \
             patch.object(ms, "_ensure_multimodal_slot", lambda: None):
            result = ms._handle_infer_with_image(
                {"image_path": tmp.name, "prompt": "Describe this"})

        os.unlink(tmp.name)

        # Cloud was attempted but its error was NOT propagated.
        m_cloud.assert_called_once()
        assert "error" not in result, f"Cloud error leaked: {result}"
        # Native path produced a result.
        assert result.get("result") is not None
        # Native slot was actually used for generation.
        model.generate.assert_called_once()

    def test_returns_cloud_success_when_cloud_works(self):
        """When cloud vision succeeds, its result is returned (no native fallback)."""
        cfg, img, model, proc = self._setup()
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_cloud_multimodal_infer",
                          return_value={"result": "cloud description"}) as m_cloud, \
             patch("PIL.Image.open", return_value=img), \
             patch.object(ms, "_audio_capable", False), \
             patch.object(ms, "_mm_model", model), \
             patch.object(ms, "_mm_processor", proc):
            result = ms._handle_infer_with_image(
                {"image_path": tmp.name, "prompt": "Describe this"})

        os.unlink(tmp.name)

        m_cloud.assert_called_once()
        assert result == {"result": "cloud description"}
        model.generate.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Audio / STT fallback
# ─────────────────────────────────────────────────────────────────────────────
class TestAudioCloudFallback:
    def _setup(self):
        """Configure cloud STT, force cloud failure, and stub the native slot."""
        cfg = {"providers": {"stt": "openrouter", "vision": "local"},
               "model_overrides": {}, "models": {}}
        model = _mock_model()
        proc = _mock_processor(model)
        return cfg, model, proc

    def test_falls_back_to_native_when_cloud_stt_402(self):
        """Cloud STT HTTP 402 must NOT be returned; native Gemma STT slot is used."""
        cfg, model, proc = self._setup()
        # Stub the audio file + ffmpeg decode so the local path runs with mocks.
        # Use a real numpy array (not a MagicMock) because the local path calls
        # np.min/np.max on it for amplitude logging.
        import numpy as np
        audio_array = np.zeros(16000, dtype=np.float32)  # 1-D array (ndim==1)

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_cloud_multimodal_infer",
                          return_value={"error": "Cloud audio failed: HTTP 402"}) as m_cloud, \
             patch.object(ms, "_ensure_model", lambda: None), \
             patch.object(ms, "_slot_registry", None), \
             patch.object(ms, "_audio_capable", False), \
             patch.object(ms, "_mm_model", model), \
             patch.object(ms, "_mm_processor", proc), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("soundfile.read", return_value=(audio_array, 16000)), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100), \
             patch("os.unlink", lambda p: None):
            result = ms._handle_infer_with_audio(
                {"audio_path": "/tmp/voice.wav", "prompt": "Transcribe.",
                 "mode": "stt", "max_new_tokens": 64})

        # Cloud was attempted but its error was NOT propagated.
        m_cloud.assert_called_once()
        assert "error" not in result, f"Cloud error leaked: {result}"
        assert result.get("result") is not None
        # Native STT slot was actually used for generation.
        model.generate.assert_called_once()

    def test_returns_cloud_success_when_cloud_works(self):
        """When cloud STT succeeds, its result is returned (no native fallback)."""
        cfg, model, proc = self._setup()

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_cloud_multimodal_infer",
                          return_value={"result": "transcribed text"}) as m_cloud, \
             patch.object(ms, "_ensure_model", lambda: None):
            result = ms._handle_infer_with_audio(
                {"audio_path": "/tmp/voice.wav", "prompt": "Transcribe.",
                 "mode": "stt", "max_new_tokens": 64})

        m_cloud.assert_called_once()
        assert result == {"result": "transcribed text"}
        model.generate.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# In-process infer_with_audio NoneType guard (model.py)
# ─────────────────────────────────────────────────────────────────────────────
class TestInProcessAudioNoneTypeGuard:
    """Regression: infer_with_audio (in-process fallback) must not crash with
    'NoneType' object has no attribute 'apply_chat_template' when no model is
    loaded (model server running / task routed to cloud). It must return a
    clean error string instead."""

    def test_returns_clean_error_when_no_processor(self):
        import core.inference.model as m
        with patch.object(m, "_processor", None), \
             patch.object(m, "_model", None), \
             patch.object(m, "load", lambda: None), \
             patch("core.inference.model_client.is_server_running", return_value=False):
            result = m.infer_with_audio("/tmp/nonexistent.wav", mode="stt")
        assert isinstance(result, str)
        assert "Audio unavailable" in result
        assert "apply_chat_template" not in result


# ─────────────────────────────────────────────────────────────────────────────
class TestInProcessLoadRaisesKeyError:
    """Regression: when the in-process fallback calls load() and load() raises
    (e.g. KeyError: 'model' when the config has no local model section because
    task_inference is cloud-routed), the audio/vision handlers must degrade to
    a clean 'unavailable' message instead of propagating the KeyError to the
    /transcribe or /describe endpoint (HTTP 500)."""

    def test_audio_returns_clean_error_when_load_raises_keyerror_model(self):
        import core.inference.model as m
        with patch.object(m, "_processor", None), \
             patch.object(m, "_model", None), \
             patch.object(m, "load", side_effect=KeyError("model")), \
             patch("core.inference.model_client.is_server_running", return_value=False):
            result = m.infer_with_audio("/tmp/nonexistent.wav", mode="stt")
        assert isinstance(result, str)
        assert "Audio unavailable" in result
        assert "KeyError" not in result

    def test_vision_returns_clean_error_when_load_raises_keyerror_model(self):
        import core.inference.model as m
        with patch.object(m, "_processor", None), \
             patch.object(m, "_model", None), \
             patch.object(m, "load", side_effect=KeyError("model")), \
             patch("core.inference.model_client.is_server_running", return_value=False):
            result = m.infer_with_image("/tmp/nonexistent.jpg", "Describe this")
        assert isinstance(result, str)
        assert "Vision unavailable" in result
        assert "KeyError" not in result


# ─────────────────────────────────────────────────────────────────────────────
# STT must NOT load the main (text-only) model — route straight to Gemma slot
# ─────────────────────────────────────────────────────────────────────────────
class TestAudioSttSkipsMainModelLoad:
    """Regression: _handle_infer_with_audio must not call _ensure_model() up
    front when the main model is text-only (e.g. Nemotron, not audio-capable).
    Loading Nemotron fails with the VRAM guard when the Gemma multimodal slot
    is already resident. STT must route directly to the Gemma slot."""

    def test_stt_routes_to_gemma_slot_without_loading_main_model(self):
        """With a non-audio-capable main model, STT uses the Gemma slot and
        _ensure_model() is never called (Nemotron not loaded)."""
        cfg = {"providers": {"stt": "local", "vision": "local"},
               "model_overrides": {}, "models": {}}
        model = _mock_model()
        proc = _mock_processor(model)
        import numpy as np
        audio_array = np.zeros(16000, dtype=np.float32)

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_ensure_model", MagicMock()) as m_ensure, \
             patch.object(ms, "_slot_registry", None), \
             patch.object(ms, "_audio_capable", False), \
             patch.object(ms, "_vllm_enabled", False), \
             patch.object(ms, "_mm_model", model), \
             patch.object(ms, "_mm_processor", proc), \
             patch.object(ms, "_ensure_multimodal_slot", lambda: None), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("soundfile.read", return_value=(audio_array, 16000)), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100), \
             patch("os.unlink", lambda p: None):
            result = ms._handle_infer_with_audio(
                {"audio_path": "/tmp/voice.wav", "prompt": "Transcribe.",
                 "mode": "stt", "max_new_tokens": 64})

        # Main model (Nemotron) must NOT be loaded for STT.
        m_ensure.assert_not_called()
        assert "error" not in result, f"Unexpected error: {result}"
        assert result.get("result") is not None
        # Gemma slot actually generated.
        model.generate.assert_called_once()

    def test_stt_loads_main_model_only_when_audio_capable(self):
        """When the main model IS audio-capable, _ensure_model() is still called
        (it is the audio model in that config)."""
        cfg = {"providers": {"stt": "local", "vision": "local"},
               "model_overrides": {}, "models": {}}
        model = _mock_model()
        proc = _mock_processor(model)
        import numpy as np
        audio_array = np.zeros(16000, dtype=np.float32)

        with patch.object(ms, "_config", cfg), \
             patch.object(ms, "_ensure_model", MagicMock()) as m_ensure, \
             patch.object(ms, "_slot_registry", None), \
             patch.object(ms, "_audio_capable", True), \
             patch.object(ms, "_vllm_enabled", False), \
             patch.object(ms, "_model", model), \
             patch.object(ms, "_processor", proc), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch("soundfile.read", return_value=(audio_array, 16000)), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=100), \
             patch("os.unlink", lambda p: None):
            result = ms._handle_infer_with_audio(
                {"audio_path": "/tmp/voice.wav", "prompt": "Transcribe.",
                 "mode": "stt", "max_new_tokens": 64})

        m_ensure.assert_called_once()
        assert "error" not in result, f"Unexpected error: {result}"
        assert result.get("result") is not None
        model.generate.assert_called_once()
