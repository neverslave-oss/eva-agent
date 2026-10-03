"""handlers/ops.py — telemetry/health handlers (issue #2 handlers split).

Extracted from src/core/inference/handlers/__init__.py: _handle_vram_free_mb,
_resolve_loaded_model_name, _handle_health. Re-exported from handlers/__init__.py.
"""
from .. import state as server_state
from .. import vllm as _vllm
from ..model_names import friendly_model_name as _friendly_model_name



def _handle_vram_free_mb(params: dict) -> dict:
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return {"vram_free_mb": free // (1024 * 1024)}
    import psutil
    return {"vram_free_mb": psutil.virtual_memory().available // (1024 * 1024)}

def _resolve_loaded_model_name() -> str:
    """Best-effort name of the model actually resident in memory/VRAM.

    Prefers runtime state (vLLM engine path, HF model's own name_or_path, slot
    registry) over config.yaml metadata, since config can silently drift from
    what's actually loaded (e.g. MS1's swap-clobber bug). Falls back to config
    only when nothing is loaded yet.
    """
    if _vllm.is_enabled() and _vllm.model_path():
        return _friendly_model_name(_vllm.model_path())
    if server_state.model is not None:
        name_or_path = getattr(server_state.model, "name_or_path", None) \
            or getattr(getattr(server_state.model, "config", None), "_name_or_path", None)
        if name_or_path:
            return _friendly_model_name(str(name_or_path))
        return server_state.model.__class__.__name__
    if server_state.slot_registry is not None:
        try:
            primary = server_state.slot_registry._loaded.get("primary")
            model_path = getattr(getattr(primary, "spec", None), "model_path", None)
            if model_path:
                return _friendly_model_name(str(model_path))
        except Exception:
            pass
    return (server_state.config or {}).get("model", {}).get("name", "unknown") if server_state.config else "unknown"

def _handle_health(params: dict) -> dict:
    vram_free = _handle_vram_free_mb({}).get("vram_free_mb", 0)
    model_name = _resolve_loaded_model_name()
    vram_warning = vram_free < 2000
    h = {
        "status": "ready",
        "model": model_name,
        "adapter": server_state.current_adapter_name,
        "vram_free_mb": vram_free,
        "vram_warning": vram_warning,
        "main_model_loaded": _vllm.is_enabled() or server_state.model is not None,
        "vllm_engine": _vllm.is_enabled(),
        "drafter_loaded": server_state.drafter is not None,
        "audio_capable": server_state.audio_capable,
        "multimodal_slot_loaded": server_state.mm_model is not None,
    }
    if server_state.is_nemotron:
        h["nemotron"] = True
        h["nemotron_mode"] = server_state.nemotron_mode
        h["nemotron_block_length"] = server_state.nemotron_block_length
    h["slots"] = server_state.slot_registry.status() if server_state.slot_registry is not None else []
    return h

