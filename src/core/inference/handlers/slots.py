"""handlers/slots.py — model slot lifecycle handlers (issue #2 handlers split).

Extracted from src/core/inference/handlers/__init__.py: _handle_load_slot,
_handle_unload_slot, _handle_slot_status, _handle_unload, _handle_swap_model.
Imports _resolve_loaded_model_name from .ops (call graph). Re-exported from
handlers/__init__.py.
"""
from .. import state as server_state
from .. import vllm as _vllm
from ..loaders import _load_hf_model, _load_nemotron
from ..model_names import is_nemotron_model as _is_nemotron_model
from ..capabilities import apply_detect as _detect_capabilities
from .ops import _resolve_loaded_model_name



def _handle_load_slot(params: dict) -> dict:
    """Load a named slot on demand. Returns slot status entry."""
    slot_name = params.get("slot")
    if not slot_name:
        return {"error": "missing required param: slot"}
    if server_state.slot_registry is None:
        return {"error": "SlotRegistry not initialised (no model_slots in config)"}
    try:
        state = server_state.slot_registry.load(slot_name)
        return {"ok": True, "slot": slot_name, "loaded_at": state.loaded_at}
    except Exception as exc:
        return {"error": str(exc)}

def _handle_unload_slot(params: dict) -> dict:
    """Unload a named slot and free its VRAM."""
    slot_name = params.get("slot")
    if not slot_name:
        return {"error": "missing required param: slot"}
    if server_state.slot_registry is None:
        return {"error": "SlotRegistry not initialised"}
    try:
        server_state.slot_registry.unload(slot_name)
        return {"ok": True, "slot": slot_name}
    except Exception as exc:
        return {"error": str(exc)}

def _handle_slot_status(params: dict) -> dict:
    """Return JSON-serialisable status of all registered slots."""
    return {"slots": server_state.slot_registry.status() if server_state.slot_registry is not None else []}

def _handle_unload(params: dict) -> dict:
    """Unload the current model from GPU memory without reloading.
    Called before switching task_inference to a cloud provider, freeing VRAM.
    """
    import gc
    import torch

    print("[model_server] unload: releasing model from VRAM...", flush=True)

    _vllm.reset()

    for attr in ("_model", "_processor", "_drafter", "_drafter_tokenizer"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
            globals()[attr] = None

    # Free the multimodal slot (Gemma) too — this is what STT/vision/audio and
    # infer_local use, and it holds a large VRAM footprint that unload() must
    # release so a different primary model can be tested.
    for attr in ("_mm_model", "_mm_processor"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
            globals()[attr] = None

    # Free all named slots in the SlotRegistry (e.g. tool_calling/Qwen).
    if server_state.slot_registry is not None:
        try:
            for _sname in list((server_state.slot_registry.loaded_slots() or {}).keys()):
                server_state.slot_registry.unload(_sname)
        except Exception as _sl_err:
            print(f"[model_server] unload: slot cleanup error ({_sl_err})", flush=True)

    server_state.is_nemotron = False
    server_state.native_agentic = False

    gc.collect()
    freed_mb = 0
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        freed_mb = (before - torch.cuda.memory_allocated()) // (1024 * 1024)
        free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        print(f"[model_server] unload: done — freed ~{freed_mb}MB, {free_mb}MB now free", flush=True)

    return {"status": "unloaded", "freed_mb": freed_mb}

def _handle_swap_model(params: dict) -> dict:
    """Hot-swap the loaded model.
    For vLLM backend: shuts down engine and reloads with new path.
    For HF backend: unloads and reloads as before.
    Routes Nemotron-Labs-Diffusion models to the dedicated loader.
    """

    import gc
    import torch

    new_path = params.get("model_path", "").strip()
    drafter_path = params.get("drafter_path", None)
    dtype_str = params.get("dtype", "bfloat16")

    if not new_path:
        return {"error": "model_path is required"}

    print(f"[model_server] swap_model: unloading current model...", flush=True)

    # Unload vLLM engine
    _vllm.reset()

    # Unload HF model
    for attr in ("_model", "_processor", "_drafter", "_drafter_tokenizer"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
            globals()[attr] = None

    server_state.is_nemotron = False

    gc.collect()
    free_before = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        print(f"[model_server] swap_model: {free_before}MB VRAM free after unload", flush=True)

    # Update config for new path
    if server_state.config:
        server_state.config["model"]["path"] = new_path
        server_state.config["model"]["name"] = new_path
        server_state.config["model"]["dtype"] = dtype_str
        # MS3: wire drafter_path through so speculative decoding survives a swap.
        # HF backend reads model.drafter_path/model.speculative_decoding; vLLM
        # backend reads inference.speculative_drafter/inference.speculative_decoding.
        # drafter_path=None (param omitted) means "keep current" — leave both untouched.
        if drafter_path is not None:
            server_state.config["model"]["drafter_path"] = drafter_path
            server_state.config["model"]["speculative_decoding"] = bool(drafter_path)
            server_state.config.setdefault("inference", {})
            server_state.config["inference"]["speculative_drafter"] = drafter_path
            server_state.config["inference"]["speculative_decoding"] = bool(drafter_path)

    # Reload via unified path
    print(f"[model_server] swap_model: loading {new_path}...", flush=True)
    try:
        if _is_nemotron_model(new_path):
            print("[model_server] swap_model: Nemotron-Labs-Diffusion detected", flush=True)
            _load_nemotron(server_state.lazy_config_path, model_path_override=new_path)
        else:
            backend = (server_state.config or {}).get("inference", {}).get("backend", "vllm")
            if backend == "vllm":
                success = _vllm.load_engine(server_state.config, _detect_capabilities, model_path_override=new_path)
                if not success:
                    _load_hf_model(server_state.lazy_config_path, model_path_override=new_path)
            else:
                _load_hf_model(server_state.lazy_config_path, model_path_override=new_path)
    except Exception as e:
        return {"error": f"model load failed: {e}"}

    model_name = _resolve_loaded_model_name()
    backend_tag = "nemotron" if server_state.is_nemotron else ("vllm" if _vllm.is_enabled() else "transformers")
    print(f"[model_server] swap_model: ready — {model_name} ({backend_tag})", flush=True)

    # Re-sync primary slot in registry so slot_status() reflects the new model
    if server_state.slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _spec = server_state.slot_registry._specs.get("primary") or SlotSpec(
                name="primary", model_path=new_path, role="primary"
            )
            _spec.model_path = new_path
            _state = SlotState(
                spec=_spec,
                model=server_state.model,
                processor=server_state.processor,
                is_nemotron=server_state.is_nemotron,
                audio_capable=server_state.audio_capable,
                supports_tools=server_state.model_supports_tools,
                native_agentic=server_state.native_agentic,
                is_omni=server_state.is_omni,
                is_janus=server_state.is_janus,
            )
            server_state.slot_registry._loaded["primary"] = _state
            server_state.slot_registry._specs["primary"] = _spec
            print("[model_server] swap_model: primary slot synced in registry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not sync primary slot after swap: {_se}", flush=True)

    return {"status": "ok", "model": model_name, "backend": backend_tag}

