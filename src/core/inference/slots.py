"""Slot lifecycle helpers for the model server.

Extracted from the original model_server.py monolith (refactor). Owns the named
model-slot registry construction, the primary `load_model` entry point, and the
lazy loaders for the multimodal (STT/vision) and tool-calling slots:

  - `_make_slot_loader`            — loader callable handed to SlotRegistry
  - `_load_model`                  — vLLM-first / HF-fallback main-model loader
  - `_ensure_model`                — idempotent main-model load (+ MODEL_ADAPTER_PATH)
  - `_ensure_multimodal_slot`      — lazy Gemma 4 E2B-it STT/vision slot
  - `_ensure_tool_calling_slot`    — lazy Qwen3-0.6B tool-calling slot

Dependencies live in their extracted homes: config.py, capabilities.py,
vllm.py, model_names.py, loaders.py (HF backends), adapters.py (LoRA).
The `_SLOTS_AVAILABLE` flag and the lazy `from model_slots import ...` pattern
are owned HERE; model_server.py imports them back so its remaining handlers
keep working unchanged.
"""

from . import state as server_state
from . import vllm as _vllm
from .config import load_config as _load_config
from .capabilities import apply_detect as _detect_capabilities
from .model_names import (
    is_janus_model as _is_janus_model,
    is_nemotron_model as _is_nemotron_model,
)
from .loaders import (
    _load_hf_model,
    _load_nemotron,
    _load_janus,
)
from .adapters import _load_adapter

import os

# Slot registry availability — imported lazily to avoid circular import; the
# flag is owned here and re-exposed to model_server.py.
try:
    from model_slots import SlotRegistry, SlotSpec, SlotState  # type: ignore
    _SLOTS_AVAILABLE = True
except ImportError:
    _SLOTS_AVAILABLE = False


def _make_slot_loader():
    """Return a loader callable for SlotRegistry.set_loader().
    Handles HF transformers load for any non-primary slot spec.
    Primary slot loading is handled by _load_model() directly.
    """
    def _load_slot_fn(spec):
        """Load a SlotSpec and return a SlotState. Called by SlotRegistry.load()."""
        import torch
        import time as _time
        from transformers import AutoProcessor, AutoModel
        if _SLOTS_AVAILABLE:
            from model_slots import SlotState  # type: ignore
        else:
            raise RuntimeError("model_slots module not available")

        model_path = spec.model_path
        if not model_path:
            raise ValueError(f"slot '{spec.name}' has no model_path configured")

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(spec.dtype or "bfloat16", torch.bfloat16)
        load_kwargs = {"device_map": spec.device or "auto", "torch_dtype": dtype}

        # Audio models: skip quantisation for stability (audio tower incompatible with BNB)
        quantize = spec.quantize
        if spec.role == "audio" and quantize in ("4bit", "8bit"):
            print(f"[slot_loader] '{spec.name}': quantize disabled for audio role", flush=True)
            quantize = None

        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[slot_loader] loading slot '{spec.name}' from {model_path}", flush=True)
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
        print(f"[slot_loader] slot '{spec.name}' loaded on {next(model.parameters()).device}", flush=True)

        audio_capable = spec.role == "audio"
        return SlotState(
            spec=spec,
            model=model,
            processor=processor,
            is_nemotron=False,
            audio_capable=audio_capable,
            supports_tools=False,
            loaded_at=_time.time(),
        )
    return _load_slot_fn


def _load_model(config_path="config.yaml", model_path_override: str | None = None):
    """Load main model — tries vLLM first, falls back to HF transformers.
    Routes Nemotron-Labs-Diffusion models to the dedicated _load_nemotron path.

    model_path_override: force a specific base model path, bypassing config.yaml's
    model.path/model.name. Used by eval tooling to load the correct base model for
    a given LoRA adapter (adapter.base_model_name_or_path), independent of whatever
    model is currently configured for live inference.
    """
    with server_state.load_lock:
        if _vllm.is_enabled() or server_state.model is not None:
            return  # already loaded

        cfg = _load_config(config_path)

        # Build SlotRegistry from config if model_slots section present
        if _SLOTS_AVAILABLE and "model_slots" in cfg and cfg["model_slots"]:
            try:
                from model_slots import SlotRegistry as _SR, SlotSpec as _SS  # type: ignore
                server_state.slot_registry = _SR(vram_threshold_mb=cfg.get("vram_threshold_mb", 3000))
                for slot_name, slot_cfg in (cfg["model_slots"] or {}).items():
                    slot_cfg = slot_cfg or {}
                    spec = _SS(
                        name=slot_name,
                        model_path=slot_cfg.get("model_path"),
                        dtype=slot_cfg.get("dtype", "bfloat16"),
                        quantize=slot_cfg.get("quantize"),
                        device=slot_cfg.get("device", "auto"),
                        role=slot_cfg.get("role", "primary"),
                        max_context_length=slot_cfg.get("max_context_length", 8192),
                        adapter_path=slot_cfg.get("adapter_path"),
                    )
                    server_state.slot_registry.register(spec)
                # Wire a slot loader so load_slot() RPC can hot-load any registered spec
                server_state.slot_registry.set_loader(_make_slot_loader())
                print(f"[model_server] SlotRegistry built with {len(cfg['model_slots'])} slot(s)", flush=True)
            except Exception as _sre:
                print(f"[model_server] WARNING: failed to build SlotRegistry: {_sre}", flush=True)
                server_state.slot_registry = None
        model_path = model_path_override or cfg["model"].get("path") or cfg["model"].get("name", "")

        # Janus uses the custom `janus` package — completely different API
        if _is_janus_model(model_path):
            print("[model_server] Janus detected — using janus package loader", flush=True)
            _load_janus(config_path, model_path_override=model_path)
            return

        # Nemotron uses its own loader — completely different API from HF standard
        if _is_nemotron_model(model_path):
            print("[model_server] Nemotron-Labs-Diffusion detected — using Nemotron loader", flush=True)
            _load_nemotron(config_path, model_path_override=model_path)
            return

        backend = cfg.get("inference", {}).get("backend", "vllm")

        if backend == "vllm":
            success = _vllm.load_engine(server_state.config, _detect_capabilities, model_path_override=model_path if model_path_override else None)
            if not success:
                # Fallback to HF
                _load_hf_model(config_path, model_path_override=model_path if model_path_override else None)
        else:
            # Explicit transformers backend
            print("[model_server] inference.backend=transformers — skipping vLLM", flush=True)
            _load_hf_model(config_path, model_path_override=model_path if model_path_override else None)


def _ensure_model():
    if not _vllm.is_enabled() and server_state.model is None:
        _load_model(server_state.lazy_config_path, model_path_override=server_state.lazy_model_override)
        adapter_path = os.environ.get("MODEL_ADAPTER_PATH")
        if adapter_path:
            _load_adapter(adapter_path)


def _ensure_multimodal_slot():
    """Lazy-load the multimodal slot (Gemma 4 E2B-it) on first use.

    Triggered by voice notes (STT), PDF/image inference (vision), or any
    combined multimodal request — whichever comes first.

    Optimisation: if the multimodal slot path matches the already-loaded
    main model, reuse _model/_processor directly to avoid a second ~9GB
    copy in VRAM.
    """
    if server_state.mm_model is not None:
        print("[model_server] Multimodal slot already loaded", flush=True)
        return
    print("[model_server] Multimodal slot not yet loaded, initializing...", flush=True)
    # Ensure config is loaded (the multimodal slot path comes from stt_model).
    # Do NOT call _ensure_model() here — that loads the main (text-only) model,
    # which is unnecessary for vision/STT and can fail if the main model path
    # is a cloud-only config (e.g. ${KERNEL_EVO_HF_HUB} unexpanded in a spawned
    # on-demand server). The multimodal slot is loaded independently below.
    if server_state.config is None:
        _load_config(server_state.lazy_config_path)
    stt_cfg = (server_state.config or {}).get("stt_model", {})
    stt_path = stt_cfg.get("path") or stt_cfg.get("name")
    if not stt_path:
        err = "stt_model not configured in config.yaml"
        print(f"[model_server] ERROR: {err}", flush=True)
        raise RuntimeError(err)

    # ── Same-model optimisation ───────────────────────────────────────────────
    # Resolve both paths and compare. If they point to the same weights, alias
    # the STT handle to the already-loaded main model — zero extra VRAM.
    import os
    main_path = ((server_state.config or {}).get("model") or {}).get("path") or ""
    stt_resolved  = os.path.realpath(os.path.expanduser(stt_path))
    main_resolved = os.path.realpath(os.path.expanduser(main_path))
    if stt_resolved == main_resolved and server_state.model is not None:
        print(
            f"[model_server] STT path == main model path — reusing loaded model (saves ~9GB VRAM)",
            flush=True,
        )
        server_state.mm_model     = server_state.model
        server_state.mm_processor = server_state.processor
        return
    # ─────────────────────────────────────────────────────────────────────────

    dtype_str = stt_cfg.get("dtype", "bfloat16")
    quantize = stt_cfg.get("quantize", None)
    device = stt_cfg.get("device", "auto")
    import torch
    from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    load_kwargs = {"device_map": device, "torch_dtype": dtype}
    # Audio path stability: Gemma-4 audio tower can fail with quantized weights
    # (torch.finfo on non-floating dtype). Force full-precision STT loading.
    if quantize in ("4bit", "8bit"):
        print(f"[model_server] STT quantization '{quantize}' disabled for audio stability", flush=True)
        quantize = None
    print(f"[model_server] Loading multimodal slot: {stt_path} (device={device}, dtype={dtype_str})", flush=True)
    try:
        server_state.mm_processor = AutoProcessor.from_pretrained(stt_path)
        print(f"[model_server] STT processor loaded", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading STT processor: {e}", flush=True)
        raise
    try:
        # AutoModel loads the base class (e.g. Gemma4Model) which lacks .generate().
        # Use AutoModelForCausalLM so GenerationMixin is always present.
        from transformers import AutoModelForCausalLM as _AutoCLM
        server_state.mm_model = _AutoCLM.from_pretrained(stt_path, **load_kwargs, trust_remote_code=True)
        if not hasattr(server_state.mm_model, 'generate'):
            # Fallback: architecture-specific class via AutoConfig
            from transformers import AutoConfig as _AConf
            _arch = getattr(_AConf.from_pretrained(stt_path, trust_remote_code=True), 'architectures', [])
            if _arch:
                import importlib as _il
                _Cls = getattr(_il.import_module('transformers'), _arch[0], None)
                if _Cls and hasattr(_Cls, 'generate'):
                    del server_state.mm_model
                    server_state.mm_model = _Cls.from_pretrained(stt_path, **load_kwargs, trust_remote_code=True)
        print(f"[model_server] Multimodal slot loaded to {next(server_state.mm_model.parameters()).device} (class: {server_state.mm_model.__class__.__name__})", flush=True)
        if not hasattr(server_state.mm_model, 'generate'):
            raise AttributeError(f"Multimodal slot class {server_state.mm_model.__class__.__name__} has no .generate() — STT unavailable")
    except Exception as e:
        server_state.mm_model = None
        server_state.mm_processor = None
        print(f"[model_server] ERROR loading multimodal slot: {e}", flush=True)
        raise

    # Wire audio slot into registry (additive — existing _mm_model/_mm_processor remain set)
    if server_state.slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _audio_spec = server_state.slot_registry._specs.get("audio") or SlotSpec(
                name="audio",
                model_path=stt_path,
                role="audio",
            )
            _audio_state = SlotState(
                spec=_audio_spec,
                model=server_state.mm_model,
                processor=server_state.mm_processor,
                is_nemotron=False,
                audio_capable=True,
                supports_tools=False,
            )
            server_state.slot_registry._loaded["audio"] = _audio_state
            print("[model_server] Audio slot wired into SlotRegistry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not wire audio slot: {_se}", flush=True)


# ---------------------------------------------------------------------------
# Tool-calling slot loader
# ---------------------------------------------------------------------------

def _ensure_tool_calling_slot():
    """Lazy-load the tool-calling slot (Qwen3-0.6B) on first use.

    Triggered on the first infer_with_tools call. Qwen3-0.6B handles
    microplanning + native tool calling without thinking overflow.
    Nemotron only gets called at the end for conversation synthesis.
    """

    if server_state.tool_calling_slot_loaded:
        return

    _ensure_model()  # config must be loaded first

    # Check if tool_calling slot is configured
    if server_state.slot_registry is None:
        print("[model_server] No SlotRegistry — tool_calling slot unavailable", flush=True)
        return

    tc_spec = server_state.slot_registry._specs.get("tool_calling")
    if tc_spec is None:
        print("[model_server] No tool_calling slot in config — skipping", flush=True)
        return

    model_path = tc_spec.model_path
    if not model_path:
        print("[model_server] tool_calling slot has no model_path — skipping", flush=True)
        return

    if not os.path.isdir(model_path):
        print(f"[model_server] tool_calling model path not found: {model_path}", flush=True)
        return

    device = tc_spec.device or "cuda:0"
    dtype_str = tc_spec.dtype or "float16"
    import torch
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)

    print(f"[model_server] Loading tool_calling slot: {model_path} (device={device}, dtype={dtype_str})", flush=True)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        server_state.tool_calling_processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[model_server] Tool-calling tokenizer loaded (vocab={server_state.tool_calling_processor.vocab_size})", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading tool-calling tokenizer: {e}", flush=True)
        return

    try:
        load_kwargs = {"device_map": device, "torch_dtype": dtype, "trust_remote_code": True}
        server_state.tool_calling_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not hasattr(server_state.tool_calling_model, 'generate'):
            raise AttributeError(f"Tool-calling model class {server_state.tool_calling_model.__class__.__name__} has no .generate()")
        print(f"[model_server] Tool-calling slot loaded to {next(server_state.tool_calling_model.parameters()).device} (class: {server_state.tool_calling_model.__class__.__name__})", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading tool-calling model: {e}", flush=True)
        server_state.tool_calling_model = None
        server_state.tool_calling_processor = None
        # Retry once after 30s — GPU may have recovered from transient OOM/fragmentation
        import time as _retry_time
        print("[model_server] Retrying tool_calling slot load in 30s...", flush=True)
        _retry_time.sleep(30)
        try:
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            print(f"[model_server] Retry attempt: loading tool_calling slot...", flush=True)
            server_state.tool_calling_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            print(f"[model_server] Tool-calling slot loaded on retry to {next(server_state.tool_calling_model.parameters()).device}", flush=True)
        except Exception as e2:
            print(f"[model_server] ERROR tool-calling slot retry failed: {e2}", flush=True)
            server_state.tool_calling_model = None
            server_state.tool_calling_processor = None
            return

    server_state.tool_calling_slot_loaded = True

    # Wire into slot registry for status reporting
    if server_state.slot_registry is not None:
        try:
            from model_slots import SlotState
            _tc_state = SlotState(
                spec=tc_spec,
                model=server_state.tool_calling_model,
                processor=server_state.tool_calling_processor,
                is_nemotron=False,
                audio_capable=False,
                supports_tools=True,
            )
            server_state.slot_registry._loaded["tool_calling"] = _tc_state
            print("[model_server] Tool-calling slot wired into SlotRegistry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not wire tool_calling slot: {_se}", flush=True)

