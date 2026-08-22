"""
model_server.py — Long-lived model server process.

Loads Gemma 4 once, stays alive, owns GPU VRAM.
Exposes JSON-RPC over Unix socket /tmp/kernel_model.sock.
API and Telegram bot connect to it and restart freely.

Backend: vLLM AsyncLLMEngine (primary) with HF transformers fallback.
         vLLM 0.21.0+ supports Gemma4ForConditionalGeneration.
         Multimodal slot (Gemma 4 E2B-it) handles STT, vision, and combined inference via HF transformers
         (vLLM multimodal audio API differs from vision — kept on HF for stability).

Usage:
    python3 src/model_server.py [--config config.yaml]
"""
import asyncio
import contextlib
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import traceback
from pathlib import Path

# Allow running from src/ or repo root
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _src not in sys.path:
    sys.path.insert(0, _src)

import yaml

from runtime_paths import MODEL_ACTIVITY_FILE as _MODEL_ACTIVITY_PATH_IMPORT
from core.hf_cache import ensure_hf_home_env

# MS4: pin HF_HOME early so huggingface_hub's own cache resolution (used when
# from_pretrained()/AsyncEngineArgs() are given a bare repo_id) can't silently
# disagree with what telegram_bot.py's /models download check reported.
ensure_hf_home_env()

# R10: gate verbose two-stage debug prints (raw/clean model output) behind an
# env var — off by default so raw model output/user messages aren't dumped
# to the production log on every request.
_DEBUG_TWO_STAGE = os.environ.get("KERNEL_EVO_DEBUG_TWO_STAGE", "0") == "1"

# Activity tracking for Think-at-Rest — mirrors model_client.py counters
# so thought_engine doesn't fire during image/audio inference.
# Resolved from runtime_paths canonical location, overridable via env.
_ACTIVITY_PATH = str(_MODEL_ACTIVITY_PATH_IMPORT)

def _mark_activity_start():
    try:
        import os, json
        os.makedirs(os.path.dirname(_ACTIVITY_PATH), exist_ok=True)
        try:
            with open(_ACTIVITY_PATH) as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["in_flight"] = int(data.get("in_flight", 0)) + 1
        data["last_start_ts"] = time.time()
        with open(_ACTIVITY_PATH, "w") as fw:
            json.dump(data, fw)
    except Exception:
        pass

def _mark_activity_end():
    try:
        import os, json
        try:
            with open(_ACTIVITY_PATH) as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["in_flight"] = max(0, int(data.get("in_flight", 0)) - 1)
        data["last_end_ts"] = time.time()
        with open(_ACTIVITY_PATH, "w") as fw:
            json.dump(data, fw)
    except Exception:
        pass

# SlotRegistry — imported lazily to avoid circular import; None until wired
try:
    from model_slots import SlotRegistry, SlotSpec, SlotState  # type: ignore
    _SLOTS_AVAILABLE = True
except ImportError:
    _SLOTS_AVAILABLE = False

SOCKET_PATH = "/tmp/kernel_evolving_model.sock"

_config = None
_lazy_config_path = "config.yaml"  # set at startup, used by lazy load in handlers


# ── Failure re-prompt helpers (Plan 007) ────────────────────────────────────
def _build_failure_reprompt(tool_name: str, status: str, reason: str, failed_count: int, total_count: int, available_tools: list[str]) -> str:
    """Build a structured re-prompt when a tool call fails.

    Instead of feeding raw error text to the model and hoping it recovers,
    this provides actionable guidance: what failed, why, and what to try next.
    """
    tool_list = ", ".join(sorted(available_tools)) if available_tools else "no alternative tools available"
    status_hints = {
        "empty": "The tool returned no output — it may need different parameters or the target may not exist.",
        "timeout": "The tool timed out — the operation took too long. Try breaking it into smaller steps or using a different approach.",
        "error": f"The tool encountered an error: {reason[:200]}",
        "skill_not_found": f"Skill '{tool_name}' not found. Use search_skills(query='{tool_name}') to find the correct name, then retry.",
        "routine_not_found": f"Routine '{tool_name}' not found. Use list_routines() to see available routines.",
        "no_results": "The tool found no results — try a different query or broader search terms.",
        "skill_execution_failed": f"The skill failed during execution. Try a different approach or use search_skills to find alternatives.",
        "routine_execution_failed": "The routine failed. Check list_routines() for alternatives or try a manual approach.",
        "permission_denied": "Permission denied — you may need to request user approval or use a different path/command.",
    }
    hint = status_hints.get(status, f"Tool call failed (status: {status}). Reason: {reason[:200]}")

    if failed_count == total_count and total_count == 1:
        return (
            f"⚠️ Tool `{tool_name}` failed: {hint}\n"
            f"Available tools: {tool_list}\n"
            f"Please try one of these approaches:\n"
            f"  1. Call the same tool with corrected arguments\n"
            f"  2. Use a different tool that could achieve the same result\n"
            f"  3. If you cannot complete the task with available tools, explain why and suggest escalation"
        )
    elif failed_count == total_count:
        return (
            f"⚠️ All {total_count} tool calls failed. Here's what went wrong with `{tool_name}`: {hint}\n"
            f"Available tools: {tool_list}\n"
            f"Re-evaluate the task and try a different approach."
        )
    else:
        return f"⚠️ This tool call failed ({status}): {reason[:200]}. Some other tools may have succeeded — continue with partial results."

# ---------------------------------------------------------------------------
# vLLM engine (main chat model)
# ---------------------------------------------------------------------------
_vllm_engine = None          # vllm.AsyncLLMEngine instance (or None if fallback)
_vllm_model_path = None      # which path is loaded into vllm
_vllm_enabled = False        # True if vLLM backend is active
_vllm_loop = None            # asyncio event loop running in background thread

# ---------------------------------------------------------------------------
# HF transformers fallback (main chat model — used when vLLM fails to load)
# ---------------------------------------------------------------------------
_model = None
_processor = None          # AutoProcessor (Gemma/Qwen) or AutoTokenizer (Nemotron)
_drafter = None
_drafter_tokenizer = None

# ---------------------------------------------------------------------------
# Nemotron-specific state
# ---------------------------------------------------------------------------
_is_nemotron = False       # True when nvidia/Nemotron-Labs-Diffusion-* is loaded
_nemotron_mode = "linear_spec"  # ar | diffusion | linear_spec (from config)
_nemotron_block_length = 32
_nemotron_threshold = 0.9

# ---------------------------------------------------------------------------
# Capability flags (set during model load)
# ---------------------------------------------------------------------------
_model_supports_tools = True  # True only for Gemma 4+ with native parse_response
_audio_capable = False         # True when main model natively handles audio (Gemma 4)

# ---------------------------------------------------------------------------
# Multimodal slot — Gemma 4 E2B-it loaded via HF transformers.
# Handles STT (audio), vision (image), and combined audio+vision inference.
# Lazy-loaded on first voice note OR first PDF/image — whichever comes first.
# ---------------------------------------------------------------------------
_mm_model = None
_mm_processor = None

# ---------------------------------------------------------------------------
# Tool-calling slot — Qwen3-0.6B loaded via HF transformers.
# Handles microplanning + tool loop execution. Lazy-loaded on first
# infer_with_tools call. Nemotron only called at the end for conversation
# synthesis (never sees <function_calls>).
# ---------------------------------------------------------------------------
_tool_calling_model = None
_tool_calling_processor = None
_tool_calling_slot_loaded = False

# ---------------------------------------------------------------------------
# Internal locks / adapter registry
# ---------------------------------------------------------------------------
# Named model slot registry (optional; None = legacy single-slot mode)
# ---------------------------------------------------------------------------
_slot_registry: "SlotRegistry | None" = None  # type: ignore[name-defined]

# ---------------------------------------------------------------------------
_load_lock = threading.Lock()
_infer_lock = threading.RLock()
_loaded_adapters: dict[str, str] = {}
_current_adapter_name: str | None = None


def _sanitize_text(val: str) -> str:
    from core.tool_arg_utils import sanitize_text
    return sanitize_text(val)


def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    from core.tool_arg_utils import normalize_tool_args
    return normalize_tool_args(tool_name, args)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path="config.yaml"):
    global _config
    from runtime_paths import load_config as _load_expanded
    _config = _load_expanded(path)
    return _config


def _inference_cfg():
    """Return inference sub-config with defaults."""
    return (_config or {}).get("inference", {})


# ---------------------------------------------------------------------------
# Slot registry helpers
# ---------------------------------------------------------------------------

def _sync_globals_from_slot(state: "SlotState") -> None:  # type: ignore[name-defined]
    """
    Copy a SlotState's fields into the module-level globals that all existing
    request handlers reference.  This keeps every handler working unchanged
    while the slot registry manages the objects.
    """
    global _model, _processor, _is_nemotron, _audio_capable, _model_supports_tools
    _model = state.model
    _processor = state.processor
    _is_nemotron = state.is_nemotron
    _audio_capable = state.audio_capable
    _model_supports_tools = state.supports_tools


# ---------------------------------------------------------------------------
# Capability detection helpers
# ---------------------------------------------------------------------------

_AUDIO_CAPABLE_PREFIXES = ("google/gemma-4", "google/gemma-3")

def _detect_capabilities(model_path: str, cfg: dict):
    """Set _model_supports_tools and _audio_capable from model_path + config."""
    global _model_supports_tools, _audio_capable
    model_name = cfg.get("model", {}).get("name", "")
    # Tool support: Gemma 4 has parse_response (detected later via processor),
    # but for vLLM path we detect by name.
    _TOOL_CAPABLE_PREFIXES = ("google/gemma-4",)
    _model_supports_tools = any(
        model_name.lower().startswith(p.lower()) for p in _TOOL_CAPABLE_PREFIXES
    ) or any(
        model_path.lower().replace("\\", "/").find(p.lower().split("/")[-1]) != -1
        for p in _TOOL_CAPABLE_PREFIXES
    )
    # Audio capability
    _audio_capable = any(
        model_name.lower().startswith(p.lower()) for p in _AUDIO_CAPABLE_PREFIXES
    ) or any(
        model_path.lower().replace("\\", "/").find(p.lower().split("/")[-1]) != -1
        for p in _AUDIO_CAPABLE_PREFIXES
    )
    print(f"[model_server] Tool-calling support: {_model_supports_tools}", flush=True)
    print(f"[model_server] Audio capability: {_audio_capable}", flush=True)


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------

def _start_vllm_event_loop():
    """Start a dedicated asyncio event loop in a background thread for vLLM."""
    global _vllm_loop
    loop = asyncio.new_event_loop()
    _vllm_loop = loop
    loop.run_forever()


def _vllm_run(coro):
    """Submit a coroutine to the vLLM event loop and block until done."""
    if _vllm_loop is None:
        raise RuntimeError("vLLM event loop not started")
    fut = asyncio.run_coroutine_threadsafe(coro, _vllm_loop)
    return fut.result(timeout=300)


def _load_vllm_engine(config_path="config.yaml", model_path_override: str | None = None):
    """Load vLLM AsyncLLMEngine. Returns True on success, False on failure.

    model_path_override: when swap_model has already updated _config in-memory,
    pass the path directly to avoid re-reading stale disk config (mirrors _load_nemotron).
    """
    global _vllm_engine, _vllm_model_path, _vllm_enabled, _vllm_loop

    # vLLM uses IPC sockets for worker communication — must be on a real Linux FS.
    # Windows-mounted drives (e.g. /mnt/...) don't support Unix sockets. Force /tmp.
    import tempfile
    if not os.environ.get("TMPDIR", "").startswith("/tmp"):
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = "/tmp"

    cfg = _config if _config is not None else _load_config(config_path)
    model_path = model_path_override \
                 or (os.environ.get("MODEL_SOURCE") == "docker-hub" and os.environ.get("MODEL_ID")) \
                 or cfg["model"].get("path") or cfg["model"]["name"]
    inf_cfg = cfg.get("inference", {})

    gpu_util = inf_cfg.get("gpu_memory_utilization", 0.80)
    max_model_len = inf_cfg.get("max_model_len") or cfg["model"].get("max_context_length", 8192)
    dtype = cfg["model"].get("dtype", "bfloat16")
    # vLLM quantization — prefer bitsandbytes if config has 4bit
    quantize_cfg = cfg["model"].get("quantize", "none")
    if quantize_cfg in ("4bit", "4"):
        quantization = "bitsandbytes"
    else:
        quantization = None  # rely on dtype + PagedAttention

    # Speculative decoding config
    use_speculative = inf_cfg.get("speculative_decoding", False)
    drafter_path = inf_cfg.get("speculative_drafter", "")
    num_spec_tokens = inf_cfg.get("speculative_num_speculative_tokens", 5)

    print(f"[model_server] Loading vLLM engine: {model_path}", flush=True)
    print(f"[model_server]   gpu_util={gpu_util}, max_model_len={max_model_len}, dtype={dtype}", flush=True)
    if quantization:
        print(f"[model_server]   quantization={quantization}", flush=True)

    try:
        from vllm import AsyncLLMEngine, AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=gpu_util,
            max_model_len=int(max_model_len),
            dtype=dtype,
            trust_remote_code=True,
            # Multimodal support (Qwen3-VL)
            limit_mm_per_prompt={"image": 4, "video": 0, "audio": 0},
            enable_log_requests=False,
        )
        if quantization:
            engine_args.quantization = quantization

        # Speculative decoding — text-only drafter for text decode steps
        if use_speculative and drafter_path:
            try:
                engine_args.speculative_config = {
                    "model": drafter_path,
                    "num_speculative_tokens": int(num_spec_tokens),
                }
                print(f"[model_server] Speculative decoding: drafter={drafter_path}, tokens={num_spec_tokens}", flush=True)
            except Exception as e:
                print(f"[model_server] WARNING: speculative config failed ({e}) — disabling", flush=True)

        # Start background event loop before creating engine
        if _vllm_loop is None:
            t = threading.Thread(target=_start_vllm_event_loop, daemon=True)
            t.start()
            # Give it a moment to initialise
            import time; time.sleep(0.1)

        async def _create():
            return AsyncLLMEngine.from_engine_args(engine_args)

        _vllm_engine = _vllm_run(_create())
        _vllm_model_path = model_path
        _vllm_enabled = True

        _detect_capabilities(model_path, cfg)
        print("[model_server] vLLM engine ready.", flush=True)
        return True

    except Exception as e:
        print(f"[model_server] WARNING: vLLM engine load failed: {e}", flush=True)
        print("[model_server] Falling back to HF transformers backend.", flush=True)
        _vllm_enabled = False
        return False


async def _vllm_generate_async(prompt: str, sampling_params, request_id: str = None) -> str:
    """Run a single vLLM generation and return the full output text."""
    import uuid
    from vllm import SamplingParams

    if request_id is None:
        request_id = str(uuid.uuid4())

    full_output = ""
    async for output in _vllm_engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            full_output = output.outputs[0].text

    return full_output


async def _vllm_generate_multimodal_async(inputs: dict, sampling_params, request_id: str = None) -> str:
    """Run a vLLM multimodal generation (image + text) and return output text."""
    import uuid

    if request_id is None:
        request_id = str(uuid.uuid4())

    full_output = ""
    async for output in _vllm_engine.generate(inputs, sampling_params, request_id):
        if output.outputs:
            full_output = output.outputs[0].text

    return full_output


def _vllm_infer(prompt: str, max_new_tokens: int = 8192, temperature: float = 1.0,
                top_p: float = 0.95, top_k: int = 64) -> str:
    """Synchronous wrapper around vLLM generation."""
    from vllm import SamplingParams
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return _vllm_run(_vllm_generate_async(prompt, sp))


def _get_vllm_processor():
    """Get (or lazily load) the tokenizer/processor for prompt formatting."""
    global _processor
    if _processor is not None:
        return _processor
    if _vllm_model_path:
        try:
            from transformers import AutoProcessor
            _processor = AutoProcessor.from_pretrained(_vllm_model_path, trust_remote_code=True)
            print("[model_server] Processor loaded for prompt formatting.", flush=True)
        except Exception as e:
            print(f"[model_server] WARNING: processor load failed ({e})", flush=True)
    return _processor


# ---------------------------------------------------------------------------
# HF transformers fallback
# ---------------------------------------------------------------------------

def _load_hf_model(config_path="config.yaml", model_path_override: str | None = None):
    """Load model via HF transformers (fallback path).

    model_path_override: when swap_model has already updated _config in-memory,
    pass the path directly to avoid re-reading stale disk config (mirrors _load_nemotron).
    """
    global _model, _processor, _config, _drafter, _drafter_tokenizer
    if _model is not None:
        return

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer

    cfg = _config if _config is not None else _load_config(config_path)
    model_source = os.environ.get("MODEL_SOURCE", "local")
    if model_path_override:
        model_path = model_path_override
    elif model_source == "docker-hub":
        model_path = os.environ.get("MODEL_ID", cfg["model"]["name"])
    else:
        model_path = cfg["model"].get("path") or cfg["model"]["name"]

    device = cfg["model"].get("device", "cuda")
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    quantize = cfg["model"].get("quantize", "none")

    bnb_config = None
    if quantize in ("4bit", "4"):
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            print(f"[model_server] HF 4-bit quantisation enabled", flush=True)
        except Exception as e:
            print(f"[model_server] WARNING: 4-bit quant failed ({e}) — full precision", flush=True)
    elif quantize in ("8bit", "8"):
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            print(f"[model_server] WARNING: 8-bit quant failed ({e}) — full precision", flush=True)

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        min_required_mb = 4000 if bnb_config else 10000
        if free_mb < min_required_mb:
            msg = (f"[model_server] VRAM guard: only {free_mb}MB free, "
                   f"need ~{min_required_mb}MB. Refusing to load to prevent OOM.")
            print(msg, flush=True)
            raise RuntimeError(msg)
        print(f"[model_server] VRAM check passed: {free_mb}/{total_mb}MB free", flush=True)
        _device_map = "auto"
    else:
        _device_map = "cpu"

    load_kwargs = dict(dtype=dtype, device_map=_device_map)
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config

    max_ctx = cfg["model"].get("max_context_length")
    if max_ctx:
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(model_path)
        text_cfg = getattr(model_cfg, 'text_config', None)
        if text_cfg is not None and hasattr(text_cfg, 'max_position_embeddings'):
            text_cfg.max_position_embeddings = int(max_ctx)
            model_cfg.text_config = text_cfg
        elif hasattr(model_cfg, 'max_position_embeddings'):
            model_cfg.max_position_embeddings = int(max_ctx)
        load_kwargs["config"] = model_cfg

    print(f"[model_server] Loading HF model: {model_path} ({quantize or 'bfloat16'}) ...", flush=True)
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # AutoModelForImageTextToText doesn't expose .generate() in transformers ≥5.8 dev;
    # use the model-specific class via AutoConfig to ensure GenerationMixin is present.
    try:
        from transformers import AutoConfig as _AutoCfg
        _arch_cfg = _AutoCfg.from_pretrained(model_path, trust_remote_code=True)
        _architectures = getattr(_arch_cfg, 'architectures', []) or []
        if _architectures:
            import importlib as _il
            _cls_name = _architectures[0]  # e.g. 'Qwen3VLForConditionalGeneration'
            _mod = _il.import_module('transformers')
            _ModelCls = getattr(_mod, _cls_name, None)
            if _ModelCls is not None and hasattr(_ModelCls, 'generate'):
                print(f"[model_server] Using architecture class: {_cls_name}", flush=True)
                _model = _ModelCls.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
            else:
                raise AttributeError(f"{_cls_name} not found in transformers or lacks .generate")
        else:
            raise AttributeError("No architectures in config")
    except Exception as _arch_err:
        print(f"[model_server] Architecture lookup failed ({_arch_err}), trying AutoModelForImageTextToText", flush=True)
        _model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
    if not hasattr(_model, 'generate'):
        # Last resort: try AutoModelForCausalLM (covers some multimodal VL models)
        print("[model_server] WARNING: loaded model has no .generate() — retrying with AutoModelForCausalLM", flush=True)
        del _model
        import gc; gc.collect()
        _model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
    print("[model_server] HF Model loaded.", flush=True)

    _detect_capabilities(model_path, cfg)

    # Processor-based tool support refinement (Gemma parse_response)
    global _model_supports_tools
    try:
        _model_supports_tools = callable(getattr(_processor, 'parse_response', None))
        _processor.parse_response("")
        _model_supports_tools = True
    except AttributeError:
        _model_supports_tools = False
    except Exception:
        _model_supports_tools = True
    print(f"[model_server] Tool-calling support (refined): {_model_supports_tools}", flush=True)

    # Load MTP drafter
    drafter_path = cfg["model"].get("drafter_path")
    use_speculative = cfg["model"].get("speculative_decoding", False)
    if drafter_path and use_speculative:
        try:
            print(f"[model_server] Loading drafter: {drafter_path}", flush=True)
            _drafter = AutoModelForCausalLM.from_pretrained(drafter_path, dtype=dtype, device_map="auto")
            _drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
            print("[model_server] Drafter loaded.", flush=True)
        except Exception as e:
            print(f"[model_server] WARNING: drafter load failed ({e})", flush=True)
            _drafter = None
            _drafter_tokenizer = None

    # Wire primary slot into registry (additive — all existing globals remain set)
    global _slot_registry
    if _slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _spec = _slot_registry._specs.get("primary") or SlotSpec(
                name="primary",
                model_path=model_path,
                role="primary",
            )
            _state = SlotState(
                spec=_spec,
                model=_model,
                processor=_processor,
                is_nemotron=_is_nemotron,
                audio_capable=_audio_capable,
                supports_tools=_model_supports_tools,
            )
            _slot_registry._loaded["primary"] = _state
            print("[model_server] Primary slot wired into SlotRegistry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not wire primary slot: {_se}", flush=True)


# ---------------------------------------------------------------------------
# Nemotron-Labs-Diffusion load + inference helpers
# ---------------------------------------------------------------------------

def _is_nemotron_model(model_path: str) -> bool:
    """Return True if the model path/name is a Nemotron-Labs-Diffusion variant."""
    name = (model_path or "").lower()
    return "nemotron" in name or "nemotron-labs-diffusion" in name


def _load_nemotron(config_path="config.yaml", model_path_override: str | None = None):
    """Load nvidia/Nemotron-Labs-Diffusion via AutoModel + AutoTokenizer.

    Nemotron uses trust_remote_code=True, returns (out_ids, nfe) from its
    custom generate methods — incompatible with the standard HF generate path.
    model_path_override: when swap_model has already updated _config in-memory,
    pass the path directly to avoid re-reading stale disk config.
    """
    global _model, _processor, _config, _is_nemotron
    global _nemotron_mode, _nemotron_block_length, _nemotron_threshold
    global _model_supports_tools, _audio_capable

    if _model is not None:
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = _config if _config is not None else _load_config(config_path)
    model_path = model_path_override or cfg["model"].get("path") or cfg["model"].get("name")
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    # Nemotron generation config
    _nemotron_mode = cfg["model"].get("generation_mode", "linear_spec")
    _nemotron_block_length = int(cfg["model"].get("block_length", 32))
    _nemotron_threshold = float(cfg["model"].get("threshold", 0.9))

    # VRAM guard — Nemotron 3B bf16 ~6GB, 4bit ~3GB
    quantize = cfg["model"].get("quantize", "none")
    bnb_config = None
    if quantize in ("4bit", "4"):
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            print("[model_server] Nemotron: 4-bit quantisation enabled", flush=True)
        except Exception as e:
            print(f"[model_server] Nemotron: 4-bit quant failed ({e}) — full precision", flush=True)

    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        min_required_mb = 3500 if bnb_config else 6500
        if free_mb < min_required_mb:
            msg = (f"[model_server] VRAM guard (Nemotron): only {free_mb}MB free, "
                   f"need ~{min_required_mb}MB. Refusing to load.")
            print(msg, flush=True)
            raise RuntimeError(msg)
        print(f"[model_server] Nemotron VRAM check passed: {free_mb}/{total_mb}MB free", flush=True)

    load_kwargs = {"trust_remote_code": True}
    if bnb_config:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = dtype

    # Apply max_context_length override from config.
    # Prefer model_context_lengths map (per-model key) over the shared model.max_context_length.
    # Nemotron native is 256k; we default to 64k from model_context_lengths.nemotron.
    _ctx_map = cfg.get("model_context_lengths", {})
    _model_key = (model_path or "").lower()
    max_ctx = None
    for _k, _v in _ctx_map.items():
        if _k.lower() in _model_key:
            max_ctx = int(_v)
            break
    if max_ctx is None:
        max_ctx = cfg["model"].get("max_context_length")
    if max_ctx:
        try:
            from transformers import AutoConfig as _AutoCfg
            _model_cfg = _AutoCfg.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(_model_cfg, 'max_position_embeddings'):
                _model_cfg.max_position_embeddings = int(max_ctx)
                load_kwargs["config"] = _model_cfg
                print(f"[model_server] Nemotron context capped to {max_ctx} tokens (native 262144)", flush=True)
        except Exception as _ctx_err:
            print(f"[model_server] Nemotron: context cap failed ({_ctx_err}) — using model default", flush=True)

    print(f"[model_server] Loading Nemotron: {model_path} (mode={_nemotron_mode}) ...", flush=True)
    # AutoTokenizer — Nemotron has no multimodal processor
    _processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    raw = AutoModel.from_pretrained(model_path, **load_kwargs)

    # If no CUDA placement already done via quantise, move to GPU
    if not bnb_config and torch.cuda.is_available():
        raw = raw.cuda()
    _model = raw

    # Optional LoRA drafter for linear_spec mode (linear_spec_lora subfolder in repo)
    adapter_path = cfg["model"].get("adapter_path")
    if _nemotron_mode == "linear_spec" and not adapter_path:
        # Try bundled linear_spec_lora subfolder
        import os as _os
        _lora_path = _os.path.join(model_path, "linear_spec_lora")
        if _os.path.isdir(_lora_path):
            adapter_path = _lora_path

    if adapter_path and _nemotron_mode == "linear_spec":
        try:
            from peft import PeftModel
            _peft = PeftModel.from_pretrained(_model, adapter_path).eval()
            _model = _peft.model  # unwrap to call linear_spec_generate directly
            print(f"[model_server] Nemotron: LoRA drafter loaded from {adapter_path}", flush=True)
        except Exception as e:
            print(f"[model_server] Nemotron: LoRA drafter load failed ({e}) — continuing without", flush=True)

    _is_nemotron = True
    # Nemotron has full tool-calling via its chat template (XML function_calls format)
    _model_supports_tools = True
    _audio_capable = False
    print(f"[model_server] Nemotron ready. mode={_nemotron_mode} block={_nemotron_block_length} threshold={_nemotron_threshold}", flush=True)


def _nemotron_infer(messages: list, max_new_tokens: int = 8192) -> str:
    """Run inference using the appropriate Nemotron generation mode.
    Returns the decoded output string.
    """
    import torch
    prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = _processor(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        prompt_ids = prompt_ids.cuda()

    eos_id = _processor.eos_token_id

    print(f"[DEBUG model_server] Nemotron messages: {messages}", flush=True)

    
    with torch.no_grad():
        if _nemotron_mode == "ar":
            out_ids, nfe = _model.ar_generate(prompt_ids, max_new_tokens=max_new_tokens)
        elif _nemotron_mode == "diffusion":
            out_ids, nfe = _model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                block_length=_nemotron_block_length,
                threshold=_nemotron_threshold,
                eos_token_id=eos_id,
            )
        else:  # linear_spec (default — fastest)
            out_ids, nfe = _model.linear_spec_generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                block_length=_nemotron_block_length,
                eos_token_id=eos_id,
            )

    new_ids = out_ids[:, prompt_ids.shape[1]:]
    text = _processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    print(f"[model_server] Nemotron NFE={nfe} mode={_nemotron_mode}", flush=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Unified load entry points
# ---------------------------------------------------------------------------

def _adapter_name(adapter_path: str) -> str:
    path = str(Path(adapter_path).expanduser())
    return f"adapter_{abs(hash(path))}"


def _is_peft_model(model_obj) -> bool:
    try:
        from peft import PeftModel
        return isinstance(model_obj, PeftModel)
    except Exception:
        return False


def _load_adapter(adapter_path: str) -> str:
    """Load a LoRA adapter onto the already-loaded HF model and return its adapter name."""
    global _model
    if _model is None:
        print("[adapter] WARNING: base model not loaded — cannot attach adapter", flush=True)
        raise RuntimeError("base model not loaded")
    if _vllm_enabled:
        raise RuntimeError("adapter-aware inference is not supported with vLLM backend")

    adapter_path = str(Path(adapter_path).expanduser())
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"adapter path not found: {adapter_path}")

    existing = _loaded_adapters.get(adapter_path)
    if existing:
        return existing

    try:
        from peft import PeftModel
        adapter_name = _adapter_name(adapter_path)
        if _is_peft_model(_model):
            _model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            _model = PeftModel.from_pretrained(_model, adapter_path, adapter_name=adapter_name)
        _loaded_adapters[adapter_path] = adapter_name
        print(f"[adapter] loaded LoRA adapter '{adapter_name}' from {adapter_path}", flush=True)
        return adapter_name
    except Exception as e:
        print(f"[adapter] ERROR loading adapter from {adapter_path}: {e}", flush=True)
        raise


@contextlib.contextmanager
def _use_adapter(adapter_path: str | None):
    """Temporarily activate an adapter for generation on the shared base model."""
    global _current_adapter_name
    _ensure_model()

    if adapter_path and _vllm_enabled:
        raise RuntimeError("adapter-aware inference is not supported with vLLM backend")

    with _infer_lock:
        if not adapter_path:
            if _model is not None and _is_peft_model(_model):
                with _model.disable_adapter():
                    yield
            else:
                yield
            return

        adapter_name = _load_adapter(adapter_path)
        previous = _current_adapter_name
        if hasattr(_model, "set_adapter"):
            _model.set_adapter(adapter_name)
        _current_adapter_name = adapter_name
        try:
            yield
        finally:
            if previous and hasattr(_model, "set_adapter"):
                _model.set_adapter(previous)
                _current_adapter_name = previous
            else:
                _current_adapter_name = None

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


def _load_model(config_path="config.yaml"):
    """Load main model — tries vLLM first, falls back to HF transformers.
    Routes Nemotron-Labs-Diffusion models to the dedicated _load_nemotron path.
    """
    global _config, _slot_registry
    with _load_lock:
        if _vllm_enabled or _model is not None:
            return  # already loaded

        cfg = _load_config(config_path)

        # Build SlotRegistry from config if model_slots section present
        if _SLOTS_AVAILABLE and "model_slots" in cfg and cfg["model_slots"]:
            try:
                from model_slots import SlotRegistry as _SR, SlotSpec as _SS  # type: ignore
                _slot_registry = _SR(vram_threshold_mb=cfg.get("vram_threshold_mb", 3000))
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
                    _slot_registry.register(spec)
                # Wire a slot loader so load_slot() RPC can hot-load any registered spec
                _slot_registry.set_loader(_make_slot_loader())
                print(f"[model_server] SlotRegistry built with {len(cfg['model_slots'])} slot(s)", flush=True)
            except Exception as _sre:
                print(f"[model_server] WARNING: failed to build SlotRegistry: {_sre}", flush=True)
                _slot_registry = None
        model_path = cfg["model"].get("path") or cfg["model"].get("name", "")

        # Nemotron uses its own loader — completely different API from HF standard
        if _is_nemotron_model(model_path):
            print("[model_server] Nemotron-Labs-Diffusion detected — using Nemotron loader", flush=True)
            _load_nemotron(config_path, model_path_override=model_path)
            return

        backend = cfg.get("inference", {}).get("backend", "vllm")

        if backend == "vllm":
            success = _load_vllm_engine(config_path)
            if not success:
                # Fallback to HF
                _load_hf_model(config_path)
        else:
            # Explicit transformers backend
            print("[model_server] inference.backend=transformers — skipping vLLM", flush=True)
            _load_hf_model(config_path)


def _ensure_model():
    if not _vllm_enabled and _model is None:
        _load_model(_lazy_config_path)
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
    global _mm_model, _mm_processor
    if _mm_model is not None:
        print("[model_server] Multimodal slot already loaded", flush=True)
        return
    print("[model_server] Multimodal slot not yet loaded, initializing...", flush=True)
    # Ensure config is loaded (the multimodal slot path comes from stt_model).
    # Do NOT call _ensure_model() here — that loads the main (text-only) model,
    # which is unnecessary for vision/STT and can fail if the main model path
    # is a cloud-only config (e.g. ${KERNEL_EVO_HF_HUB} unexpanded in a spawned
    # on-demand server). The multimodal slot is loaded independently below.
    if _config is None:
        _load_config(_lazy_config_path)
    stt_cfg = (_config or {}).get("stt_model", {})
    stt_path = stt_cfg.get("path") or stt_cfg.get("name")
    if not stt_path:
        err = "stt_model not configured in config.yaml"
        print(f"[model_server] ERROR: {err}", flush=True)
        raise RuntimeError(err)

    # ── Same-model optimisation ───────────────────────────────────────────────
    # Resolve both paths and compare. If they point to the same weights, alias
    # the STT handle to the already-loaded main model — zero extra VRAM.
    import os
    main_path = ((_config or {}).get("model") or {}).get("path") or ""
    stt_resolved  = os.path.realpath(os.path.expanduser(stt_path))
    main_resolved = os.path.realpath(os.path.expanduser(main_path))
    if stt_resolved == main_resolved and _model is not None:
        print(
            f"[model_server] STT path == main model path — reusing loaded model (saves ~9GB VRAM)",
            flush=True,
        )
        _mm_model     = _model
        _mm_processor = _processor
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
        _mm_processor = AutoProcessor.from_pretrained(stt_path)
        print(f"[model_server] STT processor loaded", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading STT processor: {e}", flush=True)
        raise
    try:
        # AutoModel loads the base class (e.g. Gemma4Model) which lacks .generate().
        # Use AutoModelForCausalLM so GenerationMixin is always present.
        from transformers import AutoModelForCausalLM as _AutoCLM
        _mm_model = _AutoCLM.from_pretrained(stt_path, **load_kwargs, trust_remote_code=True)
        if not hasattr(_mm_model, 'generate'):
            # Fallback: architecture-specific class via AutoConfig
            from transformers import AutoConfig as _AConf
            _arch = getattr(_AConf.from_pretrained(stt_path, trust_remote_code=True), 'architectures', [])
            if _arch:
                import importlib as _il
                _Cls = getattr(_il.import_module('transformers'), _arch[0], None)
                if _Cls and hasattr(_Cls, 'generate'):
                    del _mm_model
                    _mm_model = _Cls.from_pretrained(stt_path, **load_kwargs, trust_remote_code=True)
        print(f"[model_server] Multimodal slot loaded to {next(_mm_model.parameters()).device} (class: {_mm_model.__class__.__name__})", flush=True)
        if not hasattr(_mm_model, 'generate'):
            raise AttributeError(f"Multimodal slot class {_mm_model.__class__.__name__} has no .generate() — STT unavailable")
    except Exception as e:
        print(f"[model_server] ERROR loading multimodal slot: {e}", flush=True)
        raise

    # Wire audio slot into registry (additive — existing _mm_model/_mm_processor remain set)
    if _slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _audio_spec = _slot_registry._specs.get("audio") or SlotSpec(
                name="audio",
                model_path=stt_path,
                role="audio",
            )
            _audio_state = SlotState(
                spec=_audio_spec,
                model=_mm_model,
                processor=_mm_processor,
                is_nemotron=False,
                audio_capable=True,
                supports_tools=False,
            )
            _slot_registry._loaded["audio"] = _audio_state
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
    global _tool_calling_model, _tool_calling_processor, _tool_calling_slot_loaded

    if _tool_calling_slot_loaded:
        return

    _ensure_model()  # config must be loaded first

    # Check if tool_calling slot is configured
    if _slot_registry is None:
        print("[model_server] No SlotRegistry — tool_calling slot unavailable", flush=True)
        return

    tc_spec = _slot_registry._specs.get("tool_calling")
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
        _tool_calling_processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"[model_server] Tool-calling tokenizer loaded (vocab={_tool_calling_processor.vocab_size})", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading tool-calling tokenizer: {e}", flush=True)
        return

    try:
        load_kwargs = {"device_map": device, "torch_dtype": dtype, "trust_remote_code": True}
        _tool_calling_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if not hasattr(_tool_calling_model, 'generate'):
            raise AttributeError(f"Tool-calling model class {_tool_calling_model.__class__.__name__} has no .generate()")
        print(f"[model_server] Tool-calling slot loaded to {next(_tool_calling_model.parameters()).device} (class: {_tool_calling_model.__class__.__name__})", flush=True)
    except Exception as e:
        print(f"[model_server] ERROR loading tool-calling model: {e}", flush=True)
        _tool_calling_model = None
        _tool_calling_processor = None
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
            _tool_calling_model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            print(f"[model_server] Tool-calling slot loaded on retry to {next(_tool_calling_model.parameters()).device}", flush=True)
        except Exception as e2:
            print(f"[model_server] ERROR tool-calling slot retry failed: {e2}", flush=True)
            _tool_calling_model = None
            _tool_calling_processor = None
            return

    _tool_calling_slot_loaded = True

    # Wire into slot registry for status reporting
    if _slot_registry is not None:
        try:
            from model_slots import SlotState
            _tc_state = SlotState(
                spec=tc_spec,
                model=_tool_calling_model,
                processor=_tool_calling_processor,
                is_nemotron=False,
                audio_capable=False,
                supports_tools=True,
            )
            _slot_registry._loaded["tool_calling"] = _tc_state
            print("[model_server] Tool-calling slot wired into SlotRegistry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not wire tool_calling slot: {_se}", flush=True)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _target_device():
    """Return the device of the model's first parameter (HF fallback path)."""
    import torch
    if _model is not None:
        try:
            return next(_model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Prompt helpers for vLLM path
# ---------------------------------------------------------------------------

def _build_chat_prompt(messages: list, add_generation_prompt: bool = True,
                        enable_thinking: bool = False) -> str:
    """Format messages into a prompt string using the loaded processor's chat template."""
    proc = _get_vllm_processor()
    if proc is None:
        # Minimal fallback
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"<|{role}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    try:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        # Only pass enable_thinking if processor supports it (Gemma 4 / Qwen3)
        try:
            return proc.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            return proc.apply_chat_template(messages, **kwargs)
    except Exception as e:
        print(f"[model_server] WARNING: apply_chat_template failed ({e}) — using fallback", flush=True)
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"<|{role}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# RPC handlers
# ---------------------------------------------------------------------------

def _handle_infer(params: dict) -> dict:
    """Standard chat inference."""
    _ensure_model()

    messages = params.get("messages", [])
    max_new_tokens = params.get("max_new_tokens", 8192)
    adapter_path = params.get("adapter_path")

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return {"error": f"'messages' must be a list, got {type(messages).__name__}"}

    # ── System prompt guard: ensure at least a default identity exists.
    # Ported from _handle_infer_plain so callers relying on the fallback
    # identity (e.g. thought_engine reflection calls without their own
    # system message) keep working once "infer" routes here instead of
    # _handle_infer_with_tools.
    # skip_identity_guard: callers with a fully self-contained utility prompt
    # (e.g. the fact extractor's own "You are a fact extractor..." role) opt
    # out to avoid two conflicting "you are X" instructions in one request.
    # See .specs/fixes/IDENTITY-INJECTION-UTILITY-PROMPT-CONFLICT-2026-08-06.md
    _has_system = any(m.get("role") == "system" for m in messages)
    if not _has_system and not params.get("skip_identity_guard", False):
        messages = [{
            "role": "system",
            "content": (
                "You are Kernel-Evo, the core AI agent of the Kernel-Evolving project. "
                "You have access to tools, skills, routines, a file system, and the internet. "
                "Answer helpfully, concisely, and with your full personality. "
                "When asked about your capabilities, describe your tools and skills enthusiastically."
            )
        }] + messages
        print("[model_server] _handle_infer: injected default system prompt (was missing)", flush=True)

    # ── Nemotron path — custom generate API, no processor.parse_response
    if _is_nemotron:
        try:
            formatted = [
                {"role": m["role"], "content": str(m.get("content", ""))}
                for m in messages
            ]
            result = _nemotron_infer(formatted, max_new_tokens=max_new_tokens)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    formatted = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        formatted.append({"role": role, "content": content})

    try:
        with _use_adapter(adapter_path):
            if _vllm_enabled:
                try:
                    prompt = _build_chat_prompt(formatted, enable_thinking=False)
                    result = _vllm_infer(prompt, max_new_tokens=max_new_tokens)
                    return {"result": result.strip()}
                except Exception as e:
                    print(f"[model_server] vLLM infer error: {e} — falling back to HF", flush=True)
                    # Fall through to HF path if model available
                    if _model is None:
                        return {"error": f"vLLM failed and HF model not loaded: {e}"}

            # HF transformers path
            import torch
            inputs = _processor.apply_chat_template(
                formatted,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(_target_device())
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                _gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                )
                if _drafter is not None:
                    _gen_kwargs["assistant_model"] = _drafter
                out = _model.generate(**inputs, **_gen_kwargs)
            response_raw = _processor.decode(out[0][input_len:], skip_special_tokens=False)
            parsed = _processor.parse_response(response_raw)
            result = parsed.get("content", "").strip()
            return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def _handle_infer_plain(params: dict) -> dict:
    """Plain text inference for models without native tool-calling (Qwen, etc.)."""
    _ensure_model()

    messages = params["messages"]
    adapter_path = params.get("adapter_path")
    # No cap here — agent.py already manages history budget correctly.
    # model_server trusts what it receives. Hard safety ceiling only for
    # runaway cases (>500 messages from a bug), always preserving system msg.
    if len(messages) > 500:
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        messages = system_msgs + non_system[-499:]

    # ── System prompt guard: ensure at least a default identity exists
    # Fallback paths (when two-stage is unavailable) may arrive without a
    # system message, causing Nemotron to answer with raw pre-training.
    _has_system = any(m.get("role") == "system" for m in messages)
    if not _has_system:
        messages.insert(0, {
            "role": "system",
            "content": (
                "You are Kernel-Evo, the core AI agent of the Kernel-Evolving project. "
                "You have access to tools, skills, routines, a file system, and the internet. "
                "Answer helpfully, concisely, and with your full personality. "
                "When asked about your capabilities, describe your tools and skills enthusiastically."
            )
        })
        print("[model_server] _handle_infer_plain: injected default system prompt (was missing)", flush=True)

    # ── Nemotron path
    if _is_nemotron:
        try:
            chat = [
                {"role": m["role"], "content": str(m.get("content", ""))}
                for m in messages
                if m["role"] in ("system", "user", "assistant")
            ]
            result = _nemotron_infer(chat, max_new_tokens=8192)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    chat = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role in ("system", "user", "assistant"):
            chat.append({"role": role, "content": content})

    try:
        with _use_adapter(adapter_path):
            if _vllm_enabled:
                try:
                    prompt = _build_chat_prompt(chat, enable_thinking=False)
                    result = _vllm_infer(prompt, max_new_tokens=8192, temperature=0.7, top_p=0.9, top_k=50)
                    return {"result": result.strip()}
                except Exception as e:
                    print(f"[model_server] vLLM infer_plain error: {e} — falling back to HF", flush=True)
                    if _model is None:
                        return {"error": f"vLLM failed and HF model not loaded: {e}"}

            # HF path
            import torch
            try:
                text = _processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = _processor(text=text, return_tensors="pt").to(_target_device())
            except Exception as e:
                last = next((m["content"] for m in reversed(chat) if m["role"] == "user"), "")
                inputs = _processor(text=last, return_tensors="pt").to(_target_device())

            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = _model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                )
            result = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
            return {"result": result}
    except Exception as e:
        return {"error": str(e)}



# ── Two-stage tool-calling pipeline (ADR-011) ─────────────────────────────
# Qwen3.5-0.8B handles tool loop; Nemotron handles conversation synthesis.

def _run_two_stage_if_available(params: dict, send_line) -> dict | None:
    """Try the two-stage pipeline. Returns None if tool_calling slot unavailable."""
    global _tool_calling_slot_loaded, _tool_calling_model, _tool_calling_processor

    if not _tool_calling_slot_loaded:
        _ensure_tool_calling_slot()
    if not _tool_calling_slot_loaded:
        return None  # fall through to single-model path

    import json
    import torch
    from core.tools import execute_tool_with_meta

    messages = params["messages"]
    raw_tools = params.get("tools", [])

    # Two-stage only makes sense when there are actual tools to call.
    # Plain reasoning/generation requests should go straight to Nemotron.
    if not raw_tools:
        return None

    workspace = os.path.expanduser(params.get("workspace", "~/.kernel-evolving/workspace"))
    max_steps = params.get("max_steps", 15)
    # Passed explicitly (not read from core.tools._current_chat_id) since
    # model_server runs in a separate, multi-threaded process from the
    # caller that knows the real chat_id (R5/T4).
    chat_id = params.get("chat_id", "")
    # T7: caller-supplied max_new_tokens overrides the synthesis defaults
    # (2048/4096) below when given. None means "use the synthesis default".
    max_new_tokens = params.get("max_new_tokens")

    # Build OpenAI-format tools for Qwen template
    def _to_openai_tool(t):
        if "type" in t and t["type"] == "function":
            return t
        return {"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }}
    tools_openai = [_to_openai_tool(t) for t in raw_tools]

    # Import helpers
    try:
        from core.inference._two_stage_helpers import parse_qwen_tool_calls, build_nemotron_synthesis_prompt, build_qwen_system_prompt
    except ImportError:
        print("[two_stage] WARNING: _two_stage_helpers not found — falling back", flush=True)
        return None

    # ── Build a minimal context for Qwen ──────────────────────────────────
    # Qwen3-0.6B has ~8k optimal context. The full Nemotron system prompt (~4k+
    # tokens) fills it entirely, causing context overflow, #INSTRUCTIONS corruption,
    # and broken tool calls. Instead, strip it and give Qwen only what it needs:
    #   (1) Minimal microplanner system instruction (template handles tool format with `tools=`)
    #   (2) Last 2 user/assistant turns for context
    #   (3) The current user query
    #
    # The full system prompt is preserved for Stage 2 Nemotron synthesis.
    #
    # Flatten messages first to extract history
    flat_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        flat_messages.append({"role": role, "content": content})

    # Extract the system prompt (contains first-contact instructions, persona, rules)
    _system_prompt = ""
    for m in flat_messages:
        if m["role"] == "system":
            _system_prompt = m["content"]
            break

    # Extract the current user query (last user message)
    _original_query = next(
        (m["content"] for m in reversed(flat_messages) if m["role"] == "user"), ""
    )

    # Build minimal context: last 4 back-and-forth turns + current query
    non_sys = [m for m in flat_messages if m["role"] != "system"]
    qwen_history = non_sys[-8:]  # at most last 4 back-and-forth turns
    # Extract conversation history for Nemotron synthesis (exclude current query)
    # Only pass user/assistant roles — tool role breaks Nemotron's chat template
    # ADR-022: cap to recent turns only to prevent history poisoning
    _MAX_HISTORY_TURNS = 6
    _history_context = [
        m for m in non_sys[:-1]
        if m["role"] in ("user", "assistant")
    ][-_MAX_HISTORY_TURNS:]

    # Replace the full system prompt with minimal tool-calling instruction
    current_messages = [build_qwen_system_prompt()] + qwen_history

    _last_tool_sig = ""
    _repeat_count = 0
    _REPEAT_LIMIT = 3
    _tool_call_tally = {}
    _empty_reprompt_count = 0
    _EMPTY_REPROMPT_LIMIT = 3
    _last_tool_result = ""
    all_tool_results = []

    # ── Stage 1: Qwen tool loop ────────────────────────────────────────────
    print("[two_stage] Stage 1: Qwen tool-calling loop", flush=True)

    for step in range(max_steps):
        try:
            inputs = _tool_calling_processor.apply_chat_template(
                current_messages,
                tools=tools_openai,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,  # Non-thinking mode per model card — thinking loops on 0.8B
            )
            # Align tensors to tool-calling model device AND dtype.
            # Without dtype alignment, float32 token tensors hit a BFloat16 weight
            # mismatch inside model.generate(), producing:
            #   "expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float"
            _tc_param = next(_tool_calling_model.parameters())
            _tc_device = _tc_param.device
            _tc_dtype = _tc_param.dtype
            inputs = {
                k: (
                    v.to(device=_tc_device, dtype=_tc_dtype)
                    if hasattr(v, "dtype") and hasattr(v, "to") and torch.is_floating_point(v)
                    else v.to(device=_tc_device) if hasattr(v, "to") else v
                )
                for k, v in inputs.items()
            }

            input_len = inputs["input_ids"].shape[-1]
            _mark_activity_start()
            try:
                with torch.no_grad():
                    out = _tool_calling_model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        # Tool-decision profile: tighter than the model card's general
                        # non-thinking chat preset (temp=1.0/top_p=1.0). At 0.8B scale that
                        # preset is too noisy for reliable multi-token XML tag emission
                        # (<tool_call>/<function=.../<parameter=...). See
                        # .specs/fixes/QWEN-TOOL-CALLING-SAMPLING-2026-08-06.md
                        temperature=0.4,
                        top_p=0.3,
                        top_k=20,
                        repetition_penalty=1.05,
                        pad_token_id=_tool_calling_processor.eos_token_id,
                    )
            finally:
                _mark_activity_end()

            response_raw = _tool_calling_processor.decode(out[0][input_len:], skip_special_tokens=False)
            response_clean = _tool_calling_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
            token_count = out[0].shape[-1] - input_len
            if _DEBUG_TWO_STAGE:
                print(f"[DEBUG two_stage] step {step}: {token_count} tokens", flush=True)
                print(f"[DEBUG two_stage] step {step}: raw response: {response_raw[:200]}", flush=True)
                print(f"[DEBUG two_stage] step {step}: clean response: {response_clean[:200]}", flush=True)
        except Exception as e:
            print(f"[ERROR two_stage] generate error at step {step}: {e}", flush=True)
            print("[ERROR two_stage] full traceback:\n" + traceback.format_exc(), flush=True)
            return None  # fall through

        tool_calls = parse_qwen_tool_calls(response_raw)

        if not tool_calls:
            final_text = response_clean
            if not final_text.strip():
                _empty_reprompt_count += 1
                if _empty_reprompt_count >= _EMPTY_REPROMPT_LIMIT:
                    print("[ERROR two_stage] empty limit — falling through", flush=True)
                    return None
                current_messages.append({"role": "user", "content": "Continue. Use a tool or provide your final answer."})
                continue
            print(f"[two_stage] Qwen final answer after {step} step(s)", flush=True)
            # Stage 2: Nemotron synthesis
            return _nemotron_synthesize_answer(
                _original_query, final_text, all_tool_results, send_line, _system_prompt, _history_context, max_new_tokens
            )

        # Execute tool calls
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            tool_args = fn.get("arguments", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            tool_args = _normalize_tool_args(tool_name, tool_args)

            print(f"[two_stage] step {step+1}: {tool_name}({json.dumps(tool_args)[:80]})", flush=True)

            # Repetition guard
            _sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if _sig == _last_tool_sig:
                _repeat_count += 1
                if _repeat_count >= _REPEAT_LIMIT:
                    return _nemotron_synthesize_answer(
                        _original_query, _last_tool_result, all_tool_results, send_line, _system_prompt, _history_context, max_new_tokens
                    )
            else:
                _last_tool_sig = _sig
                _repeat_count = 1
            _tool_call_tally[_sig] = _tool_call_tally.get(_sig, 0) + 1
            if _tool_call_tally[_sig] >= _REPEAT_LIMIT:
                return _nemotron_synthesize_answer(
                    _original_query, _last_tool_result, all_tool_results, send_line, _system_prompt, _history_context, max_new_tokens
                )

            _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace, chat_id=chat_id)
            result_str = _meta.get("result", "")
            _last_tool_result = result_str
            _ok = bool(_meta.get("ok", False))

            print(f"[two_stage] result: {result_str[:100]}", flush=True)

            # Stream step progress
            send_line(json.dumps({
                "type": "step", "step": step + 1,
                "tool": tool_name, "args": tool_args, "result": result_str,
                "ok": _ok, "status": str(_meta.get("status", "unknown")),
                "duration_ms": _meta.get("duration_ms", 0),
            }))

            all_tool_results.append({"name": tool_name, "args": tool_args, "result": result_str})

        # Preserve the assistant's own tool-call turn in history before the tool
        # results. response_clean already contains the model's native
        # <tool_call><function=...><parameter=...> XML verbatim (those tags are
        # not "special" tokens, so skip_special_tokens=True keeps them) — without
        # this turn, subsequent steps see orphaned tool results with no record of
        # what was called or why, breaking the assistant->tool template sequence.
        current_messages.append({"role": "assistant", "content": response_clean})

        # Feed results back to Qwen as tool role messages
        for tr in all_tool_results[-len(tool_calls):]:
            current_messages.append({"role": "tool", "content": tr["result"]})

    # Max steps reached — synthesize with what we have
    return _nemotron_synthesize_answer(
        _original_query, "", all_tool_results, send_line, _system_prompt, _history_context, max_new_tokens
    )


def _nemotron_synthesize_answer(original_query, qwen_answer, tool_results, send_line, system_prompt=None, history_context=None, max_new_tokens=None):
    """Stage 2: Call Nemotron for conversation synthesis. No tools in prompt.
    
    Uses the original full system prompt (including first-contact/personality instructions)
    when available, rather than a stripped identity string. This ensures Nemotron follows
    the proper Kernel-Evo persona including 🐬 first-contact greetings.
    """
    import json

    try:
        from core.inference._two_stage_helpers import build_nemotron_synthesis_prompt
    except ImportError:
        return {"type": "result", "result": qwen_answer or "(synthesis helper missing)"}

    prompt = build_nemotron_synthesis_prompt(original_query, qwen_answer, tool_results)

    if not tool_results and not qwen_answer:
        return {"type": "result", "result": "I could not complete that request."}

    # Always route through Nemotron for conversation synthesis in Kernel-Evo identity.
    # Use the FULL original system prompt (including first-contact 🐬 instructions)
    # instead of a bare identity string that ignores personality rules.
    if not tool_results:
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        else:
            chat.append({"role": "system", "content": (
                "You are Kernel-Evo, a helpful AI assistant. "
                "Answer the user's question concisely and naturally."
            )})
        # Inject conversation history so Nemotron has context from prior turns
        # ADR-022: cap to prevent history poisoning in synthesis stage
        _MAX_SYNTHESIS_HISTORY = 6
        if history_context:
            chat.extend(history_context[-_MAX_SYNTHESIS_HISTORY:])
            print(f"[two_stage] Stage 2: {min(len(history_context), _MAX_SYNTHESIS_HISTORY)} history turns injected (cap={_MAX_SYNTHESIS_HISTORY})", flush=True)
        chat.append({"role": "user", "content": original_query})
        print("[two_stage] Stage 2: Nemotron synthesis (no tools — with full system prompt)", flush=True)
        _mnt = max_new_tokens or 2048
        try:
            if _is_nemotron:
                result = _nemotron_infer(chat, max_new_tokens=_mnt)
            elif _vllm_enabled:
                p = _build_chat_prompt(chat, enable_thinking=False)
                result = _vllm_infer(p, max_new_tokens=_mnt)
            else:
                import torch
                inputs = _processor.apply_chat_template(
                    chat, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=False,
                ).to(_target_device())
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = _model.generate(**inputs, max_new_tokens=_mnt, do_sample=True, temperature=0.7)
                result = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
            return {"type": "result", "result": result}
        except Exception as e:
            print(f"[two_stage] Nemotron no-tools synthesis error: {e}", flush=True)
            return {"type": "result", "result": qwen_answer}

    print("[two_stage] Stage 2: Nemotron synthesis (no tools)", flush=True)

    # Call Nemotron via existing infrastructure with full system prompt
    try:
        chat = []
        if system_prompt:
            chat.append({"role": "system", "content": system_prompt})
        else:
            # T12: fallback identity so this call always has exactly one
            # identity statement — not zero, not duplicated (see the
            # now-removed identity line in build_nemotron_synthesis_prompt).
            chat.append({"role": "system", "content": (
                "You are Kernel-Evo, a helpful AI assistant. "
                "Answer the user's question concisely and naturally."
            )})
        # ADR-022: cap to prevent history poisoning in synthesis stage
        _MAX_SYNTHESIS_HISTORY = 6
        if history_context:
            chat.extend(history_context[-_MAX_SYNTHESIS_HISTORY:])
            print(f"[two_stage] Stage 2 (tooled): {min(len(history_context), _MAX_SYNTHESIS_HISTORY)} history turns injected (cap={_MAX_SYNTHESIS_HISTORY})", flush=True)
        chat.append({"role": "user", "content": prompt})
        _mnt = max_new_tokens or 4096
        if _is_nemotron:
            result = _nemotron_infer(chat, max_new_tokens=_mnt)
        elif _vllm_enabled:
            p = _build_chat_prompt(chat, enable_thinking=False)
            result = _vllm_infer(p, max_new_tokens=_mnt)
        else:
            import torch
            inputs = _processor.apply_chat_template(
                chat, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
                enable_thinking=False,
            ).to(_target_device())
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=_mnt, do_sample=True, temperature=0.7)
            result = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

        return {"type": "result", "result": result}
    except Exception as e:
        print(f"[two_stage] Nemotron synthesis error: {e}", flush=True)
        return {"type": "result", "result": qwen_answer or f"(Synthesis error: {e})"}


def _handle_infer_with_tools(params: dict, send_line) -> dict:
    """
    Agentic tool-calling loop.

    Two-stage pipeline (preferred):
      Stage 1: tool_calling slot (Qwen3-0.6B) handles tool loop with native tool_call XML
      Stage 2: primary slot (Nemotron) receives tool results as plain text, synthesizes answer

    Fallback: single-model loop (Nemotron/Gemma handles everything).
    """
    _ensure_model()
    import json

    # ── Try two-stage pipeline first ───────────────────────────────────────
    # MS5: only detour through the Qwen two-stage pipeline when the primary
    # model doesn't have reliable native tool-calling support of its own.
    # Nemotron's _model_supports_tools=True is a hardcoded capability flag
    # (its chat template understands tool-call XML), but the single-model
    # native tool loop below it in this function has never been this
    # pipeline's tested/working path for Nemotron — two-stage
    # (Qwen -> Nemotron synthesis) is, and stays the route for it. Only
    # non-Nemotron models with detected native tool support (e.g. Gemma 4
    # after a swap, per _TOOL_CAPABLE_PREFIXES) skip the two-stage detour.
    if _is_nemotron or not _model_supports_tools:
        _two_stage_result = _run_two_stage_if_available(params, send_line)
        if _two_stage_result is not None:
            return _two_stage_result

    if not _model_supports_tools:
        return _handle_infer_plain(params)

    messages = params["messages"]
    raw_tools = params.get("tools", [])
    workspace = os.path.expanduser(params.get("workspace", "~/.kernel-evolving/workspace"))
    max_steps = params.get("max_steps", 15)
    enable_thinking = params.get("enable_thinking", False)
    adapter_path = params.get("adapter_path")
    # Passed explicitly to execute_tool_with_meta below — see R5/T4.
    chat_id = params.get("chat_id", "")
    # T7: honor caller-supplied max_new_tokens per generation step instead of
    # always hardcoding 8192.
    max_new_tokens = params.get("max_new_tokens", 8192)
    # T11: original query, used to synthesize a conversational reply from a
    # raw tool result when the repetition/tally guards trip below, instead
    # of surfacing the tool's raw output verbatim.
    _original_query = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )

    def _synthesize_from_tool_result(query: str, tool_result: str) -> str:
        """Turn a raw tool result into a short conversational answer (R7)."""
        try:
            return _nemotron_infer([
                {"role": "system", "content": "You are Kernel-Evo, a helpful AI assistant. Answer concisely and naturally."},
                {"role": "user", "content": (
                    f'The user asked: "{query}"\n\n'
                    f"Here is the tool result:\n{str(tool_result)[:2000]}\n\n"
                    "Answer the user's question based on this result."
                )},
            ], max_new_tokens=512)
        except Exception as _synth_err:
            print(f"[tool_loop] synthesis from tool result failed, falling back to raw: {_synth_err}", flush=True)
            return tool_result

    def _to_openai_tool(t: dict) -> dict:
        if "type" in t and t["type"] == "function":
            return t
        return {"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }}
    tools_openai = [_to_openai_tool(t) for t in raw_tools]
    available_tool_names = {
        (t.get("function", {}) or {}).get("name", "")
        for t in tools_openai
        if isinstance(t, dict)
    }

    def _looks_like_capability_refusal(text: str) -> bool:
        from core.refusal_patterns import looks_like_capability_refusal
        return looks_like_capability_refusal(text)

    current_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        current_messages.append({"role": role, "content": content})

    # No cap here — agent.py already manages history budget correctly.
    # Nemotron-Labs-Diffusion-3B has 262k context window; system prompt is always preserved.
    # Hard runaway guard only (same as _handle_infer_plain).
    sys_msgs = [m for m in current_messages if m["role"] == "system"]
    non_sys = [m for m in current_messages if m["role"] != "system"]
    if len(non_sys) > 499:
        non_sys = non_sys[-499:]
    # System prompt guard: if no system message survived (or none existed),
    # inject a default identity so Nemotron doesn't fall back to raw pre-training.
    if not sys_msgs:
        default_sys = {
            "role": "system",
            "content": (
                "You are Kernel-Evo, the core AI agent of the Kernel-Evolving project. "
                "You have access to tools, skills, routines, a file system, and the internet. "
                "Answer helpfully, concisely, and with your full personality. "
                "When asked about your capabilities, describe your tools and skills enthusiastically."
            )
        }
        sys_msgs = [default_sys]
        print("[tool_loop] injected default system prompt (was missing)", flush=True)
    current_messages = sys_msgs + non_sys

    from core.tools import execute_tool_with_meta
    _last_tool_sig: str = ""   # track repetitive tool calls
    _repeat_count: int = 0
    _REPEAT_LIMIT: int = 3      # bail if same tool+args called N times total
    _tool_call_tally: dict = {}  # total call counts per tool+args sig across all steps
    _empty_reprompt_count: int = 0
    _EMPTY_REPROMPT_LIMIT: int = 3  # max empty-content re-prompts before giving up
    _last_tool_result: str = ""  # latest successful tool output across all steps

    for step in range(max_steps):
        try:
            with _use_adapter(adapter_path):
                if _vllm_enabled:
                    # vLLM path — use processor to format with tools, then vLLM to generate
                    proc = _get_vllm_processor()
                    try:
                        kwargs = dict(tools=tools_openai, tokenize=False, add_generation_prompt=True)
                        try:
                            prompt = proc.apply_chat_template(
                                current_messages, enable_thinking=enable_thinking, **kwargs
                            )
                        except TypeError:
                            prompt = proc.apply_chat_template(current_messages, **kwargs)
                    except Exception as e:
                        print(f"[tool_loop/vllm] template error at step {step}: {e}", flush=True)
                        prompt = _build_chat_prompt(current_messages, enable_thinking=False)

                    try:
                        from vllm import SamplingParams
                        sp = SamplingParams(max_tokens=max_new_tokens, temperature=1.0, top_p=0.95, top_k=64)
                        response_raw = _vllm_run(_vllm_generate_async(prompt, sp))
                    except Exception as e:
                        print(f"[tool_loop/vllm] generate error at step {step}: {e} — aborting", flush=True)
                        return {"type": "result", "result": f"(vLLM error: {e})"}

                    # Parse response
                    try:
                        parsed = proc.parse_response(response_raw)
                    except (AttributeError, NotImplementedError):
                        parsed = {"content": response_raw.strip()}

                else:
                    # HF transformers path
                    import torch
                    try:
                        inputs = _processor.apply_chat_template(
                            current_messages,
                            tools=tools_openai,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        ).to(_target_device())
                    except Exception as e:
                        print(f"[tool_loop/hf] template error: {e}", flush=True)
                        text = _processor.apply_chat_template(
                            current_messages, tools=tools_openai, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                        )
                        inputs = _processor(text=text, return_tensors="pt").to(_target_device())

                    input_len = inputs["input_ids"].shape[-1]
                    _mark_activity_start()
                    try:
                        with torch.no_grad():
                            _gen_kwargs = dict(
                                max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=64,
                            )
                            if _drafter is not None:
                                _gen_kwargs["assistant_model"] = _drafter
                            try:
                                if _is_nemotron:
                                    # Nemotron returns (out_ids, nfe); use AR for tool loops
                                    out_ids, _nfe = _model.ar_generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
                                    out = out_ids
                                    print(f"[tool_loop/nemotron] AR NFE={_nfe}", flush=True)
                                else:
                                    out = _model.generate(**inputs, **_gen_kwargs)
                            except Exception as e:
                                import logging
                                logging.getLogger(__name__).error(f"[tool_loop/hf] generate error at step {step}: {e}")
                                print(f"[tool_loop/hf] generate error at step {step}: {e} — aborting", flush=True)
                                return {"type": "result", "result": f"(HF generate error: {e})"}
                    finally:
                        _mark_activity_end()
                    response_raw = _processor.decode(out[0][input_len:], skip_special_tokens=False)
                    try:
                        parsed = _processor.parse_response(response_raw)
                    except (AttributeError, NotImplementedError):
                        plain = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
                        parsed = {"content": plain}
        except Exception as e:
            return {"type": "result", "result": f"(adapter error: {e})"}

        tool_calls = parsed.get("tool_calls", [])

        # ── Fallback: detect <tool>name(args)</tool> text patterns the model sometimes
        # emits instead of native function calls. Parse them into tool_calls format.
        if not tool_calls:
            raw_content = parsed.get("content", "") or ""
            # Strip thinking block first
            if "<think>" in raw_content and "</think>" in raw_content:
                raw_content = raw_content[raw_content.index("</think>") + len("</think>"):].strip()
            import re as _re

            # Nemotron XML: <function_calls><invoke>...</invoke></function_calls>
            if _is_nemotron and "<function_calls>" in raw_content:
                invokes = _re.findall(r'<invoke>(.*?)</invoke>', raw_content, _re.DOTALL)
                if invokes:
                    _nemo_calls = []
                    for invoke_block in invokes:
                        tn_m = _re.search(r'<tool_name>(.*?)</tool_name>', invoke_block, _re.DOTALL)
                        if not tn_m:
                            continue
                        tool_name = tn_m.group(1).strip()
                        ptags = [(m.group(1), m.group(2)) for m in _re.finditer(r"<(\w+)>(.*?)</\1>", invoke_block, _re.DOTALL)]
                        args = {k: v.strip() for k, v in ptags if k != 'tool_name'}
                        _nemo_calls.append({"function": {"name": tool_name, "arguments": args}})
                    if _nemo_calls:
                        print(f"[tool_loop] Nemotron XML: {len(_nemo_calls)} call(s)", flush=True)
                        tool_calls = _nemo_calls

            # Nemotron alt XML: <function=name> + <parameter=key>value</parameter>
            if not tool_calls:
                fn_blocks = _re.findall(r'<function=([A-Za-z_][\w-]*)>(.*?)(?:</function>|(?=<function=)|$)', raw_content, _re.DOTALL)
                if fn_blocks:
                    _fn_calls = []
                    for tool_name, fn_body in fn_blocks:
                        params = _re.findall(r'<parameter=([A-Za-z_][\w-]*)>(.*?)</parameter>', fn_body, _re.DOTALL)
                        args = {k: v.strip() for k, v in params}
                        args = _normalize_tool_args(tool_name.strip(), args)
                        _fn_calls.append({"function": {"name": tool_name.strip(), "arguments": args}})
                    if _fn_calls:
                        print(f"[tool_loop] function= fallback: {len(_fn_calls)} call(s)", flush=True)
                        tool_calls = _fn_calls

            # Generic <tool>name(args)</tool> fallback
            if not tool_calls:
                tag_matches = _re.findall(r'<tool>(\w+)\((.*)\)</tool>', raw_content, _re.DOTALL)
                if tag_matches:
                    _tag_calls = []
                    for tool_name, args_str in tag_matches:
                        args_str = args_str.strip()
                        try:
                            args = json.loads(args_str) if args_str.startswith('{') else {}
                            if not args:
                                kv = _re.findall(r'(\w+)=["\']([^\"\']*)["\']?', args_str)
                                args = dict(kv)
                        except Exception:
                            args = {}
                        _tag_calls.append({"function": {"name": tool_name, "arguments": args}})
                    if _tag_calls:
                        print(f"[tool_loop] tag fallback: {len(_tag_calls)} call(s)", flush=True)
                        tool_calls = _tag_calls

        if not tool_calls:
            final_text = parsed.get("content", "")
            if "<think>" in final_text and "</think>" in final_text:
                final_text = final_text[final_text.index("</think>") + len("</think>"):].strip()
            # Nemotron sometimes emits stray closing think tags; always strip them.
            final_text = final_text.replace("</think>", "").replace("<think>", "").strip()

            if (
                _looks_like_capability_refusal(final_text)
                and available_tool_names
                and step < max_steps - 1
            ):
                _empty_reprompt_count += 1
                if _empty_reprompt_count >= _EMPTY_REPROMPT_LIMIT:
                    print(
                        f"[tool_loop] refusal re-prompt limit reached ({_empty_reprompt_count}x) — returning answer",
                        flush=True,
                    )
                    return {"type": "result", "result": final_text}
                tool_list = ", ".join(sorted(n for n in available_tool_names if n))
                print(
                    "[tool_loop] detected capability refusal despite available tools — re-prompting",
                    flush=True,
                )
                current_messages.append({
                    "role": "user",
                    "content": (
                        "Do not claim you lack internet/tools when tools are provided. "
                        f"Available tools: {tool_list}. "
                        "Use the appropriate tool now if needed, then provide the final answer based on tool results."
                    ),
                })
                continue

            if not final_text.strip():
                # BUG FIX: guard applied at ALL steps (was step > 0, missing step 0)
                _empty_reprompt_count += 1
                if _empty_reprompt_count >= _EMPTY_REPROMPT_LIMIT:
                    print(f"[tool_loop] empty content limit reached ({_empty_reprompt_count}x) — returning empty", flush=True)
                    return {"type": "result", "result": ""}
                print(f"[tool_loop] empty content at step {step} — re-prompting", flush=True)
                current_messages.append({
                    "role": "user",
                    "content": "Continue. Use a tool to complete the original task or provide your final answer."
                })
                continue

            print(f"[tool_loop] final answer after {step} step(s)", flush=True)
            # Guard: don't surface raw tool-error strings as the final reply.
            # If the model echoed a tool error (e.g. "(error: skill '...')"),
            # replace it with a neutral message so it doesn't get saved to history
            # and repeated on the next turn.
            _ERROR_PREFIXES = (
                "(error: skill '", "(error: routine '", "(error: file not found",
                "(error: write_file", "(error: web_search failed", "(error: browser_use",
                "(error: exec_shell", "(vllm error:", "(hf generate error:", "(adapter error:",
                "(max steps reached)",
            )
            _ft_lower = final_text.lower().lstrip()
            if any(_ft_lower.startswith(p.lower()) for p in _ERROR_PREFIXES):
                print(f"[tool_loop] suppressing raw tool-error as final answer: {final_text[:80]!r}", flush=True)
                final_text = "I encountered a tool error and could not complete the request. Please try rephrasing or try again."
            return {"type": "result", "result": final_text}

        tool_responses = []
        step_ok_count = 0
        step_fail_count = 0
        step_failures = []  # track failures for structured re-prompt
        for tc in tool_calls:
            fn = tc.get("function", {})
            tool_name = fn.get("name", "")
            tool_args = fn.get("arguments", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            tool_args = _normalize_tool_args(tool_name, tool_args)

            print(f"[tool_loop] step {step+1}: {tool_name}({json.dumps(tool_args)[:80]})", flush=True)

            # Repetition guard — Nemotron sometimes loops on the same read_file call
            _sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            if _sig == _last_tool_sig:
                _repeat_count += 1
                if _repeat_count >= _REPEAT_LIMIT:
                    print(f"[tool_loop] repetition guard: {tool_name} called {_repeat_count}x in a row — breaking", flush=True)
                    # Synthesize a conversational reply from the last tool result
                    # instead of surfacing it verbatim (R7). Do NOT inject a nudge
                    # message (it poisons the history and causes the model to echo
                    # the guard text on subsequent turns).
                    last_result = _synthesize_from_tool_result(_original_query, _last_tool_result)
                    print(f"[tool_loop] repetition guard: returning synthesized answer from last tool result", flush=True)
                    return {"type": "result", "result": last_result}
            else:
                _last_tool_sig = _sig
                _repeat_count = 1
            # Total-call tally guard — same tool+args called too many times across entire loop
            _tool_call_tally[_sig] = _tool_call_tally.get(_sig, 0) + 1
            if _tool_call_tally[_sig] >= _REPEAT_LIMIT:
                print(f"[tool_loop] tally guard: {tool_name} called {_tool_call_tally[_sig]}x total — returning synthesized answer", flush=True)
                last_result = _synthesize_from_tool_result(_original_query, _last_tool_result)
                return {"type": "result", "result": last_result}

            _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace, chat_id=chat_id)
            result_str = _meta.get("result", "")
            _last_tool_result = result_str
            _ok = bool(_meta.get("ok", False))
            _status = str(_meta.get("status", "unknown"))
            _reason = str(_meta.get("failure_reason", ""))
            print(f"[tool_loop] result: {result_str[:100]}", flush=True)

            if _ok:
                step_ok_count += 1
            else:
                step_fail_count += 1
                step_failures.append((tool_name, _status, _reason))
                print(
                    f"[tool_loop] step {step+1} FAILURE tool={tool_name} status={_status} reason={_reason}",
                    flush=True,
                )

            _step_payload = {
                "type": "step", "step": step + 1,
                "tool": tool_name, "args": tool_args, "result": result_str,
                "ok": _ok,
                "status": _status,
                "duration_ms": _meta.get("duration_ms", 0),
            }
            if _reason:
                _step_payload["failure_reason"] = _reason
            if _meta.get("backend"):
                _step_payload["backend"] = _meta.get("backend")

            send_line(json.dumps(_step_payload))
            tool_responses.append({"name": tool_name, "result": result_str})

        # ── Plan 007: Structured failure re-prompt ─────────────────────────
        # When tool calls fail, inject actionable guidance instead of raw error text.
        if step_fail_count > 0 and step < max_steps - 1:
            _fail_tool_name, _fail_status, _fail_reason = step_failures[0]
            _total = step_ok_count + step_fail_count
            _reprompt = _build_failure_reprompt(
                _fail_tool_name, _fail_status, _fail_reason,
                step_fail_count, _total, available_tool_names,
            )
            print(f"[tool_loop] injecting failure re-prompt ({step_fail_count}/{_total} failed)", flush=True)

        # Inject tool results back into the conversation.
        # Nemotron chat template reads role=tool messages via message.content (plain string).
        # The custom `tool_responses` dict format renders as empty in Nemotron's Jinja template.
        # We store the assistant turn as raw text (preserving <tool_call> blocks) and inject
        # each tool result as a separate role=tool message with plain content.
        if _is_nemotron:
            # Nemotron: assistant message = raw decoded text (preserves <tool_call> XML).
            # Note: raw_content was set from parsed["content"] earlier in this step;
            # use it directly instead of re-decoding out[0] which is only defined in the HF branch.
            raw_assistant_text = parsed.get("content", "") or ""
            current_messages.append({"role": "assistant", "content": raw_assistant_text})
            # One tool message per result, plain string content
            for tr in tool_responses:
                current_messages.append({"role": "tool", "content": tr["result"]})
        else:
            # Gemma / other models: use native tool_calls + tool_responses dict format
            current_messages.append({"role": "assistant", "tool_calls": tool_calls})
            current_messages.append({"role": "tool", "tool_responses": [{"name": tr["name"], "response": {"result": tr["result"]}} for tr in tool_responses]})

        # ── Plan 007: Inject failure re-prompt after tool results ───────────
        if step_fail_count > 0 and step < max_steps - 1:
            current_messages.append({"role": "user", "content": _reprompt})

    return {"type": "result", "result": "(max steps reached)"}


def _handle_infer_with_image(params: dict) -> dict:
    """Multimodal image inference.
    Routes to the audio/vision slot (Gemma 4) when the main model is text-only
    (e.g. Nemotron diffusion), falling back to the main model if no vision slot
    is available.
    """
    image_path = params["image_path"]
    prompt = params["prompt"]
    max_new_tokens = params.get("max_new_tokens", 1024)
    # force_local: the native eyes (look/describe) always use the local Gemma
    # E2B vision slot, ignoring the cloud-vision config (which is for Telegram
    # images). When True, skip cloud routing entirely.
    force_local = bool(params.get("force_local", False))

    # ── Cloud vision routing (only when NOT forced local) ─────────────────────
    if not force_local:
        if _config is None:
            _load_config(_lazy_config_path)
        provider_cfg = (_config or {}).get("providers", {})
        vision_provider = provider_cfg.get("vision", "local")
        if vision_provider != "local":
            vision_model = provider_cfg.get("model_overrides", {}).get("vision",
                              provider_cfg.get("models", {}).get(vision_provider, "google/gemma-4-26b-a4b-it"))
            print(f"[model_server] infer_with_image: routing to cloud ({vision_provider}/{vision_model})", flush=True)
            cloud_resp = _cloud_multimodal_infer("vision", vision_provider, vision_model,
                                                  image_path=image_path, prompt=prompt,
                                                  max_new_tokens=max_new_tokens)
            # Fallback: if the cloud provider failed (HTTP 402 out-of-credit,
            # missing key, network error), route natively to the local Gemma E2B
            # vision slot instead of erroring out.
            if "error" in cloud_resp:
                print(f"[model_server] infer_with_image: cloud vision failed ({cloud_resp['error']}); "
                      f"falling back to native vision slot", flush=True)
            else:
                return cloud_resp

    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    # Prefer the dedicated audio/vision slot (Gemma 4 E2B-it) for image inference
    # when the main model is text-only. Lazy-load via _ensure_multimodal_slot() if not yet warm
    # (same path as voice notes) — works whether PDF or voice came first.
    active_model = None
    active_processor = None
    use_hf_path = False

    if not _audio_capable:
        # Main model is text-only — ensure Gemma 4 vision slot is loaded
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_image: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        if _mm_model is not None and _mm_processor is not None:
            print("[model_server] infer_with_image: routing to audio/vision slot (Gemma 4)", flush=True)
            active_model = _mm_model
            active_processor = _mm_processor
            use_hf_path = True
    elif _mm_model is not None and _mm_processor is not None:
        # Main model IS audio-capable but we still have a warm slot — prefer it
        active_model = _mm_model
        active_processor = _mm_processor
        use_hf_path = True

    if active_model is None:
        # Main model handles vision (or nothing else is available)
        _ensure_model()
        # If the main model is loaded, use it as the processor fallback (it may
        # have a processor even when the dedicated multimodal slot is absent).
        if active_processor is None and _model is not None and _processor is not None:
            active_model = _model
            active_processor = _processor
            use_hf_path = False

    # Guard: if we are heading into the HF/slot path but no processor is
    # available, return a clean error instead of crashing on apply_chat_template.
    if active_processor is None:
        return {"error": "no vision processor available (multimodal slot failed to load)"}

    # If we have a dedicated vision model (audio slot / Gemma 4), always use HF path
    if use_hf_path and active_model is not None:
        import torch
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = active_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        target_device = next(active_model.parameters()).device
        inputs = active_processor(text=text, images=[img], return_tensors="pt").to(target_device)
        # Align floating tensors to model dtype
        target_dtype = next(active_model.parameters()).dtype
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and hasattr(v, "to") and torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            elif hasattr(v, "to"):
                inputs[k] = v.to(device=target_device)
        input_len = inputs["input_ids"].shape[-1]
        _mark_activity_start()
        try:
            with torch.no_grad():
                out = active_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            _mark_activity_end()
        result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return {"result": result}

    if _vllm_enabled:
        try:
            proc = _get_vllm_processor()
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ]
            # Format text prompt; pass image as multi_modal_data
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            import torch
            inputs_proc = proc(text=text, images=[img], return_tensors="pt")
            # vLLM multimodal: pass pixel_values via multi_modal_data
            pixel_values = inputs_proc.get("pixel_values")
            mm_data = {"image": img} if pixel_values is not None else None

            from vllm import SamplingParams
            sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
            vllm_input = {"prompt": text}
            if mm_data:
                vllm_input["multi_modal_data"] = mm_data
            result = _vllm_run(_vllm_generate_multimodal_async(vllm_input, sp))
            return {"result": result.strip()}
        except Exception as e:
            print(f"[model_server] vLLM image inference failed ({e}) — falling back to HF", flush=True)
            if _model is None:
                return {"error": f"vLLM failed and HF model not loaded: {e}"}

    # HF path
    import torch
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _processor(text=text, images=[img], return_tensors="pt").to(_target_device())
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if _drafter is not None:
            _gen_kwargs["assistant_model"] = _drafter
        out = _model.generate(**inputs, **_gen_kwargs)
    result = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return {"result": result}


def _handle_infer_with_audio(params: dict) -> dict:
    """Multimodal audio inference.
    NOTE: Uses HF transformers for audio tensor inference (stable path).
    vLLM 0.21.0 supports Gemma4ForConditionalGeneration but multimodal audio
    input via vLLM API is deferred — HF path is simpler and well-tested.
    """
    _ensure_model()

    # ── Cloud audio routing ──────────────────────────────────────────────────
    if _config is None:
        _load_config(_lazy_config_path)
    provider_cfg = (_config or {}).get("providers", {})
    audio_provider = provider_cfg.get("stt", "local")
    if audio_provider != "local":
        audio_model = provider_cfg.get("model_overrides", {}).get("stt",
                          provider_cfg.get("models", {}).get(audio_provider, "google/gemini-2.5-pro"))
        print(f"[model_server] infer_with_audio: routing to cloud STT ({audio_provider}/{audio_model})", flush=True)
        prompt = params.get("prompt", "Transcribe this audio.")
        max_new_tokens = params.get("max_new_tokens", 1024)
        cloud_resp = _cloud_multimodal_infer("audio", audio_provider, audio_model,
                                              audio_path=params["audio_path"], prompt=prompt,
                                              max_new_tokens=max_new_tokens)
        # Fallback: if the cloud STT provider failed (HTTP 402 out-of-credit,
        # missing key, network error), route natively to the local Gemma E2B
        # multimodal slot (native STT) instead of erroring out.
        if "error" in cloud_resp:
            print(f"[model_server] infer_with_audio: cloud STT failed ({cloud_resp['error']}); "
                  f"falling back to native STT slot", flush=True)
        else:
            return cloud_resp

    import torch
    import soundfile as sf
    import numpy as np
    import subprocess
    import tempfile

    audio_path = params["audio_path"]
    prompt = params.get("prompt", "The user sent you a voice message. Listen and reply naturally.")
    max_new_tokens = params.get("max_new_tokens", 1024)
    history = params.get("history", [])
    mode = params.get("mode", "stt")

    # Pick which model/processor to use
    # Priority: named slot (if registry present and slot specified/loaded) → legacy path
    _requested_slot = params.get("slot")
    _use_slot_name = _requested_slot or ("audio" if _slot_registry is not None else None)
    _slot_state = _slot_registry.get(_use_slot_name) if (_slot_registry is not None and _use_slot_name) else None

    if _slot_state is not None:
        active_model = _slot_state.model
        active_processor = _slot_state.processor
        print(f"[model_server] infer_with_audio: using named slot {_use_slot_name!r}", flush=True)
    elif _audio_capable and not _vllm_enabled:
        # Legacy: main model is audio-capable on HF path
        active_model = _model
        active_processor = _processor
        print(f"[model_server] infer_with_audio: using main model (HF, audio_capable)", flush=True)
    else:
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_audio: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        active_model = _mm_model
        active_processor = _mm_processor
        print(f"[model_server] infer_with_audio: using multimodal slot (HF)", flush=True)

    # Guard: if we are heading into the HF path but no processor is available,
    # return a clean error instead of crashing on apply_chat_template.
    if active_processor is None:
        return {"error": "no audio processor available (multimodal slot failed to load)"}

    # Normalise to 16kHz mono float32 WAV via ffmpeg
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        print(f"[infer_with_audio] Input: {audio_path} ({mode} mode)", flush=True)
        import os.path
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        print(f"[infer_with_audio] File size: {os.path.getsize(audio_path)} bytes", flush=True)
        
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", tmp_wav],
            capture_output=True, timeout=30, text=True
        )
        if result.returncode != 0:
            print(f"[infer_with_audio] ffmpeg error: {result.stderr}", flush=True)
            result.check_returncode()  # raise if failed
        
        audio_array, sample_rate = sf.read(tmp_wav, dtype="float32")
        print(f"[infer_with_audio] Decoded: {len(audio_array)} samples @ {sample_rate}Hz", flush=True)
    except Exception as e:
        print(f"[infer_with_audio] Decode via ffmpeg failed: {e}, trying raw read...", flush=True)
        try:
            audio_array, sample_rate = sf.read(audio_path, dtype="float32")
            print(f"[infer_with_audio] Raw read succeeded: {len(audio_array)} samples @ {sample_rate}Hz", flush=True)
        except Exception as e2:
            print(f"[infer_with_audio] Raw read also failed: {e2}", flush=True)
            return {"error": f"Audio decode failed: {e}"}
    finally:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
        print(f"[infer_with_audio] Converted to mono", flush=True)
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
    print(f"[infer_with_audio] Amplitude range: [{audio_array.min():.4f}, {audio_array.max():.4f}]", flush=True)

    if mode == "stt":
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sample_rate}},
                {"type": "text", "text": prompt},
            ]}
        ]
    else:
        messages = [
            {"role": "system", "content": (
                "You are Evo, a conversational AI. "
                "When given audio, listen to what the user says and reply naturally. "
                "Do NOT just transcribe — understand and respond."
            )},
        ]
        for m in history:
            role = m.get("role", "user")
            if role == "assistant":
                role = "model"
            content = str(m.get("content", ""))[:400]
            messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        messages.append({"role": "user", "content": [
            {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sample_rate}},
            {"type": "text", "text": prompt},
        ]})

    text = active_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_device = _target_device() if active_model is _model else next(active_model.parameters()).device
    target_dtype = next(active_model.parameters()).dtype
    print(f"[infer_with_audio] Target: device={target_device}, dtype={target_dtype}", flush=True)

    print(f"[infer_with_audio] Processing audio through {active_model.__class__.__name__}...", flush=True)
    inputs = active_processor(
        text=text,
        audio=[audio_array],
        sampling_rate=16000,
        return_tensors="pt"
    )
    print(f"[infer_with_audio] Input tensors prepared", flush=True)

    # Keep token ids as integer tensors, but align floating tensors to model dtype
    # to avoid errors such as "expected scalar type BFloat16 but found Float".
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and hasattr(v, "to"):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            else:
                inputs[k] = v.to(device=target_device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"[infer_with_audio] Input IDs length: {input_len}", flush=True)
    _mark_activity_start()
    try:
        with torch.no_grad():
            _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if active_model is _model and _drafter is not None:
                _gen_kwargs["assistant_model"] = _drafter
            print(f"[infer_with_audio] Generating...", flush=True)
            out = active_model.generate(**inputs, **_gen_kwargs)
            print(f"[infer_with_audio] Generation complete, output shape: {out.shape}", flush=True)
    finally:
        _mark_activity_end()
    result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    print(f"[infer_with_audio] Result: {repr(result[:100])}", flush=True)
    return {"result": result}


# ── Cloud multimodal routing for audio/vision ──────────────────────────────
# Exposed as config flags: providers.stt / providers.vision / providers.tts in config.yaml.
# When set to a cloud provider ("openrouter", "openai"), the handler calls the
# cloud API with base64-encoded content instead of the local Gemma 4 slot.


def _cloud_multimodal_infer(modality: str, provider: str, model: str,
                             audio_path: str = "", image_path: str = "",
                             prompt: str = "", max_new_tokens: int = 1024) -> dict:
    """Route audio or vision inference through a cloud API (OpenRouter / OpenAI).
    Returns {"result": text} on success or {"error": msg} on failure.
    """
    import base64, json, urllib.request, urllib.error

    # Resolve API key and base URL
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = "https://openrouter.ai/api/v1/chat/completions"
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("TMP_OPEN_AI_API_KEY", "")
        base_url = "https://api.openai.com/v1/chat/completions"
    else:
        return {"error": f"Cloud provider '{provider}' not supported for {modality}"}

    if not api_key:
        return {"error": f"No API key for {provider}"}

    # Build content parts based on modality
    content_parts = []

    if modality == "vision" and image_path:
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            import mimetypes
            mime, _ = mimetypes.guess_type(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime or 'image/jpeg'};base64,{img_b64}"}
            })
        except Exception as e:
            return {"error": f"Failed to read image: {e}"}

    elif modality == "audio" and audio_path:
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            ext = audio_path.rsplit(".", 1)[-1].lower() if "." in audio_path else "wav"
            mime_map = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
                        "webm": "audio/webm", "m4a": "audio/mp4", "flac": "audio/flac"}
            mime = mime_map.get(ext, "audio/wav")
            # OpenAI-compatible audio content format
            content_parts.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": ext if ext in ("wav", "mp3", "flac") else "wav"}
            })
        except Exception as e:
            return {"error": f"Failed to read audio: {e}"}

    else:
        return {"error": f"No content provided for {modality}"}

    content_parts.append({"type": "text", "text": prompt})

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": max_new_tokens,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8779"
        headers["X-Title"] = "kernel-evolving"

    try:
        req = urllib.request.Request(base_url, data=payload, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        result = data["choices"][0]["message"]["content"]
        print(f"[model_server] cloud_{modality}: {repr(result[:100])}", flush=True)
        return {"result": result}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        print(f"[model_server] cloud_{modality} HTTP {e.code}: {body}", flush=True)
        return {"error": f"Cloud {modality} failed: HTTP {e.code}"}
    except Exception as e:
        print(f"[model_server] cloud_{modality} error: {e}", flush=True)
        return {"error": f"Cloud {modality} failed: {e}"}


def _handle_vram_free_mb(params: dict) -> dict:
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return {"vram_free_mb": free // (1024 * 1024)}
    import psutil
    return {"vram_free_mb": psutil.virtual_memory().available // (1024 * 1024)}


def _friendly_model_name(path: str) -> str:
    """Turn a local model path into a readable name.

    HF cache paths look like .../models--org--repo/snapshots/<hash>/ — recover
    'org/repo' from that structure instead of showing the meaningless hash
    (the snapshot dir's basename).
    """
    import re
    m = re.search(r"models--([^/\\]+)--([^/\\]+)", path or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return str(path).rstrip("/\\").split("/")[-1]


def _resolve_loaded_model_name() -> str:
    """Best-effort name of the model actually resident in memory/VRAM.

    Prefers runtime state (vLLM engine path, HF model's own name_or_path, slot
    registry) over config.yaml metadata, since config can silently drift from
    what's actually loaded (e.g. MS1's swap-clobber bug). Falls back to config
    only when nothing is loaded yet.
    """
    if _vllm_enabled and _vllm_model_path:
        return _friendly_model_name(_vllm_model_path)
    if _model is not None:
        name_or_path = getattr(_model, "name_or_path", None) \
            or getattr(getattr(_model, "config", None), "_name_or_path", None)
        if name_or_path:
            return _friendly_model_name(str(name_or_path))
        return _model.__class__.__name__
    if _slot_registry is not None:
        try:
            primary = _slot_registry._loaded.get("primary")
            model_path = getattr(getattr(primary, "spec", None), "model_path", None)
            if model_path:
                return _friendly_model_name(str(model_path))
        except Exception:
            pass
    return (_config or {}).get("model", {}).get("name", "unknown") if _config else "unknown"


def _handle_health(params: dict) -> dict:
    vram_free = _handle_vram_free_mb({}).get("vram_free_mb", 0)
    model_name = _resolve_loaded_model_name()
    vram_warning = vram_free < 2000
    h = {
        "status": "ready",
        "model": model_name,
        "vram_free_mb": vram_free,
        "vram_warning": vram_warning,
        "main_model_loaded": _vllm_enabled or _model is not None,
        "vllm_engine": _vllm_enabled,
        "drafter_loaded": _drafter is not None,
        "audio_capable": _audio_capable,
        "multimodal_slot_loaded": _mm_model is not None,
    }
    if _is_nemotron:
        h["nemotron"] = True
        h["nemotron_mode"] = _nemotron_mode
        h["nemotron_block_length"] = _nemotron_block_length
    h["slots"] = _slot_registry.status() if _slot_registry is not None else []
    return h


# ---------------------------------------------------------------------------
# Slot management handlers
# ---------------------------------------------------------------------------

def _handle_load_slot(params: dict) -> dict:
    """Load a named slot on demand. Returns slot status entry."""
    slot_name = params.get("slot")
    if not slot_name:
        return {"error": "missing required param: slot"}
    if _slot_registry is None:
        return {"error": "SlotRegistry not initialised (no model_slots in config)"}
    try:
        state = _slot_registry.load(slot_name)
        return {"ok": True, "slot": slot_name, "loaded_at": state.loaded_at}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_unload_slot(params: dict) -> dict:
    """Unload a named slot and free its VRAM."""
    slot_name = params.get("slot")
    if not slot_name:
        return {"error": "missing required param: slot"}
    if _slot_registry is None:
        return {"error": "SlotRegistry not initialised"}
    try:
        _slot_registry.unload(slot_name)
        return {"ok": True, "slot": slot_name}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_slot_status(params: dict) -> dict:
    """Return JSON-serialisable status of all registered slots."""
    return {"slots": _slot_registry.status() if _slot_registry is not None else []}


def _handle_unload(params: dict) -> dict:
    """Unload the current model from GPU memory without reloading.
    Called before switching task_inference to a cloud provider, freeing VRAM.
    """
    global _model, _processor, _drafter, _drafter_tokenizer
    global _vllm_engine, _vllm_model_path, _vllm_enabled
    global _is_nemotron
    import gc
    import torch

    print("[model_server] unload: releasing model from VRAM...", flush=True)

    if _vllm_engine is not None:
        try:
            _vllm_run(_vllm_engine.abort_request("*"))
        except Exception:
            pass
        _vllm_engine = None
        _vllm_model_path = None
        _vllm_enabled = False

    for attr in ("_model", "_processor", "_drafter", "_drafter_tokenizer"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
            globals()[attr] = None

    _is_nemotron = False

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
    global _model, _processor, _drafter, _drafter_tokenizer, _config
    global _vllm_engine, _vllm_model_path, _vllm_enabled, _is_nemotron

    import gc
    import torch

    new_path = params.get("model_path", "").strip()
    drafter_path = params.get("drafter_path", None)
    dtype_str = params.get("dtype", "bfloat16")

    if not new_path:
        return {"error": "model_path is required"}

    print(f"[model_server] swap_model: unloading current model...", flush=True)

    # Unload vLLM engine
    if _vllm_engine is not None:
        try:
            _vllm_run(_vllm_engine.abort_request("*"))
        except Exception:
            pass
        _vllm_engine = None
        _vllm_model_path = None
        _vllm_enabled = False

    # Unload HF model
    for attr in ("_model", "_processor", "_drafter", "_drafter_tokenizer"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
            globals()[attr] = None

    _is_nemotron = False

    gc.collect()
    free_before = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        print(f"[model_server] swap_model: {free_before}MB VRAM free after unload", flush=True)

    # Update config for new path
    if _config:
        _config["model"]["path"] = new_path
        _config["model"]["name"] = new_path
        _config["model"]["dtype"] = dtype_str
        # MS3: wire drafter_path through so speculative decoding survives a swap.
        # HF backend reads model.drafter_path/model.speculative_decoding; vLLM
        # backend reads inference.speculative_drafter/inference.speculative_decoding.
        # drafter_path=None (param omitted) means "keep current" — leave both untouched.
        if drafter_path is not None:
            _config["model"]["drafter_path"] = drafter_path
            _config["model"]["speculative_decoding"] = bool(drafter_path)
            _config.setdefault("inference", {})
            _config["inference"]["speculative_drafter"] = drafter_path
            _config["inference"]["speculative_decoding"] = bool(drafter_path)

    # Reload via unified path
    print(f"[model_server] swap_model: loading {new_path}...", flush=True)
    try:
        if _is_nemotron_model(new_path):
            print("[model_server] swap_model: Nemotron-Labs-Diffusion detected", flush=True)
            _load_nemotron(_lazy_config_path, model_path_override=new_path)
        else:
            backend = (_config or {}).get("inference", {}).get("backend", "vllm")
            if backend == "vllm":
                success = _load_vllm_engine(_lazy_config_path, model_path_override=new_path)
                if not success:
                    _load_hf_model(_lazy_config_path, model_path_override=new_path)
            else:
                _load_hf_model(_lazy_config_path, model_path_override=new_path)
    except Exception as e:
        return {"error": f"model load failed: {e}"}

    model_name = _resolve_loaded_model_name()
    backend_tag = "nemotron" if _is_nemotron else ("vllm" if _vllm_enabled else "transformers")
    print(f"[model_server] swap_model: ready — {model_name} ({backend_tag})", flush=True)

    # Re-sync primary slot in registry so slot_status() reflects the new model
    if _slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _spec = _slot_registry._specs.get("primary") or SlotSpec(
                name="primary", model_path=new_path, role="primary"
            )
            _spec.model_path = new_path
            _state = SlotState(
                spec=_spec,
                model=_model,
                processor=_processor,
                is_nemotron=_is_nemotron,
                audio_capable=_audio_capable,
                supports_tools=_model_supports_tools,
            )
            _slot_registry._loaded["primary"] = _state
            _slot_registry._specs["primary"] = _spec
            print("[model_server] swap_model: primary slot synced in registry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not sync primary slot after swap: {_se}", flush=True)

    return {"status": "ok", "model": model_name, "backend": backend_tag}


def _ensure_drafter_only():
    """Load just the drafter + tokenizer without pulling in the full main model."""
    global _drafter, _drafter_tokenizer, _config
    if _drafter is not None and _drafter_tokenizer is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_config(_lazy_config_path)
    drafter_path = cfg.get("model", {}).get("drafter_path", "")
    if not drafter_path:
        return

    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    print(f"[model_server] Loading standalone drafter: {drafter_path}", flush=True)
    _drafter = AutoModelForCausalLM.from_pretrained(drafter_path, dtype=dtype, device_map="cuda:0")
    _drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
    print("[model_server] Standalone drafter ready", flush=True)


def _handle_infer_draft(params: dict) -> dict:
    """ADR-010: Fast draft inference.

    - Nemotron primary: use linear_spec_generate with NFE=3 (self-speculation,
      lowest cost, no external drafter model needed).
    - Legacy AR models with a loaded _drafter: use the drafter AR model.
    - Otherwise: return error so model_client falls back to full infer().
    """
    # ── Nemotron path: self-draft via NFE=3 linear_spec ───────────────────────
    if _is_nemotron and _model is not None:
        import torch
        messages = params.get("messages", [])
        prompt   = params.get("prompt", "")
        max_new_tokens = params.get("max_new_tokens", 256)  # drafts are short

        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            prompt_text = _processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = _processor(prompt_text, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                prompt_ids = prompt_ids.cuda()
            eos_id = _processor.eos_token_id
            with _infer_lock:
                with torch.no_grad():
                    # NFE=3: one speculative draft pass — fast, low-cost
                    out_ids, nfe = _model.linear_spec_generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        block_length=3,   # NFE=3 — drafter mode
                        eos_token_id=eos_id,
                    )
            new_ids = out_ids[:, prompt_ids.shape[1]:]
            text = _processor.batch_decode(new_ids, skip_special_tokens=True)[0]
            print(f"[model_server] infer_draft (Nemotron NFE={nfe} block=3)", flush=True)
            return {"result": text.strip()}
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[model_server] infer_draft Nemotron error: {e}")
            return {"error": str(e)}

    # ── Legacy AR drafter path ─────────────────────────────────────────────────
    _ensure_drafter_only()
    if _drafter is None or _drafter_tokenizer is None:
        return {"error": "drafter not available"}

    import torch
    messages = params.get("messages", [])
    prompt = params.get("prompt", "")
    max_new_tokens = params.get("max_new_tokens", 8192)

    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    text_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        text_parts.append(f"<|{role}|>\n{content}")
    text_parts.append("<|assistant|>\n")
    prompt_text = "\n".join(text_parts)

    try:
        tokenizer = _drafter_tokenizer
        inputs = tokenizer(prompt_text, return_tensors="pt").to(_drafter.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = _drafter.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        result = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return {"result": result}
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[model_server] infer_draft error: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

def _preview_payload_for_log(payload) -> str:
    """Return full, pretty-formatted JSON for readable server logs."""
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(payload)
    return text

class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline()
            if not raw:
                return
            request = json.loads(raw.decode("utf-8").strip())
            method = request.get("method", "")
            params = request.get("params", {})

            if isinstance(method, str) and method.startswith("infer"):
                print(
                    f"[model_server] request method={method} params=\n{_preview_payload_for_log(params)}",
                    flush=True,
                )

            def send_line(data: str):
                if isinstance(method, str) and method.startswith("infer"):
                    try:
                        payload = json.loads(data)
                    except Exception:
                        payload = data
                    print(
                        f"[model_server] response method={method} payload=\n{_preview_payload_for_log(payload)}",
                        flush=True,
                    )
                self.wfile.write((data + "\n").encode("utf-8"))
                self.wfile.flush()

            if method == "infer":
                resp = _handle_infer(params)
                send_line(json.dumps(resp))

            elif method == "infer_draft":
                resp = _handle_infer_draft(params)
                send_line(json.dumps(resp))

            elif method == "infer_with_tools":
                resp = _handle_infer_with_tools(params, send_line)
                send_line(json.dumps(resp))

            elif method == "infer_with_image":
                resp = _handle_infer_with_image(params)
                send_line(json.dumps(resp))

            elif method == "infer_with_audio":
                resp = _handle_infer_with_audio(params)
                send_line(json.dumps(resp))

            elif method == "vram_free_mb":
                resp = _handle_vram_free_mb(params)
                send_line(json.dumps(resp))

            elif method == "health":
                resp = _handle_health(params)
                send_line(json.dumps(resp))

            elif method == "load_slot":
                resp = _handle_load_slot(params)
                send_line(json.dumps(resp))

            elif method == "unload_slot":
                resp = _handle_unload_slot(params)
                send_line(json.dumps(resp))

            elif method == "slot_status":
                resp = _handle_slot_status(params)
                send_line(json.dumps(resp))

            elif method == "unload":
                resp = _handle_unload(params)
                send_line(json.dumps(resp))

            elif method == "swap_model":
                resp = _handle_swap_model(params)
                send_line(json.dumps(resp))

            else:
                send_line(json.dumps({"error": f"unknown method: {method}"}))

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[model_server] handler error: {exc}\n{tb}", flush=True)
            try:
                self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kernel Evolving model server")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--socket", default=None, help="Override socket path")
    parser.add_argument("--lazy", action="store_true", help="Defer model load to first inference request")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter to load after base model")
    args = parser.parse_args()

    socket_path = args.socket or SOCKET_PATH

    cfg = _load_config(args.config)
    if cfg.get("model_server", {}).get("socket"):
        socket_path = cfg["model_server"]["socket"]
    if args.socket:
        socket_path = args.socket

    if os.path.exists(socket_path):
        try:
            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            test_sock.settimeout(2)
            test_sock.connect(socket_path)
            test_sock.close()
            print(f"[model_server] Socket {socket_path} is already live — another instance is running. Exiting.", flush=True)
            sys.exit(0)
        except (ConnectionRefusedError, OSError):
            print(f"[model_server] Removing stale socket {socket_path}", flush=True)
            os.unlink(socket_path)

    global _lazy_config_path
    _lazy_config_path = os.path.abspath(args.config)

    if args.adapter:
        import os as _os
        _os.environ["MODEL_ADAPTER_PATH"] = args.adapter
        print(f"[model_server] Adapter mode: will load LoRA adapter from {args.adapter}", flush=True)

    if args.lazy:
        print(f"[model_server] Lazy mode: model will load on first inference request.", flush=True)
    else:
        _load_model(args.config)
        if args.adapter:
            _load_adapter(args.adapter)

    server = None

    def _shutdown(signum, frame):
        print(f"\n[model_server] Caught signal {signum}, shutting down...", flush=True)
        # Shut down vLLM event loop if running
        if _vllm_loop is not None:
            _vllm_loop.call_soon_threadsafe(_vllm_loop.stop)
        if server:
            threading.Thread(target=server.shutdown, daemon=True).start()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server = _ThreadingUnixServer(socket_path, _RequestHandler)
    os.chmod(socket_path, 0o600)

    print(f"[model_server] Ready on {socket_path}", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
