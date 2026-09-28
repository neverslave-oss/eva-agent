"""vLLM backend subsystem for the model server.

Extracted from the original model_server.py monolith (refactor). Owns all vLLM
state (engine, background event loop, loaded model path, enabled flag) and the
vLLM-specific generation helpers. Cross-file coupling (config dict and
capability detection) is passed in explicitly, so this module has no import-time
dependency on model_server state.

External call sites in model_server.py reached for it through:
  start_event_loop(), run(coro), load_engine(cfg, detect_capabilities, model_path_override),
  generate_async(), generate_multimodal_async(), infer(), model_path(), is_enabled(),
  reset(), shutdown().
"""

import asyncio
import os
import tempfile
import threading
import uuid


_vllm_engine = None          # vllm.AsyncLLMEngine instance (or None if fallback)
_vllm_model_path = None      # which path is loaded into vllm
_vllm_enabled = False        # True if vLLM backend is active
_vllm_loop = None            # asyncio event loop running in background thread


def start_event_loop():
    """Start a dedicated asyncio event loop in a background thread for vLLM."""
    global _vllm_loop
    loop = asyncio.new_event_loop()
    _vllm_loop = loop
    loop.run_forever()


def run(coro):
    """Submit a coroutine to the vLLM event loop and block until done."""
    if _vllm_loop is None:
        raise RuntimeError("vLLM event loop not started")
    fut = asyncio.run_coroutine_threadsafe(coro, _vllm_loop)
    return fut.result(timeout=300)


def is_enabled() -> bool:
    return _vllm_enabled


def model_path():
    return _vllm_model_path


def engine():
    return _vllm_engine


def _ensure_event_loop():
    if _vllm_loop is None:
        t = threading.Thread(target=start_event_loop, daemon=True)
        t.start()
        # Give it a moment to initialise
        import time
        time.sleep(0.1)


def load_engine(cfg: dict, detect_capabilities, model_path_override: str | None = None) -> bool:
    """Load vLLM AsyncLLMEngine. Returns True on success, False on failure.

    cfg is the (already-resolved) config dict. detect_capabilities(model_path, cfg)
    is invoked after a successful load to update capability flags. model_path_override:
    when swap_model has already updated the config in-memory, pass the path directly
    to avoid re-reading stale disk config (mirrors the Nemotron loader).
    """
    global _vllm_engine, _vllm_model_path, _vllm_enabled

    # vLLM uses IPC sockets for worker communication — must be on a real Linux FS.
    # Windows-mounted drives (e.g. /mnt/...) don't support Unix sockets. Force /tmp.
    if not os.environ.get("TMPDIR", "").startswith("/tmp"):
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = "/tmp"

    cfg = cfg or {}
    model_path = model_path_override \
                 or (os.environ.get("MODEL_SOURCE") == "docker-hub" and os.environ.get("MODEL_ID")) \
                 or cfg.get("model", {}).get("path") or cfg.get("model", {}).get("name")
    inf_cfg = cfg.get("inference", {})

    gpu_util = inf_cfg.get("gpu_memory_utilization", 0.80)
    max_model_len = inf_cfg.get("max_model_len") or cfg.get("model", {}).get("max_context_length", 8192)
    dtype = cfg.get("model", {}).get("dtype", "bfloat16")
    # vLLM quantization — prefer bitsandbytes if config has 4bit
    quantize_cfg = cfg.get("model", {}).get("quantize", "none")
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

        _ensure_event_loop()

        async def _create():
            return AsyncLLMEngine.from_engine_args(engine_args)

        _vllm_engine = run(_create())
        _vllm_model_path = model_path
        _vllm_enabled = True

        if detect_capabilities is not None:
            detect_capabilities(model_path, cfg)
        print("[model_server] vLLM engine ready.", flush=True)
        return True

    except Exception as e:
        print(f"[model_server] WARNING: vLLM engine load failed: {e}", flush=True)
        print("[model_server] Falling back to HF transformers backend.", flush=True)
        _vllm_enabled = False
        return False


async def generate_async(prompt: str, sampling_params, request_id: str = None) -> str:
    """Run a single vLLM generation and return the full output text."""
    from vllm import SamplingParams  # noqa: F401  (kept for parity/readability)

    if request_id is None:
        request_id = str(uuid.uuid4())

    full_output = ""
    async for output in _vllm_engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            full_output = output.outputs[0].text

    return full_output


async def generate_multimodal_async(inputs: dict, sampling_params, request_id: str = None) -> str:
    """Run a vLLM multimodal generation (image + text) and return output text."""

    if request_id is None:
        request_id = str(uuid.uuid4())

    full_output = ""
    async for output in _vllm_engine.generate(inputs, sampling_params, request_id):
        if output.outputs:
            full_output = output.outputs[0].text

    return full_output


def infer(prompt: str, max_new_tokens: int = 8192, temperature: float = 1.0,
          top_p: float = 0.95, top_k: int = 64) -> str:
    """Synchronous wrapper around vLLM generation."""
    from vllm import SamplingParams
    sp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return run(generate_async(prompt, sp))


def reset():
    """Abort in-flight requests and clear all vLLM state (used on unload/swap)."""
    global _vllm_engine, _vllm_model_path, _vllm_enabled
    if _vllm_engine is not None:
        try:
            run(_vllm_engine.abort_request("*"))
        except Exception:
            pass
    _vllm_engine = None
    _vllm_model_path = None
    _vllm_enabled = False


def shutdown():
    """Stop the background event loop (called on process exit)."""
    global _vllm_loop
    if _vllm_loop is not None:
        try:
            _vllm_loop.call_soon_threadsafe(_vllm_loop.stop)
        except Exception:
            pass
