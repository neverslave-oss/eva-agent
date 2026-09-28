"""LoRA adapter helpers for the model server.

Extracted from the original model_server.py monolith (refactor).

  - `_load_adapter` — attach a LoRA adapter onto the already-loaded HF model.
  - `_use_adapter`  — contextmanager that temporarily activates an adapter.

`_use_adapter` calls `_ensure_model` (owned by slots.py); to avoid an
import-time cycle (slots._ensure_model also calls `_load_adapter`) that
dependency is resolved lazily inside the function body. vLLM state lives in
vllm.py (no cycle there).
"""

from . import state as server_state
from . import vllm as _vllm
from .model_names import (
    adapter_name as _adapter_name,
    is_peft_model as _is_peft_model,
)

import contextlib
from pathlib import Path


def _load_adapter(adapter_path: str) -> str:
    """Load a LoRA adapter onto the already-loaded HF model and return its adapter name."""
    if server_state.model is None:
        print("[adapter] WARNING: base model not loaded — cannot attach adapter", flush=True)
        raise RuntimeError("base model not loaded")
    if _vllm.is_enabled():
        raise RuntimeError("adapter-aware inference is not supported with vLLM backend")

    adapter_path = str(Path(adapter_path).expanduser())
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"adapter path not found: {adapter_path}")

    existing = server_state.loaded_adapters.get(adapter_path)
    if existing:
        return existing

    try:
        from peft import PeftModel
        adapter_name = _adapter_name(adapter_path)
        if _is_peft_model(server_state.model):
            server_state.model.load_adapter(adapter_path, adapter_name=adapter_name)
        else:
            server_state.model = PeftModel.from_pretrained(server_state.model, adapter_path, adapter_name=adapter_name)
        server_state.loaded_adapters[adapter_path] = adapter_name
        print(f"[adapter] loaded LoRA adapter '{adapter_name}' from {adapter_path}", flush=True)
        return adapter_name
    except Exception as e:
        print(f"[adapter] ERROR loading adapter from {adapter_path}: {e}", flush=True)
        raise


@contextlib.contextmanager
def _use_adapter(adapter_path: str | None):
    """Temporarily activate an adapter for generation on the shared base model."""
    from .slots import _ensure_model
    _ensure_model()

    if adapter_path and _vllm.is_enabled():
        raise RuntimeError("adapter-aware inference is not supported with vLLM backend")

    with server_state.infer_lock:
        if not adapter_path:
            if server_state.model is not None and _is_peft_model(server_state.model):
                with server_state.model.disable_adapter():
                    yield
            else:
                yield
            return

        adapter_name = _load_adapter(adapter_path)
        previous = server_state.current_adapter_name
        if hasattr(server_state.model, "set_adapter"):
            server_state.model.set_adapter(adapter_name)
        server_state.current_adapter_name = adapter_name
        try:
            yield
        finally:
            if previous and hasattr(server_state.model, "set_adapter"):
                server_state.model.set_adapter(previous)
                server_state.current_adapter_name = previous
            else:
                server_state.current_adapter_name = None
