"""Shared inference helpers for the model-server request handlers.

Extracted from the original model_server.py monolith (refactor). This module
holds the prompt-formatting, vLLM-processor, Omni any-to-any text, config
wrapper, and slot-sync helpers that the handlers in handlers.py need but which
don't belong in a stateless leaf. Keeping them here means handlers.py (and
handlers_common.py itself) never import back into model_server.py — the
forbidden circular edge.
"""

import os

from . import state as server_state
from . import vllm as _vllm
from .config import load_config as _load_config
from .config import (
    inference_cfg as _inference_cfg_impl,
    critique_cfg as _critique_cfg_impl,
)
from .activity import (
    mark_activity_start as _mark_activity_start,
    mark_activity_end as _mark_activity_end,
)

# R10: gate verbose two-stage debug prints (raw/clean model output) behind an
# env var — off by default so raw model output/user messages aren't dumped
# to the production log on every request.
_DEBUG_TWO_STAGE = os.environ.get("KERNEL_EVO_DEBUG_TWO_STAGE", "0") == "1"


# ---------------------------------------------------------------------------
# Config access wrappers (preserve the lazy-load side effect _critique_cfg had
# on module state when reading server_state.config directly).
# ---------------------------------------------------------------------------

def _inference_cfg():
    """Return inference sub-config with defaults."""
    return _inference_cfg_impl(server_state.config)


def _critique_cfg():
    """Return the tool-critique/repair sub-config for this run's tool loop.

    Defaults: schema_validate + repair_critique ON (they are pure, additive,
    and fail open — a validator bug can never block execution). revisor and
    refusal-reprompt are also ON by default since they only re-inject a nudge
    and are fail-open. Read live each turn so config edits take effect without
    a restart.
    """
    if server_state.config is None:
        try:
            _load_config(server_state.lazy_config_path)
        except Exception:
            pass
    return _critique_cfg_impl(server_state.config)


# ---------------------------------------------------------------------------
# Slot globals sync
# ---------------------------------------------------------------------------

def _sync_globals_from_slot(state: "SlotState") -> None:  # type: ignore[name-defined]
    """
    Copy a SlotState's fields into the module-level globals that all existing
    request handlers reference.  This keeps every handler working unchanged
    while the slot registry manages the objects.
    """
    server_state.model = state.model
    server_state.processor = state.processor
    server_state.is_nemotron = state.is_nemotron
    server_state.audio_capable = state.audio_capable
    server_state.model_supports_tools = state.supports_tools
    server_state.native_agentic = getattr(state, 'native_agentic', False)
    server_state.is_omni = getattr(state, 'is_omni', False)
    server_state.is_janus = getattr(state, 'is_janus', False)


# ---------------------------------------------------------------------------
# vLLM processor fetch
# ---------------------------------------------------------------------------

def _get_vllm_processor():
    """Get (or lazily load) the tokenizer/processor for prompt formatting."""
    if server_state.processor is not None:
        return server_state.processor
    if _vllm.model_path():
        try:
            from transformers import AutoProcessor
            server_state.processor = AutoProcessor.from_pretrained(_vllm.model_path(), trust_remote_code=True)
            print("[model_server] Processor loaded for prompt formatting.", flush=True)
        except Exception as e:
            print(f"[model_server] WARNING: processor load failed ({e})", flush=True)
    return server_state.processor


# ---------------------------------------------------------------------------
# Qwen2.5-Omni any-to-any inference helper
# ---------------------------------------------------------------------------
# Omni's generate() has a non-standard signature: it requires input_ids and
# defaults to AUDIO output (thinker + talker + token2wav), which is extremely
# slow and memory-heavy. For text-only output (chat, vision, STT) we must pass
# generation_mode="text" so only the thinker runs and returns the token ids.
def _omni_generate_text(inputs: dict, max_new_tokens: int = 1024) -> "object":
    """Run Qwen2.5-Omni for TEXT-only output. Returns the raw token tensor.

    inputs: dict from processor(text=..., images=[...]|audio=[...], return_tensors='pt')
    """
    from .slots import _ensure_model
    import torch
    _ensure_model()
    gen_kwargs = {}
    # Route the multimodal inputs Omni's generate() understands.
    if inputs.get("attention_mask") is not None:
        gen_kwargs["attention_mask"] = inputs["attention_mask"]
    if inputs.get("pixel_values") is not None:
        gen_kwargs["pixel_values"] = inputs["pixel_values"]
    # Omni's thinker needs the grid/thw tensors to know the layout of the
    # image patches / audio features — without them it sees None and crashes.
    if inputs.get("image_grid_thw") is not None:
        gen_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
    if inputs.get("feature_attention_mask") is not None:
        gen_kwargs["feature_attention_mask"] = inputs["feature_attention_mask"]
    if inputs.get("input_features") is not None:
        gen_kwargs["input_features"] = inputs["input_features"]
    if inputs.get("audio_feature_lengths") is not None:
        gen_kwargs["audio_feature_lengths"] = inputs["audio_feature_lengths"]
    _mark_activity_start()
    try:
        with torch.no_grad():
            out = server_state.model.generate(
                input_ids=inputs["input_ids"],
                generation_mode="text",
                thinker_max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
    finally:
        _mark_activity_end()
    return out


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
