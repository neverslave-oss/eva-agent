"""HF transformers backend family for the model server.

Extracted from the original model_server.py monolith (refactor). Owns the
transformer-based engine loaders and their inference helpers:

  - `load_hf_model`   — generic AutoProcessor/AutoModelForCausalLM path (+vLLM fallback)
  - `load_nemotron`   — Nemotron-Labs-Diffusion loader + modes
  - `_load_janus`     — DeepSeek Janus / Janus-Pro custom-package loader
  - `_janus_infer_text` / `_janus_infer_image` / `_nemotron_infer`

All shared state is read/written through `server_state`; config loading and
capability detection come from their extracted homes (config.py, capabilities.py)
so this module never imports back into model_server.py.
"""

from . import state as server_state
from .config import load_config as _load_config
from .capabilities import apply_detect as _detect_capabilities
from .model_names import adapter_name as _adapter_name
from .state import target_device as _target_device
from .activity import (
    mark_activity_start as _mark_activity_start,
    mark_activity_end as _mark_activity_end,
)

import os
import sys
from pathlib import Path

# Allow running from src/ or repo root (mirrors model_server.py header).
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _src not in sys.path:
    sys.path.insert(0, _src)

# Slot registry availability — lazy import to avoid circular deps; model_server.py
# owns the flag too and imports it back from this module.
try:
    from model_slots import SlotRegistry, SlotSpec, SlotState  # type: ignore
    _SLOTS_AVAILABLE = True
except ImportError:
    _SLOTS_AVAILABLE = False


def _load_hf_model(config_path="config.yaml", model_path_override: str | None = None):
    """Load model via HF transformers (fallback path).

    model_path_override: when swap_model has already updated _config in-memory,
    pass the path directly to avoid re-reading stale disk config (mirrors _load_nemotron).
    """
    if server_state.model is not None:
        return

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer

    cfg = server_state.config if server_state.config is not None else _load_config(config_path)
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
    server_state.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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
            # Some any-to-any models (e.g. Qwen2.5-Omni) declare a base
            # architecture (Qwen2_5OmniModel) that isn't exported at the
            # transformers top level, but expose a ForConditionalGeneration
            # variant that IS. Try that before falling back to Auto classes.
            if _ModelCls is None or not hasattr(_ModelCls, 'generate'):
                _fcg_name = _cls_name.replace('Model', 'ForConditionalGeneration')
                if _fcg_name != _cls_name:
                    _ModelCls = getattr(_mod, _fcg_name, None)
                    if _ModelCls is not None:
                        _cls_name = _fcg_name
            if _ModelCls is not None and hasattr(_ModelCls, 'generate'):
                print(f"[model_server] Using architecture class: {_cls_name}", flush=True)
                server_state.model = _ModelCls.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
            else:
                raise AttributeError(f"{_cls_name} not found in transformers or lacks .generate")
        else:
            raise AttributeError("No architectures in config")
    except Exception as _arch_err:
        print(f"[model_server] Architecture lookup failed ({_arch_err}), trying AutoModelForImageTextToText", flush=True)
        server_state.model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
    if not hasattr(server_state.model, 'generate'):
        # Last resort: try AutoModelForCausalLM (covers some multimodal VL models)
        print("[model_server] WARNING: loaded model has no .generate() — retrying with AutoModelForCausalLM", flush=True)
        del server_state.model
        import gc; gc.collect()
        server_state.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **load_kwargs)
    print("[model_server] HF Model loaded.", flush=True)

    _detect_capabilities(model_path, cfg)

    # Processor-based tool support refinement (Gemma parse_response)
    try:
        server_state.model_supports_tools = callable(getattr(server_state.processor, 'parse_response', None))
        server_state.processor.parse_response("")
        server_state.model_supports_tools = True
    except AttributeError:
        server_state.model_supports_tools = False
    except Exception:
        server_state.model_supports_tools = True
    print(f"[model_server] Tool-calling support (refined): {server_state.model_supports_tools}", flush=True)

    # Load MTP drafter
    drafter_path = cfg["model"].get("drafter_path")
    use_speculative = cfg["model"].get("speculative_decoding", False)
    if drafter_path and use_speculative:
        try:
            print(f"[model_server] Loading drafter: {drafter_path}", flush=True)
            server_state.drafter = AutoModelForCausalLM.from_pretrained(drafter_path, dtype=dtype, device_map="auto")
            server_state.drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
            print("[model_server] Drafter loaded.", flush=True)
        except Exception as e:
            print(f"[model_server] WARNING: drafter load failed ({e})", flush=True)
            server_state.drafter = None
            server_state.drafter_tokenizer = None

    # Wire primary slot into registry (additive — all existing globals remain set)
    if server_state.slot_registry is not None and _SLOTS_AVAILABLE:
        try:
            from model_slots import SlotState, SlotSpec  # type: ignore
            _spec = server_state.slot_registry._specs.get("primary") or SlotSpec(
                name="primary",
                model_path=model_path,
                role="primary",
            )
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
            print("[model_server] Primary slot wired into SlotRegistry", flush=True)
        except Exception as _se:
            print(f"[model_server] WARNING: could not wire primary slot: {_se}", flush=True)


# ---------------------------------------------------------------------------
# Nemotron-Labs-Diffusion load + inference helpers
# ---------------------------------------------------------------------------

def _load_nemotron(config_path="config.yaml", model_path_override: str | None = None):
    """Load nvidia/Nemotron-Labs-Diffusion via AutoModel + AutoTokenizer.

    Nemotron uses trust_remote_code=True, returns (out_ids, nfe) from its
    custom generate methods — incompatible with the standard HF generate path.
    model_path_override: when swap_model has already updated _config in-memory,
    pass the path directly to avoid re-reading stale disk config.
    """

    if server_state.model is not None:
        return

    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = server_state.config if server_state.config is not None else _load_config(config_path)
    model_path = model_path_override or cfg["model"].get("path") or cfg["model"].get("name")
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    # Nemotron generation config
    server_state.nemotron_mode = cfg["model"].get("generation_mode", "linear_spec")
    server_state.nemotron_block_length = int(cfg["model"].get("block_length", 32))
    server_state.nemotron_threshold = float(cfg["model"].get("threshold", 0.9))

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

    print(f"[model_server] Loading Nemotron: {model_path} (mode={server_state.nemotron_mode}) ...", flush=True)
    # AutoTokenizer — Nemotron has no multimodal processor
    server_state.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    raw = AutoModel.from_pretrained(model_path, **load_kwargs)

    # If no CUDA placement already done via quantise, move to GPU
    if not bnb_config and torch.cuda.is_available():
        raw = raw.cuda()
    server_state.model = raw

    # ── Adapter selection ──────────────────────────────────────────────────
    # native_agentic mode (config: model.native_agentic=true): Nemotron drives the
    # full tool loop itself with our function-calling adapter. Keep the PeftModel
    # wrapper (DO NOT unwrap) so the FC adapter stays activatable via _use_adapter/
    # set_adapter, and skip the linear_spec speed-drafter unwrap.
    native_agentic = bool(cfg.get("model", {}).get("native_agentic", False))
    adapter_path = cfg.get("model", {}).get("adapter_path")

    if native_agentic:
        if adapter_path:
            try:
                from peft import PeftModel
                adapter_name = _adapter_name(adapter_path)
                server_state.model = PeftModel.from_pretrained(
                    server_state.model, adapter_path, adapter_name=adapter_name
                )
                server_state.loaded_adapters[adapter_path] = adapter_name
                print(f"[model_server] Nemotron: FC adapter loaded from {adapter_path}", flush=True)
            except Exception as e:
                print(f"[model_server] Nemotron: FC adapter load failed ({e}) — continuing without", flush=True)
        else:
            print("[model_server] Nemotron: native_agentic=true but no adapter_path set — running native tool loop unadaptered", flush=True)
    else:
        # Original linear_spec path — optional speed-drafter (linear_spec_lora subfolder)
        if server_state.nemotron_mode == "linear_spec" and not adapter_path:
            import os as _os
            _lora_path = _os.path.join(model_path, "linear_spec_lora")
            if _os.path.isdir(_lora_path):
                adapter_path = _lora_path
        if adapter_path and server_state.nemotron_mode == "linear_spec":
            try:
                from peft import PeftModel
                _peft = PeftModel.from_pretrained(server_state.model, adapter_path).eval()
                server_state.model = _peft.model  # unwrap to call linear_spec_generate directly
                print(f"[model_server] Nemotron: LoRA drafter loaded from {adapter_path}", flush=True)
            except Exception as e:
                print(f"[model_server] Nemotron: LoRA drafter load failed ({e}) — continuing without", flush=True)

    server_state.is_nemotron = True
    # Nemotron has full tool-calling via its chat template (XML function_calls format)
    server_state.model_supports_tools = True
    server_state.audio_capable = False
    # Nemotron stays on its tested Qwen two-stage path unless native_agentic is set.
    server_state.native_agentic = native_agentic
    print(f"[model_server] Nemotron ready. mode={server_state.nemotron_mode} block={server_state.nemotron_block_length} threshold={server_state.nemotron_threshold}", flush=True)


# ---------------------------------------------------------------------------
# DeepSeek Janus / Janus-Pro loader + inference
# ---------------------------------------------------------------------------
# Janus uses the custom `janus` package (vendored at third_party/janus):
#   - MultiModalityCausalLM (loaded via AutoModelForCausalLM + trust_remote_code)
#   - VLChatProcessor for tokenization
# Inference is NOT model.generate(**inputs); it uses prepare_inputs_embeds() then
# language_model.generate(inputs_embeds=...). We keep the model in _model and the
# tokenizer/processor in _processor so existing handlers can route to it.

def _load_janus(config_path="config.yaml", model_path_override: str | None = None):
    """Load a DeepSeek Janus/Janus-Pro model via the custom `janus` package."""
    _load_config(config_path)
    cfg = server_state.config or {}
    model_path = model_path_override or cfg.get("model", {}).get("path") or cfg.get("model", {}).get("name", "")

    try:
        import third_party.janus.models as _jm  # registers multi_modality + MultiModalityCausalLM
        from transformers import AutoModelForCausalLM
        from third_party.janus.models.processing_vlm import VLChatProcessor
        from third_party.janus.models.image_processing_vlm import VLMImageProcessor
    except Exception as e:
        print(f"[model_server] WARNING: janus package not available ({e}) — aborting Janus load", flush=True)
        raise RuntimeError(f"Janus package required: {e}")

    print(f"[model_server] Loading Janus model: {model_path} ...", flush=True)
    # Load the processor first (it carries the tokenizer)
    try:
        server_state.janus_processor = VLChatProcessor.from_pretrained(model_path)
        server_state.processor = server_state.janus_processor.tokenizer
    except Exception as e:
        print(f"[model_server] Janus processor load failed ({e})", flush=True)
        server_state.processor = None

    # Load the model in bfloat16 on GPU
    import torch
    load_kwargs = {}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    server_state.model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, **load_kwargs
    )
    server_state.model.eval()

    server_state.is_janus = True
    server_state.audio_capable = False
    server_state.model_supports_tools = False
    server_state.native_agentic = True  # Janus drives its own flow (text + vision)
    print(f"[model_server] Janus loaded: {type(server_state.model).__name__}", flush=True)


def _janus_infer_text(messages: list, max_new_tokens: int = 8192) -> str:
    """Text-only inference for Janus via prepare_inputs_embeds + language_model.generate."""
    import torch
    if server_state.janus_processor is None:
        raise RuntimeError("Janus processor not loaded")
    # Flatten to a single user prompt (Janus is not a chat model in the standard sense)
    text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") in ("user", "assistant")
    ) or "Hello"
    conversation = [
        {"role": "<|User|>", "content": text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = server_state.janus_processor(
        conversations=conversation, images=[], force_batchify=True
    ).to(_target_device())
    inputs_embeds = server_state.model.prepare_inputs_embeds(**prepare_inputs)
    tokenizer = server_state.janus_processor.tokenizer
    _mark_activity_start()
    try:
        with torch.no_grad():
            outputs = server_state.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        _mark_activity_end()
    # Janus uses a sentencepiece tokenizer whose leading-space marker (Ġ, U+0120)
    # is not a registered special token, so skip_special_tokens=True leaves it in
    # the decoded text. Normalise it back to a regular space.
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\u0120", " ")


def _janus_infer_image(image_path: str, prompt: str, max_new_tokens: int = 1024) -> str:
    """Vision inference for Janus via prepare_inputs_embeds + language_model.generate."""
    import torch
    if server_state.janus_processor is None:
        raise RuntimeError("Janus processor not loaded")
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    conversation = [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt}", "images": [img]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepare_inputs = server_state.janus_processor(
        conversations=conversation, images=[img], force_batchify=True
    ).to(_target_device())
    inputs_embeds = server_state.model.prepare_inputs_embeds(**prepare_inputs)
    tokenizer = server_state.janus_processor.tokenizer
    _mark_activity_start()
    try:
        with torch.no_grad():
            outputs = server_state.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        _mark_activity_end()
    # Janus uses a sentencepiece tokenizer whose leading-space marker (Ġ, U+0120)
    # is not a registered special token, so skip_special_tokens=True leaves it in
    # the decoded text. Normalise it back to a regular space (vision responses
    # show this artifact e.g. 'TheĠimageĠshows...').
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).replace("\u0120", " ")


def _nemotron_infer(messages: list, max_new_tokens: int = 8192) -> str:
    """Run inference using the appropriate Nemotron generation mode.
    Returns the decoded output string.
    """
    import torch
    prompt = server_state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = server_state.processor(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        prompt_ids = prompt_ids.cuda()

    eos_id = server_state.processor.eos_token_id

    print(f"[DEBUG model_server] Nemotron messages: {messages}", flush=True)

    
    with torch.no_grad():
        if server_state.nemotron_mode == "ar":
            out_ids, nfe = server_state.model.ar_generate(prompt_ids, max_new_tokens=max_new_tokens)
        elif server_state.nemotron_mode == "diffusion":
            out_ids, nfe = server_state.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                block_length=server_state.nemotron_block_length,
                threshold=server_state.nemotron_threshold,
                eos_token_id=eos_id,
            )
        else:  # linear_spec (default — fastest)
            out_ids, nfe = server_state.model.linear_spec_generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                block_length=server_state.nemotron_block_length,
                eos_token_id=eos_id,
            )

    new_ids = out_ids[:, prompt_ids.shape[1]:]
    text = server_state.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    print(f"[model_server] Nemotron NFE={nfe} mode={server_state.nemotron_mode}", flush=True)
    return text.strip()


# ---------------------------------------------------------------------------
