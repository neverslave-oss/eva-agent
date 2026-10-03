"""handlers/draft.py — drafter/speculative-decoding handlers (issue #2 split).

Extracted from src/core/inference/handlers/__init__.py: _ensure_drafter_only,
_handle_infer_draft. Re-exported from handlers/__init__.py.
"""
from .. import state as server_state
from ..handlers_common import _load_config



def _ensure_drafter_only():
    """Load just the drafter + tokenizer without pulling in the full main model."""
    if server_state.drafter is not None and server_state.drafter_tokenizer is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _load_config(server_state.lazy_config_path)
    drafter_path = cfg.get("model", {}).get("drafter_path", "")
    if not drafter_path:
        return

    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    print(f"[model_server] Loading standalone drafter: {drafter_path}", flush=True)
    server_state.drafter = AutoModelForCausalLM.from_pretrained(drafter_path, dtype=dtype, device_map="cuda:0")
    server_state.drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
    print("[model_server] Standalone drafter ready", flush=True)

def _handle_infer_draft(params: dict) -> dict:
    """ADR-010: Fast draft inference.

    - Nemotron primary: use linear_spec_generate with NFE=3 (self-speculation,
      lowest cost, no external drafter model needed).
    - Legacy AR models with a loaded _drafter: use the drafter AR model.
    - Otherwise: return error so model_client falls back to full infer().
    """
    # ── Nemotron path: self-draft via NFE=3 linear_spec ───────────────────────
    if server_state.is_nemotron and server_state.model is not None:
        import torch
        messages = params.get("messages", [])
        prompt   = params.get("prompt", "")
        max_new_tokens = params.get("max_new_tokens", 256)  # drafts are short

        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            prompt_text = server_state.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = server_state.processor(prompt_text, return_tensors="pt").input_ids
            if torch.cuda.is_available():
                prompt_ids = prompt_ids.cuda()
            eos_id = server_state.processor.eos_token_id
            with server_state.infer_lock:
                with torch.no_grad():
                    # NFE=3: one speculative draft pass — fast, low-cost
                    out_ids, nfe = server_state.model.linear_spec_generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        block_length=3,   # NFE=3 — drafter mode
                        eos_token_id=eos_id,
                    )
            new_ids = out_ids[:, prompt_ids.shape[1]:]
            text = server_state.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
            print(f"[model_server] infer_draft (Nemotron NFE={nfe} block=3)", flush=True)
            return {"result": text.strip()}
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[model_server] infer_draft Nemotron error: {e}")
            return {"error": str(e)}

    # ── Legacy AR drafter path ─────────────────────────────────────────────────
    _ensure_drafter_only()
    if server_state.drafter is None or server_state.drafter_tokenizer is None:
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
        tokenizer = server_state.drafter_tokenizer
        inputs = tokenizer(prompt_text, return_tensors="pt").to(server_state.drafter.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = server_state.drafter.generate(
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

