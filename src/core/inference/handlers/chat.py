"""handlers/chat.py — core chat inference handlers (issue #2 handlers split).

Extracted from src/core/inference/handlers/__init__.py. _handle_infer and
_handle_infer_plain. Imports resolve via the core/inference package (..);
re-exported from handlers/__init__.py so model_server keep working unchanged.
"""


from .. import state as server_state
from .. import vllm as _vllm
from ..handlers_common import (
    _build_chat_prompt,
    _omni_generate_text,
)
from ..state import target_device as _target_device
from ..loaders import (
    _janus_infer_text,
    _nemotron_infer,
)
from ..adapters import _use_adapter
from ..slots import _ensure_model


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
    if server_state.is_nemotron:
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
            if _vllm.is_enabled():
                try:
                    prompt = _build_chat_prompt(formatted, enable_thinking=False)
                    result = _vllm.infer(prompt, max_new_tokens=max_new_tokens)
                    return {"result": result.strip()}
                except Exception as e:
                    print(f"[model_server] vLLM infer error: {e} — falling back to HF", flush=True)
                    # Fall through to HF path if model available
                    if server_state.model is None:
                        return {"error": f"vLLM failed and HF model not loaded: {e}"}

            # HF transformers path
            import torch
            inputs = server_state.processor.apply_chat_template(
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
                if server_state.drafter is not None:
                    _gen_kwargs["assistant_model"] = server_state.drafter
                out = server_state.model.generate(**inputs, **_gen_kwargs)
            response_raw = server_state.processor.decode(out[0][input_len:], skip_special_tokens=False)
            parsed = server_state.processor.parse_response(response_raw)
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

    # ── Janus path
    if server_state.is_janus:
        try:
            result = _janus_infer_text(messages, max_new_tokens=8192)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    # ── Nemotron path
    if server_state.is_nemotron:
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
            if _vllm.is_enabled():
                try:
                    prompt = _build_chat_prompt(chat, enable_thinking=False)
                    result = _vllm.infer(prompt, max_new_tokens=8192, temperature=0.7, top_p=0.9, top_k=50)
                    return {"result": result.strip()}
                except Exception as e:
                    print(f"[model_server] vLLM infer_plain error: {e} — falling back to HF", flush=True)
                    if server_state.model is None:
                        return {"error": f"vLLM failed and HF model not loaded: {e}"}

            # HF path
            import torch
            try:
                text = server_state.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = server_state.processor(text=text, return_tensors="pt").to(_target_device())
            except Exception as e:
                last = next((m["content"] for m in reversed(chat) if m["role"] == "user"), "")
                inputs = server_state.processor(text=last, return_tensors="pt").to(_target_device())

            input_len = inputs["input_ids"].shape[-1]
            if server_state.is_omni:
                out = _omni_generate_text(inputs, max_new_tokens=8192)
            else:
                with torch.no_grad():
                    out = server_state.model.generate(
                        **inputs,
                        max_new_tokens=8192,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                    )
            result = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
            return {"result": result}
    except Exception as e:
        return {"error": str(e)}
