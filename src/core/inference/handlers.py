"""JSON-RPC request handlers for the model server.

Extracted from the original model_server.py monolith (refactor) — the final and
largest cut. This module owns the entire handler/inference region:

  - core chat:        _handle_infer, _handle_infer_plain, _handle_infer_draft
  - tool calling:     _run_two_stage_if_available, _nemotron_synthesize_answer,
                      _handle_infer_with_tools
  - multimodal:       _handle_infer_with_image, _handle_infer_with_audio,
                      _handle_infer_local, _cloud_multimodal_infer
  - ops/telemetry:    _handle_vram_free_mb, _resolve_loaded_model_name,
                      _handle_health, _handle_load_slot, _handle_unload_slot,
                      _handle_slot_status, _handle_unload, _handle_swap_model

Shared inference helpers (prompt building, vLLM processor, Omni text, config
wrappers, slot-globals sync) live in handlers_common.py so this module never
imports back into model_server.py (forbidden circular). model_server.py re-imports
and re-exports every handler so the thin entrypoint + HANDLERS dispatch table
keep working unchanged.
"""

import os
import traceback

from . import state as server_state
from . import vllm as _vllm
from .handlers_common import (
    _build_chat_prompt,
    _get_vllm_processor,
    _omni_generate_text,
    _inference_cfg,
    _critique_cfg,
    _load_config,
    _sync_globals_from_slot,
    _mark_activity_start,
    _mark_activity_end,
    _DEBUG_TWO_STAGE,
)
from .state import target_device as _target_device
from .activity import mark_activity_start as _mark_activity_start_noarg, mark_activity_end as _mark_activity_end_noarg  # noqa: F401

# Slot registry — mirror the same lazy import pattern model_server.py uses;
# the _SLOTS_AVAILABLE flag is owned here (and re-exported by model_server.py).
try:
    from model_slots import SlotRegistry, SlotSpec, SlotState  # type: ignore
    _SLOTS_AVAILABLE = True
except ImportError:
    _SLOTS_AVAILABLE = False

# Re-exported backend leaves (HF loaders, LoRA adapters, slot lifecycle,
# model-name predicates, parsing, prompts, validation) so handlers can call
# them exactly as the old monolith did.
from .loaders import (  # noqa: E402
    _load_hf_model,
    _load_nemotron,
    _load_janus,
    _janus_infer_text,
    _janus_infer_image,
    _nemotron_infer,
)
from .adapters import (  # noqa: E402
    _load_adapter,
    _use_adapter,
)
from .slots import (  # noqa: E402
    _ensure_model,
    _ensure_multimodal_slot,
    _ensure_tool_calling_slot,
)
from .model_names import (  # noqa: E402
    is_nemotron_model as _is_nemotron_model,
    is_janus_model as _is_janus_model,
    friendly_model_name as _friendly_model_name,
)
from .parsing import (  # noqa: E402
    _normalize_tool_args,
    _parse_function_eq_xml,
    _parse_tag_fallback,
)
from .prompts import (  # noqa: E402
    build_failure_reprompt as _build_failure_reprompt,
    build_schema_repair_hint as _build_schema_repair_hint,
)
from .validation import (  # noqa: E402
    tool_result_answers_query as _tool_result_answers_query,
)
from .capabilities import apply_detect as _detect_capabilities
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



# ── Two-stage tool-calling pipeline (ADR-011) ─────────────────────────────
# Qwen3.5-0.8B handles tool loop; Nemotron handles conversation synthesis.

def _run_two_stage_if_available(params: dict, send_line) -> dict | None:
    """Try the two-stage pipeline. Returns None if tool_calling slot unavailable."""

    if not server_state.tool_calling_slot_loaded:
        _ensure_tool_calling_slot()
    if not server_state.tool_calling_slot_loaded:
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
    max_steps = params.get("max_steps", 60)
    # Passed explicitly (not read from core.tools._current_chat_id) since
    # model_server runs in a separate, multi-threaded process from the
    # caller that knows the real chat_id (R5/T4).
    chat_id = params.get("chat_id", "")
    # A new task is starting — clear any stale transient auth state
    # (approve-all grant, deny counter, stop signal) from a previous task in
    # this chat so it cannot leak across tasks.
    if chat_id:
        try:
            from core.auth_gate import clear_task_state
            clear_task_state(chat_id)
        except Exception:
            pass
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
        # Honour a stop request (/stop or 3 consecutive denials) — abort the
        # tool loop between steps instead of continuing to call tools.
        if chat_id:
            try:
                from core.auth_gate import is_stop_requested
                if is_stop_requested(chat_id):
                    print(f"[two_stage] stop requested — aborting tool loop at step {step}", flush=True)
                    return _nemotron_synthesize_answer(
                        _original_query, "(stopped by user)", all_tool_results,
                        send_line, _system_prompt, _history_context, max_new_tokens
                    )
            except Exception:
                pass
        try:
            inputs = server_state.tool_calling_processor.apply_chat_template(
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
            _tc_param = next(server_state.tool_calling_model.parameters())
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
                    out = server_state.tool_calling_model.generate(
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
                        pad_token_id=server_state.tool_calling_processor.eos_token_id,
                    )
            finally:
                _mark_activity_end()

            response_raw = server_state.tool_calling_processor.decode(out[0][input_len:], skip_special_tokens=False)
            response_clean = server_state.tool_calling_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
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
            if server_state.is_nemotron:
                result = _nemotron_infer(chat, max_new_tokens=_mnt)
            elif _vllm.is_enabled():
                p = _build_chat_prompt(chat, enable_thinking=False)
                result = _vllm.infer(p, max_new_tokens=_mnt)
            else:
                import torch
                inputs = server_state.processor.apply_chat_template(
                    chat, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True,
                    enable_thinking=False,
                ).to(_target_device())
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = server_state.model.generate(**inputs, max_new_tokens=_mnt, do_sample=True, temperature=0.7)
                result = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
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
        if server_state.is_nemotron:
            result = _nemotron_infer(chat, max_new_tokens=_mnt)
        elif _vllm.is_enabled():
            p = _build_chat_prompt(chat, enable_thinking=False)
            result = _vllm.infer(p, max_new_tokens=_mnt)
        else:
            import torch
            inputs = server_state.processor.apply_chat_template(
                chat, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
                enable_thinking=False,
            ).to(_target_device())
            input_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = server_state.model.generate(**inputs, max_new_tokens=_mnt, do_sample=True, temperature=0.7)
            result = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

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
    # (its chat template understands tool-call XML). When native_agentic is set
    # (model.native_agentic + our FC adapter), Nemotron drives the FULL tool
    # loop itself via the single-model loop below — no Qwen detour, no
    # Nemotron-as-synthesis-only. Otherwise Nemotron stays on the tested
    # two-stage (Qwen -> Nemotron synthesis) route.
    #
    # Native-agentic models (any-to-any like Qwen2.5-Omni, or Gemma 4 with
    # parse_response — and Nemotron with native_agentic=true) handle the ENTIRE
    # flow on their own and skip the two-stage pipeline.
    if not server_state.native_agentic:
        _two_stage_result = _run_two_stage_if_available(params, send_line)
        if _two_stage_result is not None:
            return _two_stage_result

    # Native agentic models fall through to the single-model loop below, which
    # degrades gracefully when parse_response isn't available. Only non-native
    # models with no tool support go straight to plain inference. Janus is a
    # VLM without a standard tool loop, so it also goes to plain inference
    # (which routes to _janus_infer_text).
    if not server_state.model_supports_tools and (not server_state.native_agentic or server_state.is_janus):
        return _handle_infer_plain(params)

    messages = params["messages"]
    raw_tools = params.get("tools", [])
    workspace = os.path.expanduser(params.get("workspace", "~/.kernel-evolving/workspace"))
    max_steps = params.get("max_steps", 30)
    enable_thinking = params.get("enable_thinking", False)
    adapter_path = params.get("adapter_path")
    # Passed explicitly to execute_tool_with_meta below — see R5/T4.
    chat_id = params.get("chat_id", "")
    # A new task is starting — clear any stale transient auth state
    # (approve-all grant, deny counter, stop signal) from a previous task in
    # this chat so it cannot leak across tasks.
    if chat_id:
        try:
            from core.auth_gate import clear_task_state
            clear_task_state(chat_id)
        except Exception:
            pass
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
    _schema_invalid_sig: str = ""  # last schema-invalid tool+args sig (T5 steering net)
    _schema_invalid_repeat: int = 0  # consecutive repeats of that sig

    for step in range(max_steps):
        # Honour a stop request (/stop or 3 consecutive denials) — abort the
        # tool loop between steps instead of continuing to call tools.
        if chat_id:
            try:
                from core.auth_gate import is_stop_requested
                if is_stop_requested(chat_id):
                    print(f"[tool_loop] stop requested — aborting tool loop at step {step}", flush=True)
                    return {"type": "result", "result": "(stopped by user)"}
            except Exception:
                pass
        try:
            with _use_adapter(adapter_path):
                if _vllm.is_enabled():
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
                        response_raw = _vllm.run(_vllm.generate_async(prompt, sp))
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
                        inputs = server_state.processor.apply_chat_template(
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
                        text = server_state.processor.apply_chat_template(
                            current_messages, tools=tools_openai, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                        )
                        inputs = server_state.processor(text=text, return_tensors="pt").to(_target_device())

                    input_len = inputs["input_ids"].shape[-1]
                    _mark_activity_start()
                    try:
                        with torch.no_grad():
                            _gen_kwargs = dict(
                                max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=0.95, top_k=64,
                            )
                            if server_state.drafter is not None:
                                _gen_kwargs["assistant_model"] = server_state.drafter
                            try:
                                if server_state.is_nemotron:
                                    # Nemotron returns (out_ids, nfe); use AR for tool loops
                                    out_ids, _nfe = server_state.model.ar_generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
                                    out = out_ids
                                    print(f"[tool_loop/nemotron] AR NFE={_nfe}", flush=True)
                                else:
                                    out = server_state.model.generate(**inputs, **_gen_kwargs)
                            except Exception as e:
                                import logging
                                logging.getLogger(__name__).error(f"[tool_loop/hf] generate error at step {step}: {e}")
                                print(f"[tool_loop/hf] generate error at step {step}: {e} — aborting", flush=True)
                                return {"type": "result", "result": f"(HF generate error: {e})"}
                    finally:
                        _mark_activity_end()
                    response_raw = server_state.processor.decode(out[0][input_len:], skip_special_tokens=False)
                    try:
                        parsed = server_state.processor.parse_response(response_raw)
                    except (AttributeError, NotImplementedError):
                        plain = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
                        parsed = {"content": plain}
        except Exception as e:
            return {"type": "result", "result": f"(adapter error: {e})"}

        tool_calls = parsed.get("tool_calls", [])

        # ── Fallback: detect <tool>name(args)</tool> text patterns the model sometimes
        # emits instead of native function calls. Parse them into tool_calls format.
        if not tool_calls:
            raw_content = parsed.get("content", "") or ""
            # Timestamped raw-emission capture — the ONLY place we see the model's
            # actual output before any parse/fallback. Tells us if it emits empty
            # args or produces args the parser then drops.
            from datetime import datetime as _dbg_dt
            print(f"[tool_loop/raw {_dbg_dt.now().strftime('%H:%M:%S')}] RAW={raw_content!r}", flush=True)
            # Strip thinking block first
            if "<think>" in raw_content and "</think>" in raw_content:
                raw_content = raw_content[raw_content.index("</think>") + len("</think>"):].strip()
            import re as _re

            # Nemotron XML: <function_calls><invoke>...</invoke></function_calls>
            if server_state.is_nemotron and "<function_calls>" in raw_content:
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

            if not tool_calls:
                _fn_calls = _parse_function_eq_xml(raw_content)
                if _fn_calls:
                    print(f"[tool_loop] function= fallback: {len(_fn_calls)} call(s)", flush=True)
                    tool_calls = _fn_calls

            # Malformed-but-recoverable Nemotron FC recovery (captured corrupt
            # emissions that the strict parser drops to {}). Runs after the
            # strict parser and only fills in what it missed, so clean
            # emissions are untouched.
            if not tool_calls or any(
                not (f.get("function") or {}).get("arguments")
                for f in tool_calls
            ):
                try:
                    from core.inference.fc_recovery_parser import recover_function_calls
                    _rec_calls = recover_function_calls(raw_content)
                    if _rec_calls:
                        print(
                            f"[tool_loop] fc-recovery: {len(_rec_calls)} call(s) salvaged"
                            f" from malformed XML", flush=True,
                        )
                        # Prefer recovered calls that have real args; keep any
                        # clean calls the strict parser already got.
                        _merged = []
                        for _m in _rec_calls:
                            if any(_m["function"]["name"] == f.get("function", {}).get("name")
                                   and (f.get("function") or {}).get("arguments")
                                   for f in tool_calls):
                                continue  # strict parser already has clean args for this tool
                            _merged.append(_m)
                        if _merged:
                            tool_calls = _merged
                except Exception as _re:
                    print(f"[tool_loop] fc-recovery skipped ({_re})", flush=True)

            # Generic <tool>name(args)</tool> fallback
            if not tool_calls:
                _tag_calls = _parse_tag_fallback(raw_content)
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
        # Cap how many tool calls we process per model response. v7 sometimes emits
        # a degenerate burst of dozens of garbage/near-empty calls (observed 45 from
        # one 8192-NFE response). Each distinct call has its own signature so the
        # repeat/tally guards never fire, and the loop grinds on past the client's
        # socket timeout (300s), which then closes the pipe. Bail fast instead.
        _MAX_CALLS_PER_RESPONSE = 8
        if len(tool_calls) > _MAX_CALLS_PER_RESPONSE:
            print(
                f"[tool_loop] degenerate response: {len(tool_calls)} tool calls in one step "
                f"(cap {_MAX_CALLS_PER_RESPONSE}) — truncating",
                flush=True,
            )
            tool_calls = tool_calls[:_MAX_CALLS_PER_RESPONSE]
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

            # ── T1 (critique design): schema-validate args before executing ──
            # If required keys are missing/typo'd or types are wrong, do NOT execute
            # with broken args. Record a schema failure whose reason is the
            # structured repair hint (T2) so the re-prompt tells the model exactly
            # which key and what the schema expects, instead of a generic "retry".
            _ccfg = _critique_cfg()
            _schema_problems = []
            if _ccfg["enabled"] and _ccfg["schema_validate"]:
                try:
                    from core.tool_arg_utils import validate_tool_args
                    _schema_problems = validate_tool_args(tool_name, tool_args)
                except Exception:
                    _schema_problems = []  # fail open — never block execution on a validator bug
            if _schema_problems:
                result_str = _build_schema_repair_hint(
                    tool_name, tool_args, _schema_problems, _original_query
                )
                _last_tool_result = result_str
                _ok = False
                _status = "schema_invalid"
                _reason = result_str
                step_fail_count += 1
                step_failures.append((tool_name, _status, _reason))
                print(
                    f"[tool_loop] step {step+1} SCHEMA-INVALID tool={tool_name} "
                    f"problems={_schema_problems} — not executing", flush=True,
                )
                _step_payload = {
                    "type": "step", "step": step + 1,
                    "tool": tool_name, "args": tool_args, "result": result_str,
                    "ok": False, "status": _status, "failure_reason": _reason,
                    "duration_ms": 0, "schema_invalid": True,
                }
                send_line(json.dumps(_step_payload))
                tool_responses.append({"name": tool_name, "result": result_str})
                # ── Step 5: feed schema-invalid (unresolvable-at-T1) steps into the
                # trajectory flywheel. This becomes DPO/next-finetune corpus: a
                # (task, schema, faulty-args -> repair-hint) negative example.
                # Fail open — never let trajectory logging break the tool loop.
                try:
                    from core.evolution.trajectory_collector import get_collector
                    _coll = get_collector()
                    if _coll._enabled:
                        _coll.record(
                            task=_original_query,
                            provider="nemotron",
                            model_name="Nemotron-Labs-Diffusion-3B",
                            call_type="tool_schema_invalid",
                            tool_calls=[{"function": {"name": tool_name, "arguments": tool_args}}],
                            final_reply=result_str,
                            artifacts=None,
                            critic_score=0.0,
                            critic_verdict="SCHEMA_INVALID",
                        )
                except Exception:
                    pass
                # ── T5 steering net: a fixated invalid tool+args (e.g. web_search({})
                # emitted repeatedly with no args) must break early instead of
                # burning every step on `continue` — otherwise it runs to
                # (max steps reached) with no synthesized answer.
                _sig_inv = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                if _sig_inv == _schema_invalid_sig:
                    _schema_invalid_repeat += 1
                    if _schema_invalid_repeat >= _REPEAT_LIMIT:
                        print(
                            f"[tool_loop] schema-invalid guard: {tool_name}({_sig_inv}) "
                            f"repeated {_schema_invalid_repeat}x invalid — breaking",
                            flush=True,
                        )
                        return {"type": "result", "result": (
                            "I could not complete that request — the web fetch "
                            "kept being attempted without a search query. "
                            "Please rephrase."
                        )}
                else:
                    _schema_invalid_sig = _sig_inv
                    _schema_invalid_repeat = 1
                continue

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

                # ── Validation stop (Fabio) ─────────────────────────────
                # The result came back successfully — if it already answers
                # the user's single look-up/read request, STOP here instead of
                # letting the model re-call the same tool (T2 read_file / T4 ps
                # used to loop 3x on the blind repetition counter).
                if _tool_result_answers_query(_original_query, tool_name, result_str):
                    _val_msg = _synthesize_from_tool_result(_original_query, result_str)
                    print(
                        f"[tool_loop] validation stop: {tool_name} result answered the query — stopping",
                        flush=True,
                    )
                    return {"type": "result", "result": _val_msg}
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
        if server_state.is_nemotron:
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

    # ── Janus path (custom MultiModalityCausalLM — not standard generate) ───
    if server_state.is_janus:
        try:
            result = _janus_infer_image(image_path, prompt, max_new_tokens=max_new_tokens)
            return {"result": result}
        except Exception as e:
            print(f"[model_server] Janus image inference failed ({e})", flush=True)
            return {"error": f"Janus vision failed: {e}"}

    # ── Cloud vision routing (only when NOT forced local) ─────────────────────
    if not force_local:
        if server_state.config is None:
            _load_config(server_state.lazy_config_path)
        provider_cfg = (server_state.config or {}).get("providers", {})
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

    if not server_state.audio_capable:
        # Main model is text-only — ensure Gemma 4 vision slot is loaded
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_image: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        if server_state.mm_model is not None and server_state.mm_processor is not None:
            print("[model_server] infer_with_image: routing to audio/vision slot (Gemma 4)", flush=True)
            active_model = server_state.mm_model
            active_processor = server_state.mm_processor
            use_hf_path = True
    elif server_state.mm_model is not None and server_state.mm_processor is not None:
        # Main model IS audio-capable but we still have a warm slot — prefer it
        active_model = server_state.mm_model
        active_processor = server_state.mm_processor
        use_hf_path = True

    if active_model is None:
        # Main model handles vision (or nothing else is available)
        _ensure_model()
        # If the main model is loaded, use it as the processor fallback (it may
        # have a processor even when the dedicated multimodal slot is absent).
        if active_processor is None and server_state.model is not None and server_state.processor is not None:
            active_model = server_state.model
            active_processor = server_state.processor
            use_hf_path = False

    # Guard: if the multimodal slot load left partial state behind, degrade
    # cleanly instead of crashing on active_model.parameters() or processor use.
    if active_model is None or active_processor is None:
        return {"error": "no vision model available (multimodal slot failed to load)"}

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

    if _vllm.is_enabled():
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
            result = _vllm.run(_vllm.generate_multimodal_async(vllm_input, sp))
            return {"result": result.strip()}
        except Exception as e:
            print(f"[model_server] vLLM image inference failed ({e}) — falling back to HF", flush=True)
            if server_state.model is None:
                return {"error": f"vLLM failed and HF model not loaded: {e}"}

    # HF path
    import torch
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = server_state.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = server_state.processor(text=text, images=[img], return_tensors="pt").to(_target_device())
    input_len = inputs["input_ids"].shape[-1]
    if server_state.is_omni:
        # Omni's generate() is non-standard — use generation_mode="text" so it
        # returns text tokens fast instead of doing slow audio synthesis.
        out = _omni_generate_text(inputs, max_new_tokens=max_new_tokens)
    else:
        with torch.no_grad():
            _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if server_state.drafter is not None:
                _gen_kwargs["assistant_model"] = server_state.drafter
            out = server_state.model.generate(**inputs, **_gen_kwargs)
    result = server_state.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    return {"result": result}


def _handle_infer_with_audio(params: dict) -> dict:
    """Multimodal audio inference.
    NOTE: Uses HF transformers for audio tensor inference (stable path).
    vLLM 0.21.0 supports Gemma4ForConditionalGeneration but multimodal audio
    input via vLLM API is deferred — HF path is simpler and well-tested.

    IMPORTANT: do NOT call _ensure_model() up front here. That loads the main
    text model (e.g. Nemotron), which is NOT audio-capable and fails with the
    VRAM guard when the Gemma multimodal slot is already resident. STT/audio
    routes to the Gemma multimodal slot (or a named audio slot) instead; the
    main model is only loaded in the branch where it is actually the
    audio-capable model.
    """
    # ── Cloud audio routing ──────────────────────────────────────────────────
    if server_state.config is None:
        _load_config(server_state.lazy_config_path)
    provider_cfg = (server_state.config or {}).get("providers", {})
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
    _use_slot_name = _requested_slot or ("audio" if server_state.slot_registry is not None else None)
    _slot_state = server_state.slot_registry.get(_use_slot_name) if (server_state.slot_registry is not None and _use_slot_name) else None

    if _slot_state is not None:
        active_model = _slot_state.model
        active_processor = _slot_state.processor
        print(f"[model_server] infer_with_audio: using named slot {_use_slot_name!r}", flush=True)
    elif server_state.audio_capable and not _vllm.is_enabled():
        # Legacy: main model is audio-capable on HF path. Only now do we load
        # the main model — it is the audio-capable model in this config.
        _ensure_model()
        active_model = server_state.model
        active_processor = server_state.processor
        print(f"[model_server] infer_with_audio: using main model (HF, audio_capable)", flush=True)
    else:
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_with_audio: multimodal slot load failed ({e}) — falling back to main model", flush=True)
        active_model = server_state.mm_model
        active_processor = server_state.mm_processor
        print(f"[model_server] infer_with_audio: using multimodal slot (HF)", flush=True)

    # Guard: if the multimodal slot load left partial state behind, degrade
    # cleanly instead of crashing on active_model.parameters() or processor use.
    if active_model is None or active_processor is None:
        return {"error": "no audio model available (multimodal slot failed to load)"}

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
    target_device = _target_device() if active_model is server_state.model else next(active_model.parameters()).device
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
        if active_model is server_state.model and server_state.is_omni:
            # Omni any-to-any: text-only STT via generation_mode="text"
            print(f"[infer_with_audio] Omni STT generating (text mode)...", flush=True)
            out = _omni_generate_text(inputs, max_new_tokens=max_new_tokens)
        else:
            with torch.no_grad():
                _gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
                if active_model is server_state.model and server_state.drafter is not None:
                    _gen_kwargs["assistant_model"] = server_state.drafter
                print(f"[infer_with_audio] Generating...", flush=True)
                out = active_model.generate(**inputs, **_gen_kwargs)
            print(f"[infer_with_audio] Generation complete, output shape: {out.shape}", flush=True)
    finally:
        _mark_activity_end()
    result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
    print(f"[infer_with_audio] Result: {repr(result[:100])}", flush=True)
    return {"result": result}


def _handle_infer_local(params: dict) -> dict:
    """Local-slot text inference for Think-at-Rest.

    Runs plain-text chat inference against a named local model slot (default
    "audio" = Gemma 4 E2B-it), independent of cloud provider routing. This is
    what lets idle thoughts run on a resident/lazy-loadable local model even
    when `task_inference` is a cloud provider (e.g. `hf`).

    The slot is lazy-loaded on first use via _ensure_multimodal_slot() (same
    pattern as vision/STT). Do NOT call _ensure_model() here — that would load
    the cloud-only primary text model, which is unnecessary and can fail when
    the main model path is a cloud-only config.
    """
    slot = params.get("slot", "audio")
    messages = params.get("messages", [])
    max_new_tokens = params.get("max_new_tokens", 8192)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return {"error": f"'messages' must be a list, got {type(messages).__name__}"}

    # ── Resolve the slot model/processor ─────────────────────────────────────
    active_model = None
    active_processor = None

    # Prefer a registered named slot (e.g. "audio" wired into the SlotRegistry).
    if server_state.slot_registry is not None:
        try:
            _state = server_state.slot_registry.get(slot)
            if _state is not None:
                active_model = _state.model
                active_processor = _state.processor
                print(f"[model_server] infer_local: using named slot {slot!r}", flush=True)
        except Exception as _e:
            print(f"[model_server] infer_local: slot {slot!r} lookup failed ({_e})", flush=True)

    # Fallback: lazy-load the multimodal slot (Gemma 4 E2B-it).
    if active_model is None or active_processor is None:
        try:
            _ensure_multimodal_slot()
        except Exception as e:
            print(f"[model_server] infer_local: multimodal slot load failed ({e})", flush=True)
        if server_state.mm_model is not None and server_state.mm_processor is not None:
            active_model = active_model or server_state.mm_model
            active_processor = active_processor or server_state.mm_processor
            print("[model_server] infer_local: using multimodal slot (Gemma 4)", flush=True)

    if active_model is None or active_processor is None:
        return {"error": "no local slot available (multimodal slot failed to load)"}

    # ── Plain-text chat inference against the slot ───────────────────────────
    import torch
    try:
        text = active_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception as e:
        return {"error": f"apply_chat_template failed: {e}"}

    target_device = next(active_model.parameters()).device
    target_dtype = next(active_model.parameters()).dtype
    inputs = active_processor(text=text, return_tensors="pt").to(target_device)
    # Align floating tensors to model dtype to avoid BFloat16/Float mismatch.
    for k, v in list(inputs.items()):
        if hasattr(v, "dtype") and hasattr(v, "to"):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=target_device, dtype=target_dtype)
            else:
                inputs[k] = v.to(device=target_device)
    input_len = inputs["input_ids"].shape[-1]

    _mark_activity_start()
    try:
        with torch.no_grad():
            _gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=64,
            )
            out = active_model.generate(**inputs, **_gen_kwargs)
    finally:
        _mark_activity_end()
    result = active_processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
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


# ---------------------------------------------------------------------------
# Slot management handlers
# ---------------------------------------------------------------------------

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

