# Fix Spec — Qwen3.5-0.8B Tool-Calling Sampling & Dead `use_raw_api` Kwarg

**Date:** 2026-08-06
**Supersedes:** Corrects R1 and R3 diagnosis in [AUDIT-2026-08-06-inference-pipeline.md](../audits/AUDIT-2026-08-06-inference-pipeline.md)
**Related tasks:** [tasks.md](../tasks.md) — T2, T3, T5, T6

## Corrected Diagnosis

The original audit assumed the two-stage pipeline fails because (a) `parse_qwen_tool_calls()`
doesn't match Qwen3.5's actual tool-call output format, and (b) the model may not reliably
support tool calling at 0.8B scale. Both assumptions are wrong. Verified against the model's
own chat template (fetched directly from `Qwen/Qwen3.5-0.8B/raw/main/chat_template.jinja` on
Hugging Face) and its model card:

1. **The parser is correct.** When `tools=` is passed to `apply_chat_template()`, the template
   auto-injects a system block instructing the model to emit:
   ```
   <tool_call>
   <function=NAME>
   <parameter=KEY>
   value
   </parameter>
   </function>
   </tool_call>
   ```
   `parse_qwen_tool_calls()` Pattern 2 (`_two_stage_helpers.py`) already parses exactly this
   format. No parser change is needed.

2. **`use_raw_api=True` (passed to `apply_chat_template()` in `_run_two_stage_if_available()`,
   `model_server.py` ~line 1560) is a dead no-op kwarg.** It never appears anywhere in the
   Jinja chat template source. It is a `qwen_agent.llm` Python library config field (for the
   Qwen-Agent wrapper talking to a remote OpenAI-compatible server), lifted from the model
   card's "Agentic Usage" example and misapplied as a chat-template argument. It has zero
   effect on the rendered prompt — extra unknown kwargs to `apply_chat_template()` just become
   unused Jinja context variables. Harmless, but misleading; should be removed.

3. **Sampling settings are not "too noisy" relative to the model card** — the code
   (`temperature=1.0, top_p=1.0, top_k=20, repetition_penalty=1.0`) matches the model card's
   general "non-thinking mode for text tasks" preset almost exactly, missing only
   `presence_penalty=2.0` (which `transformers.generate()` doesn't natively support — that
   parameter only exists in vLLM/SGLang OpenAI-compatible serving layers, not raw HF
   `.generate()`).
   However, this preset is a **general-purpose chat recommendation** for the whole Qwen3.5
   family (0.8B through much larger variants), not a structured-output/tool-calling profile.
   At 0.8B scale, `temperature=1.0` sampling among the top-20 candidates is enough noise to
   derail precise multi-token XML tag emission (`<tool_call>`, `<function=...>`,
   `<parameter=...>`) over a long sequence — producing coherent-but-wrong free text instead of
   malformed tags. This matches the observed symptom exactly: fluent Portuguese/Italian
   smalltalk and refusals, not garbled/truncated tool-call XML.

## Root Cause (corrected)

Not a format mismatch. The Qwen tool-calling stage uses a general-purpose creative-chat
sampling profile for what is effectively a structured-decision task. Combined with a dead
`use_raw_api` kwarg (cosmetic, no functional impact), this is the likely reason zero tool
calls are observed in production logs.

## Fix

1. Remove the dead `use_raw_api=True` kwarg from the `apply_chat_template()` call in
   `_run_two_stage_if_available()`.
2. Use a tighter, lower-temperature/lower-top-p sampling profile for the Qwen tool-decision
   generation call specifically (this stage only — Nemotron synthesis sampling is untouched).
   This is a deliberate, justified deviation from the model card's general chat preset, applied
   because this call is a structured tool-selection decision, not open-ended chat.
   Applied values: `temperature=0.4, top_p=0.3, top_k=20, repetition_penalty=1.05` (top_p
   tightened further from an initial 0.8 to 0.3 after manual review — favors reliability of
   XML tag emission over output diversity, appropriate for a structured-decision task).
3. Add/keep debug logging of raw Qwen output (already present) to verify tool-call XML appears
   after the change; do not remove logging until verified in a live smoke test.

## Non-goals for this fix

- No change to `parse_qwen_tool_calls()` — confirmed correct.
- No change to `build_qwen_system_prompt()` — confirmed minimal/correct, does not conflict with
  the template's auto-injected tool instructions.
- Missing assistant tool_call message (R3, separate task T2) is a distinct, real bug — not
  addressed by this fix.

## Verification

- Live smoke test: send a tool-requiring query, confirm `model_server.log` shows
  `[two_stage] step N: <tool_name>(...)` instead of `[two_stage] Qwen final answer after 0
  step(s)` with plain-text smalltalk.
- Re-run with 5-10 varied tool-requiring prompts to confirm consistency, not a one-off sampling
  fluke.
