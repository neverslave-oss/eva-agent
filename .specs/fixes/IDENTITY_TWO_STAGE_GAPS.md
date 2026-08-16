# Bug: Two-Stage Pipeline Identity & Intermittent Failures

**Filed:** 2026-06-18  
**Last investigated on:** v1.28.3 (commit a4c9be0)  
**Priority:** High — core persona/identity breaks on cold boots and certain query types  
**Status:** Partially fixed, still has failure modes

---

## Root Cause

Message routing in `model_server.py` has **two LLM inference paths**, and the fallback path loses the full system prompt (and thus Kernel-Evo's identity, personality, first-contact 🐬 greeting, and behavioral rules).

### Path 1 — Two-Stage Pipeline (Qwen3.5-0.8B → Nemotron) ✅

`_run_two_stage_if_available()` → Qwen handles tool loop → Nemotron synthesizes answer with system prompt

This works **when available and stable**.

### Path 2 — Fallback/Plain ✗

When `_run_two_stage_if_available()` returns `None`, `_handle_infer_with_tools()` falls back to either:
- `_handle_infer_plain()` (if `_model_supports_tools` is False), or
- The single-model Nemotron tool loop (if `_model_supports_tools` is True)

**Both fallbacks can lose the system prompt** if messages are constructed without one — e.g. when `model_client.infer()` (plain, no tools) is called from a path that doesn't run through `agent.py`'s `build_system_prompt()`.

---

## Observed Symptoms

1. **Cold boot / first-contact**: Nemotron responds with generic pre-training — "I currently don't have access to a file system or a specific workspace..." instead of the 🐬 first-contact greeting with tool descriptions.

2. **Intermittent two-stage crashes**: The Qwen3.5-0.8B tool-calling slot sometimes fails to load (OOM or GPU memory fragmentation under 24GB). When the slot fails, `_ensure_tool_calling_slot()` sets `_tool_calling_slot_loaded = False` and `_run_two_stage_if_available` returns `None`. The fallback then runs Nemotron RAW with no system prompt.

3. **Plain `infer()` calls bypass identity**: `model_client.infer()` sends `method: "infer"` (no tools field). On the server, `_run_two_stage_if_available` returns `None` immediately because `raw_tools` is empty. Depending on `_model_supports_tools`, it routes to a path that may have no system prompt.

---

## Commit History

| Commit | Fix |
|---|---|
| `aabdad3` | Always route through Nemotron for conversation synthesis, even without tool results |
| `a4c9be0` | Pass full system prompt to Nemotron synthesis, not bare identity string |

Both fixes address the two-stage pipeline path only. The fallback paths still need hardening.

---

## Remaining Gaps

### 1. `_handle_infer_plain` — no system prompt guarantee

**File:** `model_server.py` lines 1324-1398

When `_model_supports_tools` is False and two-stage returns None, `_handle_infer_plain` is called with the raw `params["messages"]`. If the caller didn't include a system message, Nemotron gets no identity.

**Fix:** Add system prompt retrieval/fallback inside `_handle_infer_plain()`:

```python
# Before Nemotron inference, ensure system prompt exists
messages = params["messages"]
has_system = any(m.get("role") == "system" for m in messages)
if not has_system:
    messages.insert(0, {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT})
```

### 2. Single-model tool loop — lost identity on model_server restart

**File:** `model_server.py` lines 1697-1790+

When `_model_supports_tools` is True and two-stage returns None, the single-model tool loop runs. It preserves `current_messages` from params, but if the request came from `model_client.infer()` (no tools), the tool list is empty and it dead-ends into Nemotron with whatever messages it received.

**Fix:** Explicitly check for system prompt presence and inject default if missing, same as gap 1.

### 3. Two-stage Qwen slot crashes silently

**File:** `model_server.py` `_ensure_tool_calling_slot()` lines 579-588+

When the tool-calling slot fails to load (OOM, model not found, CUDA error), the error is printed but the system falls through silently. No user-visible error, no retry.

**Fix:** Add a retry mechanism (1 retry, 30s delay) and log a clear warning when falling back. Optionally: pre-check VRAM before attempting slot load.

### 4. `model_client.infer()` vs `infer_with_tools()` — caller confusion

**File:** `model_client.py`

Two methods send different methods (`infer` vs `infer_with_tools`), but both route to the same handler on the server. The only difference is the `tools` field in params. This is subtle and error-prone — removing `method: "infer"` and always going through `infer_with_tools` would eliminate one failure mode.

---

## Reproduction

1. Stop the model server
2. Kill the Qwen tool-calling slot cache
3. Start model server (Qwen slot fails or loads slowly)
4. Send any query via Telegram or API
5. Nemotron responds with generic pre-training response — no system prompt, no tools, no personality

---

## Suggested Fix (next pass)

1. **Centralize system prompt guard**: Before any Nemotron inference, ensure a system message exists. If not, inject a default.
2. **Add retry to `_ensure_tool_calling_slot()`**: If Qwen slot fails, retry once after 30s.
3. **Merge `infer` and `infer_with_tools` client methods**: Always send tools (even empty) so the server doesn't get confused.
4. **Add logging**: When falling back to plain Nemotron, log the decision and the first 200 chars of the input messages for debugging.

---

## Relevant Code Locations

| File | Lines | Function |
|---|---|---|
| `src/core/inference/model_server.py` | 1401-1480 | `_run_two_stage_if_available()` |
| `src/core/inference/model_server.py` | 1505-1600 | Two-stage main loop, returns None gate |
| `src/core/inference/model_server.py` | 1602-1695 | `_nemotron_synthesize_answer()` |
| `src/core/inference/model_server.py` | 1697-1790+ | `_handle_infer_with_tools()` fallback paths |
| `src/core/inference/model_server.py` | 1324-1398 | `_handle_infer_plain()` |
| `src/core/inference/model_client.py` | 78-80 | `infer()` sends `method: "infer"` no tools |
| `src/core/inference/model_client.py` | 99-102 | `infer_with_tools()` sends with tools |
| `src/core/agent.py` | 568-577 | `_prov.infer_with_tools(messages, TOOLS, ...)` — always passes tools |
