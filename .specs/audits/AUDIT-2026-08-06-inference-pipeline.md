# AUDIT: Inference Pipeline — Conversation Flow & Tool Calling

**Date:** 2026-08-06  
**Author:** Olly  
**Scope:** Full inference pipeline: `agent.triage()` → `provider.infer_with_tools()` → `model_client._call()` → `model_server._handle_infer_with_tools()` → two-stage (Qwen+Nemotron) or single-model (Nemotron) tool loop  
**Method:** Source code review + live model_server.log analysis + git history  
**Severity:** 🔴 Critical — two core features (conversation flow, tool calling) are fundamentally broken  

---

## Executive Summary

The inference pipeline has **two critical structural defects** that disable tool calling and degrade conversation flow:

1. **The two-stage pipeline (Qwen3.5-0.8B → Nemotron) is the default tool-calling path, but Qwen never emits tool calls.** Every log entry shows `[two_stage] Qwen final answer after 0 step(s)` with plain text (Portuguese/Italian smalltalk, profanity, or refusal). The result is a non-None dict → `_handle_infer_with_tools` returns it immediately → the single-model Nemotron tool loop **never runs**. Tools are functionally dead.

2. **The `infer` method dispatcher routes to `_handle_infer_with_tools` instead of `_handle_infer`.** This means every plain inference call (skills, fact extraction, critic, planning, trajectory teacher) runs through the tool-loop path with `tools=[]`. Dedicated handlers (`_handle_infer`, `_handle_infer_plain`) are dead code. Critic calls with `max_new_tokens=16` generate 8192 tokens.

---

## Architecture Map (as-built)

```
Telegram / API
  │
  ▼
api.py: POST /message ──────→ agent.triage()
  │
  ▼
triage():
  1. Build system_prompt (build_system_prompt — ~4-5k tokens, AGENTS.md + USER.md + live services + skills table + 60 skills)
  2. Load history (JSON hot-window + SQLite, char-trimmed to ~153k chars)
  3. Semantic retrieval (DEAD CODE — wrong import: `from embedding_client import cosine_similarity` raises ModuleNotFoundError)
  4. Build messages = [system, *history, user]
  5. provider.infer_with_tools(messages, TOOLS)
  │
  ▼
provider.py :: InferenceProvider
  ├─ get_provider("task_inference") → "local" (config)
  ├─ Thermal fallback → N/A (temp < 85°C)
  └─ _keyless check → openai/anthropic/openrouter/hf/copilot skipped if no API key
  │
  ▼
model_client.py :: infer_with_tools()
  JSON-RPC over Unix socket: {"method": "infer_with_tools", "params": {...}}
  │
  ▼
model_server.py :: dispatch()
  ┌─ method == "infer"           → _handle_infer_with_tools ← BUG: should be _handle_infer
  ├─ method == "infer_draft"     → _handle_infer_draft
  ├─ method == "infer_with_tools" → _handle_infer_with_tools
  └─ method == "infer_with_image" → _handle_infer_with_image
  │
  ▼
_handle_infer_with_tools()
  ├─ 1. _run_two_stage_if_available()
  │     └─ Qwen3.5-0.8B loaded → YES → Stage 1
  │        ├─ Qwen generate (1024 tokens, temp=1.0, top_k=20)
  │        ├─ parse_qwen_tool_calls ← NEVER emits tool calls (log evidence)
  │        └─ "final answer" → _nemotron_synthesize_answer()
  │           └─ Returns dict → _handle_infer_with_tools returns it ← fallback blocked
  │
  └─ 2. (never reached when Qwen slot loaded) Single-model Nemotron tool loop
        ├─ apply_chat_template with tools=11
        ├─ ar_generate (8192 tokens)
        └─ tool calls → execute_tool_with_meta
```

---

## Root Causes

### 🔴 R1: Two-stage pipeline blocks tool calling (CRITICAL)

**Evidence (model_server.log):**
```
[model_server] SlotRegistry built with 3 slot(s)
[model_server] Loading tool_calling slot: ...Qwen3.5-0.8B...
[model_server] Tool-calling slot wired into SlotRegistry
[two_stage] Stage 1: Qwen tool-calling loop
[DEBUG two_stage] step 0: 137 tokens
[DEBUG two_stage] step 0: raw response: Understood! If you're just testing the UI...
[two_stage] Qwen final answer after 0 step(s)
```
Repeated across 50+ requests — **zero tool calls ever emitted by Qwen3.5-0.8B**. Typical outputs:
- `"Eu não tenho acesso a um sistema de tradução..."` (refusal pattern in Portuguese)
- `"Ma vaffan*** porco ***! Mas como?"` (gibberish/profanity)
- `"O que você tem na mão? O que podemos fazer?"` (smalltalk, no tool call)

**Why Qwen doesn't call tools:**

| Cause | Code Location | Evidence |
|-------|--------------|----------|
| Model too small / untrained for tools | `_run_two_stage_if_available()` at line 1450 | 0.8B at temp=1.0, top_k=20 fails to emit structured tool XML |
| Missing assistant tool_call message | same fn, after `for tr in all_tool_results[...]` | Only `{"role":"tool","content"}` appended; no `{"role":"assistant","tool_calls":...}` preceding it |
| `parse_qwen_tool_calls` format mismatch | `_two_stage_helpers.py:parse_qwen_tool_calls` | Only handles `<tool_call>` and `<function=name>` XML; Qwen3.5 raw API emits `<\|tool_call\|>` (pipe-delimited) — not matched |
| Qwen context too small | `qwen_history = non_sys[-4:]` | Only 2 turns of context; no persona/system prompt beyond minimal instruction |
| generate params too noisy | `temperature=1.0, top_p=1.0, top_k=20, repetition_penalty=1.0` | 0.8B at these settings drifts into language-detection smalltalk |

**Mechanism:** `_run_two_stage_if_available()` returns a dict (non-None) on Qwen's plain-text output → `_handle_infer_with_tools` returns it at line 1794:
```python
_two_stage_result = _run_two_stage_if_available(params, send_line)
if _two_stage_result is not None:
    return _two_stage_result  # ← Never reaches the single-model Nemotron tool loop
```
The single-model fallback (which actually executes tools and has Plan 007 failure re-prompting) is **permanently blocked**.

### 🔴 R2: Dispatcher routes `infer` → `_handle_infer_with_tools` (wrong handler)

**Location:** `model_server.py` dispatch table, line 2947

```python
if method == "infer":
    resp = _handle_infer_with_tools(params, send_line)  # ← BUG
```

**Consequences:**
- `_handle_infer` (dedicated plain handler at line 1280) is **dead code** (never dispatched)
- `_handle_infer_plain` (line 1357) is only called for `not _model_supports_tools` — Nemotron always has `_model_supports_tools = True` → never called
- Every plain `infer()` call (skills, fact extraction, critic, planning, trajectory teacher) runs through the tool loop with `tools=[]` → `apply_chat_template(messages, tools=[])` — may render empty tools section
- **`max_new_tokens` is ignored** — `_handle_infer_with_tools` hardcodes 8192 tokens everywhere. Critic calls meant for `max_new_tokens=16` generate 8192 tokens (latency explosion)
- **Default system prompt injected** for every plain infer that lacks a system message: the log shows `[tool_loop] injected default system prompt (was missing)` — fact extractor prompts get "You are Kernel-Evo..." identity, altering their behavior

### 🔴 R3: Missing assistant tool_call message in two-stage loop

**Location:** `_run_two_stage_if_available()` after tool execution

```python
# Only tool results are appended — NO assistant message!
for tr in all_tool_results[-len(tool_calls):]:
    current_messages.append({"role": "tool", "content": tr["result"]})
```

Compare to the correct single-model loop (`_handle_infer_with_tools`, line 2035):
```python
if _is_nemotron:
    current_messages.append({"role": "assistant", "content": raw_assistant_text})
    for tr in tool_responses:
        current_messages.append({"role": "tool", "content": tr["result"]})
```

The two-stage loop is missing the assistant turn containing the tool call. Qwen3.5's chat template with `use_raw_api=True` expects the OpenAI-compatible protocol: `assistant {tool_calls}` → `tool {content}`. Without it, tool results are orphaned and the conversation degrades.

### 🔴 R4: Semantic retrieval is dead code (wrong import)

**Location:** `agent.py` triage(), line ~200

```python
from embedding_client import cosine_similarity  # ← ModuleNotFoundError
```

`embedding_client` module does not exist at `src/` level. The correct module is `core.memory.embedding_client`. This import always raises `ModuleNotFoundError` → caught by `except Exception` → `_retrieved_context` stays empty → **semantic retrieval has never worked in production**.

Even if the import were fixed, the implementation does N sequential HTTP embedding calls per message (one per history turn), which would add 5-20s latency for 100+ turn histories.

### 🟠 R5: exec_shell auth gate bypassed in model_server

**Location:** `tools.py` execute_tool → `_current_chat_id` check

`tools._current_chat_id` is set by `triage()` in the **API process** (python3 -m uvicorn api:app). Tools execute inside the **model_server process** (python3 model_server.py, started separately by `start.sh`). `_current_chat_id` is a module-level global that starts `""` and is never set in the server process.

```python
_chat_id = _current_chat_id  # Always "" in model_server process
if _chat_id:
    from core.auth_gate import request_auth
    ...  # This code NEVER runs for model-server tool calls
```

**Security impact:** All `exec_shell` commands from the tool loop execute without the Telegram inline-button approval gate. The authorization gate added in commit 20a0bc2 is silently bypassed.

### 🟠 R6: Capability refusal markers incomplete

**Location:** `model_server.py` _looks_like_capability_refusal(), line 1900

```python
markers = (
    "cannot access the internet",
    "can't access the internet",
    "cannot browse",
    "can't browse",
    "no internet access",
    "i cannot perform web searches",
    "i can't perform web searches",
    "no access to external databases",
)
```

Missing critical denial patterns that both Qwen and Nemotron emit:
- `"i don't have access to files"` / `"can't access files"`
- `"i cannot execute commands"` / `"can't execute commands"`
- `"i don't have tools"` / `"no tools available"`
- `"these tools are not real"` / `"i cannot access real tools"`
- `"i'm just a language model"` / `"i'm a text model"`
- `"i cannot access your filesystem"` / `"cannot access the local filesystem"`

These denial patterns pass through as final answers and get saved to conversation history, poisoning subsequent turns.

### 🟠 R7: Repetition guards return raw tool output as final answer

**Location:** `_handle_infer_with_tools()` single-model loop, lines ~1980-2000

```python
return {"type": "result", "result": last_result}  # raw tool output, no synthesis
```

When the same tool call fires 3 times (repetition guard), the user gets the raw tool output (e.g., `ls -la` directory listing) as the entire reply, with no conversational framing. This is a UX break.

### 🟠 R8: Empty/refusal answers persist to history

**Location:** `agent.py` triage(), `_mem.save()` after inference

Both `""` (empty) and `"(max steps reached)"` and refusal text are saved as the assistant turn. These get re-injected as history on the next turn, poisoning the conversation.

**Two-stage dead-end path:** When Qwen loops 15 steps with no results:
```python
return _nemotron_synthesize_answer(original_query, "", all_tool_results, ...)
# → "I could not complete that request."  (hardcoded dead-end)
```

### 🟡 R9: Two-stage synthesis prompt duplicated identity

**Location:** `_nemotron_synthesize_answer()` + `build_nemotron_synthesis_prompt()`

When tool results exist:
1. `build_nemotron_synthesis_prompt()` returns a string that starts with `"You are Kernel-Evo, a helpful AI assistant..."` 
2. `_nemotron_synthesize_answer()` prepends the FULL system prompt (also "You are Kernel-Evo 🐬...")
3. Both are sent as `system` + `history` + `user(message containing identity)` → Nemotron sees conflicting/redundant identity instructions

### 🟡 R10: Debug prints remain in production

**Location:** `_run_two_stage_if_available()`, lines 1570-1575

```python
print(f"[DEBUG two_stage] step {step}: {token_count} tokens", flush=True)
print(f"[DEBUG two_stage] step {step}: raw response: {response_raw[:200]}", flush=True)
print(f"[DEBUG two_stage] step {step}: clean response: {response_clean[:200]}", flush=True)
```

These dump raw model output to the server log on every request. Presumably from development; they expose model internals in log output.

### 🟡 R11: Tool-arg sanitization layered across 3 files

- `model.py`: `_sanitize_text()`, `_normalize_tool_args()`
- `model_server.py`: `_sanitize_text()`, `_normalize_tool_args()` (duplicate)
- `tools.py`: `_rewrite_date_tokens()`, `sanitize` in `execute_tool()` (strip newlines, shell redirects)

Three separate copies of argument sanitization, each with slightly different logic. This is a symptom of the model repeatedly emitting malformed tool arguments (`>\nvoice-clone`, `run_skill(list_routines(), [])`).

### 🟡 R12: `_handle_infer` and `_handle_infer_plain` dead code

Two dedicated handlers with proper Nemotron support (linear_spec fast mode, max_new_tokens from params, system prompt guard) are unreachable:
- `_handle_infer()` — 80 lines of Nemotron-aware inference with adapter support, VRAM-aware, vLLM fallback — **never dispatched**
- `_handle_infer_plain()` — system prompt guard, Nemotron fast path, vLLM fallback — only called when `not _model_supports_tools` (never for Nemotron)

---

## Impact Analysis

### Conversation Flow (User Experience)

| Symptom | Root Cause(s) | Frequency |
|---------|--------------|-----------|
| Agent answers in Portuguese/Italian instead of English | R1 — Qwen 0.8B language-drift at temp=1.0 | Every request |
| Agent says "I can't access files/tools" | R1, R6 — Qwen refusal + incomplete markers | Every tool-requiring request |
| Agent gives smalltalk instead of executing commands | R1 — Qwen never calls tools | Every request |
| Agent suddenly forgets conversation context | R1 — Qwen only sees last 4 messages (2 turns) | >2 turns |
| "I could not complete that request" | R1 — Qwen loops 15 steps → dead-end | Multi-step tasks |
| Agent gives raw command output as reply | R7 — Repetition guard returns raw tool result | Tool loops |
| Agent seems to ignore prior instructions | R1 — Qwen's minimal system prompt has no persona | Every request |
| Slow response latency | R4 — semantic retrieval dead but no effect; R2 — critic 16-token → 8192-token AR generation | Critic/planning calls |

### Tool Calling (Core Feature)

| Symptom | Root Cause(s) | Frequency |
|---------|--------------|-----------|
| Tools never execute | R1 — Qwen never emits tool calls → two-stage returns synthesis of plain text | Every request |
| `exec_shell` runs without approval | R5 — auth gate bypassed in model_server process | Every shell command |
| Critic/planning calls are 500x slower than needed | R2 — max_new_tokens ignored, 8192 tokens generated | Every critic/planning call |
| Skill execution with wrong identity | R2 — `infer` → tool loop → default system prompt injected | Skill calls, fact extraction |
| Semantic retrieval never works | R4 — wrong import path | Never worked |

### Security

| Finding | Severity | Detail |
|---------|----------|--------|
| exec_shell auth gate bypassed | 🔴 Critical | All shell commands from model_server execute without user approval |
| Model output logged to plaintext | 🟡 Low | DEBUG prints expose raw model output, including user messages |

---

## Fix Recommendations

### Phase 1 — Immediate (Fix the two-stage pipeline so tools work)

**Priority order — the most impactful fix has the highest priority.**

1. **Fix the two-stage tool-calling protocol** (`_run_two_stage_if_available`)
   - Append `{"role": "assistant", "content": response_clean + "\n" + tool_calls_xml}` before tool results
   - Or: use Qwen3.5's native `<|tool_call|>` format for the assistant turn
   - File: `src/core/inference/model_server.py`

2. **Fix dispatcher bug** (model_server.py dispatch table, line 2947)
   - Change `method == "infer"` → `_handle_infer` (not `_handle_infer_with_tools`)
   - `_handle_infer` already handles Nemotron, adapters, max_new_tokens, system prompt guard
   - This fixes: critic latency, identity injection, skill inference quality

3. **Fix the `infer` path in `_handle_infer` to use `_nemotron_infer` straight through** (it already does — just need to route to it)

4. **Verify Qwen3.5-0.8B tool-calling output format**
   - Check the actual format: pipe-style `<|tool_call|>` vs `<tool_call>` XML
   - Update `parse_qwen_tool_calls()` to match the actual Qwen3.5 output
   - Add `/debug/two-stage-raw` endpoint to inspect raw Qwen output without risking log dump

5. **Add `exec_shell` command approval markers to the model_server** (or pipe chat_id through the socket params)

### Phase 2 — Two-stage pipeline quality

6. **Reduce Qwen temperature** (0.7 max, top_k=40, repetition_penalty=1.08)
7. **Increase Qwen context** to last 6-8 messages (3-4 turns) for tool decisions
8. **Add a system prompt for Qwen** that includes tool-calling instructions in the correct language (English only, no multi-language drift)
9. **Add max_new_tokens parameter** to `_handle_infer_with_tools` and the two-stage loop

### Phase 3 — Conversation flow resilience

10. **Expand capability refusal markers** — add file-access, tool-denial, "I'm just a model" patterns
11. **Guard against empty/refusal answers persisting to history** — don't save result if it matches refusal patterns, empty string, or error prefix
12. **Fix semantic retrieval import** — `from core.memory.embedding_client import cosine_similarity`
13. **Add `chunk_callback` to two-stage Nemotron synthesis** — or at least real-time streaming from the model_server instead of 40-char synthetic chunks

### Phase 4 — Cleanup

14. **Remove DEBUG prints** from `_run_two_stage_if_available`
15. **Consolidate `_sanitize_text` and `_normalize_tool_args`** into a single shared module instead of 3 copies
16. **Remove dead code** — `_handle_infer` can be the dispatch target; `_handle_infer_plain` is unreachable

---

## Verification Plan

| Test | What to verify | Pass criterion |
|------|---------------|----------------|
| `test_tool_first_pipeline_basic` | "read ~/.kernel-evolving/workspace/AGENTS.md" → `read_file` tool called | Model server log shows `[tool_loop] step 1: read_file(...)` and response contains file content |
| `test_tool_first_pipeline_web_search` | "search for latest Python 3.13 news" → `web_search` tool called | Tool result contains web page content |
| `test_tool_first_pipeline_conversation` | "what was the file I asked you to read earlier?" (after a previous read_file) | Model reads from history (not re-asking) |
| `test_session_continuity_5_turns` | 5-turn conversation with tool calls, each referencing prior turn | All 5 turns show correct tool calls and correct references |
| `test_exec_shell_auth_gate` | exec_shell called from model_server | Auth gate fires (chat_id propagated) |
| `test_critic_max_new_tokens` | Critic call with max_new_tokens=16 | Model_server generates ~16 tokens, not 8192 |
| `test_semantic_retrieval_import` | `from core.memory.embedding_client import cosine_similarity` | No ImportError |
| `test_refusal_filter` | "I cannot access the internet" → filtered and re-prompted | Re-prompt fires, not returned as final answer |
| `test_refusal_filter_files` | "I don't have access to files" → filtered | Re-prompt fires |
| `test_two_stage_missing_assistant_turn` | Two-stage loop with Qwen tool call | Assistant message with tool_call appended before tool results |
| `test_two_stage_synthesis_no_duplicate_identity` | Nemotron synthesis prompt | No duplicate "You are Kernel-Evo" in system + user message |

---

## Open Questions

1. **What is the actual tool-calling output format of Qwen3.5-0.8B with `use_raw_api=True` + `tools=`?** The channel log shows `raw response` with `<|im_end|>` suffix — this implies `<|im_start|>/<|im_end|>` chat template tokens. The actual tool format might be `<|tool_call|>` blocks with JSON. Needs a direct test with `/debug/two-stage-raw` to confirm.

2. **If Qwen3.5-0.8B cannot reliably call tools, should the two-stage pipeline be disabled?** The single-model Nemotron tool loop (with proper Plan 007 failure re-prompting) works. The two-stage pipeline is an optimization that adds tool-calling quality from Qwen → but if Qwen never calls tools, it's a net negative. Consider making two-stage opt-in for models that demonstrably work (e.g., Qwen3.5-1.7B or larger).

3. **Should the model_server run inside the API process (in-process) instead of as a separate process?** The auth gate bypass, VRAM tracking, and `_current_chat_id` isolation are all cross-process issues. In-process would share state naturally. Disadvantage: API crash kills the model.

4. **Is the `infer` → `_handle_infer_with_tools` dispatcher a regression or original design?** The git blame shows `4cf183f refactor(core): move src/ modules to core/ sub-folders` — the dispatcher may have been lost during the refactor. The dedicated `_handle_infer` handler suggests it was intended to be the target.

---

## Appendix: Config Snapshot

```yaml
model:
  name: nvidia/Nemotron-Labs-Diffusion-3B
  path: ~/models/.../snapshots/0d51902...
  quantize: 4bit
  max_context_length: 65536

model_slots:
  primary:
    model_path: null  # = main model
  tool_calling:
    model_path: ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/...
    dtype: float16
    max_context_length: 32768
  audio:
    model_path: ~/models/.../gemma-4-E2B-it/...
    quantize: 4bit

providers:
  task_inference: local
  synthesis: openai        # model_override: deepseek/deepseek-v4-flash
  critic: openrouter       # model_override: deepseek/deepseek-v4-flash
  planning: openrouter     # model_override: deepseek/deepseek-v4-flash
  fallback_provider: openai
```

## Appendix: Evidence Logs

Critical log excerpt showing the two-stage failure pattern:

```
[model_server] SlotRegistry built with 3 slot(s)
[model_server] Loading tool_calling slot: ...Qwen3.5-0.8B...
[two_stage] Stage 1: Qwen tool-calling loop
[DEBUG two_stage] step 0: 137 tokens
[DEBUG two_stage] step 0: raw response: Understood! If you're just testing the UI...
[two_stage] Qwen final answer after 0 step(s)
```

Repeated 50+ times across different user prompts. Zero tool calls ever emitted.

Log excerpt showing the `infer` → tool loop misrouting:

```
[model_server] request method=infer params=
  "messages": [{"role": "user", "content": "You are a fact extractor..."}]
  "tools": []
[tool_loop] injected default system prompt (was missing)
[tool_loop/nemotron] AR NFE=3
[tool_loop] final answer after 0 step(s)
```