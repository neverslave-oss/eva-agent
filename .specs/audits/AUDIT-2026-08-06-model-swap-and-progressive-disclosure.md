# AUDIT 2026-08-06 — Model Swap & Progressive Disclosure

**Branch:** `fix/qwen-tool-calling-sampling` (no changes to source, no commit)
**System:** Kernel-Evolving (port 8779) — model_server at `src/core/inference/model_server.py`

---

## Table of Contents

1. [Audit A: Model Swap from Telegram Bot Is Broken](#audit-a-model-swap-from-telegram-bot-is-broken)
2. [Audit B: Progressive Disclosure Check](#audit-b-progressive-disclosure-check)

---

## Audit A: Model Swap from Telegram Bot Is Broken

**Reported symptom:** Using `/models` slash command and selecting Gemma 4 (E2B or E4B) does not swap the active model. The model server continues to run the original Nemotron model.

### A1 — Root Cause: `_handle_swap_model` In-Memory Config Mutation Clobbered by Loaders

**Severity:** P0 — blocking

`_handle_swap_model` (model_server.py:2760) sets the new model path in the in-memory `_config` global, but then calls `_load_hf_model(_lazy_config_path)` or `_load_vllm_engine(_lazy_config_path)` — both of which re-read the config *from disk* via `_load_config(config_path)`. This overwrites `_config` with stale disk contents, discarding the in-memory mutation.

```
_handle_swap_model("google/gemma-4-E2B-it")
  ├─ _config["model"]["path"] = "google/gemma-4-E2B-it"   ← in-memory update
  ├─ _is_nemotron_model("google/gemma-4-E2B-it") → False
  ├─ backend = "transformers"  (from in-memory _config, which still has inference.backend)
  └─ _load_hf_model(_lazy_config_path)
       └─ cfg = _load_config(config_path)
            └─ _config = yaml.safe_load(open("config.yaml"))   ← OVERWRITES global _config!
               Now _config has original model.path from disk:
                 ~/models/huggingface/hub_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/...
       └─ model_path = cfg["model"].get("path")   ← Nemotron path from disk!
       └─ _processor = AutoProcessor.from_pretrained(model_path)  ← LOADS NEMOTRON, not Gemma!
```

**Why Nemotron swap works:** `_load_nemotron` accepts `model_path_override=new_path` and uses `cfg = _config if _config is not None else _load_config(config_path)` — it reads the ALREADY-MUTATED in-memory `_config`, avoiding the disk re-read. Only this branch passes the new path through.

**Why Gemma / Qwen3-VL swaps silently fail:**
- `google/gemma-4-E2B-it` is NOT a Nemotron model → `_is_nemotron_model` returns False
- Code falls through to `_load_hf_model` / `_load_vllm_engine` → disk clobber
- The loader loads whatever `config.yaml` says (Nemotron)
- `_handle_swap_model` returns `{"status": "ok", "model": "gemma-4-E2B-it"}` — **lies about success**
- The bot message says "✅ Gemma 4 E2B loaded" — misleading

**Fix structure:**
- Option A (minimal): Have `_handle_swap_model` pass `model_path_override` to `_load_hf_model` and `_load_vllm_engine`, similar to how `_load_nemotron` does it.
- Option B (clean): `_load_vllm_engine` and `_load_hf_model` should read `model_path` from the existing in-memory `_config` global *first*, falling back to disk re-read only if `_config` is None.

### A2 — Health Endpoint Reports Config Metadata, Not Actually-Loaded Model

**Severity:** P0 — misleading diagnostics

`_handle_health` (model_server.py:2670) returns:
```python
model_name = (_config or {}).get("model", {}).get("name", "unknown")
```

This reads `_config["model"]["name"]` — a **config field**, not the actually-loaded model. After a failed swap, `_config` was clobbered back to disk values, so health reports `nvidia/Nemotron-Labs-Diffusion-3B` even though the bot *said* Gemma loaded. The `/models` menu uses `h.get("model")` → shows old model → user sees "it isn't swapping".

Even without the clobber bug, `_config["model"]["name"]` is just a string in a config file — it has no runtime guarantee of matching what's in VRAM.

**Fix:** The health endpoint should report the actual loaded model by inspecting `_vllm_model_path`, `_model.__class__.__name__`, or the `_slot_registry` primary slot's model path. `_config["model"]["name"]` should be a fallback only.

### A3 — `drafter_path` Param Read But Never Used

**Severity:** P1

`_handle_swap_model` reads `drafter_path = params.get("drafter_path", None)` at line 2763, but this variable is **never passed to any loader** in the non-Nemotron path. `_load_hf_model` reads `drafter_path = cfg["model"].get("drafter_path")` — which comes from the disk config (empty string `''`).

This means:
- `/models load e2b` passes `drafter_path=google/gemma-4-E2B-it-assistant` — silently ignored
- Even a fixed Gemma swap would load without its assistant drafter (no speculative decoding)
- `/models load nemotron3b` passes `drafter_path=""` — not used either, but Nemotron doesn't need a drafter

### A4 — Mismatched Cache Roots: Bot Checks One Dir, Server Checks Another

**Severity:** P1

The bot's "downloaded?" check at telegram_bot.py:1630:
```python
_hf_home = os.environ.get("HF_HOME") or ... ~/.cache/huggingface
_hub_dir = Path(_hf_home) / "hub"
downloaded = any(_hub_dir.glob(f"models--google--gemma-4-E2B-it*"))
```

But config.yaml model paths point to a DIFFERENT root:
```yaml
model:
  path: ~/models/huggingface/hub_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/...
```

The model_server's `_load_hf_model` uses whatever `HF_HOME` is set in the model_server process — which may differ from the telegram-bot process. If the bot's process sees Gemma in `~/.cache/huggingface/` but the model_server process has `HF_HOME=~/...`, the Load button appears but `transformers` can't find it.

**Also:** The `cache_prefix` check globs `models--google--gemma-4-E2B-it*` under `hub/` — but the actual cache folder might be `models--google--gemma-4-E2B-it` (no asterisk needed). The glob should find it if present. But if HF_HOME differs, the check is against the wrong directory.

### A5 — Two-Stage Tool-Calling Slot Unaffected by Swap

**Severity:** P1

`_handle_swap_model` only touches the primary model slot. The tool-calling slot (Qwen3.5-0.8B) remains loaded in `_tool_calling_model` — `_tool_calling_slot_loaded` stays `True`. Even if the swap worked:
- `_run_two_stage_if_available` still runs Qwen for tool calls (R1 from inference audit — Qwen never emits valid tool calls)
- Gemma 4 has native tool support (`_TOOL_CAPABLE_PREFIXES = ("google/gemma-4",)`) but never gets to use it
- The two-stage pipeline hijacks tool calling regardless of what primary model is loaded

**Fix:** `_handle_swap_model` should either:
- Unload the tool-calling slot when swapping to a model that has native tool support (`_TOOL_CAPABLE_PREFIXES`)
- OR (simpler) add a `reset_tool_calling_slot: true` hint to the swap params that the two-stage pipeline respects

### A6 — `/models reload` Has Bare Import Bug

**Severity:** P2

```python
# telegram_bot.py:1761
import yaml as _yaml, model_client as _mc
```

This imports `model_client` **bare** — without the `core.inference.` prefix used everywhere else (line 1703 does `import core.inference.model_client as _mc`). If `src/model_client.py` doesn't exist as a top-level module (it lives at `src/core/inference/model_client.py`), this raises `ImportError` → caught by the outer `except Exception` → "❌ Exception" shown to user.

What's more, the reload path also reads `cfg["model"]["path"]` from disk (re-reading config with yaml) but the config still points to the original model. So `/models reload` is mostly useful for debugging.

### A7 — Swap Response Name Is `new_path.split("/")[-1]`, Not Actual Loaded Model Name

**Severity:** P2

```python
model_name = new_path.split("/")[-1]  # "gemma-4-E2B-it"
backend_tag = "nemotron" if _is_nemotron else ("vllm" if _vllm_enabled else "transformers")
return {"status": "ok", "model": model_name, "backend": backend_tag}
```

The returned `model` field is derived from the **requested** path, not the actually-loaded model. If loading fails and falls back to another path... actually, loading either succeeds or throws. But the name is always the requested one. This is misleading if the wrong model was loaded (A1 clobber).

### A8 — `_detect_capabilities` Called with Original Config After Clobber

**Severity:** P2

`_load_hf_model` calls `_detect_capabilities(model_path, cfg)` where `model_path` is the **disk config path** (Nemotron) and `cfg` is the disk config. The tool-support detection:
```python
_TOOL_CAPABLE_PREFIXES = ("google/gemma-4",)
_model_supports_tools = any(model_name.lower().startswith(...) ... )
```
With Nemotron path → no match → `_model_supports_tools` may be set to False (after the refinement check might set it True, but it's unpredictable). This cascades into `_handle_infer_with_tools` → `if not _model_supports_tools: return _handle_infer_plain(params)` — tools support may be wrong.

### A9 — The `/models` KNOWN_MODELS Hardcodes Models in Both Bot and Server

**Severity:** P3 — maintainability

`KNOWN_MODELS` in telegram_bot.py and the model-switching logic in model_server.py are independent registries. Adding a new model requires touching both files, and they can drift. There's no shared "available models" registry. The `model_slots` config section and `_slot_registry` are separate from `KNOWN_MODELS` too. Three separate model registries that don't talk to each other.

---

## Audit B: Progressive Disclosure Check

**Question:** Are skills, routines, memory, and multi-agent collective memory progressively disclosed to keep the context window tidy?

### System Prompt Builder — `build_system_prompt` (`src/core/memory/context.py:333`)

The system prompt is rebuilt fresh on every `triage()` call. It includes these sections, listed in injection order:

| Section | What's Injected | Progressive? |
|---|---|---|
| AGENTS.md | Full file up to **20,000 chars** (`_load_agents_md`) | ❌ Full-file dump, cap only |
| USER.md | Full file up to **3,000 chars** (`_load_user_md`) | ⚠️ Capped but full content |
| User profile | Name, handle, timezone, notes from `user.json` | ✅ Tiny fundamental data |
| Recent thoughts | Up to 5 thought entries with full body text | ⚠️ Moderate (body could be long) |
| System live table | CPU%, RAM, Disk (live reads) + 8 service status checks | ⚠️ Dynamic data per message |
| Tools table | 11 tools hardcoded inline | ✅ Stable, small (no growth) |
| Skills section | **Only 10 priority skills** with 90-char descriptions + search_skills hint | ✅ **Progressive** |
| Routines section | Only count + `list_routines()` hint | ✅ **Progressive** |
| ADR table | Static list of 12 ADRs | ✅ Negligible size |
| Memory section | `last_interaction` string + turn count + `recall_memory()` hint | ✅ Stats only + tool hint |
| Active notes | Last 5 files, first line each (`_load_notes`) | ✅ Capped |
| Session Notes | Static anchor text | ✅ Negligible |

### B1 — Skills: PARTIALLY Progressive (Good Pattern)

**Verdict: ✅ Partially implemented, good pattern**

```python
# context.py:594-596
"Call `search_skills(query)` to find the right skill..."
"**Key skills (call search_skills for the full list):**"
```

Only a curated `_PRIORITY_SKILLS` set (10 skills) is shown with name + command list + 90-char description. The remaining skills are hidden behind the `search_skills` tool. `search_skills` is a real tool in `tools.py:474` (line search: found at `src/core/tools.py:474, `execute_tool` dispatches to it).

**Gap:** The priority set is hardcoded (`_PRIORITY_SKILLS: {"browser-automation", "web_search", ...}`) — it doesn't adapt based on what skills are actually installed or frequently used. If a new skill is added to the ecosystem, it won't appear in the prompt until someone adds it to `_PRIORITY_SKILLS`. This could be solved by a relevance-based selection (e.g., show the 10 most-used or most-recently-run skills).

**Also:** Each priority skill's description is capped at 90 chars (`s.get("description", "")[:90]`). Commands array is dumped as `"[c1, c2]"` — could be noisy for skills with many commands.

### B2 — Routines: Progressive (Good Pattern)

**Verdict: ✅ Properly progressive**

```python
f"**{len(unique_routines)} routines available.** "
"Call `list_routines()` to see their names and triggers before running one."
```

Only the count is in the prompt. Names and descriptions are behind the `list_routines()` tool (`tools.py:511`). **No gap** — this is the ideal pattern.

### B3 — Memory: Dead Code + Partial Disclosure

**Verdict: ⚠️`_load_long_term_memory()` is dead code; actual disclosure is via tools only**

```python
# context.py:393
long_term_memory = _load_long_term_memory()
```

This variable is **assigned but never used in the prompt** (confirmed via grep — only lines 305, 309, 393 mention `long_term_memory`). It loads up to 15 snippets (3 files × 300 chars + 12 lines × 220 chars ≈ 3540 chars maximum) from `long_term_memory` module and then **discards them**.

This is wasted I/O on every `triage()` call (reads files + queries SQLite). Either:
- Wire `long_term_memory` into the prompt (e.g., under "## Archived memories" section, capped at ~500 chars) — or
- Remove the dead `_load_long_term_memory()` call entirely

**Actual memory disclosure in prompt:**
- `last_interaction` + turn count (no inline content) → on-demand via `recall_memory` tool
- Active notes: 5 files, first line each (`_load_notes` at line 218) — good cap
- Semantic retrieval in `agent.py` triage: top-5 turns >0.55 cosine score, ≤200 chars each, injected after system prompt — proper progressive disclosure by relevance
- `recall_memory` tool exists at `tools.py:536` with query argument ✓

### B4 — Collective Memory: Bot-Side Search, Not in Agent Triage

**Verdict: ⚠️ Implemented only in Telegram bot path, not in agent.py triage**

`_search_collective_memory(query)` at telegram_bot.py:808:
- Tries `context_provider` skills first (each capped at 600 chars, 6s timeout)
- Falls back to `~/.openclaw/workspace/collective-memory/scripts/search.py --top 2`

Then at line 2606-2609 (inside `handle_message`):
```python
memory_results = _search_collective_memory(text)
if memory_results:
    system_prompt = f"Relevant memory:\n{memory_results}\n\n{system_prompt}"
```

**Critical findings:**

1. **Only in Telegram bot path.** The `agent.py` `triage()` function does NOT search collective memory at all. Any API client (or non-Telegram channel) that goes through `core.agent.triage()` will never see collective memory results. Collective memory integration is incomplete.

2. **Subprocess on the critical path.** `_search_collective_memory` runs a `subprocess.run()` with 5-6s timeout on EVERY message. If the script hangs, the entire message takes 5-6s before the model even starts thinking. No async. No caching.

3. **Prepended without token budget.** `system_prompt = f"Relevant memory:\n{memory_results}\n\n{system_prompt}"` — results are prepended at the TOP of the system prompt (highest visual attention), taking budget from identity/instructions. No token cap on total prepended content. The context-provider path caps each skill at 600 chars, but multiple skills can produce multiple 600-char blocks.

4. **Fallback output is uncapped.** The fallback `result.stdout.strip()` returns the entire script output with no length limit. If `search.py` returns 5000 chars, all of it gets prepended.

5. **Quality-gated writes exist** at `_write_collective_memory` (line 840) — skip patterns for greetings, short messages, errors, and a "has_substance" heuristic (URLs, paths, code, numbers). ✅ This is good hygiene and prevents bloat.

### B5 — No Token Budget Accounting for the System Prompt

**Verdict: ❌ Missing — history trimming assumes space that's taken by system prompt**

In `agent.py` triage (around line 280-290):
```python
_ctx_tokens = _prov.get_context_length(...)
_max_history_chars = (_ctx_tokens - 4096) / 0.4  # reserves 4096 tokens for system
```

This assumes the system prompt fits within 4096 tokens. But:
- AGENTS.md can be up to **20000 chars ≈ 5000 tokens** alone
- USER.md up to 3000 chars ≈ 750 tokens
- Skills section, tools table, routines, notes, etc. add another 3000+ chars
- Total system prompt can easily be **10,000+ tokens** — blowing past the 4096-token "reserve"

The `_max_history_chars` formula then trims history based on an incorrect assumption — the actual system prompt + history can exceed `_ctx_tokens` (the model's context length). Nemotron has 65536 tokens, so it rarely overflows in practice. But for Qwen two-stage (8k ctx) this could cause silent truncation at tokenization time.

**No `len(system_prompt)` guard exists** anywhere in the codebase (grep confirms: no `system_prompt.*token` or `len(system_prompt)` in agent.py or context.py).

### B6 — Per-Message HTTP Costs in Prompt Builder

**Verdict: ⚠️ Latency concern, not progressive disclosure but context efficiency**

`build_system_prompt` makes **8 HTTP requests on every call**:
- `_olly_alive(openclaw_endpoint)` → HTTP GET to OpenClaw (line 396)
- `_service_status(8765)` → HTTP GET to fantasia (line 397)
- `_service_status(8766)` → HTTP GET to voice server (line 398)
- `_service_status_url("http://localhost:8769")` → Kernel base (line 399)
- `_service_status_url("http://localhost:8000")` → Dashboard (line 400)
- `_service_status_url("http://localhost:3333")` → Kanban (line 401)
- `_service_status_url("http://localhost:3131")` → Mental Map (line 402)
- `_service_status_url("http://localhost:8770")` → Embed server (line 403)

Each `_service_status`/`_service_status_url`/`_olly_alive` sends an HTTP request with a short timeout. Combined with the sequential embed HTTP calls in semantic retrieval (N turns × 1 HTTP each), the per-message network overhead is significant.

**Fix candidates:**
- Cache service statuses for 30-60 seconds (they rarely change between messages)
- Make service checks lazy — only check when status section is actually relevant (unlikely for a system prompt, but the prompt could show "unknown" for infrequently-checked services without blocking)

### B7 — Full-File AGENTS.md and USER.md Injection

**Verdict: ⚠️ AGENTS.md 20k chars is heavy; USER.md 3k chars is acceptable**

`_load_agents_md()` returns the **entire AGENTS.md content** up to 20000 chars with a log warning at that threshold. AGENTS.md is a long configuration document with workspace setup, multi-agent architecture description, workflows, etc. — it's not a concise identity file. Injecting it into every message costs ~5,000 tokens of context.

**USER.md** is capped at 3000 chars with management comments stripped — acceptable but still substantial (~750 tokens).

The identity-critical parts of these files could be summarized into a brief (200-300 char) persona description, with the full file available via a tool call (`read_file`).

### Summary Table — Progressive Disclosure

| Component | Progressive? | Status | Action |
|---|---|---|---|
| Skills | ✅ Partial | Priority 10 shown; rest via `search_skills` | Make priority set adaptive (most-used / most-recent) |
| Routines | ✅ Yes | Count only + `list_routines()` tool | No change needed |
| Memory (long-term) | ⚠️ Bug | `_load_long_term_memory()` computed but **never injected** — dead code | Either wire into prompt or remove the call |
| Memory (semantic) | ✅ Yes | Per-query top-5 >0.55, ≤200 chars each | Cost: sequential embed HTTP calls per message |
| Memory (USER.md) | ⚠️ Moderate | Full file up to 3000 chars injected | Could be summarized; full via `read_file` tool |
| Memory (AGENTS.md) | ❌ Heavy | Full file up to 20000 chars (~5000 tokens) | Should be summarized; include size warning / trim |
| Collective memory | ⚠️ Partial | On-demand per-message in Telegram bot path, NOT in agent.py triage | Port to triage; add token budget; cache results; cap fallback output |
| System prompt budget | ❌ Missing | `_max_history_chars` assumes 4096-token reserve but system prompt is never measured | Add `len(system_prompt)` guard; account for actual token cost |
| Active notes | ✅ Yes | Last 5 files, first line each | No change needed |
| Recent thoughts | ✅ Moderate | 5 entries, body text | Could cap body at 200 chars if large |

### Open Questions for Audit B

1. **Should AGENTS.md be summarized?** The 20k-char full-file dump is the biggest single contributor to system prompt bloat. The identity section could be ~500 chars with the rest available via tools.

2. **Should collective memory be integrated into `agent.py` `triage()`?** Currently only the Telegram bot path (line 2606) searches it. The main API path via `core.agent.triage()` has no collective memory awareness.

3. **Should `_load_long_term_memory()` dead code be removed or wired in?** Currently wastes I/O on every prompt build. If the intent is on-demand-only via `recall_memory` tool, remove the call. If it should inject snippets, add them to the prompt under "## Archived memories".

4. **Should service status HTTP checks be cached?** 8 HTTP requests per message (with timeouts) adds latency. A 30-second TTL cache would eliminate most of the overhead.

5. **Should there be a global token budget for the system prompt?** `_max_history_chars` assumes 4096 tokens for system prompt, but AGENTS.md alone can exceed that. A `token_count(system_prompt)` call before injection would let the system trim sections proportionally (shorten AGENTS.md, drop lowest-priority sections first).