# ADR-013: Multi-Provider Inference Wrapper + Trajectory-Driven Fine-Tuning Pipeline

**Status:** Proposed — awaiting Fabio review  
**Date:** 2026-05-11  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving), ADR-008 (Critic Replica), ADR-010 (Evolution Pipeline Critic)

---

## Motivation

**Question Fabio asked:** *"If we plug in a more capable model as a cloud provider — gather real solid executions — then use those trajectories to fine-tune Gemma 4 or its drafter… would that improve quality and unlock a solid small foundation self-evolving agent?"*

**Answer: Yes, and this is the correct architecture.**

Current state:
- Tier 2 synthesis uses `code_synthesizer.py` with `TMP_OPEN_AI_API_KEY` — partially wired but not yet used for **inference** (only synthesis).
- `infer_with_tools` runs exclusively on Gemma 4 E2B-it — a 2B model that struggles with multi-step planning and reliable tool calling.
- We already collect `synthesis_trajectories` and the model server streams tool call steps as `{"type": "step", ...}` JSON lines — raw material for a fine-tuning dataset.
- Sim 8 produced labeled negative examples (Gemma's failed `write_file` / premature exits). A capable model on the same tasks produces the positive examples. The delta is the fine-tuning signal.

**The flywheel:**

```
Capable cloud model (GPT-5.4 / Claude Sonnet 4.6 / DeepSeek-V3)
    ↓ produces high-quality tool call chains (Sim 9)
Trajectory collector — automatic on PASS + artifact verified
    ↓ stores (prompt, tool_calls, result, critic_score) tuples
    ↓ exports JSONL → HuggingFace private dataset: PacificDev/kernel-evo-trajectories
HuggingFace fine-tuning job (TRL / SFT, free H100 GPU minutes)
    ↓ trains Gemma 4 E2B-it or its MTP drafter on those trajectories
Better local model
    ↓ produces better tool calls → better trajectories (Sim 10)
    (loop — publish paper + models + dataset when convergence confirmed)
```

---

## Decision

### Part 1 — Multi-Provider Inference Wrapper (`src/provider.py`)

A single `InferenceProvider` class routing all inference — not just Tier 2 synthesis but every `infer()` and `infer_with_tools()` call — with a configurable active provider **per call type** and **per model**.

```python
class InferenceProvider:
    """
    Unified inference interface. Routes calls to configured provider.

    Providers:
      local      — Gemma 4 E2B-it via model_server socket (default, always available)
      openai     — OpenAI API
      anthropic  — Anthropic API
      copilot    — GitHub Copilot API (GITHUB_TOKEN)
      hf         — HuggingFace Inference API (any hosted model)

    Per-call-type routing (config.yaml providers block):
      task_inference    → infer_with_tools for user-facing tasks
      synthesis         → Tier 2 skill synthesis
      critic            → Critic replica evaluation
      planning          → Micro-planner step generation
      trajectory_teacher → Teacher model for fine-tuning data collection
    """
```

#### Provider + model configuration (`config.yaml`)

```yaml
providers:
  # Active provider per call type
  # "local" always falls back to Gemma 4 on the socket
  task_inference:       local
  synthesis:            openai       # Tier 2 — capable model required
  critic:               local        # Drafter is fast enough
  planning:             local        # Drafter handles plan gen
  trajectory_teacher:   openai       # Teacher for Sim 9 trajectory collection

  # Model selection per provider — fully configurable, no hardcoded names
  models:
    openai:     gpt-5.4                         # GPT-5.4 for synthesis + teacher
    anthropic:  claude-sonnet-4-6               # Claude Sonnet 4.6
    hf:         deepseek-ai/DeepSeek-V3         # DeepSeek-V3.2 / 4.0 / GLM5 / GLM5.1 etc.
    local:      null                            # uses model_server socket (config.model.path)
    copilot:    null                            # uses GITHUB_TOKEN endpoint

  # Per-call-type model overrides (optional — if set, overrides providers.models.<provider>)
  model_overrides:
    synthesis:          null                    # e.g. "gpt-5.4" to lock synthesis model
    trajectory_teacher: null

  # Streaming — real-time step delivery to Telegram during infer_with_tools
  streaming:
    enabled: true
    task_inference: true    # stream tool call steps to Telegram for user tasks
    synthesis: false        # synthesis runs background, no stream needed
    trajectory_teacher: true

  # Trajectory collection — automatic on PASS + artifact verified
  collect_trajectories: true
  trajectory_min_critic_score: 0.7   # only save trajectories where critic rates ≥ 0.7
  trajectory_require_artifact: true  # require at least one file written to disk

  # HuggingFace dataset — private until paper publication
  hf_dataset_repo: PacificDev/kernel-evo-trajectories
  hf_dataset_private: true
```

#### Env vars

| Var | Provider | Current status |
|-----|----------|----------------|
| `OPENAI_API_KEY` / `TMP_OPEN_AI_API_KEY` | openai | Already in `.env` |
| `ANTHROPIC_API_KEY` | anthropic | Add to `.env` |
| `GITHUB_TOKEN` | copilot | Add to `.env` |
| `HF_TOKEN` | hf | Add to `.env` |

#### Wire-up points

| File | Current call | New call |
|------|-------------|----------|
| `agent.py triage()` | `infer_with_tools(...)` | `provider.infer_with_tools(..., call_type='task_inference')` |
| `evolution_hook._try_tier2_pipeline()` | `infer_draft()` / `infer()` | `provider.infer(..., call_type='synthesis')` |
| `thought_engine.ThoughtEvaluator` | `infer(...)` | `provider.infer(..., call_type='critic')` |
| `micro_planner.MicroPlanner.plan()` | `infer_draft()` | `provider.infer(..., call_type='planning')` |

`local` is always the fallback — if a cloud provider returns an error or is unconfigured, `InferenceProvider` retries on local automatically.

---

### Part 2 — Streaming (`stream=True` path)

Cloud providers (OpenAI, Anthropic, HF) support server-sent events on their APIs. The `InferenceProvider.infer_with_tools()` call exposes a `stream=True` path that:

1. Opens a streaming HTTP connection to the cloud provider
2. Parses tool call deltas as they arrive
3. Calls `step_callback(step_event)` with each intermediate step — the same callback already used by `model_server._handle_infer_with_tools`
4. The `telegram_bot` sends a live "🔧 calling `write_file`…" message that gets edited in-place as steps arrive

This matches the existing UX for local inference (typing keepalive + in-place edit) but with cloud providers.

```python
async def _stream_openai_tool_calls(
    self, messages, tools, step_callback, model: str
) -> str:
    """Stream OpenAI tool-calling response, firing step_callback per tool call."""
    # Uses openai streaming API with stream=True
    # Accumulates tool call deltas until complete, then executes
    # Returns final text response
```

For `task_inference` calls where `providers.streaming.task_inference: true`, the stream path is used automatically. For `synthesis` calls (`providers.streaming.synthesis: false`), standard blocking call is used.

---

### Part 3 — Trajectory Collector (`src/trajectory_collector.py`)

**Automatic** — no user flag needed. Records every `infer_with_tools` session where:
1. `providers.collect_trajectories: true`
2. Critic scores the result ≥ `trajectory_min_critic_score` (0.7)
3. At least one artifact was written to disk (when `trajectory_require_artifact: true`)

**DB schema** (new table in `~/.kernel-evolving/workspace/evolution.db`):

```sql
CREATE TABLE IF NOT EXISTS task_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    task            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model_name      TEXT,
    call_type       TEXT NOT NULL,
    tool_calls      TEXT NOT NULL,   -- JSON: [{tool, args, result}, ...]
    final_reply     TEXT NOT NULL,
    artifacts       TEXT,            -- JSON: ["/path/to/file", ...]
    critic_score    REAL,
    critic_verdict  TEXT,
    token_count     INTEGER,
    elapsed_s       REAL
);
```

**JSONL export** (HuggingFace-compatible chat template format):

```jsonl
{
  "id": "abc123",
  "ts": "2026-05-11T...",
  "task": "Write a Python script and save it to ~/...",
  "provider": "openai",
  "model": "gpt-5.4",
  "call_type": "task_inference",
  "messages": [
    {"role": "user", "content": "Write a Python script..."},
    {"role": "assistant", "tool_calls": [{"name": "write_file", "arguments": {"path": "~/...", "content": "#!/usr/bin/env python3\n..."}}]},
    {"role": "tool", "name": "write_file", "content": "Written to ~/..."},
    {"role": "assistant", "content": "Done. The script has been saved to ~/..."}
  ],
  "artifacts": ["~/kernel-evo-notes/sim8/hello_evo.py"],
  "critic_score": 0.92,
  "verdict": "PASS"
}
```

Direct input to TRL `SFTTrainer` with `dataset_text_field` or `apply_chat_template`.

**API endpoint:**

```
GET /trajectories/export?format=jsonl&min_score=0.7&call_type=task_inference
→ streams JSONL to disk: ~/.kernel-evolving/workspace/trajectories/export_<ts>.jsonl
→ returns {"path": "...", "count": N, "min_score": 0.7}
```

---

### Part 4 — HuggingFace Fine-Tuning Script (`scripts/finetune_from_trajectories.py`)

UV script (PEP 723) for HuggingFace Jobs. Targets:

**Primary: Gemma 4 MTP drafter** — 180 MB, ~10× cheaper than full E2B-it. Fine-tuning the drafter improves plan generation quality and thought seeds. Estimated cost: 15-20 GPU minutes on H100 for 100 trajectories × 3 epochs — within HF Pro free tier.

**Secondary: Gemma 4 E2B-it full model** — LoRA adapter, trains on the complete tool-call sequences. Higher quality but requires more compute.

**DPO dataset construction (automatic):**
- Positive: teacher model (GPT-5.4) trajectories where `verdict=PASS` and artifact on disk
- Negative: Gemma 4 trajectories on the same tasks where `verdict=FAIL` or no artifact
- Sim 8 already provides the negative set. Sim 9 provides the positive set.

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "trl>=0.9",
#   "transformers>=4.45",
#   "datasets",
#   "peft",
#   "trackio",
#   "torch",
# ]
# ///
"""
Fine-tune Gemma 4 MTP drafter on kernel-evolving task trajectories.
Run: uv run --with trl scripts/finetune_from_trajectories.py
Dataset: PacificDev/kernel-evo-trajectories (private)
Method: SFT on PASS trajectories, DPO when negative examples available
"""
```

---

### Part 5 — Simulation 9 design (provider: openai, model: gpt-5.4)

Set `task_inference: openai` (or env `PROVIDER_TASK=openai`) → run Sim 8 task set:

| Task | Expected outcome | Trajectory value |
|------|-----------------|-----------------|
| T01 `/markdown` exec | PASS (skill dispatch, no LLM) | n/a (exec, not LLM) |
| T02 `write_file` | PASS | High — GPT-5.4 reliably calls write_file |
| T03 `exec_shell` + write | PASS | High — multi-tool chain |
| T04 fetch + summarise + save | PASS | High — micro-planner test |
| T05 read + analyse + write | PASS | High — read_file + write_file chain |

Export → `PacificDev/kernel-evo-trajectories` (private) → SFT → fine-tuned Gemma 4 → Sim 10.

**Hypothesis:** 50-100 PASS trajectories from GPT-5.4 are sufficient to fine-tune the Gemma 4 MTP drafter to reliable `write_file` + `exec_shell` chains on the Sim 8 task distribution.

---

## Implementation plan

### Phase 1 — Provider wrapper (v1.9.3)
- [ ] `src/provider.py` — `InferenceProvider` with openai / anthropic / hf / copilot / local
- [ ] Models fully configurable via `config.yaml providers.models` — no hardcoded model names
- [ ] `config.yaml` — add `providers:` block with all defaults
- [ ] `.env` — document required keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN, GITHUB_TOKEN)
- [ ] Wire `agent.py` triage → `provider.infer_with_tools(call_type='task_inference')`
- [ ] Wire `evolution_hook._try_tier2_pipeline()` → `provider.infer(call_type='synthesis')`
- [ ] Wire `micro_planner.MicroPlanner` → `provider.infer(call_type='planning')`
- [ ] Unit tests: routing, model selection, fallback to local on error

### Phase 2 — Streaming (v1.9.3)
- [ ] `InferenceProvider._stream_openai_tool_calls()` — streams steps via `step_callback`
- [ ] `InferenceProvider._stream_anthropic_tool_calls()` — same for Anthropic
- [ ] Telegram bot: step messages edit in-place during streaming (reuse existing typing keepalive pattern)
- [ ] Config: `providers.streaming.task_inference: true/false`

### Phase 3 — Trajectory collector (v1.9.3)
- [ ] `src/trajectory_collector.py` — DB schema, collector, JSONL exporter
- [ ] Wire into `infer_with_tools` completion path — auto-record on PASS + artifact
- [ ] `GET /trajectories/export` endpoint
- [ ] Auto-upload to HF on export (if `HF_TOKEN` set): `huggingface_hub.push_to_hub()`

### Phase 4 — Sim 9 + fine-tuning
- [ ] Activate openai provider for `task_inference`: `PROVIDER_TASK=openai python3 sim8/run_sim8.py`
- [ ] Collect trajectories → export → upload to `PacificDev/kernel-evo-trajectories`
- [ ] `scripts/finetune_from_trajectories.py` — UV script for HF Jobs SFT
- [ ] Run on HF Jobs (free H100 GPU minutes from Pro account)
- [ ] Evaluate fine-tuned model on Sim 8 → Sim 10

---

## Open questions (resolved)

| Question | Decision |
|----------|----------|
| Stream=True path to Telegram? | ✅ Yes — step_callback fires for every tool call, bot edits in-place |
| Trajectory collection opt-in or automatic? | 📌 Automatic on PASS + artifact verified |
| HuggingFace dataset visibility? | 🔐 Private until convergence confirmed, then open alongside paper + models |

## Publication plan

### Public release
When Sim 10 confirms the fine-tuned model improves on Sim 8 pass rate:
- Open `PacificDev/kernel-evo-trajectories` dataset (Sim 9 teacher trajectories only — GPT-5.4 on standard tasks)
- Open-source the **first-stage LoRA adapter** on HuggingFace (`PacificDev/gemma4-kernel-evo-v1`)
- Final paper revision includes Sim 8 / 9 / 10 trajectory, multi-provider architecture, DPO construction, convergence results

### Private internal model — two-stage fine-tuning strategy

The public model is stage 1. Stage 2 is private and uses our own interaction history:

```
Stage 1 — Public
  Base:      google/gemma-4-E2B-it (or MTP drafter)
  Dataset:   PacificDev/kernel-evo-trajectories (Sim 9 teacher trajectories)
  Method:    SFT → DPO (positive=GPT-5.4 PASS, negative=Gemma FAIL from Sim 8)
  Output:    PacificDev/gemma4-kernel-evo-v1  (open, shareable)

Stage 2 — Private
  Base:      gemma4-kernel-evo-v1 (Stage 1 adapter)
  Dataset:   OpenClaw session trajectories (Olly ↔ Fabio interactions, all sessions)
             ~/.openclaw/agents/main/sessions/*.trajectory.jsonl
             ~1,379 sessions, 90 trajectory files, ~1.1 GB
  Method:    Continued SFT — reinforce patterns specific to our workflow
             (tool use patterns, multi-step planning, kernel-evolving debugging style)
  Output:    Internal only — never published
             Loaded as the active model for Olly + kernel-evolving in production
```

**Why this matters:** The OpenClaw sessions contain interaction patterns that no public dataset has — diagnosing a running agent's thought loop, fixing `_parse_tool_call` by reading live logs, writing ADRs from first principles, coordinating multi-agent workflows. That behavioural fingerprint is what makes the internal model precise in our specific context. It's not general intelligence — it's domain expertise built from real work.

### OpenClaw trajectory extraction pipeline (to implement)

The raw material already exists:
- `~/.openclaw/agents/main/sessions/*.trajectory.jsonl` — 90 files, OpenClaw trajectory schema v1
- Each file contains: `prompt.submitted`, `model.completed`, `trace.artifacts` events per turn
- Tool calls are embedded in `model.completed` (truncated at 256KB — full content in `.jsonl` sibling)

Needs a script `scripts/extract_openclaw_trajectories.py` that:
1. Reads each `.trajectory.jsonl` + matching `.jsonl` session file
2. Reconstructs `(user_message, tool_calls, assistant_reply)` tuples per turn
3. Filters: only turns where tool calls were made AND outcome was verifiable success
4. Exports to `~/.openclaw/workspace/trajectories/openclaw_sessions_export.jsonl`
5. Uploads to `PacificDev/kernel-evo-trajectories-private` (HF private dataset, never public)

Format aligned with Stage 1 JSONL so Stage 2 fine-tuning uses the same TRL pipeline.

**Privacy note:** OpenClaw trajectories contain personal context (Fabio's projects, clients, finances, legal docs). Before any upload: run the `/anonymize` skill on the export to strip PII. Stage 2 dataset stays private permanently — not part of the open paper.

---

## Addendum — Hot-swap provider via Telegram bot + API (v1.9.3)

### Motivation

During a live session — mid-task, mid-simulation, or when switching between local testing and trajectory collection — you need to flip the active provider without restarting the service. Two surfaces: **Telegram bot** (`@kernel_evo_agi_bot`) and the **REST API**.

### API endpoints

```
GET  /provider                          → current routing table + model per call-type
POST /provider/set                      → hot-swap one or more call-type routes
POST /provider/set/task_inference       → shorthand for single call-type swap
GET  /provider/available                → list providers + their readiness (key set?)
```

**GET /provider** response:
```json
{
  "routing": {
    "task_inference":     {"provider": "local",  "model": null,         "streaming": true},
    "synthesis":          {"provider": "openai", "model": "gpt-5.4",     "streaming": false},
    "critic":             {"provider": "local",  "model": null,         "streaming": false},
    "planning":           {"provider": "local",  "model": null,         "streaming": false},
    "trajectory_teacher": {"provider": "openai", "model": "gpt-5.4",     "streaming": true}
  },
  "collect_trajectories": true
}
```

**POST /provider/set** body:
```json
{
  "task_inference": "openai",
  "synthesis": "anthropic",
  "model_override": {"task_inference": "gpt-5.4"}
}
```
Hot-patches the running `InferenceProvider` singleton in-place. No restart needed. Changes are **session-scoped** — they reset on next service restart (config.yaml is source of truth for persistent config).

To make a swap **persistent**, add `persist: true` to the POST body — this writes the change back to `config.yaml` via a safe read-modify-write.

**GET /provider/available**:
```json
{
  "local":     {"ready": true,  "reason": "model_server socket connected"},
  "openai":    {"ready": true,  "reason": "OPENAI_API_KEY set"},
  "anthropic": {"ready": false, "reason": "ANTHROPIC_API_KEY not set"},
  "hf":        {"ready": false, "reason": "HF_TOKEN not set"},
  "copilot":   {"ready": false, "reason": "GITHUB_TOKEN not set"}
}
```

### Telegram bot commands

Add `/provider` command family to `telegram_bot.py`:

```
/provider                   → show current routing table as formatted message + inline swap buttons
/provider set <calltype> <provider>   → hot-swap e.g. /provider set task_inference openai
/provider set all <provider>          → swap ALL call types to one provider
/provider reset                       → revert all to config.yaml defaults
/provider status                      → which providers are ready (keys set)
```

Inline button menu on `/provider` — same pattern as `/evolve`:

```
┌─────────────────────────────────────────────────┐
│ 🔀 Provider Router                              │
│                                                 │
│ task_inference  →  local    [🔄 swap]           │
│ synthesis       →  openai   [🔄 swap]           │
│ critic          →  local    [🔄 swap]           │
│ planning        →  local    [🔄 swap]           │
│ teacher         →  openai   [🔄 swap]           │
│                                                 │
│ [🌐 All → OpenAI] [🏠 All → Local] [↺ Reset]  │
│ [📊 Availability] [🎯 Collect: ON]              │
└─────────────────────────────────────────────────┘
```

When user taps **[🔄 swap]** on a row, bot sends a follow-up inline menu:
```
Swap task_inference to:
[🏠 local] [🤖 openai] [🧠 anthropic] [🤗 hf] [⚡ copilot]
```
Tapping one fires `POST /provider/set {"task_inference": "openai"}` and edits the original message to reflect the new state.

**[🎯 Collect: ON/OFF]** — toggles `providers.collect_trajectories` live.

### Implementation in `telegram_bot.py`

Add to `handle_message()` — same `/evolve` pattern:

```python
if text.startswith("/provider"):
    parts = text.split()
    if len(parts) == 1:
        # Show routing table + inline buttons
        _handle_provider_menu(chat_id)
    elif parts[1] == "set" and len(parts) == 4:
        # /provider set <calltype> <provider>
        _handle_provider_set(chat_id, parts[2], parts[3])
    elif parts[1] == "set" and parts[2] == "all" and len(parts) == 4:
        _handle_provider_set_all(chat_id, parts[3])
    elif parts[1] == "reset":
        _handle_provider_reset(chat_id)
    elif parts[1] == "status":
        _handle_provider_status(chat_id)
```

`_handle_provider_set` calls `POST /provider/set` internally (or directly patches the singleton via `from provider import get_provider; get_provider()._routing[calltype] = new_provider`).

### Phase 5 (implementation — v1.9.3)
- [ ] `GET /provider` — returns current routing
- [ ] `POST /provider/set` — hot-patches singleton, optional `persist: true`
- [ ] `GET /provider/available` — key readiness check
- [ ] `telegram_bot.py` — `/provider` command + inline swap buttons
- [ ] Test: swap task_inference to openai via bot, send a task, verify OpenAI key used
- [ ] Test: `persist: true` writes back to config.yaml correctly
