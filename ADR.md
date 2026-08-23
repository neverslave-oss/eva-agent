# ADR.md — Kernel-Evolving Architecture Decision Records

**Last updated:** 2026-08-23
**Version:** v1.0.0-evolving
**Model:** Nemotron-Labs-Diffusion-3B (4-bit, 64k context)
**Port:** 8779 | **Socket:** `/tmp/kernel_evolving_model.sock`

---

## How to use this file

- **New agents on this repo:** Read this file + `AGENTS.md`. This gives you architecture + persona.
- **Debugging:** Check the relevant ADR section, then `git log --oneline -20`.
- **Adding a feature:** Reference the related ADRs, create a new section here if the decision is architectural.

---

## Architecture Overview

```
User (Telegram)
    ↓
api.py (FastAPI, port 8779)
    ↓
agent.py (triage → context pipeline → micro-planner → infer_with_tools)
    ↓
model_server.py (Unix socket, long-lived, owns GPU VRAM)
    ├── primary slot: Nemotron-Labs-Diffusion-3B (AR mode for tool loops)
    └── audio slot:    Gemma 4 E2B-it (STT, vision, multimodal)
    ↓
tools.py (11 tools: exec_shell, read_file, write_file, http_get, ...)
    ↓
Response → trajectory_collector (if critic score ≥ 0.7 → JSONL trajectory → HF fine-tune)
```

### Key subsystems

| Subsystem | File | Purpose |
|---|---|---|
| API server | `src/api.py` | FastAPI app, debug endpoints, Telegram webhook |
| Agent | `src/agent.py` | Triage, context pipeline, tool loop, history management |
| Model server | `src/core/inference/model_server.py` | Long-lived process, owns GPU, JSON-RPC over Unix socket |
| Model client | `src/core/inference/model_client.py` | Client for model server, streaming tool loop |
| Tools | `src/core/tools.py` | 11 tool definitions + execution + result classification |
| Context | `src/core/memory/context.py` | Builds the 21K system prompt with live data |
| Memory | `src/core/memory/memory.py` | Conversation history, long-term memory, embeddings |
| Evolution | `src/core/evolution/` | Skill synthesis, verification, recommendation, critic |
| Think-at-Rest | `src/services/thought_engine.py` | Idle reflection, gap detection, proactive Telegram |
| Skills | `src/core/skills.py` | Skill loading, discovery, execution |
| Routines | `src/core/routines.py` | Named multi-step procedures |
| Slots | `src/core/inference/model_slots.py` | Named model slots (primary, audio) |
| Prompt logger | `src/prompt_logger.py` → `database/agent/prompt_log.py` | Logs prompts to SQLite |

### Debug endpoints

```
GET /debug/prompt-logs?limit=20&chat_id=     — recent prompt/response pairs
GET /debug/prompt-log/{entry_id}             — one entry with full history
GET /debug/current-prompt?chat_id=           — live system prompt preview
GET /debug/chat-history?limit=80             — persisted chat turns
GET /debug/trajectories?limit=12             — recent tool-call trajectories
GET /debug/system-state                      — evolution state, skills, memory summary
GET /health                                  — model loaded + slot status
```

### Databases

| DB | Path | Contents |
|---|---|---|
| Chat history | `~/.kernel-evolving/workspace/memory/chat_history_evolving.db` | Conversation turns |
| Long-term memory | `~/.kernel-evolving/workspace/memory/memory_store.db` | Embedding-indexed memory |
| Evolution | `~/.kernel-evolving/workspace/data/evolution.db` | Skill synthesis state |
| Failed requests | `~/.kernel-evolving/workspace/data/failed_requests.db` | Tool failures, no_skill errors |
| Prompt log | `~/.kernel-evolving/workspace/data/prompt_log.db` | Prompt snapshots (no responses) |
| Trajectories | `~/.kernel-evolving/workspace/artifacts/trajectories/` | JSONL training data |

---

## ADR Index

| ADR | Title | Status | Key Decision |
|---|---|---|---|
| [004](#adr-004) | Self-Evolving Agent | Proposed | BDI-LLM loop: Tier 1 (skill match) → Tier 2 (synthesis) → critic gate |
| [005](#adr-005) | Think-at-Rest | Implemented | Idle reflection via speculative decoder, gap hypotheses, proactive Telegram |
| [006](#adr-006) | Capability Verification | Implemented | Prevent false-positive skill matches — binary yes/no after semantic match |
| [007](#adr-007) | Capability Recommendation | Implemented | Surface partial matches when verification fails |
| [008](#adr-008) | Critic Replica + Tools | Implemented | Replicas with tool access + inter-replica messaging |
| [009](#adr-009) | Goal Discovery Boot Cap | Implemented | Cap goals at boot, drain deferred queue slowly |
| [010](#adr-010) | Evolution Pipeline Critic | Implemented | Critic gate before skill install — quality check |
| [011](#adr-011) | Micro-Planner Triage | Implemented | Decompose multi-step requests into ordered steps |
| [012](#adr-012) | Async Pipeline | Implemented | Background job queue with TTL |
| [013](#adr-013) | Multi-Provider + Trajectories | Proposed | Inference routing, trajectory collection, HF fine-tune loop |
| [014](#adr-014) | ADR-to-Codebase Diffusion | Proposed | Prompt → ADR → codebase via DiffusionGemma (separate project) |
| [015](#adr-015) | Agent Auto-Discovery | Proposed | Peer kernels, OpenClaw, coding agents — agent discovery + peer routing |
| [016](#adr-016) | Self-Evolution Completeness | Proposed | Completeness checks for the self-evolution loop |
| [017](#adr-017) | Voice Model Routing | Proposed | Adaptive model switching for voice requests |
| [018](#adr-018) | Named Model Slots | Accepted | Named model slots (primary, audio) for deterministic routing |
| [019](#adr-019) | Think-at-Rest Observer + Critique | Accepted | Observer layer, critique layer, closed-loop evolution |
| [020](#adr-020) | User-Request-Anchored Evolution | Proposed | Evolution anchored to user requests with human-confirmation reward |
| [021](#adr-021) | Tier 2 Approval Gate | Accepted | Human-approval gate before Tier 2 skill synthesis install |
| [022](#adr-022) | Unified Tool-First Pipeline | Proposed | Unified tool-first pipeline for task execution |

---

## ADR-004: Self-Evolving Agent {#adr-004}

**Date:** 2026-05-04 | **Status:** Proposed

The agent actively closes its own capability gaps via a BDI (Belief-Desire-Intention) loop:

```
Task arrives
  ↓
Tier 1: semantic match against installed skills
  + ADR-006 capability verification (yes/no)
    ✅ verified → execute skill
    ❌ no match → ADR-007 recommendation (partial match check)
                    ↓
                 Tier 2: cloud model synthesises new skill
                    ↓ ADR-010 critic gate (quality check)
                    ↓ install into private ecosystem
                    ↓ retry task
```

**Key files:** `src/core/evolution/evolver.py`, `src/core/evolution/code_synthesizer.py`, `src/core/evolution/capability_verifier.py`

---

## ADR-005: Think-at-Rest {#adr-005}

**Date:** 2026-05-05 | **Status:** Implemented

During idle periods, a speculative decoder generates rapid gap hypotheses. The full model evaluates each (score 0.0–1.0). Confirmed gaps queue for Tier 2 synthesis. Proactive insights can be pushed to Telegram.

- Idle threshold: 300s (configurable)
- Thought interval: 1800s
- Min score: 0.65
- Max proactive messages: 4/day
- Journal: `~/.kernel-evolving/workspace/thoughts/`

**Key files:** `src/services/thought_engine.py`, `src/services/observer.py`

---

## ADR-006: Capability Verification {#adr-006}

**Date:** 2026-05-06 | **Status:** Implemented

Binary yes/no verification after semantic skill match. Prevents false positives (e.g., `kernel-doc-retrieval` matching "audio transcription"). Uses the model to verify: "Can skill X actually do task Y?"

---

## ADR-007: Capability Recommendation {#adr-007}

**Date:** 2026-05-06 | **Status:** Implemented

When verification fails, surface partial matches. The agent can recommend: "Skill X partially covers this, skill Y covers that part — want me to combine them or synthesise a new one?"

---

## ADR-008: Critic Replica + Tools {#adr-008}

**Date:** 2026-05-07 | **Status:** Implemented

Replicas (critic, planner, pipeline) run with their own system prompt, separate history, and can use tools. They can also communicate results to each other via `inter_replica_messaging`.

**Key files:** `src/core/replica/`

---

## ADR-009: Goal Discovery Boot Cap {#adr-009}

**Date:** 2026-05-11 | **Status:** Implemented

Cap the number of goals generated at boot. Deferred goals drain slowly (1 per cycle). Prevents goal explosion on startup.

---

## ADR-010: Evolution Pipeline Critic {#adr-010}

**Date:** 2026-05-11 | **Status:** Implemented

Critic gate before installing a newly synthesised skill. The critic evaluates: does this skill actually solve the gap? Is the code safe? Does it follow the skill schema?

**Key files:** `src/core/evolution/recommender.py`

---

## ADR-011: Micro-Planner Triage {#adr-011}

**Date:** 2026-05-11 | **Status:** Implemented

Complex multi-step requests are decomposed into ordered steps by a micro-planner before inference. The planner runs through the local model (not cloud).

**Key files:** `src/core/pipelines/`

---

## ADR-012: Async Pipeline {#adr-012}

**Date:** 2026-05-11 | **Status:** Implemented

Background job queue for long-running tasks (skill synthesis, evolution loops). Jobs have a TTL (default 3600s). The pipeline supports speculative stages.

---

## ADR-013: Multi-Provider + Trajectory Fine-Tuning {#adr-013}

**Date:** 2026-05-11 | **Status:** Proposed

Inference routing: local-first (Nemotron), cloud fallback on thermal/unavailable. Trajectory collection: every successful tool chain with critic score ≥ 0.7 is captured as JSONL. When enough accumulate, a TRL SFT fine-tune triggers on HuggingFace Jobs. The resulting LoRA adapter is shadow-tested before promotion.

**Providers:** local, openai, anthropic, copilot, openrouter, hf
**Thermal fallback:** GPU ≥ 85°C → route to cloud; ≤ 78°C → resume local

---

## ADR-014: ADR-to-Codebase Diffusion {#adr-014}

**Date:** 2026-06-12 | **Status:** Proposed

Train a model to go from prompt → ADR → codebase using DiffusionGemma 26B-A4B-it as base. The ADR serves as a compressed intermediate representation. Encoder-decoder architecture with bidirectional canvas generation and cross-attention to ADR context.

**Separate project** — not part of kernel-evolving runtime. See `.specs/adr/ADR-014-adr-to-codebase-diffusion.md`.

---

## Current State & Known Issues

### Tool calling issue (2026-06-12)

**Symptom:** Nemotron-Diffusion-3B consistently refuses to call tools. Responds with "I can't access the internet" or "I can't access your file system" instead of generating `<function_calls>` XML.

**Root causes identified:**
1. **Model not fine-tuned for tool calling** — Nemotron is a diffusion LM, not a function-calling model. The chat template supports tools (`<function_calls>` XML format), but the model doesn't reliably generate them.
2. **System prompt mismatch** — The 21K context prompt describes tools in markdown tables. The actual tool definitions are passed separately via chat template. The model may not connect the two.
3. **AR-only for tool loops** — `_handle_infer_with_tools` uses `_model.ar_generate()` (pure AR), not the hybrid `linear_spec` diffusion mode. The diffusion sampler might produce different results.

**Audit findings (13 failed requests):**
- `no_skill` failures: model says "I can't" when it should call a tool
- `skill_error` failures: tool was called but errored (e.g., `model_client` import)
- Repeated refusal patterns: "I can't access external data", "I can't execute commands"

**Next step:** Test tool calling in isolation — minimal system prompt, one tool at a time, verify if the model generates proper `<function_calls>` XML at all.

### Model slots

| Slot | Model | Role | VRAM |
|---|---|---|---|
| primary | Nemotron-Labs-Diffusion-3B (4-bit) | Text inference, tool calls | ~3GB |
| audio | Gemma 4 E2B-it (4-bit) | STT, vision, multimodal | LRU-evictable |

### Generation modes

| Mode | How it works | Use case |
|---|---|---|
| `ar` | Standard autoregressive | Tool loops (current) |
| `diffusion` | Full diffusion, block_size tokens at once | Bulk generation |
| `linear_spec` | Hybrid AR + diffusion with speculation | Default for chat |

### Configuration

- **Model:** `nvidia/Nemotron-Labs-Diffusion-3B`
- **Quantization:** 4-bit (BitsAndBytes NF4)
- **Context:** 64K (native 256K, capped for VRAM)
- **Backend:** HF transformers (vLLM disabled — KV cache OOM)
- **Speculative decoding:** Disabled (drafter incompatible)

---

## Workspace layout

```
~/.kernel-evolving/
├── workspace/
│   ├── data/           # databases (evolution, prompts, trajectories)
│   ├── memory/         # chat history, long-term memory, embeddings
│   ├── artifacts/      # finetune adapters, eval runs, trajectories
│   ├── thoughts/       # Think-at-Rest journal + ideas
│   ├── tmp/            # scratch files (model writes here)
│   ├── runtime/        # model activity tracking
│   └── USER.md         # user facts (auto-updated)
├── ecosystem/
│   ├── community/      # public skills (git cloned)
│   └── private/skills/ # synthesised + installed skills
└── config.yaml         # all configuration
```

---

## Related documents

- **AGENTS.md** — persona, workflow rules, tool reference (in this repo)
- **CHANGELOG.md** — version history
- **CONTRIBUTING.md** — development guidelines
- **`.specs/adr/ADR-014-*.md`** — DiffusionGemma ADR-to-codebase project
- **OpenClaw workspace** — `~/.openclaw/workspace/` (READ ONLY from kernel-evolving)
