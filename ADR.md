# ADR.md — Kernel-Evolving Architecture Decision Records

**Last updated:** 2026-10-01
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
| [023](#adr-023) | Unified Look Tool | Implemented | Vision for EVA — registry, router, eyes, dispatch |
| [024](#adr-024) | Unified Sensors Tool | Implemented | Read Pi sensor data (temp/humidity/moisture), read-only phase |
| [025](#adr-025) | Agentic Browser-Use Tool | Implemented | Multi-step web tasks via browser-use, headless |
| [026](#adr-026) | Tools Loop Cap 60 | Implemented | Max tools call loop steps 30 → 60 |

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

## ADR-015: Agent Auto-Discovery {#adr-015}

**Date:** 2026-05-25 | **Status:** Proposed

kernel-evolving today hardcodes its peer topology (`peers:` in `config.yaml`) and has no awareness of other kernel-evolving instances on the LAN, the OpenClaw gateway, or locally-active coding agents (Codex CLI, opencode, Claude Code, Aider, Cursor). This hurts portability, prevents offloading coding tasks, and blocks cooperative clustering.

**Decision:** a three-tier auto-discovery system that activates on boot and refreshes every `discovery_interval_s` (default 60s):

1. **Tier 1 — Local port scan (always on):** probe well-known agent ports (kernel-base 8769, self 8779, shadow 8780, OpenClaw 18789 `/health`, opencode 3000, Codex 8888, Claude Code 40000) with a 1s timeout; register a peer when a recognised identity field is returned. Self-detection skips a peer whose `node_id` equals `self_identity.codename`.
2. **Tier 2 — mDNS/Zeroconf (LAN, when `cluster.discovery: mdns`):** announce `{node_id}._kernel-evolving._tcp.local.` with TXT records (version, role, models, adr list) and browse for other instances + OpenClaw. Graceful degradation to Tier 1 if `zeroconf` is unavailable. New dependency (~200KB).
3. **Tier 3 — Coding agent discovery:** detect coding agents via their MCP server / HTTP API (`opencode`, Claude Code, Codex, Aider, Cursor, GitHub Copilot) and register them as `coding`-role peers. An `mcp_client.py` maps `tools/list` into the kernel-evolving skill schema so coding-agent capabilities appear as virtual skills `coding/{agent}/{tool}`.

Discovery is **passive by default** — `delegation.enabled` defaults to `false`. Routing only delegates when enabled. Found peers are kept in memory (re-discovered each boot), exposed via `GET /peers`.

**Consequences:** zero-config coding-agent discovery and cluster growth; ~50ms startup cost; added optional deps; delegation needs loop-prevention and is opt-in. Cross-machine WAN discovery, encrypted peer auth, and load balancing are explicitly out of scope.

**Key files:** `src/discovery.py`, `src/mcp_client.py` (proposed), `src/agent.py` (routing hook — proposed), `config.yaml`, `docs/PEER_PROTOCOL.md`

---

## ADR-016: Self-Evolution Completeness {#adr-016}

**Date:** 2026-05-26 | **Status:** Proposed

A wiring audit (`AUDIT-2026-05-26-self-evolving-wiring.md`) found kernel-evolving has the **scaffolding** of a self-improving agent but the loops are **not closed**: the agent acquires capability but does not measure, reflect on outcomes, or compound improvements.

**Part A — Wiring fixes (approved, in flight):**
- Pass `infer_fn` from background `maybe_evolve` callers so the capability verifier actually runs in autonomous evolution (`goal_discovery.py`, `thought_engine.py`, `evolver.py`).
- Remove duplicate sync `_pipeline_jobs`/`_prune_pipeline_jobs`/`_run_pipeline_job` leftovers from an incomplete ADR-012 migration (`api.py`).
- Persist `EvolutionResult.recommendations` and apply a small additive boost (+0.05, config-driven) to candidates on the next `search_ecosystem` round.
- Trajectory feedback into evolver scoring is **PARKED** until Nemotron-Diffusion-3B is default (markers `# TODO(nemotron-swap)`).

**Part B — ThinkAtRest audit:** the System 1 / System 2 split, idle gating, category-aware actions, memory seeker, deferred drafts, and `EvolvingThinkAtRest` subclass are all wired. Weaknesses recorded: single eval pass (malformed JSON → zero thoughts), uncalibrated `min_score`, promotion→Telegram has no de-dup, `_maybe_trigger_evolution` swallows `ImportError` silently, thought outcomes not tracked, no compounding link to a skill ledger.

**Part C — Strategic gaps (proposed, not approved for build):** C1 skill performance ledger, C2 outcome-based reward signal, C3 skill retirement/quarantine, C4 self-critique/reflection (post-inference critic already wired at `src/agent.py`; only a REVISE pass is missing), C5 goal provenance + outcome tracking.

**Non-goals:** cross-host federation, full RLHF loop.

**Key files:** `src/goal_discovery.py`, `src/thought_engine.py`, `src/evolver.py`, `src/trajectory_collector.py`, `src/evolution_hook.py`, `src/agent.py`

---

## ADR-017: Voice Model Routing {#adr-017}

**Date:** 2026-05-26 | **Status:** Proposed

Voice requests need STT → inference → TTS clone. Nemotron-Labs-Diffusion-3B is text-only, so STT currently lazy-loads Gemma 4 E2B-it, causing a VRAM spike (~4–5GB), 8–15s cold-start latency, and idle VRAM waste.

**Option A (recommended) — STT via faster-whisper, Nemotron unchanged:** route voice to olly-voice-server `/stt` instead of loading Gemma 4. New `src/stt_client.py` POSTs the audio to `http://127.0.0.1:8766/stt`, falls back to `model.infer_with_audio` if the voice server is unreachable. TTS clone already hits `/tts/clone`. Currently (v1.19.14) the voice pipeline is fully wired and end-to-end tested, but STT still uses the Gemma 4 route; this ADR proposes the cheaper path.

**Option B (deferred) — Gemma 4 ↔ Nemotron hot-swap on voice request:** unload Nemotron, use Gemma 4 for the whole multimodal voice turn, swap back. ~20–30s round-trip; becomes viable only after a pooled VRAM manager exists.

**Decision:** implement Option A; Option B is a future milestone. No change to Nemotron as default text model.

**Key files:** `src/stt_client.py` (proposed), `src/telegram_bot.py`, `src/model_server.py`, `olly-voice-server`

---

## ADR-018: Named Model Slots {#adr-018}

**Date:** 2026-05-26 | **Status:** Accepted | **Branch:** `feat/named-model-slots`

kernel-evolving previously used single-slot module globals (`_model`, `_processor`, `_stt_model`) in `model_server.py` — no way to manage multiple models, route replicas to a specific backend, expose VRAM state, or load/unload on demand.

**Decision:** a **named slot registry** (`SlotRegistry`) as an optional, additive layer; legacy mode (no `model_slots:` section) behaves identically, and globals are kept in sync via `_sync_globals_from_slot()`.

| Role | Model | Eviction |
|---|---|---|
| `primary` | Nemotron 3B | Never evict |
| `audio` | Gemma 4 E2B-it | LRU |
| `draft` / `embed` | future | LRU |

An LRU eviction guard checks `torch.cuda.mem_get_info()` before every `load()`; below `vram_threshold_mb` (default 3000) the oldest non-primary loaded slot is evicted. New RPCs: `load_slot`, `unload_slot`, `slot_status`; `health` now always includes a `slots` key. `Replica` gains an optional `slot: str = "primary"` field so audio replicas can target the audio slot. 406 tests passed at implementation time.

**Consequences:** flexible multi-model management, on-demand audio load/unload, VRAM guard against OOM when Fantasia loads FLUX (~8GB), `health()/slot_status()` observability, and config-driven extensibility (`draft`, `embed`). Known Python GC caveat: references to a released `SlotState.model` block CUDA free.

**Key files:** `src/model_slots.py`, `src/model_server.py`, `src/model_client.py`, `src/replica.py`, `src/api.py`, `config.yaml`, `tests/test_model_slots.py`

---

## ADR-019: Think-at-Rest Observer + Critique Layers {#adr-019}

**Date:** 2026-05-28 | **Status:** Accepted | **Extends:** ADR-005, ADR-010, ADR-016

Two structural open loops identified: **(1)** no evidence gate — a hallucinated high-scoring thought could trigger real skill synthesis; **(2)** no outcome measurement — after Tier 2 synthesis the `retry` flag was never re-run on the triggering task, so skills were never exercised on the problem that created them. Evolution was a write-only pipeline.

**Decision:**
- **ObserverLayer** sits between `ThoughtEvaluator` and `_on_thought_accepted()`. For any thought scoring ≥ `promote_threshold` (0.80) in `gap_reflection`/`self_improvement`, it queries memory/evolution.db/chat history for evidence and classifies `EVIDENCED | SPECULATIVE | CURIOSITY`. Only EVIDENCED gap_reflection drives full evolution; SPECULATIVE goes to **probe mode** (journal + expiry-based probe record); CURIOSITY promotes to idea without evolving. Injects last-N thought summaries as anti-seed to reduce repetition.
- **CritiqueLayer** runs after `maybe_evolve()` when a skill installs (`result.installed` non-empty): invokes the new skill on the original triggering task in a sandboxed run, assesses output vs gap, returns `RESOLVED | PARTIAL | FAILED` with a max 3-iteration retry loop feeding critique notes back into synthesis.
- **Knowledge store** (`knowledge.db`) holds falsifiable claims with confidence scores + evidence backlinks; confidence decays 0.1/week.
- **Signal resolution:** `goal_discovery.resolve_pattern()` clears promoted signals on RESOLVED; a `## Trigger Context` routing hint is written to the skill's SKILL.md.

**Amendment 2026-06-18:** Observer gating logic unchanged; the ADR-005 restructure decides *what* reaches the Observer (curiosity thoughts from real exploration + triggered probes).

**Key files:** `src/thought_engine.py`, `src/goal_discovery.py`, `src/evolver.py`, `src/evolution_hook.py`, `src/capability_verifier.py`, `src/observer_layer.py`, `src/critique_layer.py` (proposed)

---

## ADR-020: User-Request-Anchored Evolution {#adr-020}

**Date:** 2026-05-29 | **Status:** Proposed | **Supersedes:** ADR-019 (partial — extends Observer/Critique intent)

In practice ADR-019's Observer measured topic frequency rather than request fulfillment, and Critique verified description-match rather than user satisfaction. Evolution installed plausible-but-useless skills from abstract self-reflection while real user failures went unaddressed.

**Decision — anchor evolution to real user requests:**
- **Failure logging:** when `infer_with_tools` returns a fallback/refusal/partial answer, write a `failed_request` record (`no_skill | skill_error | refusal | partial`) via `telegram_bot.py` + `failed_requests.py`.
- **ThinkAtRest anchors:** `EvolvingThinkAtRest._get_recent_gaps()` queries `failed_requests WHERE resolved=0` as concrete task anchors; generated thoughts carry `_source_request_id`. Thoughts without a source anchor cannot trigger evolution.
- **Observer:** short-circuits to `EVIDENCED` on `_source_request_id`; frequency-based evidence is demoted to `SPECULATIVE` only.
- **Evolution + Critique** run against the **original user message**, not the paraphrased thought.
- **Human confirmation gate + reward:** on `RESOLVED` the skill output is sent to the user's Telegram with `✅ Yes, resolved` / `❌ No, still broken` inline buttons. Confirmation writes `reward=1` (resolved), rejection loops back with failure notes up to max_iterations then escalates; reward=0 writes a negative trajectory. 48h timeout leaves requests unconfirmed without auto-retry.
- Thought-triggered evolution restricted to `gap_reflection` with a stricter observer gate.

**Amendment 2026-06-18:** `gap_reflection` now only comes from `failed_requests`; `EvolvingThinkAtRest._run_think_cycle()` Phase 1 is the sole evolution trigger; max 3 retries per request then abandoned.

**Key files:** `src/telegram_bot.py`, `src/failed_requests.py` (proposed), `src/thought_engine.py`, `src/observer_layer.py`, `src/critique_layer.py` (proposed)

---

## ADR-021: Tier 2 Approval Gate {#adr-021}

**Date:** 2026-05-30 | **Status:** Accepted | **Supersedes:** ADR-010 (adds a gate before install)

Tier 2 (code_synthesizer + critic) autonomously synthesised and installed skills on any detected gap, hitting paid providers (OpenAI / Anthropic / Olly) each time. Past autonomous runs produced low-value or redundant skills later purged — real money wasted.

**Decision:** Tier 2 synthesis **MUST pause and ask the user before spending any tokens or installing any skill.** The gate fires at the earliest moment after Tier 1 escalates, before `_try_tier2` calls the provider or writes any file. It sends a Telegram message (`🧬 Evolution wants to create a skill … provider … This will use API credits. Approve?`) with `[✅ Approve] [❌ Reject]` buttons. `EvolutionResult` gains `pending_approval: bool` and `synthesis_id`. A `SynthesisPending` record is stored in `~/.kernel-evolving/pending_synthesis.json`; Approve runs synthesis in a background thread, Reject deletes the record and writes a negative reward. `tier2_approval_skip_local: false` by default (always ask, even for local provider).

**Consequences:** no Tier 2 synthesis without explicit approval; synthesis latency increases (acceptable — it's async background); cost message shown whenever provider is not `local`.

**Key files:** `src/evolution_hook.py`, `src/telegram_bot.py`, `~/.kernel-evolving/pending_synthesis.json`, `config.yaml`

---

## ADR-022: Unified Tool-First Pipeline {#adr-022}

**Date:** 2026-07-04 | **Status:** Proposed | **Related:** ADR-004, ADR-008, ADR-011

`agent.triage()` was routing-first, causing tool-call failures: semantic skill matching intercepted `infer_with_tools`-only paths and ran skills via plain `infer()` with **no tools**, so the model answered "I can't access files". Poisoned history and session learnings reinforced denial behaviour.

**Decision:** replace routing-first triage with a **tool-first pipeline** where the model always has tools and decides what to call. Skills and routines become tool-callable via meta-tools (`run_skill`, `search_skills`, `run_routine`, `list_routines`) added to a combined `TOOLS = NATIVE_TOOLS + META_TOOLS` (11 native + 4 meta). `triage()` becomes a thin pre-filter handling only slash commands and status queries; everything else goes to the tool-first two-stage pipeline (Qwen tool-calling loop max_steps=15 → Nemotron synthesis). The system prompt uses progressive disclosure (skill names + one-line descriptions, details via `search_skills`), and **Nemotron synthesis receives only the current turn trajectory** — not full accumulated history — preventing poisoning.

**Migration plan:** Phase 1 immediate fixes (cleared poisoned history, corrective AGENTS.md, PROMPT_LOG_DB fix) done 2026-07-03; Phase 2 tool-first pipeline; Phase 3 context management; Phase 4 validation tests (`test_tool_first_pipeline_*`, `test_nemotron_synthesis_current_turn_only`, etc.).

**Key files:** `src/core/tools.py`, `src/core/agent.py`, `src/core/inference/`, `context.py`

---

## ADR-023: Unified Look Tool {#adr-023}

**Date:** 2026-08-19 | **Status:** Implemented | **Related:** ADR-018, ADR-022

EVA had no unified way to "see": vision was scattered across ad-hoc image handling with no consistent registry of capture sources or routing to the multimodal model slot.

**Decision:** a config-driven, four-layer **`look` tool**:
- **Registry** (`src/core/vision/registry.py`) — declares capture sources (camera, screen, file) and their config.
- **Router** (`src/core/vision/router.py`) — picks the capture backend for a request and dispatches.
- **Eyes** (`src/core/vision/eyes/`) — pluggable capture implementations (`object_face`, `plant_health`).
- **Dispatch** (`src/core/tools.py`) — the `look` entry point tying registry → router → eyes together.

It is registered in the system prompt (`context.py`) and micro-planner so the model invokes it natively. Vision runs through the multimodal model slot (Gemma 4 E2B) with local fallback when cloud vision is unavailable. Verified in code: `_run_look()` at `src/core/tools.py:502`, `route_look` wiring at line 958.

**Consequences:** new vision sources are added by registering a new "eye"; `look` is a first-class native tool; multimodal inference reuses the named audio/vision slot (ADR-018).

**Key files:** `src/core/vision/`, `src/core/tools.py`, `config.yaml`, `tests/test_look_tool.py`

---

## ADR-024: Unified Sensors Tool {#adr-024}

**Date:** 2026-08-20 | **Status:** Implemented (read-only phase) | **Related:** ADR-023, ADR-022

EVA had no way to read environmental sensor data (temperature, humidity, soil moisture) from the Raspberry Pi.

**Decision:** a **`sensors` tool** mirroring the `look` vision pattern so EVA reads environmental data from the Pi's `/pico/sensors` endpoint:
- **Registry** (`src/core/sensors/registry.py`) — declares sensors and env-expands the Pi base URL (`KERNEL_EVO_SENSORS_PI_BASE`).
- **Pi client** (`src/core/sensors/pi.py`) — `read_sensors` GET; `set_relay` POST (pump override) deferred to the hardware phase.
- **Router** (`src/core/sensors/router.py`) — `route_sensors` envelope.
- **Dispatch** (`src/core/tools.py`) — the `sensors` entry point (`_run_sensors` at line 522, wiring at line 961).

Config lives under a `sensors:` section in `config.yaml`; registered in the system prompt (`context.py`), micro-planner, and arg normalization (`tool_arg_utils.py`). **Read-only now** (temp/humidity/moisture/moisture_percent); pump override + Pi display deferred to the hardware phase. Tool count bumped to 14.

**Key files:** `src/core/sensors/`, `src/core/tools.py`, `config.yaml`, `tests/test_sensors_tool.py`

---

## ADR-025: Agentic Browser-Use Tool {#adr-025}

**Date:** 2026-08-18 | **Status:** Implemented | **Related:** ADR-022

`web_search` fetches a single page but cannot perform **multi-step, interactive web tasks** (navigate, click, fill forms, operate web apps) without opening a window on the host desktop.

**Decision:** integrate **browser-use** as a new native `browser_use` tool alongside `web_search`:
- `browser-use` + `playwright` deps in `requirements.txt`; new `browser:` config section (`enabled` / `headless` / `max_steps` / `timeout` / `provider`).
- Native `browser_use` tool in `tools.py` (params `task`/`url`/`max_steps`/`save_screenshot`) with an **async bridge** via `asyncio.run()` (`_run_browser_use` at `src/core/tools.py:999`).
- Reuses the `task_inference` provider (HF Router) for the agent loop; **headless enforced** via `BROWSER_USE_HEADLESS` + `Browser(headless=...)`.
- `max_steps` capped at **50** against runaway loops (verified `max(1, min(max_steps, 50))` at `tools.py:1037`).
- Registered in the system prompt (`context.py`), micro-planner, arg normalization (`tool_arg_utils.py`), and error markers (`memory.py`, `model_server.py`).

**Consequences:** EVA executes real multi-step web interactions; browser automation is a first-class native tool (tool count grew to 12); headless default keeps the host desktop clean.

**Key files:** `src/core/tools.py`, `config.yaml`, `requirements.txt`, `tests/`

---

## ADR-026: Tools Call Loop Cap (30 → 60) {#adr-026}

**Date:** 2026-08-23 | **Status:** Implemented | **Related:** ADR-022, ADR-011

Long, multi-step tasks (browser automation, expertise-field triage, multi-source lookups) were hitting the 30-turn tools loop cap and getting truncated mid-task (raised from 15 earlier, still too low).

**Decision:** raise the maximum tools call loop steps from **30 to 60**, applied consistently across the inference stack so the budget is uniform regardless of active inference path — `src/core/agent.py` (`infer_with_tools(..., max_steps=60)`), `src/core/inference/model.py`, `model_client.py`, `model_server.py`, `provider.py` (all `max_steps=60`), `src/core/tools.py`.

**Consequences:** longer agentic tasks complete without truncation; the cap remains bounded (60 — not unlimited) preserving a safety ceiling against runaway tool loops; budget consistent across all inference providers/paths.

**Key files:** `src/core/agent.py`, `src/core/inference/*`, `src/core/tools.py`

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
