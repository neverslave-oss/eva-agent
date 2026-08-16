# ADR-004: Kernel Self-Evolving Agent — Capability Discovery via Replicas

**Date:** 2026-05-04
**Status:** Proposed
**Author:** Olly (Fabio review pending)
**Paper:** [Self-Evolving Software Agents — arXiv:2604.27264](https://arxiv.org/abs/2604.27264)
**Sandbox:** Required — must not touch production Kernel until validated

---

## Motivation

The paper (AAMAS 2026, University of Trento) introduces a BDI–LLM architecture where an **Automated Evolution Module** runs *alongside* the agent's reasoning loop — not inside it. When the agent encounters a task it cannot handle with its current knowledge, goals, or capabilities, the module triggers an evolution cycle: discover what's missing, acquire it, validate it, and retain successful behaviours for future use.

The key insight that maps directly to our stack:

> *"The evolution module is triggered when perceived information cannot be interpreted or exploited using the current knowledge and goal structures."*

Kernel already has this failure mode — it regularly hits tasks where it lacks the right skill, context, or routine. Today it just fails or halts. This ADR proposes making that failure a **trigger for autonomous capability acquisition**.

---

## What "Evolution" Means for Kernel (Full Scope)

The paper proposes full code synthesis (LLM writes new plans/actions). Gemma 4 alone is not reliable enough for this — but rather than ruling it out, we **delegate code synthesis to a more capable model** via a two-tier evolver design.

Evolution covers two levels:

### Level 1 — Capability Acquisition (Gemma 4, always available)
When Kernel lacks a skill that already exists in the ecosystem:
- Evolver replica searches `fabiopacifici-bot/*` and ecosystem SKILL.md files
- Finds a matching skill, clones it, reloads Kernel, retries the task
- No code written — pure acquisition from curated library

### Level 2 — Code Synthesis (capable model, when ecosystem has no match)
When no existing skill covers the gap, the evolver escalates to a **code-synthesis replica** backed by a more capable model. Gemma 4 detects the gap and hands off — it does not attempt to write the code itself.

**Model selection priority for the code-synthesis replica:**

| Priority | Provider | Condition |
|---|---|---|
| 1 | **Olly (OpenClaw)** | Kernel is running inside an OpenClaw environment — message Olly directly via HTTP/IPC, Olly spawns a subagent with full coding capability |
| 2 | **Cloud provider** | OpenAI / Anthropic / GitHub Copilot configured in Kernel's provider list — use the best available coding model |
| 3 | **Local bigger model** | A capable local coding model present (e.g. Qwen3-Coder, DeepSeek-Coder) and sufficient RAM available — load on demand |

This ties directly into the **multi-provider Kernel roadmap** — the evolver becomes the first real consumer of provider-agnostic model routing.

| Paper concept | Our implementation |
|---|---|
| Discover unmet goals | Kernel detects no skill/routine match |
| Evolution module triggers | Spawn `evolver` replica (Gemma 4) |
| Search existing capabilities | Ecosystem scan — Level 1 |
| Synthesize new behaviours | Escalate to code-synthesis replica — Level 2 |
| Validate | Replica tests capability, reports confidence |
| Inherit successful behaviours | Skill persisted in `private/skills/` |
| Discard failures | Clean gap report, tracker todo created |

---

## Architecture

### Trigger Condition

Kernel's agent loop currently falls through to `infer_with_tools` when no skill/routine matches. We add an **evolution hook** at this fallback point:

```
User message
    │
    ▼
Skill match? ──yes──► run skill
    │ no
Routine match? ─yes──► run routine
    │ no
[EVOLUTION HOOK] ◄── new
    │
    ▼
Spawn Evolution Replica
    │
    ▼
Replica: search → acquire → validate → report
    │
    ▼
Kernel: retry with new skill OR report gap
```

The hook is opt-in per message via `/evolve <task>` command initially, auto-triggered only after sandbox validation.

---

### Two Replica Roles

#### Role 1: `evolver` (Gemma 4 — always available)

Responsible for Level 1: ecosystem search and acquisition. Also decides whether to escalate to Level 2.

```python
"evolver": {
    "system": (
        "You are an evolution replica for Kernel. Your job is to discover and acquire "
        "the capability needed to handle a task Kernel could not complete.\n\n"
        "Steps:\n"
        "1. Analyse what the task requires that Kernel currently lacks.\n"
        "2. Search the ecosystem (read SKILL.md files in ~/.kernel/ecosystem/).\n"
        "3. Read relevant context files if needed (workspace docs, MEMORY.md).\n"
        "4. If a matching skill exists: install it (git clone into private/skills/).\n"
        "5. If a matching routine exists: install it into ecosystem/routines/.\n"
        "6. If nothing matches: set escalate=true in your report.\n"
        "7. Report: found, installed, confidence, retry, escalate, gap description."
    )
}
```

#### Role 2: `code-synthesizer` (capable model — provider-routed)

Responsible for Level 2: writing a new skill from scratch when the ecosystem has no match. Only spawned when `evolver` reports `escalate=true`.

```python
"code-synthesizer": {
    "model": "<resolved at runtime — see provider priority below>",
    "system": (
        "You are a code synthesis replica for Kernel. Write a new Kernel skill to cover "
        "a capability gap. Output a complete SKILL.md + scripts/ directory.\n\n"
        "Requirements:\n"
        "- SKILL.md must have valid frontmatter (name, description, commands, exec)\n"
        "- Script must be self-contained and runnable via exec_shell\n"
        "- Include a requirements.txt if Python deps are needed\n"
        "- Do not write to any path outside ~/.kernel/ecosystem/private/skills/<name>/"
    )
}
```

**Provider resolution at runtime:**
```
1. Olly reachable? (OPENCLAW_URL set + /health ok)
   → POST to Olly's session, Olly spawns coding subagent
2. Cloud provider configured? (OPENAI_API_KEY / ANTHROPIC_API_KEY / GITHUB_TOKEN)
   → Use best available coding model (gpt-4.1 / claude-sonnet / copilot)
3. Local coding model present? (check HF_HOME for Qwen3-Coder, DeepSeek-Coder)
   and free RAM > model size?
   → Load locally, run inference
4. None available
   → Report gap, create tracker todo, do not attempt synthesis
```

Both replicas have access to: `exec_shell`, `read_file`, `write_file`, `http_get`.

---

### Evolution Cycle (Step by Step)

```
1. DETECT
   Kernel fails to match skill/routine → logs gap: {task, missing_capability}

2. SPAWN evolver (Gemma 4)
   POST /replica/spawn {"role": "evolver", "task": "<original user task>"}

3. SEARCH (evolver)
   a. Read ~/.kernel/ecosystem/ SKILL.md files (followlinks=True)
   b. Read fabiopacifici-bot/* GitHub repo list via http_get
   c. Read relevant workspace context files if needed
   d. Match task requirements against available capabilities

4a. ACQUIRE — Level 1 (evolver, if ecosystem match found)
   a. git clone <repo> ~/.kernel/ecosystem/private/skills/<name>
      OR copy ROUTINE.md to ~/.kernel/ecosystem/routines/<name>/
   b. Verify SKILL.md frontmatter is parseable
   c. Report: found=true, installed=[...], confidence, retry=true, escalate=false

4b. ESCALATE — Level 2 (evolver, if no match found)
   a. Report: found=false, escalate=true, gap="<description>"
   b. Kernel resolves provider (Olly → cloud → local → none)
   c. SPAWN code-synthesizer with resolved model + gap description
   d. code-synthesizer writes SKILL.md + scripts/ to a temp dir
   e. evolver validates output (frontmatter parseable, script executable)
   f. On pass: move to private/skills/<name>, report confidence
   g. On fail: discard, report gap, create tracker todo

5. VALIDATE (evolver or code-synthesizer)
   Assign confidence: HIGH / MEDIUM / LOW

6. REPORT (back to Kernel main)
   {
     "found": true/false,
     "installed": ["skill-name"] or [],
     "confidence": "HIGH/MEDIUM/LOW",
     "retry": true/false,
     "escalated": true/false,
     "provider_used": "olly|openai|local|none",
     "gap": "description if nothing resolved"
   }

7. INHERIT or DISCARD (Kernel main)
   - confidence HIGH/MEDIUM + retry=true → reload skills, retry task
   - confidence LOW → ask user before retrying
   - gap unresolved → report to user + auto-create tracker todo
```

---

## Sandbox Isolation Plan

**Production Kernel must not be touched until the evolution loop is validated.**

### Sandbox setup
- Clone Kernel repo to `repositories/kernel-sandbox/`
- Run on port **8779** (non-conflicting with production 8769)
- Separate ecosystem: `~/.kernel-sandbox/ecosystem/`
- Separate `.env` with sandbox Telegram chat ID (or no Telegram)
- `EVOLUTION_ENABLED=true` env flag — evolution hook only fires when this is set

### What to test in sandbox
1. Manual trigger: `POST /evolve {"task": "search the web for X"}` — expects browser-automation skill to be found and installed
2. Auto-trigger: send a task Sandbox Kernel has no skill for — verify evolver replica spawns
3. Gap reporting: send an impossible task — verify clean gap report, no broken state
4. Double-install guard: send same unknown task twice — verify no duplicate clone
5. Stability: run 10 diverse tasks — verify production-equivalent skills still work

### Go/No-Go criteria before production
- [ ] Evolution loop completes without crashing Kernel API
- [ ] Installed skills are valid and callable after reload
- [ ] No skill installed that wasn't already in the ecosystem (no arbitrary git clones)
- [ ] Gap reports are accurate and actionable
- [ ] Kernel health endpoint still returns ok throughout

---

## New Commands

| Command | Description |
|---|---|
| `/evolve <task>` | Manually trigger evolution for a task |
| `/evolution status` | Show evolution log: gaps found, skills acquired |
| `/evolution history` | List all past evolution events with outcomes |
| `/evolution disable` | Turn off auto-evolution (manual only) |

---

## Safety Constraints (Non-Negotiable)

1. **Allowlist only (Level 1)** — evolver may only clone from `fabiopacifici-bot/*`. No arbitrary URLs ever.
2. **Code synthesis isolated (Level 2)** — code-synthesizer writes only to a temp dir first; evolver validates before any move to `private/skills/`. No direct writes to live ecosystem.
3. **exec_shell gate** — all shell commands from both replicas go through Kernel's existing inline Telegram approval gate.
4. **Sandbox first** — `EVOLUTION_ENABLED=true` flag required; production stays `false` until validated.
5. **Idempotent installs** — check skill path exists before clone or move.
6. **Max 1 install per evolution cycle** — no runaway loops.
7. **Provider credentials never logged** — API keys resolved at runtime from env, never written to disk or included in replica context.

---

## Relation to Paper

| Paper mechanism | Our status |
|---|---|
| BDI reasoning loop | ✅ Kernel's agent loop (agent.py) |
| Automated Evolution Module | 🔲 New — evolver + code-synthesizer replicas |
| Variation (discover new capabilities) | 🔲 Level 1: ecosystem search |
| Synthesis (generate new capabilities) | 🔲 Level 2: code-synthesizer via capable model |
| Selection (pick best match) | 🔲 Confidence scoring in evolver |
| Inheritance (retain successes) | 🔲 git clone / move → private/skills/ |
| Behavioural stability | 🔲 Allowlist + temp dir validation + exec_shell gate |
| Multi-provider routing | 🔲 Prerequisite for Level 2 — Kernel multi-provider roadmap |
| RAG for LLM reasoning | 🔲 Future — kernel-doc-retrieval bridge |

---

## Dependencies

- **Kernel multi-provider routing** — required for Level 2 code synthesis. Provider resolution (Olly / cloud / local) must be implemented before Level 2 is enabled. This ADR is a first-class driver of that roadmap item.
- **Olly IPC/HTTP bridge** — when Kernel runs inside OpenClaw, a lightweight endpoint to forward synthesis requests to Olly. Olly then spawns a `github-copilot/claude-sonnet-4.6` subagent with full coding capability.
- **kernel-doc-retrieval** — evolver can use `/doc query` to gather context from local documents before searching the ecosystem (e.g. read a spec PDF before deciding what skill to build).

---

## Future Extensions (Post-Sandbox Validation)

- **Context-aware evolution**: evolver reads workspace docs + collective memory before searching
- **Collective evolution**: Olly notifies evolver when a new skill is published — Kernel pre-acquires
- **RAG bridge**: document knowledge gaps trigger kernel-doc-retrieval instead of (or before) code synthesis
- **Fine-tuning signal**: every successful code synthesis event (task, gap, generated SKILL.md + script) is a labelled training pair. Combined with Kernel interaction logs, replica outputs, and evolver trajectories, this becomes a self-generated dataset for fine-tuning Gemma 4 into a model natively capable of skill synthesis — eliminating the need to delegate Tier 2 to an external provider. Target: a small Gemma 4 fine-tune with strong coding capability trained entirely on Kernel's own evolution data.

---

## Decisions (finalised 2026-05-04)

1. **Auto vs manual trigger** — `/evolve` manual only until 10+ successful sandbox cycles confirm stability. Auto-trigger never fires without explicit graduation decision from Fabio.
2. **Allowlist scope** — `fabiopacifici-bot/*` only to start. ClawHub community repos opened only once confidence scoring validates skill behaviour before inheritance. No arbitrary URLs ever.
3. **Gap persistence** — Yes: unresolved gaps auto-create a tracker todo. This closes the loop between Kernel's limits and the dev backlog without relying on memory.
4. **Evolver: ephemeral** — Spawn fresh per evolution cycle. Simpler, safer, no state leakage between unrelated tasks.
5. **Sandbox port** — 8779 confirmed.

---

## Implementation Status (May 4-5, 2026)

### Repo: `fabiopacifici-bot/kernel-evolving` (private fork)
### Branch: `feature/adr004-self-evolving`
### Latest: `v0.4.0-adr004`

| Component | File | Status |
|---|---|---|
| Tier 1 evolver | `src/evolver.py` | ✅ Done |
| Remote ecosystem discovery | `src/evolver.py` | ✅ Done (fetches GitHub API) |
| Tier 2 code synthesizer | `src/code_synthesizer.py` | ✅ Done (gpt-5.4 validated) |
| Evolution log (SQLite) | `src/evolution_log.py` | ✅ Done |
| Evolution hook | `src/evolution_hook.py` | ✅ Done (wired into agent.py) |
| **Evolution state machine** | `src/evolution_state.py` | ✅ Done |
| Goal discovery (proactive) | `src/goal_discovery.py` | ✅ Done (background thread) |
| Monitoring endpoints | `src/api.py` | ✅ Done |
| D3 dashboard | `src/evolution_dashboard.html` | ✅ Done |
| Sandbox container | `Dockerfile.sandbox-lite` | ✅ Done |
| Test suite | `tests/test_evolver.py` | ✅ 39 tests passing |

### Evolution State Machine

Evolution is now supervised by a thread-safe state machine:

```
STATES: running → paused ↔ running → stopped
```

- **`EVOLUTION_MAX_ITERATIONS`** env var (default: 10) — agent auto-pauses when cap reached
- **`POST /evolution/control`** — `{"action": "start|pause|resume|stop|reset", "cap": N}`
- **`POST /evolution/trigger`** — manually send a task, returns live result
- **`GET /evolution/state`** — current state, iterations, remaining, control log
- Auto-pause on cap: guarantees no unbounded autonomous evolution without supervision

### Monitoring Dashboard

`GET /evolution/dashboard` — full D3 interactive dashboard:
- Force-directed skill acquisition network (Kernel core → acquired/synthesised skills → gaps)
- Timeline lanes: T1 acquire / T2 synthesise / gap events
- Live event log with SSE updates
- **Control bar**: Start / Pause / Resume / Stop / Reset buttons
- **Task trigger**: type any task → 🧬 Evolve button → see result inline
- Iteration counter with cap, state badge (green/orange/red)

### Live Validation (May 5, 2026)

Evolution loop tested end-to-end in Docker sandbox (port 8779):
- 1000+ events processed
- 4 skills autonomously acquired/synthesised:
  - `markdown-to-pdf` (T1 acquired)
  - `python-security-scan` (T1 acquired)
  - `telegram-weather-report` (T1 acquired)
  - `web-latest-ai-agent-papers` (T1 acquired, GPT-5.4 synthesised)
- Tier 2 synthesis (GPT-5.4) confirmed working
- State machine: pauses at cap, manual trigger works

### Pending Before Production Merge

- [ ] Docker sandbox validation: 10+ supervised cycles with state machine
- [ ] Fabio: review and approve evolution results
- [ ] Open PR from `feature/adr004-self-evolving` to `main` in kernel-evolving
- [ ] Separate PR to production `fabiopacifici-bot/kernel` (after kernel-evolving validated)
- [ ] Wire `OPENCLAW_URL` for Olly-as-provider (Tier 2 via Olly instead of raw OpenAI)

---

## Trajectory Logging & Fine-Tune Roadmap

### What to log

Three data sources feed the fine-tuning dataset:

| Source | What it captures | Format |
|---|---|---|
| **Code synthesizer** | `(task, gap_description)` → `(SKILL.md, scripts/)` | Input/output pair per synthesis call |
| **Interaction log** | All messages where evolution triggered + outcome | Conversation trace with result label |
| **Replica outputs** | What each replica did, what skill gap it addressed, success/failure | Structured episode |
| **Evolver search** | Task → ecosystem search results → acquisition decision | Retrieval + decision trace |

### Dataset structure (JSONL)

```jsonl
{"type": "synthesis", "task": "convert markdown to PDF", "gap": "No skill found for...",
 "prompt": "<full synthesis prompt sent to provider>",
 "output": {"skill_md": "---\nname: markdown-to-pdf\n...", "scripts": {"convert.py": "..."}},
 "validation": "PASS", "provider": "openai/gpt-5.4", "timestamp": "..."}

{"type": "interaction", "message": "search the web for AI papers",
 "evolved": true, "skill_acquired": "web-ai-papers", "confidence": "MEDIUM", "timestamp": "..."}

{"type": "replica", "role": "evolver", "task": "...",
 "search_results": [...], "acquired": "skill-name", "outcome": "success", "timestamp": "..."}
```

### Fine-tune target

- **Base model**: Gemma 4 E2B-it (same model already running in Kernel)
- **Method**: SFT on synthesis pairs — teach Gemma 4 to output valid SKILL.md + scripts directly
- **Goal**: Tier 2 code synthesis runs locally on Gemma 4, no external API needed
- **Infra**: HuggingFace Jobs (TRL SFT trainer) — see hugging-face-model-trainer skill
- **Dataset location**: `~/.kernel/workspace/trajectories/` (JSONL, never in repo)

### Implementation todos

- [ ] Add trajectory logger to `evolution_log.py` — `log_synthesis_trajectory(prompt, output, validation, provider)`
- [ ] Add trajectory logging call in `code_synthesizer.synthesize()` after successful validation
- [ ] Add interaction trajectory logging in `evolution_hook.maybe_evolve()`
- [ ] Add replica trajectory logging in `replica.py` spawn/complete hooks
- [ ] Build `scripts/export_training_data.py` — exports JSONL from SQLite + trajectory files
- [ ] When dataset reaches ~500 synthesis pairs: evaluate fine-tune viability
- [ ] Fine-tune job on HF Jobs using TRL SFT, evaluate on held-out synthesis tasks
- [ ] If validated: ship fine-tuned model to HuggingFace, wire as local Tier 2 provider in code_synthesizer.py
