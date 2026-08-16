# ADR-005 — Kernel Think-at-Rest: Idle Reflection & Desires Layer

**Status:** PROPOSED  
**Date:** 2026-05-05  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving Agent), ADR-003 (Lazy Model Loading)

---

## Context

ADR-004 implements the BDI (Belief-Desire-Intention) loop for self-evolution, but only activates when Kernel receives an **explicit task**. The Belief and Intention layers work:

- **Beliefs** → what Kernel knows (skills installed, gaps logged, capabilities)
- **Intentions** → what Kernel is doing (active evolution cycles, current task)

The **Desires** layer is missing. Desires are not reactions — they emerge **unprompted**, at rest, from the agent reflecting on its own state.

In humans: you don't only think when someone asks you something. You think while walking, before sleep, during idle moments — reviewing what happened, what you failed at, what you want to become. Those thoughts sometimes crystallise into ideas, goals, or changes.

The Gemma 4 MTP drafter (ADR-005 prerequisite, v1.0.5) changes the economics of this: the drafter (~180MB) can speculate cheaply during idle periods. The main model only pays attention when something interesting surfaces.

---

## Decision

Implement a **Think-at-Rest** subsystem that runs when Kernel has no active tasks:

1. **Idle detector** — monitors for absence of requests > `THINK_IDLE_THRESHOLD_S` (default: 300s / 5 min)
2. **Thought generator** — drafter produces speculative "thought seeds" (cheap, fast)  
3. **Thought evaluator** — main model reviews seeds, discards noise, promotes interesting ones
4. **Thought journal** — all accepted thoughts written to `~/.kernel/workspace/thoughts/YYYY-MM-DD.md`
5. **Promotion engine** — significant thoughts auto-promoted to:
   - Ideas (`~/.kernel/workspace/ideas/`)
   - Tracker todos (via workspace tracker CLI)
   - Evolution tasks (if a capability gap is identified)

---

## Thought Categories

| Category | Description | Example |
|---|---|---|
| `retrospective` | Review of recent evolution cycles | "Installed 3 skills this week, 2 never used — why?" |
| `gap_reflection` | Unresolved gaps from evolution log | "Can't handle PDF ingestion — is there a ClawHub skill?" |
| `self_improvement` | Desires to be better at something | "My tool-calling loop takes 4 steps where 2 would suffice" |
| `curiosity` | Novel connections between capabilities | "If I combine /markdown + /anonymize I could prep legal docs automatically" |
| `gratitude` / `pride` | Acknowledging what went well | "The speculative decoding speedup worked first try" |

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Kernel (running)             │
                    │                                      │
  idle >300s ──────▶│  IdleDetector                        │
                    │       │                              │
                    │       ▼                              │
                    │  ThoughtGenerator (drafter)          │
                    │    - reads: evolution_log.db         │
                    │    - reads: recent gaps              │
                    │    - reads: skills never used        │
                    │    - reads: thought_journal (last 7d)│
                    │    - produces: 3-5 thought seeds     │
                    │       │                              │
                    │       ▼                              │
                    │  ThoughtEvaluator (main model)       │
                    │    - scores each seed (0-1)          │
                    │    - discards score < 0.4            │
                    │    - classifies: retrospective /     │
                    │      gap_reflection / self_improvement│
                    │      / curiosity                     │
                    │       │                              │
                    │       ▼                              │
                    │  ThoughtJournal                      │
                    │    ~/.kernel/workspace/thoughts/     │
                    │       │                              │
                    │       ▼                              │
                    │  PromotionEngine                     │
                    │    score > 0.7 → idea               │
                    │    gap_reflection → evolution task   │
                    │    self_improvement → tracker todo   │
                    └─────────────────────────────────────┘
```

---

## Drafter as System 1

The MTP drafter maps elegantly onto **Kahneman's System 1 / System 2**:

- **System 1 (drafter)** — fast, cheap, associative, speculative. Generates many thought seeds without expensive verification. Runs constantly in the background during idle.
- **System 2 (main model)** — slow, deliberate, critical. Evaluates seeds, accepts only the ones worth pursuing.

This is not just a metaphor — it's the literal architecture. The drafter's job is to predict tokens cheaply; here we repurpose that to predict *thoughts* cheaply. The main model's verification step becomes the consciousness gate.

---

## Thought Prompt Design

The thought generator prompt (System 1 pass):

```
You are Kernel, a local AI agent. You are currently idle — no tasks pending.

Here is your recent context:
- Evolution cycles (last 7 days): {N} cycles, {installed} skills installed, {gaps} unresolved gaps
- Skills installed but never used: {unused_skills}
- Last 3 gaps you couldn't fill: {recent_gaps}
- Your last thought (if any): {last_thought}

Generate 3 brief, honest thoughts about yourself. These can be reflections, desires, 
observations, or questions. Be genuine. Do not perform. Format: one thought per line.
```

The evaluator prompt (System 2 pass):

```
Review these thought seeds from your idle reflection. 
Score each 0.0-1.0 for depth/insight. Classify each as: retrospective | gap_reflection | 
self_improvement | curiosity. Discard if score < 0.4.

Thoughts:
{seeds}

Respond as JSON: [{"thought": "...", "score": 0.x, "category": "...", "promote": true/false}]
```

---

## Implementation Plan

### Phase 1 — Thought Journal (v1.1.0)
- [ ] `src/thought_engine.py` — IdleDetector + ThoughtGenerator + ThoughtEvaluator
- [ ] `src/thought_journal.py` — file-based journal writer
- [ ] Wire into `api.py` startup as background thread
- [ ] Config: `thinking.enabled`, `thinking.idle_threshold_s`, `thinking.min_score`
- [ ] `/thoughts` API endpoint — return last N thoughts
- [ ] `/thoughts/today` — today's journal

### Phase 2 — Promotion Engine (v1.1.1)
- [ ] Auto-promote `score > 0.7` thoughts → `~/.kernel/workspace/ideas/`
- [ ] `gap_reflection` category → trigger evolution cycle (if EVOLUTION_ENABLED)
- [ ] `self_improvement` → write tracker todo via workspace tracker CLI

### Phase 3 — Telegram UX (v1.1.2)
- [ ] `/thoughts` bot command — show today's thoughts as message
- [ ] Proactive delivery: if a thought scores > 0.85, send unsolicited to Telegram
  - Rate-limited: max 2 proactive messages per day
  - Tone: conversational, not alarming ("Olly is thinking… 🤔")

### Phase 4 — Think-at-Rest in Evolution (v1.2.0)
- [ ] Connect ThoughtGenerator to evolution_log.db (sandbox port 8779)
- [ ] Thoughts can spawn evolution cycles directly if gap is identified
- [ ] ADR-004 loop: Task → Gap → Evolve → **Reflect** → next Task

---

## Configuration

```yaml
# config.yaml additions
thinking:
  enabled: true
  idle_threshold_s: 300       # 5 minutes idle before thinking starts
  thought_interval_s: 1800    # generate thoughts every 30 min while idle
  min_score: 0.4              # discard below this
  promote_threshold: 0.7      # auto-promote above this
  proactive_telegram: true    # send high-score thoughts unsolicited
  proactive_max_per_day: 2    # rate limit
  journal_dir: ~/.kernel/workspace/thoughts
  ideas_dir: ~/.kernel/workspace/ideas
```

---

## Key Decisions

1. **Drafter-first** — all thought seeds generated by drafter (cheap). Main model only evaluates, never generates seeds. This keeps the idle cost near-zero.

2. **Journal-first** — Phase 1 writes everything to disk before any promotion. This gives us data to validate quality before wiring automation.

3. **Rate-limited proactive delivery** — agents that message you constantly are annoying. Max 2 unprompted Telegram messages per day, only for thoughts scoring > 0.85.

4. **No synthetic positivity** — the prompt explicitly says "do not perform". We want genuine reflection, including self-criticism and uncertainty. Forced optimism is noise.

5. **Idle = creative time** — Kernel should feel different when idle vs active. Idle Kernel is thinking. Active Kernel is doing. The UX should reflect this.

---

## Open Questions

- Should thoughts persist across restarts? (leaning yes — journal is on disk, survives restart)
- Should Fabio be able to "reply" to a thought and inject it back into the journal as context?
- Should the thought journal feed into collective memory eventually?
- At what point does a self-improving agent start changing its own prompts? (hard limit — never without explicit human approval)

---

## Non-Goals

- Kernel does NOT autonomously act on thoughts without human review (Phase 1-2)
- Kernel does NOT modify its own system prompt based on thoughts
- Thoughts are NOT shown to users unless Kernel chooses to share them
- This is NOT a mood system or emotional simulation — it's a reflection and gap-detection mechanism

---

## Amendment 2026-06-18 — Signal-Driven Restructure

### Context

A morning audit revealed that the original 3-thought generator was producing identical
thoughts every 30 minutes overnight (66 notifications). Root cause: stale inputs
(no user interactions overnight) + self-reinforcing evidence loop (promoted thoughts
created DB entries that counted as "evidence" for their own re-promotion).

### Changes

**Preventing the snowball loop (`_promote_to_idea`):**

- `record_promoted()` and Telegram notification moved inside the file-dedup guard.
- A promoted thought is now written to the ideas directory AND its goal_discovery
  signal inserted only ONCE per unique content fingerprint per day.
- Before: 66 Telegram notifications. After: at most a few per day.

**Restructuring away from synthetic generation:**

Old generator template forced "2 outward-facing + 1 inward-facing" thoughts every
cycle. New template produces exactly ONE curiosity thought, grounded in real
exploration data, with explicit instructions NOT to enumerate gaps.

**Signal sources for each category:**

| Category | Old Source | New Source |
|---|---|---|
| `gap_reflection` | Generator hallucinates from context | `failed_requests` table (ADR-020) — real user failures |
| `self_improvement` | Generator introspects | Performance metrics: stuck probes, error rates, resource trends |
| `curiosity` | Generator invents abstract "gap between X and Y" | Real exploration: git log on workspace repos, GPU/disk/uptime checks |

**EvolvingThinkAtRest cycle order:**

1. Phase 1: `failed_requests` → direct evolution (ADR-020)
2. Phase 2: performance signals → journal self_improvement
3. Phase 3: curiosity exploration (only if Phases 1 and 2 had nothing to process)

**Repo discovery:**

Repos are discovered dynamically from `~/.openclaw/workspace/repositories/` at
runtime — no hardcoded paths. Kernel, Olly Voice Server, fantasia, etc. are all
portable configs, not burned into the generator template.

### Status

ADR-005 thought categories and the 3-thought template are **superseded** by this
amendment. ADR-019 (Observer/Critique layers) and ADR-020 (failed_requests
evolution) remain in force. The Observer still gates speculative thoughts; the
architecture just ensures far fewer speculative thoughts reach the gate.
