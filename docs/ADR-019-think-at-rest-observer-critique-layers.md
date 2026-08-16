# ADR-019 — Think-at-Rest: Observer Layer, Critique Layer, and Closed-Loop Evolution

**Status:** ACCEPTED  
**Date:** 2026-05-28  
**Author:** Fabio Pacifici (design), Olly (spec)  
**Extends:** ADR-005 (Think-at-Rest), ADR-010 (evolution pipeline critic), ADR-016 (self-evolution completeness)  
**Supersedes (partially):** ADR-016 Part B gap items: "no evidence gate", "retry=True never used", "gap signal not resolved on success"

---

## Context

A session audit on 2026-05-28 revealed that the current Think-at-Rest chain has two structural open loops:

**Open loop 1 — no evidence gate before evolution**  
`ThoughtGenerator` can emit a high-scoring thought with no empirical basis (e.g. "voice-clone is inconsistent") and `_on_thought_accepted()` immediately routes it into `goal_discovery` (weight=3) and/or `_maybe_trigger_evolution()`. The capability verifier (ADR-006) only checks whether a skill *description* matches the task string — it does not check whether the underlying claim in the thought is evidenced by real interactions, failures, or observations. Result: hallucinated self-critique triggers real skill synthesis.

**Open loop 2 — no outcome measurement after evolution**  
After Tier 2 synthesis, `maybe_evolve()` returns `EvolutionResult(retry=True)`. The `retry` flag is set but `agent.triage()` does not re-run the triggering task with the new skill. The synthesised skill is never exercised on the problem that triggered it. The evolution pipeline critic (ADR-010) evaluates SKILL.md text quality only — not behavioural correctness. Nothing closes the "gap detected → gap resolved" signal in `goal_discovery`.

**Root cause summary:** Evolution is a write-only pipeline. Skills accumulate. Learning does not.

Additionally, the following secondary gaps compound the above:

- `ThoughtGenerator` receives no anti-seed context from the recent journal → semantic repetition across cycles despite NFE diversity fix (PR #58)
- `goal_discovery` promoted signals are never resolved on success → pattern weight persists indefinitely
- No distinction between a *claim* (falsifiable, evidence-dependent) and a *desire* (preference, always valid as signal)
- Skill routing knowledge (when to invoke a new skill) is not updated after synthesis

---

## Decision

### 1. Observer Layer

Insert an `ObserverLayer` between `ThoughtEvaluator` and `_on_thought_accepted()`.

**Trigger:** any thought with `score ≥ promote_threshold` (currently 0.80) whose category is `gap_reflection` or `self_improvement` (i.e., any thought that would trigger evolution or a goal signal).

**Responsibilities:**

1. **Evidence query.** Query the memory layer for real evidence supporting the thought's claim:
   - Recent evolution events (`evolution.db`) matching the thought's subject
   - Chat history turns in the current session where the claim's topic appeared
   - Known failures in `promoted_signals.db` with matching text
   - User-written todos or idea files matching the subject

2. **Claim classification.** Classify the thought as one of:
   - `EVIDENCED` — at least one real failure event, user complaint, or test error found
   - `SPECULATIVE` — no real evidence found; thought is model-generated introspection
   - `CURIOSITY` — a desire or idea without a failure claim (valid even without evidence)

3. **Evolution path decision.** Based on classification:

   | Classification | Category | Evolution Path |
   |---|---|---|
   | EVIDENCED | gap_reflection | → full evolution pipeline (Tier 1 → Tier 2) |
   | EVIDENCED | self_improvement | → tracker todo + goal_discovery (weight=1) |
   | SPECULATIVE | gap_reflection | → **probe mode**: journal thought, set a probe trigger on next relevant interaction; do NOT evolve |
   | SPECULATIVE | self_improvement | → journal only, weight=0 in goal_discovery |
   | CURIOSITY | any | → promote to idea, weight=1 in goal_discovery, no evolution |

4. **Probe mode.** When a speculative gap_reflection is parked, the Observer writes a probe record:
   - Subject of the claim (e.g. "voice-clone consistency")
   - What evidence would confirm it (e.g. failed tool call, user complaint, error in log)
   - When the probe expires (default: 7 days without a trigger → discard)
   - If a real interaction triggers the probe before expiry → re-evaluate as EVIDENCED

5. **Anti-seed injection.** The Observer also injects the last N thought summaries (configurable, default 10) into `ThoughtGenerator`'s seed prompt as a "do not repeat" context block, reducing semantic duplication.

**Interface (proposed):**

```python
class ObserverLayer:
    def evaluate(self, thought: dict, memory_ctx: dict) -> ObserverVerdict:
        # returns: classification, evolution_path, evidence_summary, probe_record|None
```

**Config keys (config.yaml):**

```yaml
think_at_rest:
  observer_enabled: true
  observer_evidence_lookback_turns: 100   # chat history turns to search
  observer_evidence_lookback_days: 7      # evolution_events recency window
  observer_anti_seed_count: 10            # recent thoughts injected as anti-seed
  observer_probe_ttl_days: 7              # probe expiry
```

---

### 2. Critique Layer

Insert a `CritiqueLayer` after `maybe_evolve()` completes (i.e. after Tier 1 acquisition or Tier 2 synthesis).

**Trigger:** any evolution cycle that installed at least one skill (`result.installed` non-empty).

**Responsibilities:**

1. **Skill execution.** Invoke the newly installed (or acquired) skill on the original triggering task text. This is a sandboxed dry-run via the agent's `triage()` path with the new skill in scope.

2. **Outcome assessment.** Ask the local model to evaluate the skill's output against the original gap:
   - Did the skill produce output? (non-empty, non-error response)
   - Does the output address the identified gap?
   - Is the output coherent and safe to deliver to the user?

3. **Verdict:**
   - `RESOLVED` — skill ran, output addresses gap → close gap signal, write to knowledge store
   - `PARTIAL` — skill ran, output partially addresses gap → annotate, queue for next iteration
   - `FAILED` — skill errored or output is irrelevant → pass critique notes back to synthesiser for one retry (max 3 total iterations across Observer → Evolution → Critique)

4. **Retry loop (max iterations = 3):**
   ```
   Observer → Evolution (Tier 2 synthesise) → Critique
     FAILED / PARTIAL → re-enter Evolution with critique notes injected into synth prompt
     FAILED after 3 → escalate: write gap todo, notify Telegram, mark signal as UNRESOLVED
   ```

5. **Signal resolution.** On `RESOLVED`:
   - Remove or down-weight the promoted signal in `goal_discovery.promoted_signals.db`
   - Write a resolution record: `{thought, skill_name, evidence_used, run_output_summary, resolved_at}`
   - Update the knowledge store (see §3)

6. **Journal the full cycle.** The `ThoughtJournal` entry is expanded to include: Observer verdict, evolution path taken, Critique verdict, skill run output summary, resolution status.

**Interface (proposed):**

```python
class CritiqueLayer:
    def assess(self, thought: dict, evolution_result: EvolutionResult,
               run_output: str | None) -> CritiqueVerdict:
        # returns: verdict, notes, should_retry, retry_context
```

**Config keys:**

```yaml
think_at_rest:
  critique_enabled: true
  critique_max_iterations: 3
  critique_run_timeout_s: 30
```

---

### 3. Knowledge Store

A lightweight facts store distinct from skills and from the journal.

**Purpose:** Hold falsifiable claims about the system's own state and the user's context, with confidence scores and evidence backlinks.

**Schema (`~/.kernel-evolving/workspace/knowledge.db`):**

```sql
CREATE TABLE facts (
  id           INTEGER PRIMARY KEY,
  subject      TEXT NOT NULL,          -- e.g. "voice-clone"
  claim        TEXT NOT NULL,          -- e.g. "produces inconsistent output"
  confidence   REAL NOT NULL DEFAULT 0.5,
  evidence     TEXT,                   -- JSON array of {source, ts, summary}
  last_updated TEXT NOT NULL,
  resolved     INTEGER DEFAULT 0       -- 1 = claim was tested and confirmed false
);
```

**Lifecycle:**
- Observer adds a fact with low confidence when a speculative thought is parked as a probe
- Critique raises confidence on RESOLVED, lowers on FAILED
- ThoughtGenerator queries the knowledge store as grounding context: *"known claims: {subject} — {claim} (confidence={n})"*
- Facts decay: confidence -= 0.1/week without new evidence; removed at confidence < 0.1

---

### 4. Signal Resolution in goal_discovery

After a Critique `RESOLVED` verdict, `goal_discovery` must be notified to clear or down-weight the triggering pattern:

```python
goal_discovery.resolve_pattern(signal_text: str) -> None
```

Clears matching rows from `promoted_signals` and removes the pattern from `_interaction_log` ring buffer. Prevents already-solved problems from continuing to accumulate weight.

---

### 5. Skill Routing Knowledge Update

After a skill is installed AND the Critique confirms `RESOLVED`, write a routing hint to the skill's SKILL.md under a `## Trigger Context` section:

```markdown
## Trigger Context
<!-- auto-generated by CritiqueLayer on {date} -->
Invoke this skill when: {original_gap_text}
Evidence that prompted creation: {evidence_summary}
Confirmed effective on: {resolved_at}
```

This gives the semantic skill router more signal to match future similar requests.

---

## Updated Think-at-Rest Chain

```
IdleDetector (idle ≥ threshold_s, model not active)
  │
  ▼
ThoughtGenerator
  · seeds: unused_skills, recent_interactions, recent_gaps, evolution_context
  · anti-seed: last N thought summaries (injected by ObserverLayer on next cycle)
  │
  ▼
ThoughtEvaluator (System 2 — score + classify)
  · discard score < 0.65
  │
  ▼
[NEW] ObserverLayer
  · query memory: evidence for this claim?
  · classify: EVIDENCED | SPECULATIVE | CURIOSITY
  · decide: full evolution | probe | journal-only | idea
  · write anti-seed context for next ThoughtGenerator call
  │
  ├─ CURIOSITY / journal-only → ThoughtJournal.write() → done
  ├─ probe mode → write probe record → ThoughtJournal.write() → done
  │
  ▼ (EVIDENCED → evolution path)
_on_thought_accepted()
  · promote_to_idea (score ≥ 0.80)
  · goal_discovery.record_promoted (weight per Observer verdict)
  │
  ▼
maybe_evolve(task, config, skills_dir, infer_fn)
  · Tier 1: search → verify → acquire
  · Tier 2: synthesise → critic (text) → install
  │
  ▼
[NEW] CritiqueLayer
  · run skill on original task (sandboxed)
  · assess output vs gap
  · verdict: RESOLVED | PARTIAL | FAILED
  │
  ├─ RESOLVED → goal_discovery.resolve_pattern()
  │             knowledge_store.update(confidence+)
  │             skill routing hint written
  │             ThoughtJournal.write(full cycle)
  │
  ├─ PARTIAL/FAILED (attempt < 3) → re-enter Tier 2 with critique notes
  │
  └─ FAILED (attempt = 3) → tracker todo, Telegram alert, mark UNRESOLVED
                             ThoughtJournal.write(full cycle, outcome=UNRESOLVED)
```

---

## What Evolution Means

Evolution is not skill accumulation. A new skill represents a *hypothesis* that a capability gap has been filled. The Critique Layer is what tests the hypothesis. A skill that has passed the Critique is an *accepted* capability improvement. A skill that has not been tested is an untested assumption.

Learning = applying skills to real problems + measuring outcomes + updating beliefs (knowledge store).

A system that writes skills without measuring outcomes is a writer, not a learner.

---

## Implementation Plan

| Phase | Component | Est. complexity | Depends on |
|---|---|---|---|
| 1 | `ObserverLayer` class + evidence query (chat_history + evolution.db) | M | — |
| 2 | Anti-seed injection into `ThoughtGenerator` prompt | S | Phase 1 |
| 3 | Probe record store + probe expiry sweep | M | Phase 1 |
| 4 | `CritiqueLayer` class + sandboxed skill runner | L | Phase 1 |
| 5 | Retry loop wiring in `_on_thought_accepted` / `maybe_evolve` | M | Phase 4 |
| 6 | `goal_discovery.resolve_pattern()` | S | Phase 4 |
| 7 | Knowledge store (`knowledge.db`) + ThoughtGenerator grounding | M | Phase 4 |
| 8 | Skill routing hint writer | S | Phase 4 |
| 9 | Extended ThoughtJournal schema (full cycle logging) | S | Phase 4 |

**Phase 1–3 first.** The Observer alone stops hallucinated evolution cycles — highest leverage, lowest risk. Critique layers can follow once the Observer gate is validated.

---

## Acceptance Criteria

- [ ] A speculative thought with no evidence in memory/evolution.db does NOT trigger `maybe_evolve()`
- [ ] A speculative thought IS journaled and a probe record is written
- [ ] An evidenced thought (real failure event found) DOES trigger evolution
- [ ] After Tier 2 synthesis, the new skill is invoked on the triggering task in a sandboxed run
- [ ] RESOLVED verdicts remove the pattern from `promoted_signals.db`
- [ ] Three consecutive FAILED critique attempts produce a tracker todo and Telegram alert
- [ ] ThoughtGenerator receives the last 10 thought summaries as anti-seed context
- [ ] All existing tests pass; new unit tests for ObserverLayer and CritiqueLayer

---

## References

- `src/thought_engine.py` — ThinkAtRest, ThoughtGenerator, ThoughtEvaluator, IdleDetector
- `src/goal_discovery.py` — promoted_signals.db, record_promoted(), _interaction_log
- `src/evolver.py` — Tier 1 search, acquire, validate
- `src/evolution_hook.py` — maybe_evolve(), _try_tier2_pipeline(), _SYNTH_PROMPT, _CRITIC_PROMPT
- `src/capability_verifier.py` — verify() (text-only, not behavioural)
- ADR-005: Think-at-Rest design (amended 2026-06-18 — signal-driven restructure)
- ADR-010: Evolution pipeline critic (text quality critic — complementary to CritiqueLayer)
- ADR-016: Self-evolution completeness audit (see Part B gap items)

### Amendment 2026-06-18

The ObserverLayer design is unchanged — it still gates thoughts via evidence
classification, anti-seed injection, and probe mode. The restructure (ADR-005
amendment) affects WHAT reaches the Observer: only curiosity thoughts from real
exploration data, plus triggered probes. The Observer's core logic is unaffected.
