# ADR-021 — Tier 2 Synthesis Human-Approval Gate

**Status:** Accepted  
**Date:** 2026-05-30  
**Supersedes:** ADR-010 (adds a gate before install; ADR-010 pipeline unchanged)

---

## Context

Tier 2 (code_synthesizer + critic pipeline) autonomously creates and installs skills whenever it
detects a capability gap.  Each synthesis call hits a paid provider (OpenAI / Anthropic / Olly).
Past autonomous runs produced many low-value or redundant skills that were later purged — real
money wasted on no-value output.

Core problem: the agent has no cost awareness before committing to an LLM synthesis call and
installing the result.

---

## Decision

**Tier 2 synthesis MUST pause and ask the user before spending any tokens or installing any skill.**

The gate fires at the earliest moment after Tier 1 escalates — *before* `_try_tier2` calls the
provider or writes any file.

### Flow

```
Tier 1 gap detected
        │
        ▼
 [ADR-021 GATE] Send Telegram message:
   "🧬 Evolution wants to create a skill
    Skill: `<proposed-name>`
    Gap:   <gap description>
    Provider: <olly|openai|anthropic|local>
    This will use API credits. Approve?"
   Buttons: [✅ Approve] [❌ Reject]
        │
   ┌────┴────┐
 Approve   Reject
   │          │
   ▼          ▼
Run Tier 2  Return EvolutionResult(
pipeline    found=False, gap="User rejected synthesis")
   │
   ▼
Normal ADR-010 pipeline (synthesise → critic → install → notify)
```

### Pending state

A `SynthesisPending` record is stored in `~/.kernel-evolving/pending_synthesis.json` (array of
objects, keyed by `id`).  Keys: `id`, `task`, `gap`, `provider`, `chat_id`, `created_at`.

The Tier 2 call returns an `EvolutionResult(pending_approval=True)` immediately.
When the user taps Approve, a background thread runs the actual synthesis and notifies the user.
When the user taps Reject, the record is deleted and evolution is notified via negative reward.

### EvolutionResult additions

```python
pending_approval: bool = False   # waiting for user gate decision
synthesis_id: str | None = None  # key in pending_synthesis.json
```

---

## Cost awareness

The gate message shows which provider will be used so the user can make an informed decision.
Provider cost indicators are best-effort — the message tells the user "this uses API credits"
only when the provider is not `local`.

---

## Consequences

- No Tier 2 synthesis ever happens without explicit user approval.
- Synthesis latency increases by the time it takes the user to tap a button — acceptable because
  Tier 2 is an async background activity.
- Rejected synthesis writes a negative reward in the evolution log (same as ADR-020 rejection).
- Local provider (HF model) may optionally bypass the cost warning but still requires approval.
  (config flag: `tier2_approval_skip_local: false` — default false = always ask)
