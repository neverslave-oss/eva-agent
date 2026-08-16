# ADR-006: Capability Verification Layer

**Status:** Approved — implement  
**Date:** 2026-05-06  
**Motivation:** Simulation 3 exposed a new failure mode introduced by semantic matching: 6/15 tasks were resolved via Tier 1 with MEDIUM confidence but matched to skills that cannot actually execute the requested task (e.g., `kernel-doc-retrieval` for "audio transcription"). The agent treats these as resolved, silently skipping Tier 2 synthesis and corrupting Think-at-Rest feedback loops.

---

## Problem

Semantic cosine similarity ranks skill candidates by *surface meaning proximity*, not *capability coverage*. A 768-dim embedding for "transcribe meeting audio" is close to `kernel-doc-retrieval` because both involve documents and extraction — but the skill cannot handle audio. The evolver cannot distinguish:

1. **True match** — skill covers the task ✅
2. **False positive** — similarity high but capability wrong ❌
3. **Partial match** — skill partially covers the task ⚠️

All three cases produce `found=True, confidence=MEDIUM`. Think-at-Rest then sees "no gap" and stops generating gap_reflection thoughts for that domain. The acquisition trajectory is corrupted.

---

## Solution: Two-Stage Pipeline

Insert a lightweight **capability verification step** between semantic ranking and acquisition:

```
search_ecosystem(task)
    → semantic ranking (existing)
    → capability_verify(skill, task)   ← NEW (ADR-006)
        → YES: acquire() as before
        → NO:  try next candidate
        → all fail: escalate to Tier 2
```

### Verification method: Gemma 4 yes/no inference

Single short prompt to the local model server (already running, zero extra infrastructure):

```
Does the skill "{name}" ({description}) have the capability to handle this task:
"{task}"
Answer with exactly YES or NO, then one sentence of reasoning.
```

- ~50 token prompt, ~20 token response — fast
- Binary outcome with reasoning logged to evolution_events
- No external API dependency
- Fail-open on inference error (preserves existing behaviour in degraded environments)

### Confidence mapping

| Semantic score | Verification | Confidence | Action |
|----------------|-------------|-----------|--------|
| ≥ 0.55 | YES | HIGH | Acquire + install |
| ≥ 0.55 | NO | — | Try next candidate |
| 0.3–0.55 | YES | MEDIUM | Acquire + install |
| 0.3–0.55 | NO | — | Try next candidate |
| < 0.3 | — | — | Skip |
| All candidates fail | — | — | Escalate Tier 2 |

---

## Implementation

### New file: `src/capability_verifier.py`

```python
"""
ADR-006: Capability verification via local Gemma 4 inference.
Asks the model whether a skill can handle a task before acquisition.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

VERIFY_PROMPT = """Does the skill "{name}" ({description}) have the capability to handle this task:
"{task}"
Answer with exactly YES or NO, then one sentence of reasoning."""

def verify(skill: dict, task: str, infer_fn) -> tuple[bool, str]:
    """
    Ask Gemma 4 whether skill can handle task.
    Returns (can_handle: bool, reasoning: str).
    Fails open on inference error.
    """
    name = skill.get("name", "")
    description = skill.get("description", "")
    prompt = VERIFY_PROMPT.format(name=name, description=description, task=task)
    try:
        response = infer_fn(prompt, max_new_tokens=60)
        text = response.strip().upper()
        can_handle = text.startswith("YES")
        logger.info(f"[ADR-006] verify '{name}' for '{task[:40]}' → {can_handle}")
        return can_handle, response.strip()
    except Exception as e:
        logger.warning(f"[ADR-006] verification failed: {e} — failing open")
        return True, f"skipped (error: {e})"
```

### Modify `evolver.py` — `run()` method

Accept optional `infer_fn` in `Evolver.__init__`. Iterate top-3 candidates through `verify()`:

```python
for candidate in matches[:3]:
    if self._infer_fn is not None:
        can_handle, reasoning = capability_verifier.verify(
            candidate, task, self._infer_fn
        )
        if not can_handle:
            logger.info(f"[ADR-006] '{candidate['name']}' rejected — {reasoning[:60]}")
            continue
    best = candidate
    break
else:
    # All candidates failed verification → Tier 2
    return EvolutionResult(
        found=False, confidence="LOW", escalated=True,
        gap=f"No verified skill for task: {task}"
    )
```

### Modify `evolution_log.py`

Add two nullable columns (migration-safe):
```sql
ALTER TABLE evolution_events ADD COLUMN verification_result TEXT;
ALTER TABLE evolution_events ADD COLUMN verification_reasoning TEXT;
```

### Modify `evolution_hook.py`

Pass `infer_fn` (from `model_client.infer`) into `Evolver.__init__` at construction time.

---

## Fail-open policy

If model server is unavailable or inference fails → `verify()` returns `True`. Existing Tier 1 resolution rate is preserved in degraded environments. Failure is always logged at WARNING level.

---

## Expected impact (Sim 4 hypothesis)

| Metric | Sim 3 (before) | Sim 4 hypothesis (after) |
|--------|---------------|--------------------------|
| Tier 1 resolved | 15/15 (100%) | ~9/15 (60%) |
| False positives | 6/15 (40%) | 0 |
| Escalated to Tier 2 | 0 | ~6 |
| New syntheses | 0 | ~4–6 |

Resolution rate drops — but all resolutions are trustworthy. Genuinely unhandled tasks get synthesised.

---

## Tests required

- `test_verify_returns_true_on_yes_response`
- `test_verify_returns_false_on_no_response`
- `test_verify_fails_open_on_inference_error`
- `test_evolver_skips_false_positive_tries_next`
- `test_evolver_escalates_when_all_candidates_fail`

---

## Sim 4 protocol

1. Stop background evolution loop on kernel-evolving
2. Restart api.py to pick up ADR-006
3. Re-run same 15 Sim 3 tasks via `/evolution/trigger`
4. Timestamp-slice DB for Sim 4 boundary
5. Compare false positives, escalations, syntheses
6. Update paper to v5 with Sim 4 results + cross-condition table

## GH issue to open

"feat(ADR-006): capability verification layer — eliminate Tier 1 false positives"
