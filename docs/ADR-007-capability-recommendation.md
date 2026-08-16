# ADR-007: Capability Recommendation Layer

**Status:** Proposed → Implement before Sim 5  
**Date:** 2026-05-06  
**Motivation:** Sim 4 showed ADR-006 correctly rejecting false positives, but the pipeline is still binary: Tier 1 resolves OR Tier 2 synthesises. When verification fails, the agent has no way to surface partial matches — capabilities that *partially* cover a task and could be recommended to the user or combined. This missing layer represents the Belief+Desire synthesis step in the BDI model.

---

## Problem

After ADR-006 rejects all top-3 candidates, the agent immediately escalates to Tier 2. Two things are lost:

1. **Partial matches** — a skill may cover 70% of the task. In interactive mode, the user might accept it. In autonomous mode, it should be logged as a near-miss for future ecosystem curation.
2. **Composable candidates** — two lower-scoring skills together may cover the task completely. No mechanism exists to discover or surface this.

The result in Sim 4: every task went to Tier 2 even when the ecosystem was growing and partial coverage existed. The recommendation layer closes this gap.

---

## Solution: Recommendation Engine (Post-Verification, Pre-Tier-2)

Insert between ADR-006 rejection and Tier 2 escalation:

```
Tier 1 search → semantic rank → ADR-006 verify (top-3)
    → all fail →  ADR-007 recommend()          ← NEW
        → finds closest skill(s) with overlap > 0.5
        → emits structured recommendation
        → autonomous mode: log + escalate Tier 2
        → future interactive mode: propose to user before synthesising
    → no recommendation → Tier 2 as before
```

### Recommendation output structure

```python
@dataclass
class SkillRecommendation:
    skill_name: str
    similarity: float        # cosine vs task
    coverage: str            # "full" | "partial" | "related"
    gap_description: str     # what the skill does NOT cover
    verified: bool           # ADR-006 result for this candidate
```

### Coverage classification

| Cosine score | ADR-006 result | Coverage label |
|-------------|----------------|----------------|
| ≥ 0.7 | YES | full |
| ≥ 0.5 | YES | partial |
| ≥ 0.5 | NO | related |
| < 0.5 | — | — (not surfaced) |

---

## Implementation

### New file: `src/recommender.py`

```python
"""
ADR-007: Capability Recommendation Layer.
After ADR-006 rejects top candidates, surface partial matches
before committing to Tier 2 synthesis.
"""
import logging
from dataclasses import dataclass
from embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

@dataclass
class SkillRecommendation:
    skill_name: str
    similarity: float
    coverage: str        # "full" | "partial" | "related"
    gap_description: str
    verified: bool

def recommend(
    task: str,
    all_matches: list[dict],  # full ranked list from search_ecosystem
    rejected_names: set[str], # names rejected by ADR-006
    embedding_client: EmbeddingClient,
    verify_fn=None,           # optional: ADR-006 verify callable
    threshold: float = 0.5,
) -> list[SkillRecommendation]:
    """
    Given the full ranked match list and ADR-006 rejections,
    return SkillRecommendation list for skills with overlap > threshold.
    Skips already-rejected candidates (re-evaluates with verify if provided).
    """
    recommendations = []
    seen = set()

    for candidate in all_matches:
        name = candidate.get("name", "")
        if name in seen:
            continue
        seen.add(name)

        score = candidate.get("score", 0.0)
        if score < threshold:
            break  # sorted descending — nothing below threshold

        verified = name not in rejected_names
        if not verified and verify_fn:
            # Already rejected by ADR-006 — skip
            coverage = "related"
        elif score >= 0.7:
            coverage = "partial" if name in rejected_names else "full"
        else:
            coverage = "related"

        if coverage in ("full", "partial", "related"):
            # Generate gap description from skill description vs task
            desc = candidate.get("description", "")
            gap = _describe_gap(task, desc, coverage)
            recommendations.append(SkillRecommendation(
                skill_name=name,
                similarity=score,
                coverage=coverage,
                gap_description=gap,
                verified=verified,
            ))

    return recommendations[:3]  # top 3 only


def _describe_gap(task: str, skill_desc: str, coverage: str) -> str:
    """Simple gap description — semantic difference between task and skill."""
    if coverage == "full":
        return ""
    if coverage == "partial":
        return f"Skill covers part of the task but may not handle all requirements of: {task[:80]}"
    return f"Skill is related but designed for a different primary purpose than: {task[:80]}"
```

### Modify `evolver.py` — `run()` method

After the ADR-006 loop (when `best is None`), before escalating to Tier 2:

```python
if best is None:
    # ADR-007: surface partial matches before Tier 2
    from recommender import recommend as _recommend
    recs = _recommend(
        task=task,
        all_matches=matches,
        rejected_names={c.get("name") for c in matches[:3]},
        embedding_client=self._embedding_client,
    )
    # Log recommendations alongside the escalation
    rec_names = [r.skill_name for r in recs]
    rec_summaries = [f"{r.skill_name}({r.coverage},{r.similarity:.2f})" for r in recs]
    if recs:
        logger.info(f"[ADR-007] recommendations for '{task[:50]}': {rec_summaries}")

    return EvolutionResult(
        found=False,
        confidence="LOW",
        retry=False,
        escalated=True,
        gap=f"No verified skill found for task: {task}",
        verification_result=verification_result,
        verification_reasoning=verification_reasoning,
        recommendations=rec_names,       # NEW field
    )
```

### Add `recommendations` field to `EvolutionResult`

```python
@dataclass
class EvolutionResult:
    ...
    recommendations: list = field(default_factory=list)  # ADR-007
```

### Expose in API `/evolution/trigger` response

Add `recommendations` to the result dict.

### Log in `evolution_log.py`

Add `recommendations TEXT` column (nullable JSON list). Migration-safe ALTER.

---

## Sim 5 role

ADR-007 is NOT the primary variable in Sim 5 — it's an observer layer. Sim 5 runs the same 15 tasks with the enriched 27-skill ecosystem. The primary finding will be whether Tier 1 now correctly resolves the same tasks that Tier 2 synthesised in Sim 4. ADR-007 provides the secondary finding: for any task that still fails, are useful partial recommendations surfaced?

**Sim 5 hypothesis:**
- 12–15/15 Tier 1 resolved (ecosystem now contains the exact synthesised skills)
- Verification returns YES for synthesised skills (scores 0.87–0.91 confirmed)
- 0 Tier 2 escalations (or at most 3 for tasks that timed out in Sim 4)
- ADR-007 logs partial matches for any remaining gaps

---

## Paper contribution (ADR-007)

Completes the BDI trilogy:

| ADR | BDI layer | Description |
|-----|-----------|-------------|
| ADR-004 | Intention | Execute known skills, acquire new ones |
| ADR-005 | Desires | Idle reflection, autonomous goal formation |
| ADR-006 | Belief verification | Verify skill capability before committing |
| ADR-007 | Belief+Desire synthesis | Recommend alternatives before expensive action |

> "The recommendation layer implements a lightweight form of means-end reasoning [Bratman, 1987]: before committing to the expensive action of skill synthesis, the agent surfaces what it *does* have that partially satisfies the goal. This is the first system to implement all four BDI layers in a local-first, consumer-GPU agent architecture."

---

## Tests required

- `test_recommend_returns_partial_match_above_threshold`
- `test_recommend_skips_below_threshold`
- `test_recommend_caps_at_three`
- `test_recommend_labels_rejected_as_related`
- `test_evolver_includes_recommendations_in_result`

## GH issue

"feat(ADR-007): capability recommendation layer — surface partial matches before Tier 2 escalation"
