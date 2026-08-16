"""
ADR-007: Capability Recommendation Layer.
After ADR-006 rejects top candidates, surface partial matches
before committing to Tier 2 synthesis.
"""
import logging
from dataclasses import dataclass

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
    all_matches: list,       # full ranked list from search_ecosystem
    rejected_names: set,     # names rejected by ADR-006
    embedding_client=None,   # EmbeddingClient (unused currently — kept for future use)
    verify_fn=None,          # optional: ADR-006 verify callable
    threshold: float = 0.5,
) -> list:
    """
    Given the full ranked match list and ADR-006 rejections,
    return SkillRecommendation list for skills with similarity >= threshold.
    Caps at 3 results.
    """
    recommendations = []
    seen: set = set()

    for candidate in all_matches:
        name = candidate.get("name", "")
        if name in seen:
            continue
        seen.add(name)

        score = candidate.get("score", 0.0)
        if score < threshold:
            break  # sorted descending — nothing further above threshold

        verified = name not in rejected_names

        # Coverage classification per ADR-007 spec table
        if score >= 0.7 and verified:
            coverage = "full"
        elif score >= 0.5 and verified:
            coverage = "partial"
        else:
            # score >= threshold but either rejected or below 0.7 unverified
            coverage = "related"

        desc = candidate.get("description", "")
        gap = _describe_gap(task, desc, coverage)
        recommendations.append(SkillRecommendation(
            skill_name=name,
            similarity=score,
            coverage=coverage,
            gap_description=gap,
            verified=verified,
        ))

        if len(recommendations) >= 3:
            break

    return recommendations


def _describe_gap(task: str, skill_desc: str, coverage: str) -> str:
    """Simple gap description — semantic difference between task and skill."""
    if coverage == "full":
        return ""
    if coverage == "partial":
        return f"Skill covers part of the task but may not handle all requirements of: {task[:80]}"
    return f"Skill is related but designed for a different primary purpose than: {task[:80]}"
