"""
Tests for ADR-007: Capability Recommendation Layer (recommender.py)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from core.evolution.recommender import recommend, SkillRecommendation


def _make_matches(*entries):
    """Helper: build a sorted matches list from (name, score) tuples."""
    return [{"name": n, "score": s, "description": f"desc for {n}"} for n, s in entries]


# ---------------------------------------------------------------------------
# test_recommend_returns_candidates_above_threshold
# ---------------------------------------------------------------------------
def test_recommend_returns_candidates_above_threshold():
    matches = _make_matches(("skill-a", 0.75), ("skill-b", 0.55), ("skill-c", 0.30))
    rejected = {"skill-a", "skill-b"}  # ADR-006 rejected both
    recs = recommend(task="do something", all_matches=matches, rejected_names=rejected)
    # skill-a and skill-b are above 0.5, skill-c is below → 2 results
    assert len(recs) == 2
    names = {r.skill_name for r in recs}
    assert "skill-a" in names
    assert "skill-b" in names
    assert "skill-c" not in names


# ---------------------------------------------------------------------------
# test_recommend_skips_below_threshold
# ---------------------------------------------------------------------------
def test_recommend_skips_below_threshold():
    matches = _make_matches(("low-skill", 0.40), ("very-low", 0.10))
    recs = recommend(task="something", all_matches=matches, rejected_names=set())
    assert recs == []


# ---------------------------------------------------------------------------
# test_recommend_caps_at_three
# ---------------------------------------------------------------------------
def test_recommend_caps_at_three():
    matches = _make_matches(
        ("a", 0.90), ("b", 0.85), ("c", 0.80), ("d", 0.75), ("e", 0.70)
    )
    recs = recommend(task="task", all_matches=matches, rejected_names=set())
    assert len(recs) == 3


# ---------------------------------------------------------------------------
# test_recommend_labels_coverage_correctly
# ---------------------------------------------------------------------------
def test_recommend_labels_coverage_correctly():
    # score >= 0.7 + not rejected → "full"
    # score >= 0.5 + not rejected → "partial"
    # any + rejected → "related"
    matches = _make_matches(("full-skill", 0.80), ("partial-skill", 0.55), ("rejected-skill", 0.75))
    rejected = {"rejected-skill"}
    recs = recommend(task="task", all_matches=matches, rejected_names=rejected)
    by_name = {r.skill_name: r for r in recs}

    assert by_name["full-skill"].coverage == "full"
    assert by_name["partial-skill"].coverage == "partial"
    assert by_name["rejected-skill"].coverage == "related"

    # verified flag should be False for rejected
    assert by_name["rejected-skill"].verified is False
    assert by_name["full-skill"].verified is True


# ---------------------------------------------------------------------------
# test_evolver_result_includes_recommendations
# ---------------------------------------------------------------------------
def test_evolver_result_includes_recommendations():
    """
    When ADR-006 rejects all top-3 candidates, EvolutionResult.recommendations
    should be populated with partial match names (those above 0.5 threshold).
    """
    from unittest.mock import MagicMock, patch
    from core.evolution.evolver import Evolver, EvolutionResult

    # Build a fake match list: top-3 will be rejected (all ≥ 0.5 → surfaced as 'related')
    # skill-d is below threshold so it should NOT appear
    fake_matches = [
        {"name": "skill-a", "score": 0.80, "description": "a", "repo": "", "path": ""},
        {"name": "skill-b", "score": 0.78, "description": "b", "repo": "", "path": ""},
        {"name": "skill-c", "score": 0.76, "description": "c", "repo": "", "path": ""},
        {"name": "skill-d", "score": 0.30, "description": "d", "repo": "", "path": ""},
    ]

    config = {"embedding_backend": "native", "embedding_server_url": "http://localhost:8770/embeddings"}
    evolver = Evolver(config=config, skills_dir="/tmp/fake_skills")

    # Make _infer_fn always reject
    evolver._infer_fn = MagicMock(return_value="NO — not a match")

    with patch.object(evolver, "search_ecosystem", return_value=fake_matches), \
         patch("core.evolution.capability_verifier.verify", return_value=(False, "not a match")):
        result = evolver.run("some task")

    assert result.escalated is True
    assert result.found is False
    # Recommendations should contain the 3 rejected skills (all ≥ 0.5), NOT skill-d (0.30)
    assert len(result.recommendations) == 3
    assert "skill-a" in result.recommendations
    assert "skill-b" in result.recommendations
    assert "skill-c" in result.recommendations
    assert "skill-d" not in result.recommendations
