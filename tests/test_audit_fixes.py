"""
test_audit_fixes.py — unit tests for 2026-05-26 audit fixes

Covers:
  FIX #1: Background maybe_evolve callers pass infer_fn (verifier exercised)
  FIX #4: RecommendationStore persistence + score boost in search_ecosystem
"""
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Stub heavy optional deps
import sys as _sys
for _mod in ("torch", "transformers", "sounddevice", "soundfile", "accelerate"):
    _sys.modules.setdefault(_mod, MagicMock())


# ──────────────────────────────────────────────────────────────────────────────
# FIX #1 — infer_fn wired into background maybe_evolve callers
# ──────────────────────────────────────────────────────────────────────────────

def test_goal_discovery_set_infer_fn_sets_module_level():
    """set_infer_fn() stores the callable at module level."""
    import core.pipelines.goal_discovery as gd
    sentinel = lambda prompt, **kw: "yes"
    gd.set_infer_fn(sentinel)
    assert gd._infer_fn is sentinel
    gd.set_infer_fn(None)  # reset


def test_goal_discovery_maybe_evolve_receives_infer_fn(monkeypatch, tmp_path):
    """_run_single_pattern passes _infer_fn to maybe_evolve when guards pass."""
    import core.pipelines.goal_discovery as gd
    import core.evolution.evolution_state as evolution_state
    import core.evolution.evolution_log as evolution_log

    sentinel_fn = lambda prompt, **kw: "yes"
    gd.set_infer_fn(sentinel_fn)

    captured = {}

    # Satisfy guards
    monkeypatch.setattr(gd, "EVOLUTION_ENABLED", True)
    monkeypatch.setattr(evolution_state, "should_evolve", lambda: True)
    monkeypatch.setattr(gd, "_is_system_idle", lambda: True)
    monkeypatch.setattr(gd, "_already_installed", lambda pattern, sd: False)

    fake_evo_log = MagicMock()
    fake_evo_log.get_gaps.return_value = []
    fake_evo_log.get_history.return_value = []
    monkeypatch.setattr(evolution_log, "EvolutionLog", lambda: fake_evo_log)

    import types
    fake_hook = types.ModuleType("evolution_hook")
    fake_hook.maybe_evolve = lambda t, c, skills_dir=None, infer_fn=None: captured.update({"infer_fn": infer_fn}) or None
    monkeypatch.setitem(sys.modules, "core.evolution.evolution_hook", fake_hook)

    gd._run_single_pattern("test task", {"skills_dir": str(tmp_path)}, str(tmp_path))

    assert "infer_fn" in captured, "maybe_evolve was not called"
    assert captured["infer_fn"] is sentinel_fn, (
        f"Expected infer_fn={sentinel_fn!r}, got {captured['infer_fn']!r}"
    )
    gd.set_infer_fn(None)


def test_thought_engine_maybe_trigger_evolution_passes_infer_fn(monkeypatch, tmp_path):
    """_maybe_trigger_evolution passes goal_discovery._infer_fn to maybe_evolve."""
    import core.pipelines.goal_discovery as gd

    sentinel_fn = lambda p, **kw: "yes"
    gd.set_infer_fn(sentinel_fn)

    captured = {}

    import core.evolution.evolution_hook as evolution_hook
    monkeypatch.setattr("core.evolution.evolution_hook.maybe_evolve",
                        lambda task, cfg, sd, infer_fn=None: captured.update({"infer_fn": infer_fn}) or None)

    # Write a minimal config.yaml into tmp_path
    import yaml
    cfg = {"skills_dir": str(tmp_path)}
    (tmp_path / "config.yaml").write_text(yaml.dump(cfg))

    import services.thought_engine as thought_engine
    engine = thought_engine.ThinkAtRest.__new__(thought_engine.ThinkAtRest)

    # Patch the config path lookup so it finds our tmp config
    orig = thought_engine.os.path.join
    def patched_join(*parts):
        if parts[-1] == "config.yaml" and ".." in parts:
            return str(tmp_path / "config.yaml")
        return orig(*parts)
    monkeypatch.setattr(thought_engine.os.path, "join", patched_join)

    thought = {"thought": "I need a git skill", "category": "gap_reflection"}
    engine._maybe_trigger_evolution(thought)

    assert captured.get("infer_fn") is sentinel_fn, (
        f"Expected sentinel_fn, got {captured.get('infer_fn')!r}"
    )
    gd.set_infer_fn(None)


# ──────────────────────────────────────────────────────────────────────────────
# FIX #4 — RecommendationStore + score boost
# ──────────────────────────────────────────────────────────────────────────────

def test_recommendation_store_record_and_get_hits(tmp_path):
    """RecommendationStore.record() increments hit counts correctly."""
    from core.evolution.evolver import RecommendationStore

    store = RecommendationStore(db_path=str(tmp_path / "evo.db"))
    assert store.get_hits() == {}

    store.record(["skill-a", "skill-b"])
    store.record(["skill-a"])  # second hit for skill-a

    hits = store.get_hits()
    assert hits["skill-a"] == 2
    assert hits["skill-b"] == 1
    assert "skill-c" not in hits


def test_recommendation_boost_applied_in_search_ecosystem(tmp_path):
    """search_ecosystem applies recommendation_boost to previously recommended skills."""
    from core.evolution.evolver import Evolver, RecommendationStore

    # Write a SKILL.md that looks like our target skill
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: does something useful\ncommands:\n  - /myskill\n---\n"
    )

    cfg = {
        "skills_dir": str(tmp_path / "skills"),
        "evolution": {
            "min_candidate_score": 0.0,   # accept everything
            "recommendation_boost": 0.05,
        },
        "embedding_server_url": "http://localhost:8770/embeddings",
        "embedding_backend": "native",
        "embedding_model_path": "",
    }

    # Create evolver with mocked embedding (returns 0 so keyword path fires)
    evolver = Evolver(config=cfg, skills_dir=str(tmp_path / "skills"))

    # Use a separate store pointing at same DB to pre-populate a hit
    store = RecommendationStore(db_path=evolver._rec_store.db_path)
    store.record(["my-skill"])

    # Reload hits into the evolver's store (same file)
    evolver._rec_store = store

    # Patch remote fetch to return nothing
    evolver._fetch_remote_skills = lambda: []

    # Patch embedding to return 0.0 (triggers keyword fallback)
    evolver._embedding_client.similarity = MagicMock(return_value=None)

    results = evolver.search_ecosystem("something useful")
    assert any(r["name"] == "my-skill" for r in results), "my-skill not in results"

    my_skill_entry = next(r for r in results if r["name"] == "my-skill")
    # The keyword score for "my-skill" + "does something useful" vs "something useful"
    # should be boosted — we just verify it's > raw keyword score without boost
    # Raw: keyword score without boost; we run evolver without boost for comparison
    evolver2 = Evolver(config={**cfg, "evolution": {"min_candidate_score": 0.0, "recommendation_boost": 0.0}},
                       skills_dir=str(tmp_path / "skills"))
    evolver2._fetch_remote_skills = lambda: []
    evolver2._embedding_client.similarity = MagicMock(return_value=None)
    results2 = evolver2.search_ecosystem("something useful")
    raw_score = next((r["score"] for r in results2 if r["name"] == "my-skill"), 0.0)

    assert my_skill_entry["score"] == pytest.approx(min(raw_score + 0.05, 1.0), abs=1e-6), (
        f"Boosted score {my_skill_entry['score']} != raw+boost {raw_score + 0.05}"
    )


def test_recommendation_boost_not_applied_to_non_recommended(tmp_path):
    """Skills not in the recommendation store do not get a boost."""
    from core.evolution.evolver import Evolver, RecommendationStore
    import pytest

    skill_dir = tmp_path / "skills" / "other-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: other-skill\ndescription: does something else\ncommands:\n  - /other\n---\n"
    )

    cfg = {
        "skills_dir": str(tmp_path / "skills"),
        "evolution": {"min_candidate_score": 0.0, "recommendation_boost": 0.05},
        "embedding_server_url": "http://localhost:8770/embeddings",
        "embedding_backend": "native",
        "embedding_model_path": "",
    }
    evolver = Evolver(config=cfg, skills_dir=str(tmp_path / "skills"))
    evolver._fetch_remote_skills = lambda: []
    evolver._embedding_client.similarity = MagicMock(return_value=None)

    # No prior hits for "other-skill"
    results = evolver.search_ecosystem("something else")
    entry = next((r for r in results if r["name"] == "other-skill"), None)
    assert entry is not None

    # Without boost, score from _combined_score should not be inflated
    evolver2 = Evolver(config={**cfg, "evolution": {"min_candidate_score": 0.0, "recommendation_boost": 0.0}},
                       skills_dir=str(tmp_path / "skills"))
    evolver2._fetch_remote_skills = lambda: []
    evolver2._embedding_client.similarity = MagicMock(return_value=None)
    results2 = evolver2.search_ecosystem("something else")
    raw_score = next((r["score"] for r in results2 if r["name"] == "other-skill"), 0.0)

    import pytest
    assert entry["score"] == pytest.approx(raw_score, abs=1e-6), (
        "Non-recommended skill score should not differ"
    )


def test_run_persists_recommendations(tmp_path, monkeypatch):
    """core.evolution.evolver.run() persists recommendation names when best is None."""
    from core.evolution.evolver import Evolver, RecommendationStore
    from unittest.mock import MagicMock
    import core.evolution.recommender as rec_mod
    import core.evolution.capability_verifier as capability_verifier

    # Create a skill so matches list is non-empty
    skill_dir = tmp_path / "skills" / "partial-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: partial-skill\ndescription: partial match skill\ncommands:\n  - /partial\n---\n"
    )

    cfg = {
        "skills_dir": str(tmp_path / "skills"),
        "evolution": {"min_candidate_score": 0.0, "recommendation_boost": 0.05},
        "embedding_server_url": "http://localhost:8770/embeddings",
        "embedding_backend": "native",
        "embedding_model_path": "",
        "allowlist": [],
    }

    evolver = Evolver(config=cfg, skills_dir=str(tmp_path / "skills"))
    # Use isolated DB for this test
    evolver._rec_store = RecommendationStore(db_path=str(tmp_path / "test_evo.db"))
    evolver._fetch_remote_skills = lambda: []
    evolver._embedding_client.similarity = MagicMock(return_value=None)
    # Force verifier to reject all candidates so best=None and recommender path runs
    evolver._infer_fn = lambda prompt, **kw: "NO"
    monkeypatch.setattr(capability_verifier, "verify", lambda candidate, task, fn: (False, "rejected"))

    # Stub recommender.recommend on the actual module object (evolver does a local import)
    fake_recs = [MagicMock(skill_name="near-miss-skill", coverage="partial", similarity=0.6)]
    monkeypatch.setattr(rec_mod, "recommend", lambda task, all_matches, rejected_names, **kw: fake_recs)

    result = evolver.run("some unknown task")

    hits = evolver._rec_store.get_hits()
    assert "near-miss-skill" in hits, "near-miss-skill not persisted in recommendation_hits"


import pytest
