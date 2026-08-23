"""
test_expertise_field.py — Tests for the expertise_field module (Step 1: registry + router).

Run: python -m pytest tests/  (from repo root, with src on path)
Or:  python tests/test_expertise_field.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from expertise_field.registry import HotFieldState, load_registry
from expertise_field import router


def make_repo(tmp: Path) -> Path:
    """Build a minimal field repo in tmp/ for testing."""
    (tmp / "fields").mkdir(parents=True)
    registry = {
        "active": ["plant-science", "bioinformatics"],
        "fields": [
            {"id": "plant-science", "path": "fields/plant-science.json", "version": "1.0.0"},
            {"id": "bioinformatics", "path": "fields/bioinformatics.json", "version": "1.0.0"},
        ],
    }
    (tmp / "expertise_registry.json").write_text(json.dumps(registry))
    (tmp / "fields/plant-science.json").write_text(json.dumps({
        "id": "plant-science",
        "name": "Plant Science & Greenhouse Ops",
        "version": "1.0.0",
        "description": "Indoor growing and plant health.",
        "trigger_terms": ["plant", "greenhouse", "grow", "leaf", "nutrient", "harvest", "crop"],
        "skills": ["open-workspace-tracker", "timeseries-anomaly-detector"],
        "knowledge_base": [],
    }))
    (tmp / "fields/bioinformatics.json").write_text(json.dumps({
        "id": "bioinformatics",
        "name": "Bioinformatics",
        "version": "1.0.0",
        "description": "Sequence analysis and genomics.",
        "trigger_terms": ["sequence", "genome", "dna", "protein", "variant"],
        "skills": ["vcftools-helper"],
        "knowledge_base": [],
    }))
    return tmp


def test_load_and_active(tmp_path):
    repo = make_repo(tmp_path)
    reg = load_registry(repo)
    assert reg.is_active("plant-science")
    assert reg.is_active("bioinformatics")
    assert not reg.is_active("nope")
    assert reg.skill_names_for("plant-science") == ["open-workspace-tracker", "timeseries-anomaly-detector"]


def test_dedup_priority(tmp_path):
    """A private/ field shadows a top-level field with the same id (kernel parity)."""
    repo = make_repo(tmp_path)
    (repo / "fields/private").mkdir(parents=True)
    (repo / "fields/private/plant-science.json").write_text(json.dumps({
        "id": "plant-science",
        "name": "plant-science PRIVATE override",
        "version": "9.9.9",
        "description": "private version",
        "trigger_terms": ["priv"],
        "skills": ["private-skill"],
        "knowledge_base": [],
    }))
    reg = load_registry(repo)
    pf = reg.field("plant-science")
    assert pf["name"] == "plant-science PRIVATE override"
    assert pf["skills"] == ["private-skill"]


def test_hotfield_json_persistence(tmp_path):
    st = HotFieldState(tmp_path / "hot.json", max_hot=2)
    evicted = st.set_hot("chatA", "plant-science")
    assert not evicted
    st.set_hot("chatA", "bioinformatics")
    assert st.hot_fields("chatA") == ["plant-science", "bioinformatics"]
    # exceeds max_hot -> evict LRU (front)
    evicted = st.set_hot("chatA", "newfield")
    assert evicted is True
    assert len(st.hot_fields("chatA")) == 2
    assert "plant-science" not in st.hot_fields("chatA")
    # reload from disk (round-trip)
    st2 = HotFieldState(tmp_path / "hot.json", max_hot=2)
    assert st2.hot_fields("chatA") == ["bioinformatics", "newfield"]


def test_hotfield_chat_isolation():
    """Decision #5 / ADR-009: hot fields must not leak across chats."""
    with tempfile.TemporaryDirectory() as d:
        st = HotFieldState(Path(d) / "hot.json", max_hot=3)
        st.set_hot("chatA", "plant-science")
        st.set_hot("chatB", "bioinformatics")
        assert st.hot_fields("chatA") == ["plant-science"]
        assert st.hot_fields("chatB") == ["bioinformatics"]


def test_router_trigger_match(tmp_path):
    repo = make_repo(tmp_path)
    reg = load_registry(repo)
    st = HotFieldState(tmp_path / "hot.json")
    hot, cands = router.choose_fields("the plant leaves look yellow, check nutrient", reg, st, chat_id="chatA")
    assert "plant-science" in hot
    assert "bioinformatics" not in hot
    # candidate skills fed to semantic matcher (decision #3)
    assert "timeseries-anomaly-detector" in cands


def test_router_fallback_to_hot(tmp_path):
    repo = make_repo(tmp_path)
    reg = load_registry(repo)
    st = HotFieldState(tmp_path / "hot.json")
    # seed hot field, then give unrelated text -> falls back to existing hot
    st.set_hot("chatA", "bioinformatics")
    hot, cands = router.choose_fields("hello there how are you", reg, st, chat_id="chatA")
    assert hot == ["bioinformatics"]
    assert "vcftools-helper" in cands


def test_field_tagged_skills_narrowing(tmp_path):
    """Decision #5: skills tagged `field:` feed candidate narrowing."""
    repo = make_repo(tmp_path)
    reg = load_registry(repo)
    st = HotFieldState(tmp_path / "hot.json")
    # field-tagged refs of shape 'name::field_id' plus explicit field skills
    tagged = ["soil-analyzer::plant-science", "yield-predictor::plant-science"]
    hot, cands = router.choose_fields("soil analysis for my greenhouse crop", reg, st,
                                      chat_id="chatA", all_skill_names=tagged)
    assert "plant-science" in hot
    assert "soil-analyzer" in cands
    assert "yield-predictor" in cands


def test_debug_snapshot(tmp_path):
    """/debug/fields backing data (decision #4)."""
    repo = make_repo(tmp_path)
    reg = load_registry(repo)
    st = HotFieldState(tmp_path / "hot.json")
    st.set_hot("chatA", "plant-science")
    snap = st.snapshot("chatA")
    assert snap["hot"] == ["plant-science"]
    regd = reg.to_dict()
    assert len(regd["fields"]) == 2
    assert regd["active"] == ["plant-science", "bioinformatics"]


if __name__ == "__main__":
    import traceback
    import tempfile as _tf
    fails = 0
    with _tf.TemporaryDirectory() as d:
        from pathlib import Path as _P
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn(_P(d))
                    print(f"  PASS  {name}")
                except Exception:
                    fails += 1
                    print(f"  FAIL  {name}")
                    traceback.print_exc()
    sys.exit(1 if fails else 0)