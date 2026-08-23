"""
test_discovered_fields.py — Verify the validated discovered fields (finance,
media-content, engineering) load from the real registry, match by trigger terms,
and narrow candidate skills (decision #3: feed, not bypass).

Run: python -m pytest tests/test_discovered_fields.py  (from module root)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from expertise_field.registry import HotFieldState, load_registry
from expertise_field import router


def test_registry_loads_all_four_fields():
    reg = load_registry(REPO)
    ids = [f["id"] for f in reg.all_fields()]
    assert "plant-science" in ids
    assert "finance" in ids
    assert "media-content" in ids
    assert "engineering" in ids
    # all active
    for fid in ids:
        assert reg.is_active(fid), f"{fid} should be active"


def test_finance_trigger_match():
    reg = load_registry(REPO)
    state = HotFieldState()
    hot, cands = router.choose_fields(
        "should I rebalance my portfolio of stocks today",
        reg, state, chat_id="test-c",
        all_skill_names=[s["name"] for s in reg.active_fields()],
    )
    assert "finance" in hot, f"expected finance, got {hot}"
    assert "open-agentic-investor" in cands


def test_media_content_trigger_match():
    reg = load_registry(REPO)
    state = HotFieldState()
    hot, cands = router.choose_fields(
        "write a blog article and generate a cover image",
        reg, state, chat_id="test-c",
        all_skill_names=[s["name"] for s in reg.active_fields()],
    )
    assert "media-content" in hot, f"expected media-content, got {hot}"
    assert "open-fantasia-imagegen" in cands


def test_engineering_trigger_match():
    reg = load_registry(REPO)
    state = HotFieldState()
    hot, cands = router.choose_fields(
        "a lambda in my deployment keeps 403'ing, debug the auth",
        reg, state, chat_id="test-c",
        all_skill_names=[s["name"] for s in reg.active_fields()],
    )
    assert "engineering" in hot, f"expected engineering, got {hot}"
    assert "github" in cands


def test_plant_trigger_match():
    reg = load_registry(REPO)
    state = HotFieldState()
    hot, cands = router.choose_fields(
        "my cannabis leaves look yellow, check soil moisture",
        reg, state, chat_id="test-c",
        all_skill_names=[s["name"] for s in reg.active_fields()],
    )
    assert "plant-science" in hot, f"expected plant-science, got {hot}"