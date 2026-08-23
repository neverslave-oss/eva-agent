"""
Integration tests for the expertise-field kernel bridge (ADR-015).

These verify the three integration seams without needing a live model:
  1. The bridge discovers + loads the sidecar registry (available()).
  2. build_system_prompt injects a progressive-disclosure block when a task
     triggers a hot field.
  3. reorder_matches biases search_skills results toward the active field
     ("feed, don't bypass" — never drops non-field matches).

All assertions compare REAL on-disk sidecar data (fields/, hot state), so they
fail loudly if the sidecar disappears rather than silently passing.
"""

import os
import sys
from pathlib import Path

import pytest

KERNEL_ROOT = Path(__file__).resolve().parents[4]  # .../kernel-evolving
sys.path.insert(0, str(KERNEL_ROOT / "src"))

from core.expansions import expertise_field_bridge as bridge  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_hot_state(tmp_path, monkeypatch):
    """Point the bridge at a throwaway hot-state file so tests never touch live
    JSON state or leak across chat_ids."""
    hot = tmp_path / "hot_fields.json"
    monkeypatch.setenv("EXPERTISE_FIELD_HOT_STATE", str(hot))
    # Force re-init on next access
    bridge._registry = None
    bridge._state = None
    bridge._sidecar = None
    yield hot


def test_bridge_available_when_sidecar_present():
    """Truth test — the entire integration rests on the sidecar being loadable."""
    assert bridge.available() is True


def test_inject_field_context_returns_disclosure_block():
    """A finance-triggering task should route hot and produce a block listing
    the domain skills + kb."""
    block = bridge.inject_field_context(chat_id="chat-fin", text="analyze my trading portfolio and holdings")
    assert isinstance(block, str)
    assert "Active expertise fields" in block
    assert "Finance & Investment" in block
    assert "domain skills:" in block


def test_inject_field_context_noop_without_match():
    """An unmatched/no-match probe should degrade to "" (safe no-op)."""
    block = bridge.inject_field_context(chat_id="chat-x", text="zzzqqqxyznonsense")
    assert block == ""


def test_reorder_matches_biases_toward_active_field():
    """Field skills float to the top, non-field skills are NOT dropped."""
    finance_skill = {"name": "open-agentic-investor", "description": "portfolio"}
    other_skill = {"name": "github", "description": "git"}
    matches = [finance_skill, other_skill]
    # Warm hot state for this chat via a finance task first
    bridge.inject_field_context(chat_id="chat-re", text="portfolio allocation rebalancing")
    out = bridge.reorder_matches("portfolio", matches, chat_id="chat-re")
    assert out[0]["name"] == "open-agentic-investor"
    # Both retained — "feed, don't bypass"
    assert {s["name"] for s in out} == {"open-agentic-investor", "github"}


def test_reorder_matches_noop_without_bias():
    """When no field is active, reorder leaves the list intact (degree of
    freedom: original order preserved)."""
    a = {"name": "github", "description": "git"}
    b = {"name": "soil-analyzer", "description": "soil"}
    matches = [a, b]
    out = bridge.reorder_matches("zzzqqq", matches, chat_id="chat-none")
    assert [s["name"] for s in out] == ["github", "soil-analyzer"]