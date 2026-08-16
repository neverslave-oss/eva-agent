"""
tests/test_model_slots.py — Unit tests for SlotRegistry, SlotSpec, SlotState.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.inference.model_slots import SlotRegistry, SlotSpec, SlotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spec(name: str, role: str = "primary", model_path: str = "/fake/model") -> SlotSpec:
    return SlotSpec(name=name, model_path=model_path, role=role)


def make_state(spec: SlotSpec, loaded_at: float = None) -> SlotState:
    state = SlotState(
        spec=spec,
        model=MagicMock(),
        processor=MagicMock(),
        is_nemotron=False,
        audio_capable=(spec.role == "audio"),
        supports_tools=True,
    )
    if loaded_at is not None:
        state.loaded_at = loaded_at
    return state


def simple_loader(spec: SlotSpec) -> SlotState:
    """Fake loader that returns a mock SlotState."""
    return make_state(spec)


# ---------------------------------------------------------------------------
# Tests: register / get
# ---------------------------------------------------------------------------

class TestRegisterGet:
    def test_get_returns_none_before_load(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("primary"))
        assert registry.get("primary") is None

    def test_get_unregistered_returns_none(self):
        registry = SlotRegistry(loader=simple_loader)
        assert registry.get("nonexistent") is None

    def test_load_registered_slot(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("primary"))
        state = registry.load("primary")
        assert state is not None
        assert isinstance(state, SlotState)
        assert registry.get("primary") is state

    def test_load_unregistered_raises_key_error(self):
        registry = SlotRegistry(loader=simple_loader)
        with pytest.raises(KeyError):
            registry.load("doesnotexist")

    def test_load_without_loader_raises_runtime_error(self):
        registry = SlotRegistry()  # no loader
        registry.register(make_spec("primary"))
        with pytest.raises(RuntimeError, match="no loader"):
            registry.load("primary")

    def test_load_twice_returns_cached(self):
        call_count = {"n": 0}

        def counting_loader(spec):
            call_count["n"] += 1
            return make_state(spec)

        registry = SlotRegistry(loader=counting_loader)
        registry.register(make_spec("primary"))
        s1 = registry.load("primary")
        s2 = registry.load("primary")
        assert s1 is s2
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Tests: loaded_slots
# ---------------------------------------------------------------------------

class TestLoadedSlots:
    def test_empty_initially(self):
        registry = SlotRegistry(loader=simple_loader)
        assert registry.loaded_slots() == {}

    def test_loaded_after_load(self):
        registry = SlotRegistry(loader=simple_loader)
        spec = make_spec("audio", role="audio")
        registry.register(spec)
        registry.load("audio")
        loaded = registry.loaded_slots()
        assert "audio" in loaded


# ---------------------------------------------------------------------------
# Tests: unload
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_removes_from_loaded(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("audio", role="audio"))
        registry.load("audio")
        assert registry.get("audio") is not None
        registry.unload("audio")
        assert registry.get("audio") is None

    def test_unload_not_loaded_is_noop(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("audio", role="audio"))
        # Should not raise
        registry.unload("audio")


# ---------------------------------------------------------------------------
# Tests: eviction
# ---------------------------------------------------------------------------

class TestEviction:
    def test_non_primary_slot_evicted_on_vram_low(self):
        """Load slot A (non-primary), mock VRAM low, load slot B → A gets evicted."""
        registry = SlotRegistry(vram_threshold_mb=3000, loader=simple_loader)

        spec_a = make_spec("audio", role="audio")
        spec_b = make_spec("draft", role="draft")
        registry.register(spec_a)
        registry.register(spec_b)

        # Load A first
        registry.load("audio")
        assert registry.get("audio") is not None

        # Simulate VRAM below threshold
        with patch.object(registry, "_free_vram_mb", return_value=500.0):
            # Also patch _release to avoid torch teardown
            with patch.object(SlotRegistry, "_release", staticmethod(lambda name, state: None)):
                registry.load("draft")

        # A should have been evicted
        assert registry.get("audio") is None
        # B should be loaded
        assert registry.get("draft") is not None

    def test_primary_slot_never_evicted(self):
        """Primary slot must never be evicted even if VRAM is critically low."""
        registry = SlotRegistry(vram_threshold_mb=3000, loader=simple_loader)

        primary_spec = make_spec("primary", role="primary")
        audio_spec = make_spec("audio", role="audio")
        registry.register(primary_spec)
        registry.register(audio_spec)

        # Load primary
        registry.load("primary")
        assert registry.get("primary") is not None

        # Simulate VRAM below threshold — no non-primary candidates (primary loaded, audio not loaded)
        with patch.object(registry, "_free_vram_mb", return_value=100.0):
            with patch.object(SlotRegistry, "_release", staticmethod(lambda name, state: None)):
                # Loading audio when no candidates → primary stays, audio loads
                registry.load("audio")

        assert registry.get("primary") is not None, "Primary slot must never be evicted"

    def test_lru_eviction_evicts_oldest(self):
        """When two non-primary slots are loaded, the oldest is evicted first."""
        registry = SlotRegistry(vram_threshold_mb=3000, loader=simple_loader)

        spec_a = make_spec("audio", role="audio")
        spec_b = make_spec("draft", role="draft")
        spec_c = make_spec("embed", role="embed")
        registry.register(spec_a)
        registry.register(spec_b)
        registry.register(spec_c)

        # Load A then B with explicit timestamps so A is older
        registry.load("audio")
        registry._loaded["audio"].loaded_at = 1000.0
        registry.load("draft")
        registry._loaded["draft"].loaded_at = 2000.0

        # Mock VRAM low, load embed → A (older) should be evicted, not B
        with patch.object(registry, "_free_vram_mb", return_value=100.0):
            with patch.object(SlotRegistry, "_release", staticmethod(lambda name, state: None)):
                registry.load("embed")

        assert registry.get("audio") is None, "Oldest slot (audio) should be evicted"
        assert registry.get("draft") is not None, "Newer slot (draft) should survive"


# ---------------------------------------------------------------------------
# Tests: status()
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_returns_list(self):
        registry = SlotRegistry(loader=simple_loader)
        assert isinstance(registry.status(), list)

    def test_status_empty_registry(self):
        registry = SlotRegistry(loader=simple_loader)
        assert registry.status() == []

    def test_status_unloaded_slot(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("primary"))
        result = registry.status()
        assert len(result) == 1
        entry = result[0]
        assert entry["name"] == "primary"
        assert entry["loaded"] is False
        assert "model_path" in entry
        assert "role" in entry

    def test_status_loaded_slot(self):
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("audio", role="audio"))
        registry.load("audio")
        result = registry.status()
        entry = next(e for e in result if e["name"] == "audio")
        assert entry["loaded"] is True
        assert "loaded_at" in entry
        assert "is_nemotron" in entry
        assert "audio_capable" in entry
        assert "supports_tools" in entry
        assert "adapter_count" in entry

    def test_status_json_serialisable(self):
        import json
        registry = SlotRegistry(loader=simple_loader)
        registry.register(make_spec("primary"))
        registry.load("primary")
        # Should not raise
        json.dumps(registry.status())


# ---------------------------------------------------------------------------
# Tests: backwards-compat (empty registry)
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    def test_empty_registry_status_is_empty_list(self):
        registry = SlotRegistry()
        assert registry.status() == []

    def test_empty_registry_loaded_slots_is_empty_dict(self):
        registry = SlotRegistry()
        assert registry.loaded_slots() == {}

    def test_empty_registry_get_returns_none(self):
        registry = SlotRegistry()
        assert registry.get("anything") is None

    def test_set_loader_after_init(self):
        registry = SlotRegistry()
        registry.set_loader(simple_loader)
        registry.register(make_spec("primary"))
        state = registry.load("primary")
        assert state is not None
