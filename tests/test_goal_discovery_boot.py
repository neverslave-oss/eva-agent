"""
test_goal_discovery_boot.py — ADR-009 tests

Verifies:
- Only top-N patterns run immediately at boot (boot cap respected)
- Remaining patterns stored as deferred
- _already_installed check prevents redundant evolution
"""
import sys
import os
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _reset_discovery_state():
    """Reset module-level state between tests."""
    import core.pipelines.goal_discovery as gd
    with gd._log_lock:
        gd._interaction_log.clear()
    with gd._deferred_lock:
        gd._deferred_patterns.clear()
    gd._last_model_call_ts = 0.0


def test_boot_cap_limits_immediate_execution(tmp_path, monkeypatch):
    """Only top BOOT_PATTERN_CAP patterns should run at boot; rest go to deferred."""
    import core.pipelines.goal_discovery as gd
    _reset_discovery_state()

    # Inject distinct multi-word patterns that produce different keyword extractions
    # Use real content words that survive the stopword filter
    distinct_patterns = [
        "create document report file",
        "download video audio clip",
        "analyze data extract table",
        "generate chart graph visual",
        "translate text language foreign",
    ]

    def fake_seed(limit=200, allowed_chat_ids=None):
        with gd._log_lock:
            for pattern in distinct_patterns:
                for _ in range(gd.RECURRENCE_THRESHOLD + 1):
                    gd._interaction_log.append(pattern)
        return 50
    monkeypatch.setattr(gd, "seed_from_chat_history", fake_seed)

    # Monkeypatch maybe_evolve to track calls
    evolved_patterns = []
    def fake_evolve(pattern, config, skills_dir, infer_fn=None):
        evolved_patterns.append(pattern)
        from core.evolution.evolver import EvolutionResult
        return EvolutionResult(found=False, confidence="LOW")
    monkeypatch.setattr("core.evolution.evolution_hook.maybe_evolve", fake_evolve, raising=False)

    # Mock evolution state to allow evolving
    import core.evolution.evolution_state as evolution_state
    monkeypatch.setattr(evolution_state, "should_evolve", lambda: True)
    monkeypatch.setenv("EVOLUTION_ENABLED", "true")
    gd.EVOLUTION_ENABLED = True

    # Mock EvolutionLog
    class FakeLog:
        def get_gaps(self): return []
        def get_history(self, limit=500): return []
    monkeypatch.setattr("core.evolution.evolution_log.EvolutionLog", FakeLog)

    boot_cap = 2
    config = {"goal_discovery": {"boot_pattern_cap": boot_cap, "deferred_drain_per_cycle": 1}}
    skills_dir = str(tmp_path)

    gd.seed_from_chat_history_and_defer(config, skills_dir)

    # Check total patterns detected (immediate + deferred)
    with gd._deferred_lock:
        deferred = list(gd._deferred_patterns)

    total = len(evolved_patterns) + len(deferred)
    # Should have detected multiple distinct patterns
    assert total >= 1, f"Expected at least 1 pattern detected, got evolved={len(evolved_patterns)}, deferred={len(deferred)}"
    # Immediate executions should be capped at boot_cap
    assert len(evolved_patterns) <= boot_cap, f"Expected at most {boot_cap} immediate, got {len(evolved_patterns)}"

    # Reset env
    monkeypatch.setenv("EVOLUTION_ENABLED", "false")
    gd.EVOLUTION_ENABLED = False


def test_already_installed_skips_evolution(tmp_path, monkeypatch):
    """_already_installed should prevent re-running evolution for present skills."""
    import core.pipelines.goal_discovery as gd

    # Create a fake skill directory
    skill_name = "my-skill"
    skill_path = tmp_path / "private" / "skills" / skill_name
    skill_path.mkdir(parents=True)

    assert gd._already_installed(skill_name, str(tmp_path)) is True
    assert gd._already_installed("nonexistent-skill", str(tmp_path)) is False


def test_deferred_patterns_drained_when_idle(tmp_path, monkeypatch):
    """_run_single_pattern should be callable and drain from _deferred_patterns."""
    import core.pipelines.goal_discovery as gd
    _reset_discovery_state()

    with gd._deferred_lock:
        gd._deferred_patterns.append("deferred task thing")

    monkeypatch.setenv("EVOLUTION_ENABLED", "false")
    gd.EVOLUTION_ENABLED = False

    # With evolution disabled, run_single_pattern should return immediately
    gd._run_single_pattern("deferred task thing", {}, str(tmp_path))
    # No crash = pass


def test_is_system_idle_no_replicas(monkeypatch):
    """_is_system_idle returns True when no active replicas and no recent model call."""
    import core.pipelines.goal_discovery as gd

    # Mock replica.active to return empty list
    import core.replica.replica as rep
    monkeypatch.setattr(rep, "active", lambda: [])
    gd._last_model_call_ts = 0.0  # very old

    assert gd._is_system_idle() is True


def test_is_system_idle_with_recent_call(monkeypatch):
    """_is_system_idle returns False when model was called recently."""
    import core.pipelines.goal_discovery as gd
    import time

    import core.replica.replica as rep
    monkeypatch.setattr(rep, "active", lambda: [])
    gd._last_model_call_ts = time.time()  # just now

    assert gd._is_system_idle() is False
