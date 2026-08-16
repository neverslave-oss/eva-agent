"""
test_observer_layer.py — Tests for ADR-019 ObserverLayer.

All DB operations use test-local temp dirs (tmp_path fixture).
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_config(tmp_path, journal_dir=None, obs_overrides=None):
    """Build a minimal config dict pointing DBs to tmp_path."""
    obs = {
        "enabled": True,
        "evidence_lookback_turns": 100,
        "evidence_lookback_days": 7,
        "chat_history_db_path": str(tmp_path / "chat_history_evolving.db"),
        "evolution_db_path": str(tmp_path / "evolution.db"),
        "promoted_signals_db": str(tmp_path / "promoted_signals.db"),
    }
    if obs_overrides:
        obs.update(obs_overrides)
    return {
        "thinking": {
            "enabled": True,
            "journal_dir": str(journal_dir or tmp_path / "thoughts"),
            "ideas_dir": str(tmp_path / "ideas"),
            "observer": obs,
        }
    }


def _setup_evolution_db(db_path: Path, rows: list):
    """
    Create evolution_events table with given rows.
    rows: list of dicts with keys: gap, task, verification_result, created_at
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evolution_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gap TEXT,
            task TEXT,
            verification_result TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    for row in rows:
        conn.execute(
            "INSERT INTO evolution_events (gap, task, verification_result, created_at) VALUES (?, ?, ?, ?)",
            (row.get("gap", ""), row.get("task", ""), row.get("verification_result", ""), row.get("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))),
        )
    conn.commit()
    conn.close()


def _setup_chat_db(db_path: Path, messages: list):
    """
    Create messages table with given messages.
    messages: list of dicts with keys: role, content
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    for msg in messages:
        conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (msg.get("role", "user"), msg.get("content", "")),
        )
    conn.commit()
    conn.close()


def _setup_signals_db(db_path: Path, signals: list):
    """
    Create promoted_signals table.
    signals: list of dicts with keys: text
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS promoted_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    for sig in signals:
        conn.execute("INSERT INTO promoted_signals (text) VALUES (?)", (sig.get("text", ""),))
    conn.commit()
    conn.close()


# ── Classification tests ──────────────────────────────────────────────────────

def test_evidenced_when_no_verification_in_evolution(tmp_path):
    """EVIDENCED classification when evolution_events has a NO-verification row."""
    from core.layers.observer_layer import ObserverLayer

    _setup_evolution_db(tmp_path / "evolution.db", [
        {"gap": "memory retrieval broken", "task": "fix memory", "verification_result": "NO"},
    ])

    cfg = make_config(tmp_path, obs_overrides={"evidence_lookback_days": 30})
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I notice my memory retrieval is broken and needs fixing",
        "category": "gap_reflection",
        "score": 0.75,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "EVIDENCED"
    assert verdict.evolution_path == "full_evolution"
    assert verdict.evidence_count >= 1


def test_evidenced_gap_reflection_path_is_full_evolution(tmp_path):
    """EVIDENCED + gap_reflection → full_evolution."""
    from core.layers.observer_layer import ObserverLayer

    # Two sources: chat + signals  (total >= 2)
    _setup_chat_db(tmp_path / "chat_history_evolving.db", [
        {"role": "user", "content": "you should handle memory better"},
        {"role": "user", "content": "memory search is broken"},
    ])
    _setup_signals_db(tmp_path / "promoted_signals.db", [
        {"text": "memory retrieval signal"},
    ])

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I need to improve memory retrieval for better answers",
        "category": "gap_reflection",
        "score": 0.75,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "EVIDENCED"
    assert verdict.evolution_path == "full_evolution"


def test_evidenced_self_improvement_path_is_goal_signal(tmp_path):
    """EVIDENCED + self_improvement → goal_signal."""
    from core.layers.observer_layer import ObserverLayer

    _setup_evolution_db(tmp_path / "evolution.db", [
        {"gap": "speed improvement needed", "task": "improve speed", "verification_result": "NO"},
    ])

    cfg = make_config(tmp_path, obs_overrides={"evidence_lookback_days": 30})
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I should focus on speed improvement and faster responses",
        "category": "self_improvement",
        "score": 0.75,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "EVIDENCED"
    assert verdict.evolution_path == "goal_signal"


def test_speculative_gap_reflection_path_is_probe(tmp_path):
    """SPECULATIVE + gap_reflection → probe."""
    from core.layers.observer_layer import ObserverLayer

    # No DBs → zero evidence
    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "Perhaps I should integrate better tools for searching documents",
        "category": "gap_reflection",
        "score": 0.65,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "SPECULATIVE"
    assert verdict.evolution_path == "probe"
    assert verdict.evidence_count == 0


def test_speculative_self_improvement_path_is_journal_only(tmp_path):
    """SPECULATIVE + self_improvement → journal_only."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I could potentially learn better communication patterns",
        "category": "self_improvement",
        "score": 0.65,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "SPECULATIVE"
    assert verdict.evolution_path == "journal_only"


def test_curiosity_always_returns_idea(tmp_path):
    """CURIOSITY classification always returns path='idea' regardless of evidence."""
    from core.layers.observer_layer import ObserverLayer

    # Set up lots of evidence — should still be CURIOSITY/idea
    _setup_evolution_db(tmp_path / "evolution.db", [
        {"gap": "curious wonder", "task": "curious wonder", "verification_result": "NO"},
        {"gap": "curious wonder", "task": "curious wonder", "verification_result": "YES"},
    ])
    _setup_chat_db(tmp_path / "chat_history_evolving.db", [
        {"role": "user", "content": "curious wonder about things"},
        {"role": "user", "content": "curious wonder"},
    ])

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I am curious about wonder and beautiful things in science",
        "category": "curiosity",
        "score": 0.80,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "CURIOSITY"
    assert verdict.evolution_path == "idea"


def test_curiosity_with_no_dbs_still_idea(tmp_path):
    """CURIOSITY with no DBs → always idea."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "What if there was a better approach to creativity?",
        "category": "curiosity",
        "score": 0.70,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "CURIOSITY"
    assert verdict.evolution_path == "idea"


# ── Anti-seed tests ───────────────────────────────────────────────────────────

def test_anti_seed_returns_at_most_n(tmp_path):
    """get_anti_seed_context() returns at most n items."""
    from core.layers.observer_layer import ObserverLayer

    journal_dir = tmp_path / "thoughts"
    journal_dir.mkdir()

    # Write 15 journal files
    for i in range(15):
        day = f"2026-05-{i+1:02d}.md"
        (journal_dir / day).write_text(
            f"## 10:{i:02d} — curiosity\n\nThought number {i}: exploring the ideas of the universe\n",
            encoding="utf-8",
        )

    cfg = make_config(tmp_path, journal_dir=journal_dir)
    obs = ObserverLayer(config=cfg)
    results = obs.get_anti_seed_context(n=10)

    assert len(results) <= 10


def test_anti_seed_entries_max_80_chars(tmp_path):
    """Each anti-seed entry is ≤ 80 chars."""
    from core.layers.observer_layer import ObserverLayer

    journal_dir = tmp_path / "thoughts"
    journal_dir.mkdir()

    long_thought = "This is a very long thought that goes on and on and exceeds eighty characters easily and keeps going"
    (journal_dir / "2026-05-01.md").write_text(
        f"## 10:00 — curiosity\n\n{long_thought}\n",
        encoding="utf-8",
    )

    cfg = make_config(tmp_path, journal_dir=journal_dir)
    obs = ObserverLayer(config=cfg)
    results = obs.get_anti_seed_context(n=5)

    for entry in results:
        assert len(entry) <= 80, f"Entry too long ({len(entry)}): {entry!r}"


def test_anti_seed_empty_when_no_journal(tmp_path):
    """get_anti_seed_context() returns [] when journal dir doesn't exist."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path, journal_dir=tmp_path / "nonexistent_dir")
    obs = ObserverLayer(config=cfg)
    results = obs.get_anti_seed_context(n=10)

    assert results == []


def test_anti_seed_sorted_descending(tmp_path):
    """Most recent journal entries come first."""
    from core.layers.observer_layer import ObserverLayer

    journal_dir = tmp_path / "thoughts"
    journal_dir.mkdir()

    (journal_dir / "2026-05-01.md").write_text(
        "## 10:00 — curiosity\n\nOldest thought about nothing\n", encoding="utf-8"
    )
    (journal_dir / "2026-05-28.md").write_text(
        "## 10:00 — curiosity\n\nNewest thought about everything\n", encoding="utf-8"
    )

    cfg = make_config(tmp_path, journal_dir=journal_dir)
    obs = ObserverLayer(config=cfg)
    results = obs.get_anti_seed_context(n=2)

    assert len(results) == 2
    # Most recent first (2026-05-28 > 2026-05-01)
    assert "Newest" in results[0]
    assert "Oldest" in results[1]


# ── Graceful fallback tests ───────────────────────────────────────────────────

def test_observer_instantiates_cleanly_when_dbs_missing(tmp_path):
    """ObserverLayer instantiates cleanly when DBs don't exist (graceful fallback)."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path)
    # Don't create any DB files
    obs = ObserverLayer(config=cfg)
    assert obs is not None


def test_observer_evaluate_with_missing_dbs_is_speculative(tmp_path):
    """evaluate() with no DBs returns SPECULATIVE for gap_reflection."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I should build better search capability",
        "category": "gap_reflection",
        "score": 0.70,
    }
    verdict = obs.evaluate(thought)

    # No DBs → zero evidence → SPECULATIVE
    assert verdict.classification == "SPECULATIVE"
    assert verdict.evidence_count == 0


def test_observer_evaluate_with_missing_dbs_curiosity_still_idea(tmp_path):
    """evaluate() with no DBs returns CURIOSITY/idea for curiosity category."""
    from core.layers.observer_layer import ObserverLayer

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "What makes consciousness interesting?",
        "category": "curiosity",
        "score": 0.80,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "CURIOSITY"
    assert verdict.evolution_path == "idea"


def test_observer_verdict_has_required_fields(tmp_path):
    """ObserverVerdict always has all required fields."""
    from core.layers.observer_layer import ObserverLayer, ObserverVerdict

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "Testing field completeness in verdicts",
        "category": "gap_reflection",
        "score": 0.75,
    }
    verdict = obs.evaluate(thought)

    assert hasattr(verdict, "classification")
    assert hasattr(verdict, "evolution_path")
    assert hasattr(verdict, "evidence_summary")
    assert hasattr(verdict, "evidence_count")
    assert verdict.classification in ("EVIDENCED", "SPECULATIVE", "CURIOSITY")


# ── Single-hit boundary (evidenced requires >=2 hits or NO-verification) ─────

def test_single_chat_hit_without_no_verification_is_speculative(tmp_path):
    """Only 1 total hit (no NO-verification) → SPECULATIVE."""
    from core.layers.observer_layer import ObserverLayer

    _setup_chat_db(tmp_path / "chat_history_evolving.db", [
        {"role": "user", "content": "could you improve search results"},
    ])

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I should improve search results for better user experience",
        "category": "gap_reflection",
        "score": 0.70,
    }
    verdict = obs.evaluate(thought)

    # 1 hit, no NO-verification → SPECULATIVE
    assert verdict.classification == "SPECULATIVE"


def test_two_hits_without_no_verification_is_evidenced(tmp_path):
    """2+ hits (even without NO-verification) → EVIDENCED."""
    from core.layers.observer_layer import ObserverLayer

    _setup_chat_db(tmp_path / "chat_history_evolving.db", [
        {"role": "user", "content": "improve search results"},
        {"role": "user", "content": "search results are not good"},
    ])

    cfg = make_config(tmp_path)
    obs = ObserverLayer(config=cfg)
    thought = {
        "thought": "I should improve search results",
        "category": "gap_reflection",
        "score": 0.70,
    }
    verdict = obs.evaluate(thought)

    assert verdict.classification == "EVIDENCED"
