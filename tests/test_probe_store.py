"""
tests/test_probe_store.py — Unit tests for ProbeStore (ADR-019 Phase 2)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# Make src importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.evolution.probe_store import ProbeStore


def _make_thought(text="I wonder about Python async patterns", category="gap_reflection", score=0.5):
    return {"thought": text, "category": category, "score": score}


# ── helpers ──────────────────────────────────────────────────────────────────

def _store(tmp_path, ttl_days=7):
    return ProbeStore(db_path=str(tmp_path / "probes.db"), ttl_days=ttl_days)


# ── tests ────────────────────────────────────────────────────────────────────

class TestAddAndListActive:
    def test_add_returns_id(self, tmp_path):
        s = _store(tmp_path)
        probe_id = s.add(_make_thought(), subject="python async")
        assert isinstance(probe_id, int)
        assert probe_id >= 1

    def test_list_active_returns_inserted(self, tmp_path):
        s = _store(tmp_path)
        s.add(_make_thought(), subject="python async")
        active = s.list_active()
        assert len(active) == 1
        p = active[0]
        assert p["subject"] == "python async"
        assert p["resolved"] == 0


class TestMatch:
    def test_match_keyword_in_text(self, tmp_path):
        s = _store(tmp_path)
        s.add(_make_thought(), subject="python async")
        results = s.match("I want to learn about python today")
        assert len(results) == 1

    def test_match_no_keyword_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        s.add(_make_thought(), subject="python async")
        results = s.match("the weather is nice today")
        assert results == []

    def test_match_any_keyword_triggers(self, tmp_path):
        s = _store(tmp_path)
        s.add(_make_thought(), subject="python async patterns")
        results = s.match("looking at async behavior")
        assert len(results) == 1

    def test_match_resolved_probe_excluded(self, tmp_path):
        s = _store(tmp_path)
        pid = s.add(_make_thought(), subject="python async")
        s.resolve(pid)
        results = s.match("python is great")
        assert results == []

    def test_match_expired_probe_excluded(self, tmp_path):
        s = _store(tmp_path, ttl_days=0)
        # ttl_days=0 means expires immediately (same second or before)
        # We'll manually insert with a past expires_at
        import sqlite3, datetime as dt
        db_path = str(tmp_path / "probes.db")
        s2 = ProbeStore(db_path=db_path, ttl_days=0)
        # Insert probe with past expiry
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO probes (subject, thought_text, category, score, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("python", "test thought", "gap_reflection", 0.5,
                 datetime.now(timezone.utc).isoformat(), past)
            )
        results = s2.match("python")
        assert results == []


class TestTrigger:
    def test_trigger_sets_fields(self, tmp_path):
        s = _store(tmp_path)
        pid = s.add(_make_thought(), subject="async patterns")
        s.trigger(pid, triggered_by="user asked about async")
        active = s.list_active()
        assert len(active) == 1
        p = active[0]
        assert p["triggered_at"] is not None
        assert "async" in p["triggered_by"]


class TestResolve:
    def test_resolve_sets_flag(self, tmp_path):
        s = _store(tmp_path)
        pid = s.add(_make_thought(), subject="test subject")
        s.resolve(pid)
        active = s.list_active()
        assert active == []

    def test_resolve_does_not_delete(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "probes.db")
        s = ProbeStore(db_path=db_path)
        pid = s.add(_make_thought(), subject="test subject")
        s.resolve(pid)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT resolved FROM probes WHERE id=?", (pid,)).fetchone()
        assert row[0] == 1


class TestExpireOld:
    def test_expire_deletes_past_expiry(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "probes.db")
        s = ProbeStore(db_path=db_path)
        # manually insert expired probe
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO probes (subject, thought_text, category, score, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("old subject", "old thought", "gap_reflection", 0.5, past, past)
            )
        deleted = s.expire_old()
        assert deleted == 1
        assert s.list_active() == []

    def test_expire_keeps_active(self, tmp_path):
        s = _store(tmp_path)
        s.add(_make_thought(), subject="active probe")
        deleted = s.expire_old()
        assert deleted == 0
        assert len(s.list_active()) == 1

    def test_expire_returns_count(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "probes.db")
        s = ProbeStore(db_path=db_path)
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with sqlite3.connect(db_path) as conn:
            for i in range(3):
                conn.execute(
                    "INSERT INTO probes (subject, thought_text, category, score, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (f"subj{i}", f"thought{i}", "gap_reflection", 0.5, past, past)
                )
        deleted = s.expire_old()
        assert deleted == 3


class TestObserverLayerProbeIntegration:
    """Test ObserverLayer.check_interaction_for_probes()."""

    def _make_observer(self, tmp_path):
        """Create an ObserverLayer with in-memory probe store."""
        from core.layers.observer_layer import ObserverLayer
        from core.evolution.probe_store import ProbeStore

        obs = ObserverLayer.__new__(ObserverLayer)
        obs._chat_history_db = str(tmp_path / "chat.db")
        obs._evolution_db = str(tmp_path / "evo.db")
        obs._promoted_signals_db = str(tmp_path / "signals.db")
        obs._journal_dir = str(tmp_path / "journal")
        obs._lookback_turns = 100
        obs._lookback_days = 7
        obs._probe_store = ProbeStore(db_path=str(tmp_path / "probes.db"), ttl_days=7)
        return obs

    def test_check_interaction_triggers_matching_probe(self, tmp_path):
        obs = self._make_observer(tmp_path)
        # seed a probe directly via probe_store
        pid = obs._probe_store.add(
            {"thought": "wondering about async patterns", "category": "gap_reflection", "score": 0.6},
            subject="async patterns"
        )
        triggered = obs.check_interaction_for_probes("tell me about async usage")
        assert len(triggered) == 1
        assert triggered[0]["id"] == pid
        # confirm it was marked triggered
        active = obs._probe_store.list_active()
        assert active[0]["triggered_at"] is not None

    def test_check_interaction_no_match_returns_empty(self, tmp_path):
        obs = self._make_observer(tmp_path)
        obs._probe_store.add(
            {"thought": "rust memory management", "category": "gap_reflection", "score": 0.5},
            subject="rust memory"
        )
        triggered = obs.check_interaction_for_probes("what is the weather today")
        assert triggered == []
