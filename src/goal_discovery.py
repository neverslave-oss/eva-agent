"""
goal_discovery.py — ADR-004 proactive goal discovery.

Runs as a background thread. Periodically scans interaction logs for
recurring unhandled task patterns and triggers the evolver to fill gaps
before users encounter them.

This implements the paper's "variation" step running alongside the reasoning loop:
  - Monitor: track all messages that fall through to infer_with_tools (no skill match)
  - Identify: cluster recurring patterns (simple keyword frequency)
  - Propose: when a pattern recurs >= THRESHOLD times, trigger evolution
  - Inherit: successful evolutions remembered; failures logged as gaps
"""

import os
import re
import json
import threading
import time
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

DISCOVERY_INTERVAL = int(os.environ.get("DISCOVERY_INTERVAL_S", "300"))  # 5 min default
RECURRENCE_THRESHOLD = int(os.environ.get("DISCOVERY_THRESHOLD", "3"))    # 3 hits → evolve
EVOLUTION_ENABLED = os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true"

_interaction_log: list[str] = []   # in-memory ring buffer
_log_lock = threading.Lock()
_MAX_LOG = 500


def record_unhandled(text: str):
    """Called by agent.triage() when falling through to LLM (no skill match)."""
    with _log_lock:
        _interaction_log.append(text.strip().lower())
        if len(_interaction_log) > _MAX_LOG:
            _interaction_log.pop(0)


def _extract_intent_keywords(text: str) -> str:
    """Strip common words, return core intent (first 5 content words)."""
    stopwords = {"the", "a", "an", "i", "me", "my", "please", "can", "you", "help",
                 "want", "need", "would", "like", "how", "to", "do", "make", "get", "us"}
    words = [w for w in re.findall(r'\b[a-z]{3,}\b', text) if w not in stopwords]
    return " ".join(words[:5])


def _discover_patterns() -> list[str]:
    """Return recurring intent patterns above threshold."""
    with _log_lock:
        log_copy = list(_interaction_log)
    if not log_copy:
        return []
    counts = Counter(_extract_intent_keywords(t) for t in log_copy)
    return [pattern for pattern, count in counts.items() if count >= RECURRENCE_THRESHOLD and pattern]


def _run_discovery_cycle(config: dict, skills_dir: str):
    """One discovery cycle — find patterns and trigger evolution for new ones."""
    import evolution_state as _evo_state
    if not EVOLUTION_ENABLED or not _evo_state.should_evolve():
        return

    patterns = _discover_patterns()
    if not patterns:
        return

    from evolution_log import EvolutionLog
    evo_log = EvolutionLog()
    known_gaps = {g["task"] for g in evo_log.get_gaps()}
    known_resolved = {e["task"] for e in evo_log.get_history(limit=500) if e.get("found")}

    for pattern in patterns:
        if pattern in known_gaps or pattern in known_resolved:
            continue  # Already handled or tracked

        print(f"[goal_discovery] Recurring pattern detected ({RECURRENCE_THRESHOLD}+ hits): '{pattern}'")

        from evolution_hook import maybe_evolve
        result = maybe_evolve(pattern, config, skills_dir=skills_dir)

        if result:
            status = "resolved" if result.found else "gap"
            print(f"[goal_discovery] Pattern '{pattern}': {status} | installed={result.installed}")


def start_discovery_thread(config: dict, skills_dir: str) -> threading.Thread:
    """Start the background goal discovery thread. Returns the thread."""
    def _loop():
        print(f"[goal_discovery] Started — interval={DISCOVERY_INTERVAL}s threshold={RECURRENCE_THRESHOLD}")
        time.sleep(60)  # Initial delay — let agent warm up
        while True:
            try:
                _run_discovery_cycle(config, skills_dir)
            except Exception as e:
                print(f"[goal_discovery] Error in discovery cycle: {e}")
            time.sleep(DISCOVERY_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True, name="goal-discovery")
    t.start()
    return t
