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
import logging
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from runtime_paths import PROMOTED_SIGNALS_DB

logger = logging.getLogger(__name__)

DISCOVERY_INTERVAL = int(os.environ.get("DISCOVERY_INTERVAL_S", "300"))  # 5 min default
RECURRENCE_THRESHOLD = int(os.environ.get("DISCOVERY_THRESHOLD", "3"))    # 3 hits → evolve
EVOLUTION_ENABLED = os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true"
BOOT_PATTERN_CAP = int(os.environ.get("DISCOVERY_BOOT_CAP", "3"))         # ADR-009: max patterns at boot

# Module-level infer_fn for background evolution paths (FIX #1: capability verifier)
_infer_fn = None


def set_infer_fn(fn) -> None:
    """Inject the model infer callable so background evolution passes it to the
    capability verifier.  Call from api.py startup after mdl.load()."""
    global _infer_fn
    _infer_fn = fn


_interaction_log: list[str] = []   # in-memory ring buffer
_log_lock = threading.Lock()
_MAX_LOG = 500

# ADR-009: deferred patterns (seeded at boot but not yet processed)
_deferred_patterns: list[str] = []
_deferred_lock = threading.Lock()

# ADR-009: timestamp of last model call (for idle detection)
_last_model_call_ts: float = 0.0

# ── Promoted-signals SQLite store ───────────────────────────────────────────
from database.agent import PromotedSignalsRepository as _PromotedSignalsRepo

_DB_FILE = PROMOTED_SIGNALS_DB
_signals_repo = _PromotedSignalsRepo(_DB_FILE)
_signals_repo_path = _DB_FILE


def _get_signals_repo() -> _PromotedSignalsRepo:
    """Get or recreate repo when _DB_FILE is patched (test isolation)."""
    global _signals_repo, _signals_repo_path
    if _signals_repo_path != _DB_FILE:
        _signals_repo = _PromotedSignalsRepo(_DB_FILE)
        _signals_repo_path = _DB_FILE
    return _signals_repo


def _db_conn():
    """Deprecated: kept for compatibility. Use _get_signals_repo() directly."""
    return _get_signals_repo().connection()


def seed_from_chat_history(limit: int = 200, allowed_chat_ids: list | None = None) -> int:
    """Prime _interaction_log with user messages from SQLite chat history.

    When `allowed_chat_ids` is provided, only seeds messages from sessions
    that have at least one attachment row with a matching chat_id — this
    filters out eval/test sessions that wrote to the DB without a real
    Telegram chat_id.

    Also replays persisted promoted signals (at their original weight) so
    high-confidence ideas survive process restarts with full pattern strength.

    Returns total number of entries seeded.
    """
    seeded = 0
    # 1. Chat history (user messages)
    try:
        import core.memory.memory as _mem

        # Build allowed session_id set when chat_id filtering is requested
        allowed_sessions: set | None = None
        if allowed_chat_ids:
            try:
                from core.memory.memory import DB_FILE
                conn = sqlite3.connect(str(DB_FILE))
                placeholders = ",".join("?" * len(allowed_chat_ids))
                rows = conn.execute(
                    f"SELECT DISTINCT session_id FROM attachments WHERE chat_id IN ({placeholders})",
                    allowed_chat_ids,
                ).fetchall()
                conn.close()
                allowed_sessions = {r[0] for r in rows}
            except Exception as _e:
                logger.debug(f"[goal_discovery] chat_id filter build error: {_e}")
                allowed_sessions = None  # fall back to unfiltered

        turns = _mem.history(limit=limit)
        user_turns = [
            t["content"] for t in turns
            if t.get("role") == "user"
            and (allowed_sessions is None or t.get("session_id") in allowed_sessions)
        ]
        if user_turns:
            with _log_lock:
                for text in user_turns:
                    _interaction_log.append(text.strip().lower())
                    if len(_interaction_log) > _MAX_LOG:
                        _interaction_log.pop(0)
            seeded += len(user_turns)
    except Exception as e:
        logger.debug(f"[goal_discovery] seed chat history error: {e}")

    # 2. Promoted signals (high-weight, persisted from prior think cycles)
    try:
        rows = _get_signals_repo().get_all(limit=100)
        promoted_count = 0
        with _log_lock:
            for row in rows:
                text, weight = row["text"], row["weight"]
                for _ in range(weight):
                    _interaction_log.append(text.strip().lower())
                    if len(_interaction_log) > _MAX_LOG:
                        _interaction_log.pop(0)
                promoted_count += weight
        seeded += promoted_count
        if promoted_count:
            logger.debug(f"[goal_discovery] replayed {len(rows)} promoted signals ({promoted_count} entries)")
    except Exception as e:
        logger.debug(f"[goal_discovery] seed promoted signals error: {e}")

    return seeded


def record_unhandled(text: str):
    """Called by agent.triage() when falling through to LLM (no skill match)."""
    with _log_lock:
        _interaction_log.append(text.strip().lower())
        if len(_interaction_log) > _MAX_LOG:
            _interaction_log.pop(0)


def record_promoted(text: str, weight: int = 3, category: str = "", score: float = 0.0):
    """Called when a thought is promoted to an idea.

    Inserts `weight` copies into the in-memory ring buffer so the pattern
    immediately counts toward RECURRENCE_THRESHOLD.  Also persists the signal
    to SQLite so the weight is restored on the next process restart.

    weight=3 means one promoted idea counts as 3 normal user interactions —
    enough to hit threshold on its own when combined with real usage.
    """
    normalised = text.strip().lower()
    with _log_lock:
        for _ in range(weight):
            _interaction_log.append(normalised)
            if len(_interaction_log) > _MAX_LOG:
                _interaction_log.pop(0)
    # Persist to SQLite
    try:
        _get_signals_repo().insert(text.strip(), weight=weight, category=category, score=score)
        logger.debug(f"[goal_discovery] promoted signal persisted (weight={weight}): {text[:60]}")
    except Exception as e:
        logger.debug(f"[goal_discovery] record_promoted DB error: {e}")


def resolve_pattern(text: str) -> None:
    """
    Remove matching promoted signals from the DB and the in-memory log.
    Called when CritiqueLayer returns RESOLVED.
    """
    _STOPWORDS = {
        "this", "that", "with", "have", "from", "they", "will", "what",
        "when", "your", "been", "were", "there", "their", "which", "about",
        "would", "could", "should", "these", "those",
    }
    keywords = [w for w in text.lower().split() if len(w) > 4 and w not in _STOPWORDS]
    if not keywords:
        return
    try:
        for kw in keywords:
            _get_signals_repo().delete_by_keyword(kw)
        logger.info(f"[goal_discovery] resolve_pattern removed signals matching {keywords[:5]}")
    except Exception as e:
        logger.warning(f"[goal_discovery] resolve_pattern error: {e}")
    # Also clear from in-memory ring buffer
    with _log_lock:
        global _interaction_log
        _interaction_log = [
            entry for entry in _interaction_log
            if not any(kw in entry.lower() for kw in keywords)
        ]


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


def _already_installed(skill_name: str, skills_dir: str) -> bool:
    """ADR-009: Check if skill is already installed in any ecosystem tier."""
    base = Path(skills_dir).expanduser()
    return any(
        (base / tier / skill_name).exists()
        for tier in ["private/skills", "community/skills", "third-party/skills"]
    )


def _is_system_idle() -> bool:
    """ADR-009: Returns True when no active replicas and no recent model call."""
    # Never considered idle during SIM_MODE — avoid loading skill models mid-inference
    if os.environ.get("SIM_MODE") == "true":
        return False
    try:
        import core.replica.replica as replica  # import inside function to avoid circular import
        if replica.active():
            return False
    except Exception:
        pass
    idle_threshold = 120.0
    return (time.time() - _last_model_call_ts) > idle_threshold


def _run_single_pattern(pattern: str, config: dict, skills_dir: str):
    """Run discovery cycle for a single pattern (used for deferred drain)."""
    import core.evolution.evolution_state as _evo_state
    if not EVOLUTION_ENABLED or not _evo_state.should_evolve() or not _is_system_idle():
        return

    from core.evolution.evolution_log import EvolutionLog
    evo_log = EvolutionLog()
    known_gaps = {g["task"] for g in evo_log.get_gaps()}
    known_resolved = {e["task"] for e in evo_log.get_history(limit=500) if e.get("found")}

    if pattern in known_gaps or pattern in known_resolved:
        return
    if _already_installed(pattern, skills_dir):
        logger.debug(f"[goal_discovery] Pattern '{pattern}' already installed — skipping")
        return

    print(f"[goal_discovery] Deferred pattern '{pattern}' — running evolution")
    global _last_model_call_ts
    _last_model_call_ts = time.time()

    from core.evolution.evolution_hook import maybe_evolve
    result = maybe_evolve(pattern, config, skills_dir=skills_dir, infer_fn=_infer_fn)
    if result:
        status = "resolved" if result.found else "gap"
        print(f"[goal_discovery] Deferred pattern '{pattern}': {status} | installed={result.installed}")


def _run_discovery_cycle(config: dict, skills_dir: str):
    """One discovery cycle — find patterns and trigger evolution for new ones."""
    # Skip during SIM_MODE to avoid model-loading interference with active inference
    if os.environ.get("SIM_MODE") == "true":
        return
    if not _is_system_idle():
        return
    import core.evolution.evolution_state as _evo_state
    if not EVOLUTION_ENABLED or not _evo_state.should_evolve():
        return

    patterns = _discover_patterns()
    if not patterns:
        return

    from core.evolution.evolution_log import EvolutionLog
    evo_log = EvolutionLog()
    known_gaps = {g["task"] for g in evo_log.get_gaps()}
    known_resolved = {e["task"] for e in evo_log.get_history(limit=500) if e.get("found")}

    global _last_model_call_ts
    for pattern in patterns:
        if pattern in known_gaps or pattern in known_resolved:
            continue  # Already handled or tracked
        if _already_installed(pattern, skills_dir):
            logger.debug(f"[goal_discovery] Pattern '{pattern}' already installed — skipping")
            continue

        print(f"[goal_discovery] Recurring pattern detected ({RECURRENCE_THRESHOLD}+ hits): '{pattern}'")
        _last_model_call_ts = time.time()

        from core.evolution.evolution_hook import maybe_evolve
        result = maybe_evolve(pattern, config, skills_dir=skills_dir, infer_fn=_infer_fn)

        if result:
            status = "resolved" if result.found else "gap"
            print(f"[goal_discovery] Pattern '{pattern}': {status} | installed={result.installed}")


def seed_from_chat_history_and_defer(config: dict, skills_dir: str, limit: int = 200) -> int:
    """ADR-009: Seed from chat history, run top-N immediately, defer the rest.

    Returns total number of entries seeded.
    """
    global _last_model_call_ts
    # Restrict seeding to real Telegram chat sessions (filter out eval/test DB pollution)
    telegram_cfg = config.get("telegram", {})
    allowed_ids = telegram_cfg.get("allowed_chat_ids") or telegram_cfg.get("chat_ids", [])
    # Fallback: read from env (same var used by telegram_bot.py)
    if not allowed_ids:
        env_chat_id = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID", "").strip()
        if env_chat_id:
            allowed_ids = [env_chat_id]
    allowed_chat_ids = [str(c) for c in allowed_ids] if allowed_ids else None
    seeded = seed_from_chat_history(limit=limit, allowed_chat_ids=allowed_chat_ids)

    boot_cap = config.get("goal_discovery", {}).get("boot_pattern_cap", BOOT_PATTERN_CAP)

    # Detect all patterns, sort by frequency descending
    with _log_lock:
        log_copy = list(_interaction_log)
    if not log_copy:
        return seeded

    counts = Counter(_extract_intent_keywords(t) for t in log_copy)
    all_patterns = sorted(
        [p for p, c in counts.items() if c >= RECURRENCE_THRESHOLD and p],
        key=lambda p: counts[p],
        reverse=True,
    )

    if not all_patterns:
        return seeded

    immediate = all_patterns[:boot_cap]
    deferred = all_patterns[boot_cap:]

    with _deferred_lock:
        _deferred_patterns.extend(deferred)

    if deferred:
        print(f"[goal_discovery] Boot cap={boot_cap}: {len(immediate)} patterns immediate, {len(deferred)} deferred")

    # Run top-N immediately (blocks briefly but bounded) only if the system is idle.
    import core.evolution.evolution_state as _evo_state
    if EVOLUTION_ENABLED and _evo_state.should_evolve() and _is_system_idle():
        from core.evolution.evolution_log import EvolutionLog
        evo_log = EvolutionLog()
        known_gaps = {g["task"] for g in evo_log.get_gaps()}
        known_resolved = {e["task"] for e in evo_log.get_history(limit=500) if e.get("found")}

        for pattern in immediate:
            if pattern in known_gaps or pattern in known_resolved:
                continue
            if _already_installed(pattern, skills_dir):
                logger.debug(f"[goal_discovery] Boot pattern '{pattern}' already installed — skipping")
                continue
            print(f"[goal_discovery] Boot pattern (immediate): '{pattern}'")
            _last_model_call_ts = time.time()
            from core.evolution.evolution_hook import maybe_evolve
            result = maybe_evolve(pattern, config, skills_dir=skills_dir, infer_fn=_infer_fn)
            if result:
                status = "resolved" if result.found else "gap"
                print(f"[goal_discovery] Boot pattern '{pattern}': {status}")

    return seeded


def start_discovery_thread(config: dict, skills_dir: str) -> threading.Thread:
    """Start the background goal discovery thread. Returns the thread."""
    def _loop():
        print(f"[goal_discovery] Started — interval={DISCOVERY_INTERVAL}s threshold={RECURRENCE_THRESHOLD}")
        # ADR-009: Seed with boot cap — run top-N immediately, defer the rest
        seeded = seed_from_chat_history_and_defer(config, skills_dir)
        if seeded:
            print(f"[goal_discovery] Seeded {seeded} user messages from chat history")
        time.sleep(60)  # Initial delay — let agent warm up
        drain_per_cycle = config.get("goal_discovery", {}).get("deferred_drain_per_cycle", 1)
        while True:
            try:
                _run_discovery_cycle(config, skills_dir)
                # ADR-009: Drain deferred patterns one per cycle when idle
                with _deferred_lock:
                    has_deferred = bool(_deferred_patterns)
                if has_deferred and _is_system_idle():
                    for _ in range(drain_per_cycle):
                        with _deferred_lock:
                            if not _deferred_patterns:
                                break
                            pattern = _deferred_patterns.pop(0)
                        _run_single_pattern(pattern, config, skills_dir)
            except Exception as e:
                print(f"[goal_discovery] Error in discovery cycle: {e}")
            time.sleep(DISCOVERY_INTERVAL)

    t = threading.Thread(target=_loop, daemon=True, name="goal-discovery")
    t.start()
    return t
