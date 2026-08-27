"""
thought_engine.py — ADR-005 Think-at-Rest subsystem.

Implements:
  IdleDetector    — fires callback after no requests for threshold_s seconds
  ThoughtGenerator — System 1 (drafter) produces thought seeds
  ThoughtEvaluator — System 2 (main model) scores + classifies seeds
  ThinkAtRest     — orchestrator daemon thread

Categories & signal sources (2026-06-18 restructure):
  gap_reflection:
    ONLY from failed_requests table. A real user request that could not be fulfilled.
    At rest: try evolution for each unresolved request (max 3 retries), then mark resolved/abandoned.
    Skip entirely when no failed_requests exist.

  self_improvement:
    From performance metrics: evolution retry loops, error rate trends, response time degradation,
    recurring error patterns, stuck probes. Never from "introspection".

  curiosity:
    Outward exploration: git log on active repos, system health checks, web-accessible feeds,
    benchmark comparisons. Gather new information that might surface real gaps.
    Never "the thought focuses on the gap between X and Y".
"""
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Callable, Optional
from runtime_paths import MODEL_ACTIVITY_FILE
from runtime_paths import NOTES_DIR

logger = logging.getLogger(__name__)
_ACTIVITY_PATH = str(MODEL_ACTIVITY_FILE)

_THOUGHT_SYSTEM_PROMPT = (
    "You are Kernel Evo, a local self-evolving AI agent. You are currently idle.\n"
    "Use idle time productively: think about your user, the world, and your own growth — in that order of priority.\n"
)

_GENERATOR_TEMPLATE = """\
You are Kernel Evo, a local self-evolving AI agent. You are idle.

Your context:
- Skills you haven't used recently: {unused_skills}
- Evolution history (last 5): {evolution_context}

Generate exactly ONE exploratory curiosity thought.
This is NOT about what capabilities you lack — it's about exploring the outside world.

Good examples:
- "I wonder what changed in the fabiopacifici.com repo this week — let me check git log"
- "I should benchmark my speculative decoding speed against last week's baseline"
- "What's the GPU temperature trend over the last 24 hours?"
- "I want to check if there are new releases of my dependencies"

Bad examples (do NOT write):
- "I lack a robust X pipeline" or any gap/inability framing
- "The thought focuses on the gap between X and Y"
- "I should prioritize integrating Y" (that's a gap, not curiosity)

Format: one line, actionable, outward-facing.
"""

_EVALUATOR_TEMPLATE = """\
Review these thought seeds from your idle reflection.
Score each 0.0-1.0 for depth/insight. Classify each as: retrospective | gap_reflection | \
self_improvement | curiosity. Discard if score < {min_score}.

Thoughts:
{seeds}

Respond as JSON array: [{{"thought": "...", "score": 0.x, "category": "...", "promote": true/false}}]
Only return the JSON array, nothing else.
"""


class IdleDetector:
    """Tracks last request timestamp and fires callback when idle > threshold_s."""

    def __init__(self, threshold_s: float, callback: Callable, poll_interval_s: float = 10.0,
                 startup_grace_s: float = 600.0):
        self._threshold_s = threshold_s
        self._callback = callback
        self._poll_interval_s = poll_interval_s
        # startup_grace_s: extra time after boot before idle detection begins.
        # Prevents ThinkAtRest firing immediately after every restart.
        self._startup_grace_s = startup_grace_s
        self._boot_time = time.monotonic()
        self._last_request = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def ping(self):
        """Call this on every incoming request to reset the idle clock."""
        self._last_request = time.monotonic()

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="idle-detector")
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        poll = min(self._poll_interval_s, max(0.1, self._threshold_s / 10))
        while not self._stop_event.wait(poll):
            # Respect startup grace period — don't fire during boot window
            if (time.monotonic() - self._boot_time) < self._startup_grace_s:
                continue
            elapsed = time.monotonic() - self._last_request
            if elapsed >= self._threshold_s:
                try:
                    self._callback()
                except Exception as e:
                    logger.error(f"[IdleDetector] callback error: {e}")


class ThoughtGenerator:
    """System 1: prefer the configured local thought slot (default Gemma 'audio'),
    fall back to drafter-only generation when the local slot is unavailable."""

    def __init__(self, unused_skills_fn: Optional[Callable] = None,
                 recent_gaps_fn: Optional[Callable] = None,
                 last_thought_fn: Optional[Callable] = None,
                 recent_interactions_fn: Optional[Callable] = None,
                 thought_slot: str = "audio"):
        self._unused_skills_fn = unused_skills_fn
        self._recent_gaps_fn = recent_gaps_fn
        self._last_thought_fn = last_thought_fn
        self._recent_interactions_fn = recent_interactions_fn
        self._thought_slot = thought_slot
        self._recent_thoughts: list = []

    def set_recent_thoughts(self, thoughts: list) -> None:
        """Store recent thought summaries for anti-seed injection in generate()."""
        self._recent_thoughts = thoughts or []

    def generate(self, exploration_summary: str = "") -> list:
        """Return list of raw thought strings (1-3). Local model only.

        When exploration_summary is provided, generates a curiosity thought
        based on real outward data. When empty, returns nothing — no synthetic
        generation from vacuum.
        """
        import core.inference.model_client as model_client
        if not model_client.is_server_running():
            logger.debug("[ThoughtGenerator] model server not running — skipping")
            return []

        if not exploration_summary:
            logger.debug("[ThoughtGenerator] no exploration data — skipping synthetic generation")
            return []

        unused_skills = "none"
        try:
            if self._unused_skills_fn:
                unused_skills = self._unused_skills_fn() or "none"
        except Exception:
            pass

        # Get evolution context for enrichment
        evolution_context = "none"
        try:
            if self._recent_gaps_fn:
                evolution_context = self._recent_gaps_fn() or "none"
        except Exception:
            pass

        prompt = _GENERATOR_TEMPLATE.format(
            unused_skills=unused_skills,
            evolution_context=evolution_context,
        )
        prompt += f"\n\nRecent exploration data:\n{exploration_summary[:1200]}"

        # Anti-seed to prevent repetition
        if self._recent_thoughts:
            anti_seed_block = "\n".join(f"- {s}" for s in self._recent_thoughts[:5])
            prompt += (
                "\n\nDo not repeat or closely paraphrase any of these recent thoughts:\n"
                + anti_seed_block
            )

        messages = [
            {"role": "system", "content": _THOUGHT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Priority #0 + thought_slot config (2026-08-26): run thoughts on the
        # configured local slot (default Gemma 'audio') so idle reflection works
        # even in cloud mode and only ONE local model loads (no Nemotron drafter).
        raw = ""
        try:
            raw = model_client.infer_local(messages, max_new_tokens=8192, slot=self._thought_slot)
        except Exception as e:
            logger.error(f"[ThoughtGenerator] local slot infer error: {e}")

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtGenerator] local slot '{self._thought_slot}' unavailable, falling back to drafter: {raw!r}")
            try:
                raw = model_client.infer_draft(prompt, max_new_tokens=8192)
            except Exception as e:
                logger.error(f"[ThoughtGenerator] infer_draft error: {e}")
                return []

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtGenerator] bad response: {raw!r}")
            return []

        # Parse one thought per line, skip empty lines
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # Remove numbered prefixes like "1. " or "- "
        cleaned = []
        for line in lines:
            line = re.sub(r"^[\d]+\.\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            if line:
                cleaned.append(line)
        return cleaned[:5]


class ThoughtEvaluator:
    """System 2: uses the configured local thought slot to score + classify seeds."""

    def __init__(self, min_score: float = 0.65, thought_slot: str = "audio"):
        self._min_score = min_score
        self._thought_slot = thought_slot

    def evaluate(self, seeds: list) -> list:
        """
        Score and classify thought seeds.
        Returns list of dicts: {thought, score, category, promote}
        Discards score < min_score.
        """
        if not seeds:
            return []

        import core.inference.model_client as model_client
        if not model_client.is_server_running():
            logger.debug("[ThoughtEvaluator] model server not running — skipping")
            return []

        seeds_text = "\n".join(f"- {s}" for s in seeds)
        prompt = _EVALUATOR_TEMPLATE.format(
            seeds=seeds_text,
            min_score=self._min_score,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            raw = model_client.infer_local(messages, max_new_tokens=8192, slot=self._thought_slot)
        except Exception as e:
            logger.error(f"[ThoughtEvaluator] local slot infer error: {e}")
            return []

        if not raw or raw.startswith("[model_"):
            logger.warning(f"[ThoughtEvaluator] bad response: {raw!r}")
            return []

        # Extract JSON array from response
        try:
            # Try to find JSON array in response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                results = json.loads(match.group())
            else:
                results = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"[ThoughtEvaluator] JSON parse error: {e} — raw: {raw[:200]}")
            return []

        if not isinstance(results, list):
            return []

        accepted = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            if score < self._min_score:
                continue
            # Normalise category
            category = item.get("category", "curiosity")
            valid_categories = {"retrospective", "gap_reflection", "self_improvement", "curiosity"}
            if category not in valid_categories:
                category = "curiosity"
            accepted.append({
                "thought": item.get("thought", ""),
                "score": score,
                "category": category,
                "promote": bool(item.get("promote", score >= 0.7)),
            })
        return accepted


class ThinkAtRest:
    """
    Orchestrates: IdleDetector → ThoughtGenerator → ThoughtEvaluator → ThoughtJournal.
    Runs as a daemon thread. Config-driven.
    """

    def _model_is_active(self) -> bool:
        """Return True when model inference is in-flight or just finished recently."""
        try:
            with open(_ACTIVITY_PATH) as f:
                data = json.load(f)
            if int(data.get("in_flight", 0)) > 0:
                # Staleness guard: if last_start_ts is older than 120s and we still
                # see in_flight > 0, the counter is stuck (process was hard-killed).
                # Treat it as idle so thought cycles are not permanently blocked.
                last_start = float(data.get("last_start_ts", 0.0) or 0.0)
                if last_start > 0 and (time.time() - last_start) > 120.0:
                    logger.warning(
                        "[ThinkAtRest] in_flight counter appears stuck (>120s since last_start) — treating as idle"
                    )
                    # Auto-heal: reset the stuck counter
                    try:
                        data["in_flight"] = 0
                        import os as _os
                        _os.makedirs(_os.path.dirname(_ACTIVITY_PATH), exist_ok=True)
                        with open(_ACTIVITY_PATH, "w") as fw:
                            json.dump(data, fw)
                    except Exception:
                        pass
                    return False
                return True
            quiet_period_s = 30.0
            last_end = float(data.get("last_end_ts", 0.0) or 0.0)
            return last_end > 0 and (time.time() - last_end) < quiet_period_s
        except Exception:
            return False

    def _main_model_loaded(self) -> bool:
        """Return True only when the full main model is already resident."""
        try:
            import core.inference.model_client as model_client
            health = model_client.health()
            return bool(health.get("main_model_loaded", False))
        except Exception:
            return False

    def _local_slot_available(self) -> bool:
        """Return True when the local thought slot can be used for idle thoughts.

        Priority #0: idle thoughts run on a resident/lazy-loadable local slot
        (Gemma 4 E2B-it, the `audio` slot), independent of the cloud text
        provider. The slot is lazy-loaded on first use by `infer_local()`, so
        this only needs to confirm the model server is running.
        """
        try:
            import core.inference.model_client as model_client
            return model_client.is_server_running()
        except Exception:
            return False

    def _deferred_path(self) -> Path:
        d = Path(self._ideas_dir) / "deferred_thoughts"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{date.today().strftime('%Y-%m-%d')}.jsonl"

    def _store_deferred_drafts(self, seeds: list) -> None:
        if not seeds:
            return
        path = self._deferred_path()
        record = {
            "ts": datetime.now().isoformat(),
            "status": "pending_score",
            "seeds": seeds,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"[ThinkAtRest] deferred {len(seeds)} draft thought(s) for later scoring")

    def _pop_deferred_drafts(self) -> list:
        path = self._deferred_path()
        if not path.exists():
            return []
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return []
        first = json.loads(lines[0])
        remaining = lines[1:]
        if remaining:
            path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return first.get("seeds", [])

    def __init__(self, config: dict):
        self._cfg_full = config           # full config (paths, thinking, etc.)
        self._cfg = config.get("thinking", {})
        self._enabled = self._cfg.get("enabled", False)
        self._idle_threshold_s = float(self._cfg.get("idle_threshold_s", 300))
        self._startup_grace_s = float(self._cfg.get("startup_grace_s", 600))
        self._thought_interval_s = float(self._cfg.get("thought_interval_s", 1800))
        self._min_score = float(self._cfg.get("min_score", 0.65))
        self._promote_threshold = float(self._cfg.get("promote_threshold", 0.80))
        # Which local model slot the Think-at-Rest engine uses for BOTH thought
        # generation and evaluation. Default "audio" = Gemma 4 E2B-it (multimodal,
        # lazy-loaded). Set to e.g. "primary" for Nemotron. This makes the engine
        # load only ONE local model instead of two.
        self._thought_slot = self._cfg.get("thought_slot", "audio") or "audio"
        self._proactive_telegram = self._cfg.get("proactive_telegram", True)
        self._proactive_max_per_day = int(self._cfg.get("proactive_max_per_day", 2))
        self._journal_dir = os.path.expanduser(
            self._cfg.get("journal_dir", "~/.kernel-evolving/workspace/thoughts")
        )
        self._ideas_dir = os.path.expanduser(
            self._cfg.get("ideas_dir", "~/.kernel-evolving/workspace/thoughts/ideas")
        )

        from core.memory.thought_journal import ThoughtJournal
        self._journal = ThoughtJournal(journal_dir=self._journal_dir)

        # ADR-019: ObserverLayer — evidence gate + anti-seed injection
        _obs_cfg = self._cfg.get("observer", {})
        _obs_enabled = _obs_cfg.get("enabled", True)
        if _obs_enabled:
            from core.layers.observer_layer import ObserverLayer
            self._observer = ObserverLayer(config=config)
        else:
            self._observer = None

        # ADR-019 Phase 3: CritiqueLayer
        _crit_cfg = self._cfg.get("critique", {})
        _crit_enabled = _crit_cfg.get("enabled", True)
        if _crit_enabled:
            from core.layers.critique_layer import CritiqueLayer
            self._critique = CritiqueLayer(config=self._cfg)
        else:
            self._critique = None

        self._generator = ThoughtGenerator(
            unused_skills_fn=self._get_unused_skills,
            recent_gaps_fn=self._get_recent_gaps,
            last_thought_fn=self._get_last_thought,
            recent_interactions_fn=self._get_recent_interactions,
            thought_slot=self._thought_slot,
        )
        self._evaluator = ThoughtEvaluator(min_score=self._min_score, thought_slot=self._thought_slot)

        # Idle state
        self._is_idle = False
        self._last_think_time = 0.0
        self._idle_detector = IdleDetector(
            threshold_s=self._idle_threshold_s,
            callback=self._on_idle,
            startup_grace_s=self._startup_grace_s,
        )

        # Proactive rate limiting
        self._proactive_count_today = 0
        self._proactive_date = date.today()

    def ping(self):
        """Signal that a request was received (resets idle timer)."""
        self._is_idle = False
        self._idle_detector.ping()

    def mark_active(self, user_text: str = "") -> None:
        """Alias for ping() — called by API middleware on every request."""
        self._idle_detector.ping()
        self._is_idle = False
        if user_text and self._observer is not None:
            triggered = self._observer.check_interaction_for_probes(user_text)
            if triggered:
                for p in triggered:
                    logger.info(
                        f"[ThinkAtRest] probe re-surfaced by user interaction: "
                        f"'{p['subject']}' (id={p['id']}) — "
                        f"will evaluate as EVIDENCED on next think cycle"
                    )

    def start(self):
        """Start the think-at-rest subsystem."""
        if not self._enabled:
            logger.info("[ThinkAtRest] disabled in config, not starting")
            return
        self._idle_detector.start()
        logger.info(
            f"[ThinkAtRest] started — idle_threshold={self._idle_threshold_s}s, "
            f"interval={self._thought_interval_s}s"
        )

    def stop(self):
        self._idle_detector.stop()

    def _on_idle(self):
        """Called by IdleDetector when idle threshold is crossed."""
        # Idle is not just "no HTTP requests" — ongoing/recent model work counts as active.
        if self._model_is_active():
            self._is_idle = False
            logger.debug("[ThinkAtRest] skipping idle cycle — model still active/recently active")
            return
        if not self._is_idle:
            self._is_idle = True
            logger.info("[ThinkAtRest] entered idle state")

        # Priority #6 (delta-aware cadence): the clock is a FLOOR, not the sole
        # driver. A changed performance signal is the primary event trigger — if
        # the anomaly state changed since the last cycle, run immediately (subject
        # to a short minimum-interval floor to avoid thundering). Otherwise fall
        # back to the regular cadence so slow drift (e.g. RAM creep) is still caught.
        elapsed = time.monotonic() - self._last_think_time
        min_floor_s = min(self._thought_interval_s, 300.0)
        try:
            perf_changed = bool(self._gather_performance_signals())
        except Exception:
            perf_changed = False
        if elapsed >= self._thought_interval_s or (perf_changed and elapsed >= min_floor_s):
            self._run_think_cycle()

    def _get_triggered_probe_seeds(self) -> list[str]:
        """Return thoughts from probes that were triggered but not yet resolved."""
        if self._observer is None:
            return []
        try:
            active = self._observer._probe_store.list_active()
            return [
                p["thought_text"][:120]
                for p in active
                if p.get("triggered_at") and not p.get("resolved")
            ]
        except Exception:
            return []

    # ── Exploration & signal gathering ─────────────────────────────────

    def _discover_repos(self, max_repos: int = 8) -> list[dict]:
        """Discover git repos in the workspace repositories/ folder.
        Uses config paths.REPOSITORIES as primary source (deployment-configurable).
        Falls back to WORKSPACE_ROOT/repositories/ or scans WORKSPACE_ROOT for git dirs.
        Returns list of {name, path} sorted by last activity (newest first)."""
        repos: list[dict] = []

        # Primary: config.yaml paths.REPOSITORIES (deployment-specific)
        repos_root = None
        try:
            repos_root_str = self._cfg_full.get("paths", {}).get("REPOSITORIES", "")
            if repos_root_str:
                repos_root = Path(os.path.expanduser(repos_root_str))
        except Exception:
            pass

        # Fallback: scan the kernel's own workspace for git repos
        if not repos_root or not repos_root.is_dir():
            try:
                from runtime_paths import WORKSPACE_ROOT
                prepos_ws = WORKSPACE_ROOT / "repositories"
                if prepos_ws.is_dir():
                    repos_root = prepos_ws
                else:
                    # Last resort: scan workspace for any dirs with .git
                    repos_root = WORKSPACE_ROOT
            except Exception:
                pass

        if not repos_root or not repos_root.is_dir():
            return repos

        for entry in sorted(repos_root.iterdir()):
            if not entry.is_dir():
                continue
            git_dir = entry / ".git"
            if not git_dir.exists():
                continue
            if len(repos) >= max_repos:
                break
            repos.append({"name": entry.name, "path": str(entry)})

        # Sort by most recent git activity
        def _last_commit_date(p: str) -> str:
            try:
                import subprocess
                out = subprocess.check_output(
                    ["git", "-C", p, "log", "-1", "--format=%ct"],
                    stderr=subprocess.DEVNULL, text=True, timeout=5,
                ).strip()
                return out or "0"
            except Exception:
                return "0"

        repos.sort(key=lambda r: _last_commit_date(r["path"]), reverse=True)
        return repos

    def _gather_exploration_data(self) -> dict:
        """Gather real outward data for curiosity generation.
        Returns dict with keys: git_log, system_health.
        Each is a short text summary or empty string if nothing changed.
        Repos are discovered dynamically — no hardcoded paths.
        """
        results: dict[str, str] = {}

        # ── Git activity across discovered repos ──
        git_lines = []
        for repo in self._discover_repos(max_repos=8):
            try:
                import subprocess
                out = subprocess.check_output(
                    ["git", "-C", repo["path"], "log", "--oneline", "-5", "--since=48 hours ago"],
                    stderr=subprocess.DEVNULL, text=True, timeout=10,
                ).strip()
                if out:
                    git_lines.append(f"{repo['name']}:\n{out}")
            except Exception:
                pass
        results["git_log"] = "\n".join(git_lines) if git_lines else "no recent commits"

        # ── Universal system health checks ──
        health_lines = []
        checks = [
            ("GPU", "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU'"),
            ("Memory", "free -h 2>/dev/null | grep Mem"),
            ("Disk", "df -h / 2>/dev/null | tail -1"),
            ("Uptime", "uptime 2>/dev/null"),
        ]
        for name, cmd in checks:
            try:
                import subprocess
                out = subprocess.check_output(
                    cmd, shell=True, stderr=subprocess.DEVNULL, text=True, timeout=10,
                ).strip()
                if out:
                    health_lines.append(f"{name}: {out[:120]}")
            except Exception:
                pass
        results["system_health"] = "\n".join(health_lines) if health_lines else ""

        return results

    def _gather_performance_signals(self) -> list[dict]:
        """Return real self-improvement signals from metrics.

        Each is {message, metric, severity}. Returns empty list if nothing notable.

        Priority #6 (state-change suppression): a signal is only emitted when its
        underlying value *changes* since the last cycle (or crosses a threshold
        boundary). A confirmed anomaly that merely persists at the same value is
        NOT re-emitted — otherwise the same observation is re-journaled every
        cycle (the "uniform heartbeat" failure mode).
        """
        signals = []
        try:
            # Check for stuck probes (> 7 days old, unanswered)
            if self._observer is not None:
                active = self._observer._probe_store.list_active()
                old_probes = [p for p in active if p.get("created_at")]
                probe_count = len(old_probes)
                metric = "stuck_probes"
                last = getattr(self, "_last_signal_values", {}).get(metric)
                # Emit only when the count changed, or crossed the threshold
                # (>10) for the first time. Persist the last-observed value so
                # a stable count goes quiet.
                if probe_count > 10 and (
                    last is None or last <= 10 or probe_count != last
                ):
                    signals.append({
                        "message": f"{probe_count} active probes accumulating — may indicate unresolved patterns",
                        "metric": metric,
                        "severity": 0.75,
                    })
                self._last_signal_values = getattr(self, "_last_signal_values", {})
                self._last_signal_values[metric] = probe_count
        except Exception:
            pass
        return signals

    def _has_exploration_changed(self) -> bool:
        """Track whether exploration data changed since last cycle.
        Uses a simple hash. Returns True if data is new/different."""
        import hashlib
        try:
            data = self._gather_exploration_data()
            h = hashlib.md5(str(data).encode()).hexdigest()
            changed = (h != getattr(self, "_last_exploration_hash", ""))
            self._last_exploration_hash = h
            return changed
        except Exception:
            return True  # on error, assume changed to be safe

    # ── Core cycle ─────────────────────────────────────────────────────

    def _run_think_cycle(self):
        """Restructured think cycle (2026-06-18).

        Real-world-first order:
          1. Housekeeping (probe expiry, memory seeker, identity consolidation)
          2. gap_reflection  — ONLY from real failed_requests (handled in EvolvingThinkAtRest override)
          3. self_improvement — from performance metrics (handled in EvolvingThinkAtRest override)
          4. curiosity       — real exploration (git, health, feeds) → LLM-generated insight

        No synthetic gap enumeration. No forced 3-thought output.
        """
        self._last_think_time = time.monotonic()
        logger.info("[ThinkAtRest] running think cycle")

        # ── Housekeeping (always) ──
        if self._observer is not None:
            expired = self._observer.expire_probes()
            if expired:
                logger.info(f"[ThinkAtRest] expired {expired} old probe(s)")
        self._run_memory_seeker()
        self._run_identity_consolidator()

        # ── Curiosity: real exploration → LLM insight ──
        # Priority #0: the old hard `_main_model_loaded()` gate blocked all
        # curiosity in cloud mode (no resident primary model). Replace it with a
        # local-slot-available check: if the model server is running, the local
        # thought slot (Gemma 4 E2B-it) will be lazy-loaded on first use by
        # infer_local(). Skip only if the server is not running at all.
        if not self._local_slot_available():
            logger.debug("[ThinkAtRest] no local slot available (model server not running) — skipping curiosity generation")
            return

        # Only generate if exploration data actually changed since last cycle
        if not self._has_exploration_changed():
            logger.debug("[ThinkAtRest] no change in exploration data — skipping")
            return

        exploration = self._gather_exploration_data()
        has_data = any(v and v != "no recent commits" for v in exploration.values())
        if not has_data:
            logger.debug("[ThinkAtRest] nothing to explore — cycle complete")
            return

        # Build a prompt from real exploration data
        summary_parts = []
        if exploration.get("git_log") and exploration["git_log"] != "no recent commits":
            summary_parts.append(f"Recent repo activity:\n{exploration['git_log'][:500]}")
        if exploration.get("system_health"):
            summary_parts.append(f"System health:\n{exploration['system_health'][:300]}")
        exploration_summary = "\n\n".join(summary_parts) if summary_parts else ""

        try:
            seeds = self._generator.generate(exploration_summary=exploration_summary)
        except Exception as e:
            logger.error(f"[ThinkAtRest] generator error: {e}")
            return

        # Also inject triggered probe seeds
        probe_seeds = self._get_triggered_probe_seeds()
        if probe_seeds:
            seeds.extend(probe_seeds)

        if not seeds:
            logger.debug("[ThinkAtRest] no curiosity seeds generated")
            return

        try:
            thoughts = self._evaluator.evaluate(seeds)
        except Exception as e:
            logger.error(f"[ThinkAtRest] evaluator error: {e}")
            return

        # Anti-seed for next cycle
        if self._observer is not None:
            try:
                self._generator.set_recent_thoughts(self._observer.get_anti_seed_context(n=10))
            except Exception as e:
                logger.debug(f"[ThinkAtRest] anti-seed injection error: {e}")

        for thought in thoughts:
            try:
                # ADR-019: Observer gate
                if self._observer is not None:
                    verdict = self._observer.evaluate(thought)
                    thought["_observer_verdict"] = verdict.classification
                    thought["_observer_path"]    = verdict.evolution_path
                    thought["_observer_evidence"]= verdict.evidence_summary
                    if verdict.evolution_path in ("probe", "journal_only"):
                        self._journal.write(thought)
                        continue

                self._journal.write(thought)
                self._on_thought_accepted(thought)
            except Exception as e:
                logger.error(f"[ThinkAtRest] journal write error: {e}")

        logger.info(f"[ThinkAtRest] curiosity cycle complete — {len(thoughts)} thought(s)")

    def _on_thought_accepted(self, thought: dict):
        """Handle a thought that passed evaluation.

        gap_reflection evolution is NOT triggered here — it comes directly from
        failed_requests in EvolvingThinkAtRest._run_think_cycle(), no generation step.

        self_improvement goal_discovery is NOT triggered here — it comes from
        real performance metrics with concrete recommendations.

        Only curiosity thoughts reach here: just promote high-score ideas.
        """
        score = thought.get("score", 0.0)
        category = thought.get("category", "curiosity")

        # Only promote curiosity thoughts — gap_reflection/self_improvement are
        # handled via direct paths, not this generation pipeline.
        if category == "gap_reflection":
            # Synthetic gap_reflection — journal only, no evolution.
            # Real gaps are handled by failed_requests in EvolvingThinkAtRest.
            logger.debug(f"[ThinkAtRest] synthetic gap_reflection — journal only, no auto-evolution")
            return

        if category == "self_improvement":
            # Synthetic self_improvement — tracker todo only, no goal_discovery.
            # Real improvement signals come from performance metrics.
            thought_text = thought.get("thought", "")
            self._add_tracker_todo(thought_text)
            return

        # curiosity: promote high-score to ideas
        if score >= self._promote_threshold:
            self._promote_to_idea(thought)

        # Proactive Telegram for accepted curiosity thoughts (respects the
        # proactive_max_per_day cap). Actionable thoughts get ✅/❌ buttons.
        try:
            self._maybe_send_telegram(thought)
        except Exception as e:
            logger.error(f"[ThinkAtRest] proactive telegram error: {e}")

    def _promote_to_idea(self, thought: dict):
        """Write thought to ideas directory and feed as high-weight signal into goal_discovery."""
        try:
            os.makedirs(self._ideas_dir, exist_ok=True)
            today = date.today().strftime("%Y-%m-%d")
            slug = thought["thought"][:40].lower()
            slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
            filename = f"{today}-{slug}.md"
            path = os.path.join(self._ideas_dir, filename)

            # Dedup: skip file write if same idea already promoted today
            _file_already_exists = os.path.exists(path)
            if not _file_already_exists:
                content = (
                    f"# {thought['thought'][:80]}\n\n"
                    f"**Date:** {today}\n"
                    f"**Category:** {thought.get('category', 'unknown')}\n"
                    f"**Score:** {thought.get('score', 0.0):.2f}\n\n"
                    f"{thought['thought']}\n"
                )
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"[ThinkAtRest] promoted idea → {path}")

                # Feed as high-weight signal into goal_discovery
                # Only on first promotion to prevent self-reinforcing snowball loop.
                try:
                    import core.pipelines.goal_discovery as goal_discovery
                    goal_discovery.record_promoted(
                        thought["thought"],
                        weight=3,
                        category=thought.get("category", ""),
                        score=thought.get("score", 0.0),
                    )
                    logger.info("[ThinkAtRest] promoted idea fed into goal_discovery (weight=3)")
                except Exception as e:
                    logger.error(f"[ThinkAtRest] goal_discovery promotion error: {e}")

                # Notify Telegram — desire promoted, invite action
                # Only on first promotion to avoid 66-notification morning.
                try:
                    import services.channels.telegram_bot as _tb
                    chat_id = _tb.ALLOWED_CHAT_ID
                    if chat_id:
                        cat = thought.get("category", "curiosity")
                        score = thought.get("score", 0.0)
                        cat_emoji = {"gap_reflection": "\U0001f9e0", "self_improvement": "\U0001f4aa",
                                     "curiosity": "\U0001f50d", "retrospective": "\U0001f4d6"}.get(cat, "\U0001f4ad")
                        _tb.send_message(chat_id,
                            f"{cat_emoji} *Desire promoted* (score {score:.2f})\n"
                            f"Category: `{cat}`\n\n"
                            f"_{thought['thought'][:200]}_\n\n"
                            f"Queued as goal signal. Reply /evolve to trigger now, or let discovery pick it up."
                        )
                except Exception:
                    pass
            else:
                logger.debug(f"[ThinkAtRest] idea already exists today, skipping all side effects: {filename}")
        except Exception as e:
            logger.error(f"[ThinkAtRest] idea promotion error: {e}")

    def _run_evolution_with_critique(self, thought: dict) -> None:
        """
        Full evolution cycle with CritiqueLayer feedback loop.
        Max 3 attempts. Wraps maybe_evolve() and assesses with CritiqueLayer.
        """
        try:
            import core.evolution.evolution_hook as evolution_hook
            from core.evolution.evolution_hook import EVOLUTION_ENABLED
            if not EVOLUTION_ENABLED:
                return
        except ImportError:
            return  # evolution_hook not available in base kernel

        try:
            import core.pipelines.goal_discovery as _gd
            import yaml
        except ImportError as e:
            logger.warning(f"[ThinkAtRest] _run_evolution_with_critique import error: {e}")
            return

        thought_text = thought.get("thought", "")
        max_iter = 3
        critique_notes = ""

        _cfg_ovr = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
        cfg_path = (
            os.path.abspath(os.path.expanduser(_cfg_ovr))
            if _cfg_ovr
            else os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
        )
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[ThinkAtRest] could not load config for evolution: {e}")
            return
        skills_dir = os.path.expanduser(
            os.environ.get("SKILLS_DIR") or cfg.get("skills_dir", "~/.kernel-evolving/ecosystem")
        )
        infer_fn = _gd._infer_fn

        for attempt in range(1, max_iter + 1):
            task = thought_text
            if critique_notes:
                task = f"{thought_text}\n\nPrevious attempt notes: {critique_notes}"

            # ADR-021 (fix 2026-08-26): carry the originating chat_id into the
            # evolution pipeline so the Tier 2 approval gate can ask the user
            # instead of being silently skipped.
            source_chat_id = thought.get("_source_chat_id")

            try:
                evo_result = evolution_hook.maybe_evolve(
                    task,
                    cfg,
                    skills_dir=skills_dir,
                    infer_fn=infer_fn,
                    chat_id=source_chat_id,
                )
            except Exception as e:
                logger.error(f"[ThinkAtRest] evolution attempt {attempt} error: {e}")
                break

            if not evo_result or not evo_result.found or not evo_result.installed:
                logger.info(f"[ThinkAtRest] evolution attempt {attempt}: no skill found/installed")
                break

            logger.info(f"[ThinkAtRest] evolution attempt {attempt}: installed {evo_result.installed}")

            if self._critique is None:
                break

            try:
                verdict = self._critique.assess(
                    thought=thought,
                    evolution_result=evo_result,
                    infer_fn=infer_fn,
                    attempt=attempt,
                )
            except Exception as e:
                logger.error(f"[ThinkAtRest] critique assess error: {e}")
                break

            logger.info(f"[ThinkAtRest] critique verdict: {verdict.verdict} (attempt {attempt})")

            thought["_critique_verdict"] = verdict.verdict
            thought["_critique_attempt"] = attempt
            thought["_critique_notes"] = verdict.notes
            # ADR-019: persist critique result into the audit log
            try:
                self._journal.append_critique_audit(thought)
            except Exception as _ce:
                logger.warning(f"[ThinkAtRest] critique audit write error: {_ce}")

            if verdict.verdict in ("RESOLVED", "PENDING_CONFIRM"):
                # PENDING_CONFIRM: confirmation buttons already sent to user — wait for callback
                # RESOLVED: critique disabled path — resolve immediately (no user confirmation)
                if verdict.verdict == "RESOLVED":
                    # Close the goal_discovery signal
                    try:
                        import core.pipelines.goal_discovery as goal_discovery
                        goal_discovery.resolve_pattern(thought_text)
                    except Exception:
                        pass
                    # Resolve any matching probe
                    if self._observer is not None:
                        try:
                            active = self._observer._probe_store.list_active()
                            for p in active:
                                if any(kw.lower() in thought_text.lower()
                                       for kw in p.get("subject", "").split()):
                                    self._observer._probe_store.resolve(p["id"])
                        except Exception:
                            pass
                    # Write routing hint
                    evidence_summary = thought.get("_observer_evidence", "")
                    resolved_at = datetime.now(timezone.utc).isoformat()
                    installed = evo_result.installed
                    if isinstance(installed, str):
                        installed = [installed]
                    for skill_name in (installed or []):
                        self._critique._write_routing_hint(
                            skill_name, thought, evidence_summary, resolved_at
                        )
                # PENDING_CONFIRM: routing hint + goal_discovery cleanup happen in callback
                break

            elif not verdict.should_retry:
                # Max attempts reached — escalate
                self._add_tracker_todo(f"[UNRESOLVED evolution] {thought_text[:120]}")
                try:
                    import services.channels.telegram_bot as _tb
                    chat_id = _tb.ALLOWED_CHAT_ID
                    if chat_id:
                        _tb.send_message(
                            chat_id,
                            f"⚠️ *Evolution unresolved after {attempt} attempt(s)*\n"
                            f"Gap: _{thought_text[:100]}_\n"
                            f"Last verdict: `{verdict.verdict}`\n"
                            f"Notes: {verdict.notes[:150]}",
                        )
                except Exception:
                    pass
                break
            else:
                critique_notes = verdict.notes

    def _maybe_trigger_evolution(self, thought: dict):
        """Trigger evolution if evolution_hook is available."""
        try:
            import core.evolution.evolution_hook as evolution_hook
            from core.pipelines import goal_discovery as _gd
            import yaml, os
            if hasattr(evolution_hook, "maybe_evolve"):
                _cfg_ovr = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
                cfg_path = (
                    os.path.abspath(os.path.expanduser(_cfg_ovr))
                    if _cfg_ovr
                    else os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
                )
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                skills_dir = os.path.expanduser(cfg.get("skills_dir", "~/.kernel-evolving/ecosystem"))
                # FIX #1: pass infer_fn so capability verifier runs in background path
                evolution_hook.maybe_evolve(thought["thought"], cfg, skills_dir, infer_fn=_gd._infer_fn)
                logger.info("[ThinkAtRest] triggered evolution for gap_reflection thought")
        except ImportError:
            pass  # evolution_hook not available in base kernel
        except Exception as e:
            logger.error(f"[ThinkAtRest] evolution trigger error: {e}")

    def _add_tracker_todo(self, thought_text: str):
        """Append to workspace notes/todos.md."""
        try:
            todos_path = str(NOTES_DIR / "todos.md")
            os.makedirs(os.path.dirname(todos_path), exist_ok=True)
            today = date.today().strftime("%Y-%m-%d")
            entry = f"- [ ] [{today}] (self_improvement) {thought_text[:200]}\n"
            with open(todos_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.info("[ThinkAtRest] appended self_improvement todo")
        except Exception as e:
            logger.error(f"[ThinkAtRest] todo append error: {e}")

    def _maybe_send_telegram(self, thought: dict):
        """Proactively send high-score thought to Telegram.
        Actionable thoughts (request for user action) get inline buttons.
        Regular thoughts are plain text.
        """
        today = date.today()
        if today != self._proactive_date:
            self._proactive_count_today = 0
            self._proactive_date = today
        if self._proactive_count_today >= self._proactive_max_per_day:
            return
        try:
            import services.channels.telegram_bot as _tb
            chat_id = _tb.ALLOWED_CHAT_ID
            if not chat_id:
                return
            thought_text = thought.get("thought", "")
            category = thought.get("category", "curiosity")
            score = thought.get("score", 0.0)
            cat_emoji = {
                "gap_reflection": "🧠",
                "self_improvement": "💪",
                "curiosity": "🔍",
                "retrospective": "📖",
            }.get(category, "💭")

            # Detect if this thought is actionable (requests something from the user)
            action_keywords = (
                "should i", "do you want", "would you like", "shall i",
                "remind", "todo", "add a task", "set a reminder",
                "notify", "let me know", "you should", "consider",
                "action needed", "decision", "approve", "confirm",
            )
            is_actionable = any(kw in thought_text.lower() for kw in action_keywords)

            text = (
                f"{cat_emoji} *Kernel is thinking…*\n\n"
                f"_{thought_text[:300]}_\n\n"
                f"_— {category} · score {score:.2f}_"
            )
            if is_actionable:
                # Send with action buttons only for actionable thoughts.
                # Use send_buttons (not send_message) — buttons are a separate
                # render path in telegram_bot.
                _tb.send_buttons(
                    chat_id, text,
                    [[{"text": "✅ Do it", "callback_data": f"thought_do_{score:.2f}"}, {"text": "❌ Skip", "callback_data": "thought_skip"}]]
                )
            else:
                _tb.send_message(chat_id, text)
            self._proactive_count_today += 1
            logger.info(f"[ThinkAtRest] proactive Telegram sent ({self._proactive_count_today}/{self._proactive_max_per_day}), actionable={is_actionable}")
        except Exception as e:
            logger.error(f"[ThinkAtRest] telegram send error: {e}")

    # ── Memory seeker ───────────────────────────────────────────────────

    def _run_memory_seeker(self):
        """Scan recent chat turns for user facts and append them to USER.md.
        Runs during every idle think cycle, before thought generation.
        USER.md is injected verbatim into every system prompt by context.py.
        """
        try:
            import core.memory.memory as _mem
            import core.inference.model_client as model_client
            if not model_client.is_server_running():
                return

            turns = _mem.load()
            # Only look at last 40 turns — enough context, not too expensive
            recent = [
                m for m in turns[-40:]
                if m.get("role") == "user"
                and str(m.get("content", "")).strip()
                # Skip user messages that look like tool invocations — those are
                # automation artifacts, not personal facts.
                and not str(m.get("content", "")).strip().startswith((
                    "run_skill(", "write_file(", "http_get(", "send_file(",
                    "exec_shell(", "read_file(", "web_search(", "search_skills(",
                    "list_routines()", "run_routine(", "recall_memory(",
                ))
            ]
            if not recent:
                return
            user_lines = "\n".join(
                f"User: {str(m['content'])[:300]}" for m in recent
            )
            prompt = (
                "You are a fact extractor. Read the following messages from a user and extract "
                "any personal facts they revealed about themselves. "
                "\n\n"
                "ALLOWED KEYS (only use these exact keys):\n"
                "- name: the user's name or handle\n"
                "- job: their profession or role\n"
                "- location: where they live or work\n"
                "- pets: any pets they mentioned\n"
                "- preferences: personal likes/dislikes\n"
                "\n"
                "RULES:\n"
                "- Values MUST be plain strings. Reject list/array/dict — those are noise.\n"
                "- If a value looks like a tool command, filename, or system path, omit it.\n"
                "- Only extract facts the user explicitly shared about THEMSELVES.\n"
                "- If nothing new is revealed, output {}.\n"
                "- Output ONLY a valid JSON object with string keys and string values.\n"
                "- No explanation, no markdown, no arrays.\n\n"
                + user_lines
            )
            raw = model_client.infer(
                [{"role": "user", "content": prompt}],
                max_new_tokens=8192,
                skip_identity_guard=True,  # self-contained fact-extractor role — avoid duplicate identity
            )
            if not raw:
                return
            m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if not m:
                return
            facts = json.loads(m.group(0))
            if not isinstance(facts, dict) or not facts:
                return

            # Sanitize: whitelist allowed keys and reject non-string values
            ALLOWED_KEYS = {"name", "job", "location", "pets", "preferences"}
            facts = {
                k: v for k, v in facts.items()
                if k in ALLOWED_KEYS
                and isinstance(v, str)
                and v.strip()
                and len(v) < 200
                # Reject values that look like hallucinated artifact strings
                and not v.startswith("[")
                and not v.startswith("run_skill")
                and not v.startswith("http")
                and not v.startswith("None")
                and not v.startswith("none")
                and not v.startswith("unknown")
                and not v.startswith("Unknown")
                and v.lower() != "n/a"
            }
            if not facts:
                return

            # Write new facts to USER.md (the context injection file)
            user_md_path = Path.home() / ".kernel-evolving" / "workspace" / "USER.md"
            user_md_path.parent.mkdir(parents=True, exist_ok=True)
            existing_text = user_md_path.read_text(encoding="utf-8") if user_md_path.exists() else ""

            # Consolidate: replace old Known facts section entirely, never append
            new_lines = []
            for k, v in facts.items():
                fact_line = f"- {k}: {v}"
                new_lines.append(fact_line)

            if new_lines:
                import re as _re
                cleaned = _re.sub(
                    r"## Known facts\n(?:- .*\n?)*",
                    "",
                    existing_text,
                )
                insert = "\n".join(new_lines) + "\n"
                updated = cleaned.rstrip() + "\n\n## Known facts\n" + insert
                user_md_path.write_text(updated, encoding="utf-8")
                logger.info(f"[MemorySeeker] replaced Known facts with: {new_lines}")

            # Reset user.json — don't sync hallucinated names
            user_json_path = Path.home() / ".kernel-evolving" / "workspace" / "user.json"
            try:
                if user_json_path.exists():
                    existing_json = json.loads(user_json_path.read_text())
                    # Only update name if user actually told us their name
                    if "name" in facts and isinstance(facts["name"], str) and len(facts["name"]) > 1:
                        existing_json["name"] = facts["name"]
                        user_json_path.write_text(json.dumps(existing_json, indent=2, ensure_ascii=False))
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"[MemorySeeker] error: {e}")

    def _run_identity_consolidator(self):
        """Once-per-day session scan: update AGENTS.md with behaviour patterns/lessons.
        Reads today's + yesterday's chat history, asks model what the agent should
        remember about itself, and appends non-duplicate notes to AGENTS.md.
        """
        try:
            # Rate-limit: max once per 20 hours
            stamp_path = Path.home() / ".kernel-evolving" / "workspace" / "runtime" / "last_identity_consolidation.txt"
            stamp_path.parent.mkdir(parents=True, exist_ok=True)
            import time as _time
            if stamp_path.exists():
                last = float(stamp_path.read_text().strip() or "0")
                if _time.time() - last < 72000:  # 20 hours
                    return

            import core.memory.memory as _mem
            import core.inference.model_client as model_client
            if not model_client.is_server_running():
                return

            turns = _mem.load()
            if len(turns) < 6:
                return  # not enough history to consolidate

            # Sample: last 60 turns (both user + assistant)
            sample = turns[-60:]
            convo = "\n".join(
                f"{m['role'].upper()}: {str(m.get('content',''))[:200]}"
                for m in sample
            )

            agents_path = Path.home() / ".kernel-evolving" / "workspace" / "AGENTS.md"
            existing_agents = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""

            prompt = (
                "You are reviewing a conversation between an AI agent (Kernel-Evo) and a user.\n"
                "Based on the conversation, identify:\n"
                "1. Behaviour patterns the agent should remember (e.g. tools it used well, mistakes to avoid)\n"
                "2. User preferences the agent should incorporate into its identity\n"
                "3. Any identity updates (e.g. the agent learned a new skill, the user gave it a new responsibility)\n\n"
                "Output ONLY a JSON array of strings. Each string is a short note (max 120 chars). "
                "Only include genuinely new insights. If nothing new, output [].\n\n"
                f"Conversation sample:\n{convo[:3000]}\n\n"
                f"Existing AGENTS.md excerpt (to avoid duplicates):\n{existing_agents[:500]}"
            )

            raw = model_client.infer(
                [{"role": "user", "content": prompt}],
                max_new_tokens=8192,
                skip_identity_guard=True,  # self-contained reviewer role — avoid duplicate identity
            )
            if not raw:
                return

            arr_match = re.search(r'\[.*?\]', raw, re.DOTALL)
            if not arr_match:
                return
            notes = json.loads(arr_match.group(0))
            if not isinstance(notes, list) or not notes:
                return

            new_notes = [n for n in notes if isinstance(n, str) and n.strip() and n.strip() not in existing_agents]
            if not new_notes:
                stamp_path.write_text(str(_time.time()))
                return

            # Append under ## Session learnings section
            insert = "\n".join(f"- {n.strip()}" for n in new_notes) + "\n"
            today = __import__('datetime').date.today().isoformat()
            section = f"\n## Session learnings ({today})\n" + insert

            # Defense in depth: never let the consolidator strip the identity template.
            # If the runtime AGENTS.md lost its core content, restore it from the
            # canonical template (preserving any learnings) before appending, so the
            # file always keeps the template + append-only session learnings.
            _sentinel_markers = ("## How a request flows", "## Who you are", "## Architecture reference")
            if not agents_path.exists() or not any(m in existing_agents for m in _sentinel_markers):
                try:
                    _tpl = Path(__file__).parent.parent.parent / "src" / "assets" / "agent-templates" / "AGENTS.md"
                    if _tpl.exists():
                        _template_text = _tpl.read_text(encoding="utf-8")
                        _learnings = ""
                        if "## Session learnings" in existing_agents:
                            _learnings = existing_agents[existing_agents.find("## Session learnings"):].rstrip() + "\n"
                        _restored = _template_text.rstrip() + "\n"
                        if _learnings:
                            _restored += "\n" + _learnings
                        agents_path.write_text(_restored, encoding="utf-8")
                        existing_agents = _restored
                        logger.info("[IdentityConsolidator] restored AGENTS.md template before append")
                except Exception as _rexc:
                    logger.debug(f"[IdentityConsolidator] template restore failed: {_rexc}")

            if agents_path.exists():
                agents_path.write_text(existing_agents.rstrip() + section, encoding="utf-8")
            else:
                agents_path.write_text(section, encoding="utf-8")

            stamp_path.write_text(str(_time.time()))
            logger.info(f"[IdentityConsolidator] appended {len(new_notes)} note(s) to AGENTS.md")

        except Exception as e:
            logger.debug(f"[IdentityConsolidator] error: {e}")


# ── Context provider helpers ──────────────────────────────────────────

    def _get_unused_skills(self) -> str:
        """Return skill/routine counts with sample names, plus installed private skills."""
        try:
            import core.agent as agent
            skills = getattr(agent, "_skills", []) or []
            routines = getattr(agent, "_routines", []) or []
            skill_names = [s["name"] for s in skills]
            count_str = f"{len(skills)} skills, {len(routines)} routines loaded"
            sample = ", ".join(skill_names[:10]) if skill_names else "none"
            # Also list installed private ecosystem skills so the generator
            # does NOT flag already-solved capabilities as gaps.
            private_dir = Path.home() / ".kernel-evolving" / "ecosystem" / "private" / "skills"
            installed = []
            if private_dir.exists():
                installed = [d.name for d in private_dir.iterdir() if d.is_dir()]
            installed_str = ", ".join(installed) if installed else "none"
            return (
                f"{count_str} — sample: {sample}\n"
                f"Already installed private skills (DO NOT flag these as gaps): {installed_str}"
            )
        except Exception:
            return "none"

    def _get_recent_gaps(self) -> str:
        """Return recent unresolved gaps (stub — overridden in kernel-evolving)."""
        return "none"

    def _get_last_thought(self) -> str:
        """Return the most recent journal entry text."""
        try:
            thoughts = self._journal.read_today()
            if thoughts:
                return thoughts[-1].get("thought", "none")
            # Try yesterday
            from datetime import timedelta
            yesterday = date.today() - timedelta(days=1)
            thoughts = self._journal.read_date(yesterday)
            if thoughts:
                return thoughts[-1].get("thought", "none")
        except Exception:
            pass
        return "none"

    def _get_recent_interactions(self) -> str:
        """Return recent user interactions — stub overridden in EvolvingThinkAtRest."""
        return "none"


class EvolvingThinkAtRest(ThinkAtRest):
    """
    ThinkAtRest extension for kernel-evolving.
    Wires ThoughtGenerator to read from evolution_log.db for richer context.
    """

    def _get_recent_interactions(self) -> str:
        """Return last 6 user messages from SQLite chat history as grounding context."""
        try:
            import core.memory.memory as _mem
            turns = _mem.history(limit=6)
            user_turns = [t for t in turns if t.get("role") == "user"]
            if not user_turns:
                return "none"
            lines = [f"- {t['content'][:120]}" for t in user_turns[-6:]]
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[EvolvingThinkAtRest] get_recent_interactions error: {e}")
            return "none"

    def _get_recent_gaps(self) -> str:
        """
        ADR-020: Return unresolved failed user requests as concrete task anchors.
        Falls back to evolution_log gaps if no failed_requests exist.
        Stores pending request ids on self for thought tagging.
        """
        try:
            import database.agent.failed_requests as _fr
            pending = _fr.get_unresolved(limit=3)
            if pending:
                self._pending_failed_requests = pending  # stash for _tag_thought_with_source
                lines = []
                for req in pending:
                    lines.append(f"[id:{req['id']} type:{req['failure_type']}] {req['user_message'][:120]}")
                return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[EvolvingThinkAtRest] failed_requests query error: {e}")
        # Fallback: evolution_log gaps (old behaviour)
        try:
            from core.evolution.evolution_log import EvolutionLog
            elog = EvolutionLog()
            gaps = elog.get_gaps()
            recent = [g.get("gap", "") for g in gaps[:3] if g.get("gap")]
            if not recent:
                return "none"
            return "; ".join(recent)
        except Exception as e:
            logger.debug(f"[EvolvingThinkAtRest] get_gaps error: {e}")
            return "none"

    def _get_recent_evolution_context(self) -> str:
        """Return last 5 evolution cycles summary for prompt enrichment."""
        try:
            from core.evolution.evolution_log import EvolutionLog
            elog = EvolutionLog()
            history = elog.get_history(limit=5)
            if not history:
                return "none"
            lines = []
            for h in history:
                ts = h.get("ts", "?")[:10]
                task = h.get("task", "?")[:80]
                found = "✓" if h.get("found") else "✗"
                lines.append(f"[{ts}] {found} {task}")
            return "\n".join(lines)
        except Exception:
            return "none"

    def _run_think_cycle(self):
        """Restructured think cycle (2026-06-18).

        Signal-driven, not generation-driven:
          Phase 1: gap_reflection  — directly evolve from real failed_requests.
          Phase 2: self_improvement — real performance metrics.
          Phase 3: curiosity        — real exploration data → LLM insight.

        NEVER generates synthetic gaps from a template. If there's nothing
        real to act on, the system stays quiet.
        """
        self._last_think_time = time.monotonic()
        logger.info("[EvolvingThinkAtRest] running think cycle")

        # ADR-019: expire old probes (EvolvingThinkAtRest)
        if self._observer is not None:
            expired = self._observer.expire_probes()
            if expired:
                logger.info(f"[EvolvingThinkAtRest] expired {expired} old probe(s)")
        self._run_memory_seeker()
        self._run_identity_consolidator()

        # ── Phase 1: gap_reflection — real failed requests → direct evolution ──
        handled = False
        try:
            import database.agent.failed_requests as _fr
            pending = _fr.get_unresolved(limit=3)
            if pending:
                self._pending_failed_requests = pending
                # Only count the cycle as "handled" if at least one request was
                # ACTUALLY evolved. A request skipped by the recency/yield gate
                # (e.g. stale backlog) must not suppress curiosity exploration —
                # otherwise a permanent stale backlog starves Phase 3 forever.
                evolved_any = False
                for req in pending:
                    if self._evolve_from_failed_request(req):
                        evolved_any = True
                handled = evolved_any
                logger.info(
                    f"[EvolvingThinkAtRest] processed {len(pending)} failed request(s) "
                    f"(evolved={evolved_any})"
                )
        except Exception as e:
            logger.warning(f"[EvolvingThinkAtRest] failed_requests error: {e}")

        # ── Phase 2: self_improvement — real performance signals ──
        perf_signals = self._gather_performance_signals()
        if perf_signals:
            for sig in perf_signals:
                thought = {
                    "thought": sig["message"],
                    "category": "self_improvement",
                    "score": sig["severity"],
                    "_observer_verdict": "EVIDENCED",
                    "_observer_path": "journal_only",
                    "_observer_evidence": f"perf:{sig['metric']}",
                }
                self._journal.write(thought)
                logger.info(f"[EvolvingThinkAtRest] perf signal: {sig['metric']}")
            handled = True

        # ── Only run curiosity if nothing urgent was found ──
        if handled:
            logger.info("[EvolvingThinkAtRest] real signals processed — skipping curiosity exploration")
            return

        # Phase 3: curiosity — outward exploration (only model-assisted when data changed)
        super()._run_think_cycle()

    def _evolve_from_failed_request(self, req: dict) -> bool:
        """ADR-020: directly trigger evolution for a concrete failed user request.

        Returns True if evolution actually ran for this request, False if it was
        skipped (evolution disabled/paused, stale beyond the recency window, or
        yielding to an active chat request). The caller uses this so a cycle where
        every request was merely skipped does NOT count as "handled" — otherwise
        stale backlog would permanently suppress curiosity exploration.

        Recency gate (2026-08-26): enabling EVOLUTION_ENABLED caused every stale
        unresolved failed request in the backlog to be evolved during idle,
        which starved active chat. Only evolve requests that failed within the
        configured recency window (default 24h). Use the request's `ts` (ISO UTC).
        """
        try:
            import core.evolution.evolution_hook
            from core.evolution.evolution_hook import EVOLUTION_ENABLED
            if not EVOLUTION_ENABLED:
                return False
            import core.evolution.evolution_state as _evo_state
            if not _evo_state.should_evolve():
                logger.info(f"[EvolvingThinkAtRest] evolution paused/stopped — skipping failed_request id={req['id']}")
                return False
        except ImportError:
            return False

        # Recency gate: skip stale failed requests so idle evolution doesn't
        # replay old backlog and preempt active chat.
        try:
            recency_hours = float(self._cfg.get("failed_request_recency_hours", 24))
            ts_raw = req.get("ts") or ""
            if ts_raw:
                ts_naive = ts_raw.split("+")[0].split("Z")[0]
                req_ts = datetime.fromisoformat(ts_naive).replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - req_ts).total_seconds() / 3600.0
                if age_h > recency_hours:
                    logger.info(
                        f"[EvolvingThinkAtRest] skipping failed_request id={req.get('id')} "
                        f"(age {age_h:.1f}h > {recency_hours}h recency window)"
                    )
                    return False
        except Exception as _e:
            logger.debug(f"[EvolvingThinkAtRest] recency check skipped ({_e})")

        # Yield gate (2026-08-26): if a user/model request is now in-flight or
        # recently finished, do NOT run evolution — it would compete for the same
        # inference slot and starve active chat. Reuse the same activity check the
        # idle detector uses. The failed request stays unresolved and will be
        # reconsidered on a later idle cycle.
        if self._model_is_active():
            logger.info(
                f"[EvolvingThinkAtRest] model active — yielding evolution for "
                f"failed_request id={req.get('id')} to active chat"
            )
            return False

        request_id = req["id"]
        user_message = req["user_message"]
        logger.info(f"[EvolvingThinkAtRest] direct evolution for failed_request id={request_id}: {user_message[:80]!r}")

        # Build a synthetic thought anchored to this request
        thought = {
            "thought": f"User asked: {user_message[:200]}",
            "category": "gap_reflection",
            "score": 1.0,
            "_source_request_id": request_id,
            "_source_user_message": user_message,
            "_source_failure_type": req.get("failure_type", "unknown"),
            "_source_chat_id": req.get("chat_id"),
            "_observer_evidence": f"user_request:{request_id}",
        }

        # Run full evolution + critique loop
        self._run_evolution_with_critique(thought)

        # ADR-020: always increment retry_count after each attempt to prevent
        # the same failed_request from being re-queued on every idle cycle.
        # Once retry_count reaches 3, mark resolved (abandoned) so it stops.
        try:
            import database.agent.failed_requests as _fr
            new_count = _fr.increment_retry(request_id)
            if new_count >= 3:
                _fr.mark_resolved(request_id, reward=0)
                logger.info(
                    f"[EvolvingThinkAtRest] failed_request id={request_id} abandoned "
                    f"after {new_count} evolution attempts — marked resolved"
                )
        except Exception as _re:
            logger.warning(f"[EvolvingThinkAtRest] retry_count update error: {_re}")

        return True

    def _on_thought_accepted(self, thought: dict):
        """Override: gap_reflection no longer reaches here — handled directly
        via _evolve_from_failed_request(). Only curiosity thoughts flow through."""
        super()._on_thought_accepted(thought)
