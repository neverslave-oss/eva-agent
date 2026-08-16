"""
observer_layer.py — ADR-019 ObserverLayer: evidence gate + anti-seed injection.

Sits between ThoughtEvaluator and _on_thought_accepted(). Classifies thoughts
as EVIDENCED / SPECULATIVE / CURIOSITY based on evidence from DBs, and
determines the appropriate evolution path.
"""
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from runtime_paths import CHAT_HISTORY_DB, EVOLUTION_DB, PROMOTED_SIGNALS_DB, THOUGHTS_DIR

logger = logging.getLogger(__name__)

# Stopwords to skip when extracting keywords
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "have", "from",
    "are", "was", "were", "been", "will", "would", "could", "should",
    "more", "also", "when", "what", "about", "which", "their", "into",
    "than", "then", "them", "they", "there", "some", "such", "just",
    "very", "much", "even", "like", "well", "only", "over", "back",
    "after", "being", "each", "other", "your", "how", "can", "not",
    "all", "its", "but", "our", "out", "has", "had", "may",
})


@dataclass
class ObserverVerdict:
    classification: str  # "EVIDENCED" | "SPECULATIVE" | "CURIOSITY"
    evolution_path: str  # "full_evolution" | "goal_signal" | "probe" | "journal_only" | "idea"
    evidence_summary: str = ""
    evidence_count: int = 0


class ObserverLayer:
    """
    Evidence gate: examines thought context against DB evidence before accepting.
    Anti-seed: provides recent thought summaries to prevent repetition.
    """

    def __init__(self, config: dict):
        thinking_cfg = config.get("thinking", {})
        obs_cfg = thinking_cfg.get("observer", {})

        # DB paths
        self._chat_history_db = obs_cfg.get(
            "chat_history_db_path",
            str(CHAT_HISTORY_DB),
        )
        self._evolution_db = obs_cfg.get(
            "evolution_db_path",
            str(EVOLUTION_DB),
        )
        self._promoted_signals_db = obs_cfg.get(
            "promoted_signals_db",
            str(PROMOTED_SIGNALS_DB),
        )

        # Journal dir for anti-seed reads
        self._journal_dir = os.path.expanduser(
            thinking_cfg.get("journal_dir", str(THOUGHTS_DIR))
        )

        # Config knobs
        self._lookback_turns = int(obs_cfg.get("evidence_lookback_turns", 100))
        self._lookback_days = int(obs_cfg.get("evidence_lookback_days", 7))

        # ADR-019 Phase 2: probe store
        from core.evolution.probe_store import ProbeStore
        _probe_ttl = int(obs_cfg.get("probe_ttl_days", 7))
        self._probe_store = ProbeStore(ttl_days=_probe_ttl)

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, thought: dict) -> ObserverVerdict:
        """
        Evaluate a thought and return an ObserverVerdict.

        ADR-020 fast path:
          If thought has _source_request_id → immediately EVIDENCED (a real user failure anchors it).

        ADR-019 fallback classification:
          EVIDENCED   — ≥1 NO-verification in evolution_events, OR ≥2 hits across sources
          SPECULATIVE — 0-1 hits, category in (gap_reflection, self_improvement)
          CURIOSITY   — category == "curiosity" (always valid, no evidence gate)
        """
        category = thought.get("category", "curiosity")

        # ADR-020: direct evidence from a real failed user request — skip all heuristics
        if thought.get("_source_request_id"):
            return ObserverVerdict(
                classification="EVIDENCED",
                evolution_path="full_evolution",
                evidence_summary=f"user_request:{thought['_source_request_id']} type:{thought.get('_source_failure_type', '?')}",
                evidence_count=1,
            )

        # CURIOSITY is always valid — no evidence gate
        if category == "curiosity":
            return ObserverVerdict(
                classification="CURIOSITY",
                evolution_path="idea",
                evidence_summary="curiosity — no gate",
                evidence_count=0,
            )

        text = thought.get("thought", "")
        keywords = self._extract_keywords(text)

        if not keywords:
            return ObserverVerdict(
                classification="SPECULATIVE",
                evolution_path=self._speculative_path(category),
                evidence_summary="no keywords extracted",
                evidence_count=0,
            )

        # Query evidence sources
        hits_chat = self._search_chat_history(keywords)
        hits_evo, has_no_verification = self._search_evolution_events(keywords)
        hits_signals = self._search_promoted_signals(keywords)

        total_hits = hits_chat + hits_evo + hits_signals
        evidence_parts = []
        if hits_chat:
            evidence_parts.append(f"chat:{hits_chat}")
        if hits_evo:
            evidence_parts.append(f"evolution:{hits_evo}")
        if hits_signals:
            evidence_parts.append(f"signals:{hits_signals}")
        evidence_summary = ", ".join(evidence_parts) if evidence_parts else "none"

        # Classify
        if has_no_verification or total_hits >= 2:
            classification = "EVIDENCED"
            evolution_path = self._evidenced_path(category)
        else:
            classification = "SPECULATIVE"
            evolution_path = self._speculative_path(category)

        verdict = ObserverVerdict(
            classification=classification,
            evolution_path=evolution_path,
            evidence_summary=evidence_summary,
            evidence_count=total_hits,
        )

        # ADR-019 Phase 2: persist probe record
        if verdict.evolution_path == "probe":
            try:
                self._probe_store.add(thought, subject=" ".join(keywords))
                logger.info(f"[ObserverLayer] probe recorded for: {keywords}")
            except Exception as e:
                logger.warning(f"[ObserverLayer] probe_store.add error: {e}")

        return verdict

    def get_anti_seed_context(self, n: int = 10) -> list:
        """
        Return the last n thought summaries from the journal directory.
        Each entry is ≤80 chars, for injection as anti-seed into ThoughtGenerator.
        """
        journal_path = Path(self._journal_dir)
        if not journal_path.exists():
            return []

        md_files = sorted(journal_path.glob("*.md"), reverse=True)
        results = []

        for md_file in md_files:
            if len(results) >= n:
                break
            try:
                summary = self._extract_first_thought(md_file)
                if summary:
                    results.append(summary[:80])
            except Exception as e:
                logger.debug(f"[ObserverLayer] anti-seed read error {md_file}: {e}")

        return results[:n]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list:
        """Extract 2–3 meaningful keywords from thought text."""
        words = text.lower().split()
        keywords = []
        seen = set()
        for word in words:
            # Strip punctuation
            clean = word.strip(".,!?;:\"'()[]{}").strip()
            if (
                len(clean) > 4
                and clean not in _STOPWORDS
                and clean not in seen
                and clean.isalpha()
            ):
                keywords.append(clean)
                seen.add(clean)
            if len(keywords) >= 3:
                break
        return keywords

    def _search_chat_history(self, keywords: list) -> int:
        """Count distinct matching messages in chat_history_evolving.db → messages table."""
        if not os.path.exists(self._chat_history_db):
            return 0
        try:
            conn = sqlite3.connect(self._chat_history_db)
            try:
                cursor = conn.cursor()
                # Build OR clause for all keywords
                if not keywords:
                    return 0
                conditions = " OR ".join(["content LIKE ?" for _ in keywords])
                params = [f"%{kw}%" for kw in keywords]
                cursor.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                        SELECT id FROM messages
                        WHERE role = 'user'
                        AND ({conditions})
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    """,
                    params + [self._lookback_turns],
                )
                row = cursor.fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[ObserverLayer] chat_history query error: {e}")
            return 0

    def _search_evolution_events(self, keywords: list) -> tuple:
        """
        Count distinct matches in evolution.db → evolution_events.
        Returns (hit_count, has_no_verification).
        """
        if not os.path.exists(self._evolution_db):
            return 0, False
        try:
            conn = sqlite3.connect(self._evolution_db)
            try:
                cursor = conn.cursor()
                if not keywords:
                    return 0, False

                conditions = " OR ".join(
                    ["gap LIKE ? OR task LIKE ?" for _ in keywords]
                )
                params = []
                for kw in keywords:
                    params.extend([f"%{kw}%", f"%{kw}%"])

                cursor.execute(
                    f"""
                    SELECT id, verification_result FROM evolution_events
                    WHERE ({conditions})
                    AND datetime(created_at) >= datetime('now', ?)
                    """,
                    params + [f"-{self._lookback_days} days"],
                )
                rows = cursor.fetchall()
                total_hits = len(rows)
                has_no_verification = any(row[1] == "NO" for row in rows)

                return total_hits, has_no_verification
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[ObserverLayer] evolution_events query error: {e}")
            return 0, False

    def _search_promoted_signals(self, keywords: list) -> int:
        """Count distinct matching rows in promoted_signals.db → promoted_signals."""
        if not os.path.exists(self._promoted_signals_db):
            return 0
        try:
            conn = sqlite3.connect(self._promoted_signals_db)
            try:
                cursor = conn.cursor()
                if not keywords:
                    return 0
                conditions = " OR ".join(["text LIKE ?" for _ in keywords])
                params = [f"%{kw}%" for kw in keywords]
                cursor.execute(
                    f"SELECT COUNT(*) FROM promoted_signals WHERE {conditions}",
                    params,
                )
                row = cursor.fetchone()
                return row[0] if row else 0
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[ObserverLayer] promoted_signals query error: {e}")
            return 0

    def _evidenced_path(self, category: str) -> str:
        """Map EVIDENCED + category → evolution_path."""
        if category == "gap_reflection":
            return "full_evolution"
        elif category == "self_improvement":
            return "goal_signal"
        else:
            # retrospective or unknown — just journal it
            return "journal_only"

    def _speculative_path(self, category: str) -> str:
        """Map SPECULATIVE + category → evolution_path."""
        if category == "gap_reflection":
            return "probe"
        elif category == "self_improvement":
            return "journal_only"
        else:
            return "journal_only"

    def _extract_first_thought(self, md_file: Path) -> Optional[str]:
        """Parse a journal .md file and extract the first thought text line."""
        import re

        text = md_file.read_text(encoding="utf-8", errors="ignore")
        # Look for section headers like: ## HH:MM — category
        # Then grab the first non-empty line after the header
        lines = text.splitlines()
        in_section = False
        for line in lines:
            if re.match(r"^## \d{2}:\d{2}", line):
                in_section = True
                continue
            if in_section and line.strip() and not line.startswith("#"):
                return line.strip()
        return None

    # ── ADR-019 Phase 2: probe interaction methods ────────────────────────────

    def check_interaction_for_probes(self, user_text: str) -> list[dict]:
        """
        Called on every incoming user message.
        Returns list of probes triggered by this message.
        """
        triggered = []
        try:
            matches = self._probe_store.match(user_text)
            for probe in matches:
                self._probe_store.trigger(probe["id"], triggered_by=user_text[:200])
                triggered.append(probe)
                logger.info(
                    f"[ObserverLayer] probe triggered: id={probe['id']} subject={probe['subject']}"
                )
        except Exception as e:
            logger.warning(f"[ObserverLayer] probe match error: {e}")
        return triggered

    def expire_probes(self) -> int:
        """Run TTL expiry sweep.  Returns count deleted."""
        try:
            return self._probe_store.expire_old()
        except Exception as e:
            logger.warning(f"[ObserverLayer] probe expiry error: {e}")
            return 0
