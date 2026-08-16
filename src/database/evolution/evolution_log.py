"""
database/evolution/evolution_log.py — EvolutionLogRepository.

Single owner of evolution.db; extracts ALL SQL from src/evolution_log.py.
Replaces the inline ALTER TABLE migration hacks (ADR-006/007/010) with
proper migration files from database/migrations/evolution/.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from database.base import BaseRepository

logger = logging.getLogger(__name__)

# Path to the migration directory for this domain
_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations" / "evolution"


class EvolutionLogRepository(BaseRepository):
    """SQLite repository for evolution_events and synthesis_trajectories."""

    def __init__(self, db_path: Path | str) -> None:
        super().__init__(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Run numbered migration files instead of inline ALTER TABLE hacks."""
        self.migrate(_MIGRATIONS_DIR)

    # ── evolution_events ──────────────────────────────────────────────────────

    def record(self, task: str, result, verification_result=None,
               verification_reasoning=None) -> int:
        """Persist an EvolutionResult. Returns the row id."""
        ts = datetime.now(timezone.utc).isoformat()
        resolved = 1 if result.found and result.retry else 0
        vr = verification_result or getattr(result, "verification_result", None)
        vreason = verification_reasoning or getattr(result, "verification_reasoning", None)
        recs = getattr(result, "recommendations", None) or []
        recs_json = json.dumps(recs) if recs else None
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO evolution_events
                    (ts, task, found, installed, confidence, retry, escalated, provider_used, gap, resolved,
                     verification_result, verification_reasoning, recommendations, event_type, entity_name, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, task, int(result.found), json.dumps(result.installed),
                    result.confidence, int(result.retry), int(result.escalated),
                    result.provider_used, result.gap, resolved,
                    vr, vreason, recs_json, "evolution", None, None,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def record_activity(self, *, event_type: str, task: str, entity_name: str = None,
                        provider_used: str = "runtime", found: bool = True,
                        confidence: str = "EXEC", gap: str = "",
                        meta: dict | None = None) -> int:
        """Persist non-synthesis runtime activity."""
        ts = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(meta or {})
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO evolution_events
                    (ts, task, found, installed, confidence, retry, escalated, provider_used, gap, resolved,
                     verification_result, verification_reasoning, recommendations, event_type, entity_name, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, task, int(found), json.dumps([]), confidence, int(found), 0,
                    provider_used, gap, int(found), None, None, None,
                    event_type, entity_name, meta_json,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_gaps(self) -> list[dict]:
        """Return unresolved gap entries, deduplicated by semantic content."""
        import re as _re
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_events WHERE resolved = 0 ORDER BY ts DESC"
            ).fetchall()
        result = []
        seen_keys: set[str] = set()
        for r in rows:
            d = dict(r)
            if isinstance(d.get("installed"), str):
                try:
                    d["installed"] = json.loads(d["installed"])
                except Exception:
                    d["installed"] = []
            if "ts" in d and "timestamp" not in d:
                d["timestamp"] = d["ts"]
            gap_text = (d.get("gap") or d.get("task") or "").lower()[:80]
            dedup_key = _re.sub(r"\b(sim\d+|\d{8}|\d{14}|20\d{6}[_\d]*)\b", "", gap_text).strip()
            if dedup_key and dedup_key in seen_keys:
                continue
            if dedup_key:
                seen_keys.add(dedup_key)
            result.append(d)
        return result

    def resolve_gap(self, gap_id: int) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE evolution_events SET resolved = 1 WHERE id = ?", (gap_id,))
            conn.commit()

    def reconcile_gaps_with_skills(self, skills: list[dict]) -> int:
        """Mark gaps resolved when a matching skill exists. Returns count resolved."""
        skill_fragments: list[str] = []
        for s in skills:
            name = (s.get("name") or "").lower()
            if name:
                skill_fragments.extend(name.split("-"))
                skill_fragments.append(name)
        skill_fragments = [f for f in skill_fragments if len(f) > 3]

        with self.connection() as conn:
            open_gaps = conn.execute(
                "SELECT id, gap, task FROM evolution_events WHERE resolved = 0"
            ).fetchall()

        resolved_count = 0
        for row in open_gaps:
            gap_text = ((row["gap"] or "") + " " + (row["task"] or "")).lower()
            for fragment in skill_fragments:
                if fragment in gap_text:
                    self.resolve_gap(row["id"])
                    resolved_count += 1
                    break
        if resolved_count:
            logger.info(f"[EvolutionLogRepo] reconcile_gaps: resolved {resolved_count}")
        return resolved_count

    def get_history(self, limit: int = 50) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("installed"), str):
                try:
                    d["installed"] = json.loads(d["installed"])
                except Exception:
                    d["installed"] = []
            if isinstance(d.get("meta"), str):
                try:
                    d["meta"] = json.loads(d["meta"])
                except Exception:
                    d["meta"] = {}
            if "ts" in d and "timestamp" not in d:
                d["timestamp"] = d["ts"]
            result.append(d)
        return result

    # ── synthesis_trajectories ─────────────────────────────────────────────

    def log_synthesis_trajectory(self, task: str, gap: str, prompt: str,
                                  output_skill: str, output_script: str,
                                  validation: str, provider: str, skill_name: str,
                                  critic_score: float = None, critic_issues: list = None,
                                  suggested_name: str = None) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        critic_issues_json = json.dumps(critic_issues) if critic_issues is not None else None
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO synthesis_trajectories
                   (ts, task, gap, prompt, output_skill, output_script, validation, provider, skill_name,
                    critic_score, critic_issues, suggested_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, task, gap, prompt, output_skill, output_script, validation, provider, skill_name,
                 critic_score, critic_issues_json, suggested_name),
            )
            conn.commit()

    def get_trajectories(self, limit: int = 100) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id,ts,task,gap,output_skill,validation,provider,skill_name,"
                "critic_score,critic_issues,suggested_name FROM synthesis_trajectories "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("critic_issues"), str):
                try:
                    d["critic_issues"] = json.loads(d["critic_issues"])
                except Exception:
                    pass
            result.append(d)
        return result
