"""
evolution_log.py — ADR-004 persistent SQLite log of all evolution events.

This module is a thin shim over database.evolution.EvolutionLogRepository.
All SQL logic (including migration-based schema setup) lives in
database/evolution/evolution_log.py.
The EvolutionLog class interface is preserved for backward compatibility.
"""
import os
import json
import logging
import subprocess
from pathlib import Path
from runtime_paths import EVOLUTION_DB, ARTIFACTS_TRAJECTORIES_DIR
from database.evolution import EvolutionLogRepository

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = str(EVOLUTION_DB)


class EvolutionLog:
    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.expanduser(db_path or _DEFAULT_DB_PATH)
        self._repo = EvolutionLogRepository(self.db_path)

    # ── delegation ────────────────────────────────────────────────────────────

    def record(self, task: str, result, verification_result=None,
               verification_reasoning=None) -> int:
        return self._repo.record(task, result, verification_result, verification_reasoning)

    def record_activity(self, *, event_type: str, task: str, entity_name: str = None,
                        provider_used: str = "runtime", found: bool = True,
                        confidence: str = "EXEC", gap: str = "",
                        meta: dict | None = None) -> int:
        return self._repo.record_activity(
            event_type=event_type, task=task, entity_name=entity_name,
            provider_used=provider_used, found=found, confidence=confidence,
            gap=gap, meta=meta,
        )

    def get_gaps(self) -> list[dict]:
        return self._repo.get_gaps()

    def resolve_gap(self, gap_id: int) -> None:
        self._repo.resolve_gap(gap_id)

    def reconcile_gaps_with_skills(self, skills: list[dict]) -> int:
        return self._repo.reconcile_gaps_with_skills(skills)

    def get_history(self, limit: int = 50) -> list[dict]:
        return self._repo.get_history(limit)

    def log_synthesis_trajectory(self, task: str, gap: str, prompt: str,
                                  output_skill: str, output_script: str,
                                  validation: str, provider: str, skill_name: str,
                                  critic_score: float = None, critic_issues: list = None,
                                  suggested_name: str = None):
        self._repo.log_synthesis_trajectory(
            task, gap, prompt, output_skill, output_script, validation, provider, skill_name,
            critic_score=critic_score, critic_issues=critic_issues, suggested_name=suggested_name,
        )
        # Also write JSONL to trajectory file for easy export
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        traj_dir = ARTIFACTS_TRAJECTORIES_DIR
        traj_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "type": "synthesis",
            "ts": ts,
            "task": task,
            "gap": gap,
            "prompt": prompt,
            "output": {"skill_md": output_skill, "script": output_script},
            "validation": validation,
            "provider": provider,
            "skill_name": skill_name,
            "critic_score": critic_score,
            "critic_issues": critic_issues,
            "suggested_name": suggested_name,
        }
        with open(traj_dir / "synthesis.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trajectories(self, limit: int = 100) -> list[dict]:
        return self._repo.get_trajectories(limit)

    def create_tracker_todos_for_gaps(self):
        gaps = self.get_gaps()
        tracker = os.path.expanduser(
            "~/.openclaw/workspace/repositories/open-workspace-tracker/scripts/tracker.py"
        )
        if not os.path.exists(tracker):
            return
        for gap in gaps:
            description = (gap.get("gap") or gap.get("task") or "Unknown gap")[:120]
            try:
                subprocess.run(
                    ["python3", tracker, "todo", "add", f"[ADR-004 gap] {description}",
                     "--project", "kernel-evolving", "--category", "research"],
                    capture_output=True, timeout=10,
                )
            except Exception:
                continue

    def record_synthesis_rejection(self, task: str, gap: str, provider: str) -> None:
        import time, json as _json
        record = {
            "ts": time.time(),
            "event": "tier2_rejected",
            "task": task,
            "gap": gap,
            "provider": provider,
            "reward": 0,
        }
        try:
            log_path = os.path.expanduser("~/.kernel-evolving/evolution_log.jsonl")
            with open(log_path, "a") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:
            pass
