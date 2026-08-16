"""
trajectory_collector.py — ADR-013: Automatic trajectory collection.

Records infer_with_tools sessions where:
  - critic score >= trajectory_min_critic_score (default 0.7)
  - at least one artifact written to disk (when trajectory_require_artifact: true)

Exports JSONL compatible with HuggingFace TRL SFTTrainer / DPO format.
"""

import os
import json
import sqlite3
import logging
import datetime
from contextlib import contextmanager
from runtime_paths import EVOLUTION_DB, ARTIFACTS_TRAJECTORIES_DIR
from database.evolution import EvolutionLogRepository as _EvoRepo

logger = logging.getLogger(__name__)

_DB_PATH = str(EVOLUTION_DB)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    task            TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model_name      TEXT,
    call_type       TEXT NOT NULL,
    tool_calls      TEXT NOT NULL,
    final_reply     TEXT NOT NULL,
    artifacts       TEXT,
    critic_score    REAL,
    critic_verdict  TEXT,
    token_count     INTEGER,
    elapsed_s       REAL
);
"""


class TrajectoryCollector:
    def __init__(self, config: dict):
        providers_cfg = config.get("providers", {})
        self._enabled = providers_cfg.get("collect_trajectories", False)
        self._min_score = providers_cfg.get("trajectory_min_critic_score", 0.7)
        self._require_artifact = providers_cfg.get("trajectory_require_artifact", True)

        # Use EvolutionLogRepository to ensure schema (migrations) are applied
        # Uses _DB_PATH which tests may patch for isolation
        _EvoRepo(_DB_PATH)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def should_record(self, critic_score: float | None, artifacts: list | None, tool_calls: list | None = None) -> bool:
        """Check if this trajectory meets collection thresholds."""
        if not self._enabled:
            return False
        # Must have at least one tool call — plain text replies are not trajectories
        if not tool_calls:
            return False
        if critic_score is not None and critic_score < self._min_score:
            return False
        if self._require_artifact:
            if not artifacts:
                return False
        return True

    def record(self, task: str, provider: str, model_name: str | None, call_type: str,
               tool_calls: list, final_reply: str, artifacts: list | None = None,
               critic_score: float | None = None, critic_verdict: str | None = None,
               token_count: int | None = None, elapsed_s: float | None = None):
        """Insert one trajectory row."""
        ts = datetime.datetime.utcnow().isoformat()
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT INTO task_trajectories
                       (ts, task, provider, model_name, call_type, tool_calls, final_reply,
                        artifacts, critic_score, critic_verdict, token_count, elapsed_s)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ts, task, provider, model_name, call_type,
                        json.dumps(tool_calls),
                        final_reply,
                        json.dumps(artifacts) if artifacts else None,
                        critic_score,
                        critic_verdict,
                        token_count,
                        elapsed_s,
                    )
                )
            logger.info(f"[trajectory] recorded task={task[:50]!r} provider={provider} score={critic_score}")
        except Exception as e:
            logger.warning(f"[trajectory] record failed: {e}")

    def export_jsonl(self, path: str | None = None, min_score: float = 0.7,
                     call_type: str | None = None, limit: int = 1000):
        """
        Query DB and write HF-compatible JSONL.
        Returns (path, count).
        """
        export_dir = str(ARTIFACTS_TRAJECTORIES_DIR)
        os.makedirs(export_dir, exist_ok=True)

        if path is None:
            ts_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(export_dir, f"export_{ts_str}.jsonl")

        query = "SELECT * FROM task_trajectories WHERE (critic_score IS NULL OR critic_score >= ?)"
        params: list = [min_score]
        if call_type:
            query += " AND call_type = ?"
            params.append(call_type)
        query += f" ORDER BY id DESC LIMIT {limit}"

        count = 0
        try:
            with self._conn() as conn:
                rows = conn.execute(query, params).fetchall()

            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    row_dict = dict(row)
                    tool_calls_raw = row_dict.get("tool_calls") or "[]"
                    try:
                        tool_calls = json.loads(tool_calls_raw)
                    except Exception:
                        tool_calls = []

                    artifacts_raw = row_dict.get("artifacts")
                    artifacts = json.loads(artifacts_raw) if artifacts_raw else []

                    # Reconstruct messages array for HF format
                    messages = [{"role": "user", "content": row_dict["task"]}]
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            messages.append({
                                "role": "assistant",
                                "tool_calls": [{
                                    "type": "function",
                                    "function": {
                                        "name": tc.get("tool", tc.get("name", "")),
                                        "arguments": json.dumps(tc.get("args", tc.get("arguments", {}))),
                                    }
                                }]
                            })
                            messages.append({
                                "role": "tool",
                                "name": tc.get("tool", tc.get("name", "")),
                                "content": str(tc.get("result", "")),
                            })
                    messages.append({"role": "assistant", "content": row_dict["final_reply"]})

                    record = {
                        "id": str(row_dict["id"]),
                        "ts": row_dict["ts"],
                        "task": row_dict["task"],
                        "provider": row_dict["provider"],
                        "model": row_dict.get("model_name"),
                        "call_type": row_dict["call_type"],
                        "messages": messages,
                        "artifacts": artifacts,
                        "critic_score": row_dict.get("critic_score"),
                        "verdict": row_dict.get("critic_verdict"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

            logger.info(f"[trajectory] exported {count} records to {path}")
        except Exception as e:
            logger.warning(f"[trajectory] export_jsonl failed: {e}")

        return path, count

    def recent(self, limit: int = 20, call_type: str | None = None) -> list[dict]:
        """Return recent trajectories with parsed tool call payloads."""
        query = "SELECT * FROM task_trajectories"
        params: list = []
        if call_type:
            query += " WHERE call_type = ?"
            params.append(call_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        try:
            with self._conn() as conn:
                rows = conn.execute(query, params).fetchall()
            items: list[dict] = []
            for row in rows:
                row_dict = dict(row)
                try:
                    row_dict["tool_calls"] = json.loads(row_dict.get("tool_calls") or "[]")
                except Exception:
                    row_dict["tool_calls"] = []
                try:
                    row_dict["artifacts"] = json.loads(row_dict.get("artifacts") or "[]")
                except Exception:
                    row_dict["artifacts"] = []
                items.append(row_dict)
            return items
        except Exception:
            return []


def recent_trajectories(limit: int = 20, call_type: str | None = None) -> list[dict]:
    """Convenience wrapper for the module-level collector singleton."""
    try:
        return get_collector().recent(limit=limit, call_type=call_type)
    except Exception:
        return []


# ── Module-level singleton ────────────────────────────────────────────────────

_collector: TrajectoryCollector | None = None


def get_collector(config: dict = None) -> TrajectoryCollector:
    """Get or create the module-level collector singleton."""
    global _collector
    if _collector is None or config is not None:
        import yaml
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.yaml')
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        _collector = TrajectoryCollector(config)
    return _collector


def purge_bad_trajectories(dry_run: bool = False) -> dict:
    """
    Remove corrupt / useless rows from the trajectory DB:
      1. tool_calls is empty ([]) — plain text reply, not a trajectory
      2. final_reply contains model error markers (OOM, CUDA, timeout)
      3. critic_score IS NULL and provider = 'local' (uncored live rows)

    Returns a dict with counts per reason and total deleted.
    """
    _ERROR_MARKERS = [
        "[model_server error]", "[model_client error]", "CUDA error",
        "out of memory", "cudaErrorMemoryAllocation",
        "(max steps reached)", "model_server timeout",
    ]

    deleted: dict[str, int] = {
        "empty_tool_calls": 0,
        "error_reply": 0,
        "unscored_local": 0,
    }

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row

        rows = conn.execute("SELECT id, tool_calls, final_reply, critic_score, provider FROM task_trajectories").fetchall()

        ids_to_delete: set[int] = set()
        for row in rows:
            row_id = row["id"]
            tc = (row["tool_calls"] or "").strip()
            reply = row["final_reply"] or ""
            score = row["critic_score"]
            provider = row["provider"] or ""

            if tc in ("", "[]", "null"):
                ids_to_delete.add(row_id)
                deleted["empty_tool_calls"] += 1
                continue

            if any(m in reply for m in _ERROR_MARKERS):
                ids_to_delete.add(row_id)
                deleted["error_reply"] += 1
                continue

            if score is None and provider == "local":
                ids_to_delete.add(row_id)
                deleted["unscored_local"] += 1
                continue

        total = len(ids_to_delete)
        deleted["total"] = total

        if not dry_run and ids_to_delete:
            conn.execute(
                f"DELETE FROM task_trajectories WHERE id IN ({','.join('?' * total)})",
                list(ids_to_delete)
            )
            conn.commit()

        conn.close()
        action = "[dry-run] would delete" if dry_run else "deleted"
        logger.info(f"[trajectory_purge] {action} {total} bad rows: {deleted}")

    except Exception as e:
        logger.warning(f"[trajectory_purge] failed: {e}")
        deleted["error"] = str(e)

    return deleted
