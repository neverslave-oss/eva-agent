"""telemetry.py — real trace collector for computer-use runs.

Emits structured, append-only JSONL traces for every run and every step,
including input/output context (goal, planned actions, per-action execution
results, observation snapshots, verification outcomes, and errors).

Traces are written to <expansion>/tmp/traces/<run_id>.jsonl so they can be
replayed or inspected after the fact. The collector is a no-op-safe singleton:
if the trace directory cannot be created, events are still returned (for the
in-memory envelope) but nothing is persisted — the live kernel never breaks.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]  # expansions/computer-use
_LOCK = threading.Lock()


class TraceCollector:
    """Append-only JSONL trace store with an in-memory buffer per run."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.trace_dir = self.root / "tmp" / "traces"
        self._buffer: dict[str, list[dict]] = {}
        try:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._writable = True
        except Exception:
            self._writable = False

    # ── event emission ────────────────────────────────────────────────────
    def event(self, run_id: str, name: str, **fields) -> dict:
        """Record one structured event and persist it (best-effort)."""
        rec = {
            "event": name,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with _LOCK:
            self._buffer.setdefault(run_id, []).append(rec)
            if self._writable:
                try:
                    line = json.dumps(rec, ensure_ascii=False)
                    with open(self.trace_dir / f"{run_id}.jsonl", "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
        return rec

    # ── convenience events ───────────────────────────────────────────────
    def run_started(self, run_id: str, goal: str, target: dict, dry_run: bool) -> dict:
        return self.event(
            run_id, "run_started",
            goal=goal, target=target, dry_run=dry_run,
        )

    def observation(self, run_id: str, source: str, url: str | None, text: str | None,
                    state_hash: str | None, screenshot: str | None = None) -> dict:
        return self.event(
            run_id, "observation",
            source=source, url=url, text=text, state_hash=state_hash,
            screenshot=screenshot,
        )

    def action_planned(self, run_id: str, index: int, action: dict) -> dict:
        return self.event(run_id, "action_planned", index=index, action=action)

    def action_executed(self, run_id: str, index: int, action: dict, result: dict) -> dict:
        return self.event(run_id, "action_executed", index=index, action=action, result=result)

    def action_blocked(self, run_id: str, index: int, action: dict, reason: str) -> dict:
        return self.event(run_id, "action_blocked", index=index, action=action, reason=reason)

    def verification(self, run_id: str, ok: bool, message: str) -> dict:
        return self.event(run_id, "verification", ok=ok, message=message)

    def run_finished(self, run_id: str, status: str, completed: bool, message: str) -> dict:
        return self.event(run_id, "run_finished", status=status, completed=completed, message=message)

    def error(self, run_id: str, stage: str, message: str) -> dict:
        return self.event(run_id, "error", stage=stage, message=message)

    # ── readback ─────────────────────────────────────────────────────────
    def events(self, run_id: str) -> list[dict]:
        """Return buffered events for a run (memory), falling back to disk."""
        with _LOCK:
            if run_id in self._buffer:
                return list(self._buffer[run_id])
        path = self.trace_dir / f"{run_id}.jsonl"
        if path.exists():
            try:
                return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            except Exception:
                return []
        return []

    def list_runs(self, limit: int = 50) -> list[dict]:
        """List persisted traces newest-first with a summary of each run."""
        if not self.trace_dir.exists():
            return []
        runs = []
        for p in sorted(self.trace_dir.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            evs = self.events(p.stem)
            started = next((e for e in evs if e["event"] == "run_started"), {})
            finished = next((e for e in evs if e["event"] == "run_finished"), {})
            runs.append({
                "run_id": p.stem,
                "goal": started.get("goal"),
                "dry_run": started.get("dry_run"),
                "status": finished.get("status"),
                "completed": finished.get("completed"),
                "events": len(evs),
                "file": str(p),
            })
        return runs


# Singleton shared across the bridge and orchestrator.
_collector = TraceCollector()


def get_collector(root: str | Path | None = None) -> TraceCollector:
    """Return the shared collector (optionally re-pointed at a test root)."""
    global _collector
    if root is not None:
        _collector = TraceCollector(root)
    return _collector
