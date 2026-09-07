from __future__ import annotations

import json
from pathlib import Path

from .telemetry import TraceCollector


def snapshot(root: str | Path | None = None) -> dict:
    base = Path(root) if root else Path(__file__).resolve().parents[2]
    reg = base / "computer_use_registry.json"
    policies = sorted(p.name for p in (base / "policies").glob("*.json")) if (base / "policies").exists() else []

    registry = {}
    if reg.exists():
        registry = json.loads(reg.read_text(encoding="utf-8"))

    tracer = TraceCollector(base)
    runs = tracer.list_runs(limit=20)

    return {
        "available": True,
        "root": str(base),
        "registry": registry,
        "policy_files": policies,
        "recent_runs": runs,
    }
