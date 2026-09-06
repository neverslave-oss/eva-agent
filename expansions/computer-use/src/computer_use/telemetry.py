from __future__ import annotations

from datetime import datetime, timezone


def event(name: str, **fields):
    """Emit a structured telemetry event payload."""
    return {
        "event": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
