from __future__ import annotations


def event(name: str, **fields):
    """Telemetry stub for scaffold phase."""
    return {"event": name, **fields}
