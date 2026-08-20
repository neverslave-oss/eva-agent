"""
sensors/router.py — Action routing for the `sensors` tool.

Maps a semantic action ("read", future "water on"/"water off") to the configured
sensor devices and returns a unified response envelope, mirroring the `look`
tool's router. Read-only for now; pump override is deferred to the hardware phase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .registry import SensorRegistry
from . import pi as pi_client

logger = logging.getLogger(__name__)

# Valid actions for the `sensors` tool. `water on`/`water off` are reserved for
# the hardware phase (pump override) and are not wired yet.
VALID_ACTIONS = ("read",)

# Actions that require a target device id (defaults to "pi" if omitted).
DEFAULT_DEVICE = "pi"


def _envelope(dev_id: str, action: str, status: str, observations: Any = None,
              raw: str = "", error: str = "") -> Dict[str, Any]:
    return {
        "sensor": dev_id,
        "action": action,
        "observations": observations,
        "raw": raw,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }


def route_sensors(action: str, registry: SensorRegistry,
                  target: Optional[str] = None) -> Dict[str, Any]:
    """Route a `sensors` action to the configured device(s).

    - `target` (optional device id) overrides the default; falls back to "pi".
    - `read` returns the sensor observations for the target device.
    """
    action = (action or "").strip()
    if action not in VALID_ACTIONS:
        return _envelope("", action, "error", error=f"unsupported sensors action '{action}'")

    dev_id = target or DEFAULT_DEVICE
    dev = registry.get(dev_id)
    if dev is None:
        return _envelope(dev_id, action, "error", error=f"unknown sensor device '{dev_id}'")
    if not dev.base:
        return _envelope(dev_id, action, "offline", error="sensor device has no base configured")

    if action == "read":
        return _run_read(dev)

    return _envelope(dev_id, action, "error", error=f"unsupported sensors action '{action}'")


def _run_read(dev) -> Dict[str, Any]:
    """Read sensor data from a device and return a unified envelope."""
    try:
        result = pi_client.read_sensors(dev.base, timeout_s=dev.timeout_s)
        if result.get("error"):
            return _envelope(dev.id, "read", "offline", error=result["error"], raw=str(result))
        return _envelope(
            dev.id, "read", "ok",
            observations=result,
            raw=str(result),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[sensors] read failed for %s: %s", dev.id, exc)
        return _envelope(dev.id, "read", "offline", error=str(exc))
