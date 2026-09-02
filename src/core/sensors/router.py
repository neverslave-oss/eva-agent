"""
sensors/router.py — Action routing for the `sensors` tool.

Maps a semantic action ("read", "control") to the configured sensor devices and
returns a unified response envelope, mirroring the `look` tool's router.

- `read` fetches environmental sensor data (temp/humi/moisture/...).
- `control` actuates a device (relay/motor/lcd) through the Pi proxy.
- `water on` / `water off` are aliases for `control(device="relay", ...)`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .registry import SensorRegistry
from . import pi as pi_client

logger = logging.getLogger(__name__)

# Valid actions for the `sensors` tool.
VALID_ACTIONS = ("read", "control", "water on", "water off")

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
                  target: Optional[str] = None, device: Optional[str] = None,
                  command: Optional[str] = None,
                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Route a `sensors` action to the configured device(s).

    - `target` (optional device id) overrides the default; falls back to "pi".
    - `read` returns the sensor observations for the target device.
    - `control` actuates a device; `device`/`command`/`params` describe it.
    - `water on` / `water off` are aliases for relay control.
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

    if action == "control":
        return _run_control(dev, device, command, params)

    # Aliases: "water on"/"water off" -> relay control.
    if action == "water on":
        return _run_control(dev, "relay", "on", None)
    if action == "water off":
        return _run_control(dev, "relay", "off", None)

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


def _run_control(dev, device: Optional[str], command: Optional[str],
                 params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Actuate a device (relay/motor/lcd) via the Pi proxy."""
    device = (device or "").strip().lower()
    command = (command or "").strip().lower()
    if not device:
        return _envelope(dev.id, "control", "error", error="control requires a 'device'")
    if not command:
        return _envelope(dev.id, "control", "error", error="control requires a 'command'")
    try:
        result = pi_client.control(
            dev.base, device=device, command=command,
            params=params or {}, timeout_s=dev.timeout_s,
        )
        if result.get("error"):
            return _envelope(dev.id, "control", "offline", error=result["error"], raw=str(result))
        return _envelope(
            dev.id, "control", "ok",
            observations=result,
            raw=str(result),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[sensors] control failed for %s: %s", dev.id, exc)
        return _envelope(dev.id, "control", "offline", error=str(exc))
