"""
sensors/pi.py — Pi sensor client for the `sensors` tool.

Calls the Pi's `sensors_data_api.py` endpoints (config-driven via the eye's
`base`). The Pi exposes environmental sensor data at `/pico/sensors`:

  GET /pico/sensors → {temp, humi, moisture, moisture_percent}
  POST /pico/sensors → {relay: bool, watering: bool} (pump override — deferred)

Endpoints are config-driven via the sensor registry's `base`, never hardcoded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Sensor keys returned by the Pi endpoint.
SENSOR_KEYS = ("temp", "humi", "moisture", "moisture_percent")


def read_sensors(base: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    """GET /pico/sensors → dict of {temp, humi, moisture, moisture_percent}.

    Returns the raw sensor JSON on success, or a dict with an "error" key on
    failure (network/HTTP) so the caller can surface a helpful message.
    """
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/pico/sensors"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code >= 500:
            return {"error": f"sensor endpoint returned HTTP {r.status_code}"}
        data = r.json()
        # Normalise to a stable subset (ignore extra fields like raw/relay/ts).
        return {k: data.get(k) for k in SENSOR_KEYS}
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("[sensors] read failed from %s: %s", url, exc)
        return {"error": str(exc)}


def set_relay(base: str, relay: bool, watering: bool, timeout_s: float = 5.0) -> Dict[str, Any]:
    """POST /pico/sensors → control the water-pump relay (manual override).

    NOTE: actuation is deferred to the hardware phase (Pico W firmware flash).
    This client is implemented so the tool can be wired later, but it is NOT
    exposed via `sensors(action=...)` yet.
    """
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/pico/sensors"
    payload = {"relay": bool(relay), "watering": bool(watering)}
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code >= 500:
            return {"error": f"relay control returned HTTP {r.status_code}"}
        return {"ok": True, "relay": bool(relay), "watering": bool(watering)}
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("[sensors] relay control failed from %s: %s", url, exc)
        return {"error": str(exc)}
