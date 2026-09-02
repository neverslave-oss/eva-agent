"""
sensors/pi.py — Pi sensor client for the `sensors` tool.

Calls the Pi's `sensors_data_api.py` endpoints (config-driven via the eye's
`base`). The Pi exposes environmental sensor data at `/pico/sensors` and
actuator control at `/pico/control`:

  GET  /pico/sensors   -> {temp, humi, moisture, moisture_percent, ...}
  POST /pico/control   -> {device, command, params} proxied to the Pico-W

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


def control(base: str, device: str, command: str, params: Dict[str, Any] = None,
            timeout_s: float = 5.0) -> Dict[str, Any]:
    """POST /pico/control to actuate a device via the Pi proxy.

    The Pi proxies the call to the Pico-W. Supported devices/commands:

      - device="relay",  command="on"|"off"          -> Pico POST /relay (pump)
      - device="motor",  command="move", params={"angle": 0..180}
                                                      -> Pico POST /motor (servo head)
      - device="lcd",    command="message", params={"line1","line2"}
                                                      -> Pico POST /message (LCD)

    Returns the proxy's JSON on success, or a dict with an "error" key on
    failure so the caller can surface a helpful message.
    """
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/pico/control"
    payload = {"device": device, "command": command, "params": params or {}}
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code >= 500:
            return {"error": f"control endpoint returned HTTP {r.status_code}"}
        try:
            return r.json()
        except Exception:
            return {"ok": True, "raw": r.text}
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("[sensors] control failed from %s: %s", url, exc)
        return {"error": str(exc)}
