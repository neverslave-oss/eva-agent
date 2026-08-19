"""
router.py — Intent-based routing for the `look` tool.

Maps a semantic intent to the right eye(s), runs them, and returns a unified
response envelope. Supports `scan` (all eyes, merged) and plug-and-play
availability (never routes to a dead eye).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .registry import EyeRegistry, EyeStatus
from .eyes import object_face, plant_health

logger = logging.getLogger(__name__)

# Intent -> eye kind mapping. `scan` is special-cased (all eyes).
INTENT_EYE_KIND = {
    "what's there": "object_face",
    "who is it": "object_face",
    "plant health": "plant_health",
}

VALID_INTENTS = ("what's there", "who is it", "plant health", "scan")


def _envelope(eye_id: str, intent: str, status: str, observations: Any = None,
              raw: str = "", error: str = "") -> Dict[str, Any]:
    return {
        "eye": eye_id,
        "intent": intent,
        "observations": observations,
        "raw": raw,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }


def _run_eye(eye, intent: str) -> Dict[str, Any]:
    """Run a single eye for the given intent. Returns a response envelope."""
    base = eye.base
    try:
        if eye.kind == "plant_health":
            result = plant_health.capture_and_detect(base, timeout_s=eye.timeout_s)
            crops = result.get("crops", [])
            urls = plant_health.crop_urls(base, crops)
            return _envelope(
                eye.id, intent, EyeStatus.ONLINE,
                observations={
                    "num_crops": result.get("num_crops", len(crops)),
                    "crops": urls,
                },
                raw=str(result),
            )
        # object_face (default)
        result = object_face.detect_objects(base, timeout_s=eye.timeout_s)
        return _envelope(
            eye.id, intent, EyeStatus.ONLINE,
            observations=result,
            raw=str(result),
        )
    except Exception as exc:  # pragma: no cover - network/parse errors
        logger.warning("[vision] eye %s failed for intent %s: %s", eye.id, intent, exc)
        return _envelope(eye.id, intent, EyeStatus.OFFLINE, error=str(exc))


def route_look(intent: str, registry: EyeRegistry,
               target: Optional[str] = None) -> Dict[str, Any]:
    """Route an intent to the appropriate eye(s) and return a unified envelope.

    - `target` (optional eye id) overrides intent mapping if the eye is online.
    - `scan` runs all online eyes and merges results.
    """
    intent = (intent or "").strip()
    if intent not in VALID_INTENTS:
        return _envelope("", intent, "error", error=f"unknown intent '{intent}'")

    # Explicit target eye id wins.
    if target:
        eye = registry.get(target)
        if eye is None:
            return _envelope(target, intent, "error", error=f"unknown eye '{target}'")
        if registry.is_online(target):
            return _run_eye(eye, intent)
        return _envelope(target, intent, EyeStatus.OFFLINE, error="eye offline")

    # `scan` → all online eyes.
    if intent == "scan":
        online = registry.online()
        if not online:
            return _envelope("all", intent, EyeStatus.OFFLINE, error="no eyes online")
        results = [_run_eye(e, intent) for e in online]
        return {
            "eye": "all",
            "intent": intent,
            "observations": results,
            "raw": "",
            "error": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": EyeStatus.ONLINE,
        }

    # Intent-based routing.
    kind = INTENT_EYE_KIND.get(intent)
    candidates = [e for e in registry.by_kind(kind) if registry.is_online(e.id)]
    if not candidates:
        return _envelope(kind or "?", intent, EyeStatus.OFFLINE,
                         error=f"no online eye for intent '{intent}'")
    return _run_eye(candidates[0], intent)
