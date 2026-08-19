"""
vision — EVA's perception layer: the unified `look` tool.

Abstracts the physical LAN cameras ("eyes") behind a single semantic interface.
All endpoints/IPs are config-driven (never hardcoded), so it's portable across
installs — hub IPs come from each user's settings (set via the companion
desktop app).

Modules:
  registry.py    — load eye config, health-check, status tracking (plug-and-play)
  router.py      — intent -> eye(s) mapping + scan merge
  eyes/          — concrete eye clients (object_face, plant_health)
  capture.py     — grab a single frame from an eye's MJPEG stream
"""

from .registry import EyeRegistry, EyeStatus
from .router import route_look

__all__ = ["EyeRegistry", "EyeStatus", "route_look"]
