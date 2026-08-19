"""
eyes/object_face.py — Eye client for the object + face recognition camera.

Calls the YOLOv8n brain (`yolo_server.py`, port 5010) that backs the XIAO
ESP32S3 Sense camera. Endpoints are all config-driven via the eye's `base`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def detect_objects(base: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    """POST /detect → JSON detections [{label, conf, box}]."""
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/detect"
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def detect_text(base: str, timeout_s: float = 5.0) -> str:
    """POST /detect_text → plain-text detection summary."""
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/detect_text"
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.text


def detect_annotated(base: str, timeout_s: float = 8.0) -> bytes:
    """POST /detect_annotated → JPEG with bounding boxes drawn."""
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/detect_annotated"
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def collect_status(base: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    """GET /collect/status → face-collection pipeline status."""
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/collect/status"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def collect_toggle(base: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    """POST /collect/toggle → toggle the face-collection pipeline."""
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/collect/toggle"
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()
