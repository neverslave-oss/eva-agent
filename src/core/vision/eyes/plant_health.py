"""
eyes/plant_health.py — Eye client for the plant/leaf health camera.

Calls the YOLOv5 Nano brain (`prototype_leaf_detection.py`, port 5000) that
backs the Pi webcam. Endpoints are config-driven via the eye's `base`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def capture_and_detect(base: str, timeout_s: float = 10.0) -> Dict[Any, Any]:
    """POST /plant_health/capture_and_detect → leaf crops.

    Returns: {"status": ..., "num_crops": N, "crops": ["/crops/...", ...]}
    """
    import requests  # type: ignore

    url = f"{base.rstrip('/')}/plant_health/capture_and_detect"
    r = requests.post(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def crop_urls(base: str, crops: List[str]) -> List[str]:
    """Resolve relative crop paths to absolute URLs (GET /crops/<filename>)."""
    base = base.rstrip("/")
    urls = []
    for c in crops or []:
        if c.startswith("http"):
            urls.append(c)
        else:
            urls.append(f"{base}{c if c.startswith('/') else '/' + c}")
    return urls
