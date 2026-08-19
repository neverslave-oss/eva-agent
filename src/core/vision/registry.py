"""
registry.py — Eye registry for the `look` tool.

Loads the eye configuration (config-driven, never hardcoded), health-checks
each eye, and tracks plug-and-play status as eyes come online/offline.

The config lives in config.yaml under the `vision.eyes` key, populated by the
companion desktop app's settings. All endpoints/IPs are resolved from config at
runtime so EVA is portable across installs.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Repo root — two levels up from this file (src/core/vision/registry.py -> repo root)
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Default config path (config.yaml at repo root). Overridable via env for tests.
CONFIG_PATH = os.environ.get("KERNEL_EVO_CONFIG", str(_REPO_ROOT / "config.yaml"))


class EyeStatus:
    """Health states an eye can be in."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class Eye:
    """A single configured eye (camera + its brain endpoints)."""

    id: str
    kind: str  # object_face | plant_health
    base: str  # brain base URL (config-driven)
    stream: str = ""  # MJPEG stream URL (config-driven)
    health: str = "/"  # health-check path
    status: str = EyeStatus.UNKNOWN
    timeout_s: float = 3.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def health_url(self) -> str:
        return f"{self.base.rstrip('/')}/{self.health.lstrip('/')}"


class EyeRegistry:
    """Loads and health-checks the configured eyes.

    Plug-and-play: call `refresh()` to re-check each eye's health endpoint; eyes
    that respond flip to ONLINE, unreachable ones flip to OFFLINE. The router
    never routes to a dead eye.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._eyes: Dict[str, Eye] = {}
        self._config_path = config_path or CONFIG_PATH
        if config is not None:
            self._load_from_dict(config)
        else:
            self._load_from_file()

    # ── loading ──────────────────────────────────────────────────────────────
    def _load_from_file(self) -> None:
        try:
            import yaml  # type: ignore

            if not os.path.isfile(self._config_path):
                logger.warning("[vision] config not found at %s — no eyes registered", self._config_path)
                return
            with open(self._config_path, "r", encoding="utf-8") as _f:
                cfg = yaml.safe_load(_f) or {}
            self._load_from_dict(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[vision] failed to load eye config: %s", exc)

    @staticmethod
    def _expand(value: Any) -> Any:
        """Resolve ${VAR} / ${VAR:-default} env placeholders in config values.

        Matches the shell-style syntax used across config.yaml (e.g.
        `${KERNEL_EVO_EYE_LEFT_BASE:-}`). Unknown vars without a default expand
        to empty string.
    """
        if not isinstance(value, str):
            return value

        def _sub(m):
            name, default = m.group(1), m.group(2)
            val = os.environ.get(name)
            if val is not None and val != "":
                return val
            return default if default is not None else ""

        import re as _re
        return _re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", _sub, value)

    def _load_from_dict(self, cfg: Dict[str, Any]) -> None:
        eyes_cfg = (cfg.get("vision") or {}).get("eyes") or {}
        with self._lock:
            self._eyes = {}
            for eye_id, spec in eyes_cfg.items():
                if not isinstance(spec, dict):
                    continue
                self._eyes[eye_id] = Eye(
                    id=str(eye_id),
                    kind=str(self._expand(spec.get("kind", "object_face"))),
                    base=str(self._expand(spec.get("base", ""))).rstrip("/"),
                    stream=str(self._expand(spec.get("stream", ""))),
                    health=str(self._expand(spec.get("health", "/"))),
                    status=str(self._expand(spec.get("status", EyeStatus.UNKNOWN))),
                    timeout_s=float(spec.get("timeout_s", 3.0)),
                    meta=dict(spec.get("meta") or {}),
                )
        logger.info("[vision] registered %d eye(s): %s", len(self._eyes), list(self._eyes))

    # ── access ──────────────────────────────────────────────────────────────
    def all(self) -> List[Eye]:
        with self._lock:
            return list(self._eyes.values())

    def get(self, eye_id: str) -> Optional[Eye]:
        with self._lock:
            return self._eyes.get(eye_id)

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._eyes.keys())

    def by_kind(self, kind: str) -> List[Eye]:
        return [e for e in self.all() if e.kind == kind]

    # ── health / plug-and-play ───────────────────────────────────────────────
    def _check_one(self, eye: Eye) -> str:
        """Return ONLINE if the eye's health endpoint responds, else OFFLINE."""
        if not eye.base:
            return EyeStatus.OFFLINE
        try:
            import requests  # type: ignore

            r = requests.get(eye.health_url, timeout=eye.timeout_s)
            return EyeStatus.ONLINE if r.status_code < 500 else EyeStatus.OFFLINE
        except Exception:
            return EyeStatus.OFFLINE

    def refresh(self) -> Dict[str, str]:
        """Re-check every eye's health endpoint. Returns {eye_id: status}."""
        statuses: Dict[str, str] = {}
        with self._lock:
            for eye_id, eye in self._eyes.items():
                eye.status = self._check_one(eye)
                statuses[eye_id] = eye.status
        return statuses

    def online(self) -> List[Eye]:
        """Eyes currently ONLINE (after a refresh)."""
        self.refresh()
        return [e for e in self.all() if e.status == EyeStatus.ONLINE]

    def is_online(self, eye_id: str) -> bool:
        eye = self.get(eye_id)
        if eye is None:
            return False
        return self._check_one(eye) == EyeStatus.ONLINE
