"""
sensors/registry.py — Sensor registry for the `sensors` tool.

Loads the `sensors` section from config.yaml, resolves ${VAR} env placeholders
(e.g. ${KERNEL_EVO_SENSORS_PI_BASE}), and exposes each configured sensor device.
Mirrors the vision `EyeRegistry` pattern so the registry scales to N devices.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Repo root — four levels up from this file (src/core/sensors/registry.py -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = os.environ.get("KERNEL_EVO_CONFIG", str(_REPO_ROOT / "config.yaml"))


@dataclass
class SensorDevice:
    """A single configured sensor device (e.g. the Pi)."""
    id: str
    base: str = ""
    timeout_s: float = 5.0


class SensorRegistry:
    """Loads the configured sensor devices from `sensors` in config.yaml.

    Config-driven, never hardcoded. Each entry is a read source (temp/humi/
    moisture). The registry scales to N devices like the eye registry.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self._lock = threading.Lock()
        self._devices: Dict[str, SensorDevice] = {}
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
                logger.warning("[sensors] config not found at %s — no sensors registered", self._config_path)
                return
            with open(self._config_path, "r", encoding="utf-8") as _f:
                cfg = yaml.safe_load(_f) or {}
            self._load_from_dict(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[sensors] failed to load sensor config: %s", exc)

    @staticmethod
    def _expand(value: Any) -> Any:
        """Resolve ${VAR} / ${VAR:-default} env placeholders in config values.

        Matches the shell-style syntax used across config.yaml (e.g.
        `${KERNEL_EVO_SENSORS_PI_BASE:-}`). Unknown vars without a default expand
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

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", _sub, value)

    def _load_from_dict(self, cfg: Dict[str, Any]) -> None:
        sensors_cfg = (cfg.get("sensors") or {})
        with self._lock:
            self._devices = {}
            for dev_id, spec in sensors_cfg.items():
                if not isinstance(spec, dict):
                    continue
                self._devices[dev_id] = SensorDevice(
                    id=str(dev_id),
                    base=str(self._expand(spec.get("base", ""))).rstrip("/"),
                    timeout_s=float(spec.get("timeout_s", 5.0)),
                )
        logger.info("[sensors] registered %d device(s): %s", len(self._devices), list(self._devices))

    # ── access ──────────────────────────────────────────────────────────────
    def all(self) -> List[SensorDevice]:
        with self._lock:
            return list(self._devices.values())

    def get(self, dev_id: str) -> Optional[SensorDevice]:
        with self._lock:
            return self._devices.get(dev_id)

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._devices.keys())
