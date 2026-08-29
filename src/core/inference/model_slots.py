"""
model_slots.py — Named model slot registry for kernel-evolving.

Provides SlotSpec, SlotState, and SlotRegistry to allow multiple named model
slots (e.g. "primary", "audio") to be managed independently with LRU eviction
when VRAM is constrained.

Usage:
    from model_slots import SlotRegistry, SlotSpec

    registry = SlotRegistry(vram_threshold_mb=3000)
    # model_path overridable via KERNEL_AUDIO_MODEL env var; falls back to audio section in config
    _audio_cfg = cfg.get("model_slots", {}).get("audio", {}) if cfg else {}
    _audio_path = os.environ.get("KERNEL_AUDIO_MODEL_PATH") or _audio_cfg.get("model_path", os.path.expanduser("~/models/gemma-4-E2B-it"))
    registry.register(SlotSpec(
        name="audio",
        model_path=_audio_path,
        role="audio",
    ))
    state = registry.load("audio")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SlotSpec:
    """Specification for a named model slot."""
    name: str
    model_path: Optional[str] = None       # None → defer to caller / existing config
    dtype: str = "bfloat16"
    quantize: Optional[str] = None         # e.g. "4bit", "8bit", or None
    device: str = "auto"
    role: str = "primary"                  # primary | audio | draft | embed
    max_context_length: int = 8192
    adapter_path: Optional[str] = None     # LoRA / PEFT adapter


@dataclass
class SlotState:
    """Runtime state for a loaded model slot."""
    spec: SlotSpec
    model: Any                             # transformers PreTrainedModel
    processor: Any                         # AutoProcessor / AutoTokenizer
    is_nemotron: bool = False
    audio_capable: bool = False
    supports_tools: bool = False
    native_agentic: bool = False
    is_omni: bool = False
    is_janus: bool = False
    loaded_at: float = field(default_factory=time.time)
    adapter_registry: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader type alias
# ---------------------------------------------------------------------------

# External callers (model_server.py) inject a loader callable so that
# model_slots.py has zero import-time dependency on transformers.
LoaderFn = Callable[[SlotSpec], SlotState]


# ---------------------------------------------------------------------------
# SlotRegistry
# ---------------------------------------------------------------------------

class SlotRegistry:
    """
    Registry that manages named model slots with LRU eviction.

    Eviction policy:
    - Never evict the "primary" slot (role == "primary").
    - When free VRAM drops below *vram_threshold_mb*, evict the non-primary
      slot with the oldest `loaded_at` timestamp before loading a new slot.
    """

    def __init__(
        self,
        vram_threshold_mb: int = 3000,
        loader: Optional[LoaderFn] = None,
    ) -> None:
        self._vram_threshold_mb = vram_threshold_mb
        self._loader: Optional[LoaderFn] = loader
        self._specs: Dict[str, SlotSpec] = {}
        self._loaded: Dict[str, SlotState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_loader(self, loader: LoaderFn) -> None:
        """Inject the loader callable (called by model_server at startup)."""
        self._loader = loader

    def register(self, spec: SlotSpec) -> None:
        """Register a slot spec. Does not load the model."""
        self._specs[spec.name] = spec
        logger.debug("SlotRegistry: registered slot %r (role=%s)", spec.name, spec.role)

    def load(self, name: str) -> SlotState:
        """
        Load the named slot.  If already loaded, returns the cached state.
        If VRAM is low, evicts an LRU non-primary slot first.
        """
        if name in self._loaded:
            return self._loaded[name]

        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"Slot {name!r} is not registered")

        if self._loader is None:
            raise RuntimeError(
                "SlotRegistry has no loader set. "
                "Call registry.set_loader(fn) before loading slots."
            )

        # Evict if needed
        self._maybe_evict(exclude=name)

        logger.info("SlotRegistry: loading slot %r from %s", name, spec.model_path)
        state = self._loader(spec)
        state.loaded_at = time.time()
        self._loaded[name] = state
        logger.info("SlotRegistry: slot %r loaded OK", name)
        return state

    def unload(self, name: str) -> None:
        """Unload a specific slot and release its VRAM."""
        if name not in self._loaded:
            logger.debug("SlotRegistry: unload(%r) — slot not loaded, no-op", name)
            return
        state = self._loaded.pop(name)
        self._release(name, state)

    def get(self, name: str) -> Optional[SlotState]:
        """Return the loaded SlotState, or None if not loaded."""
        return self._loaded.get(name)

    def loaded_slots(self) -> Dict[str, SlotState]:
        """Return all currently loaded slots."""
        return dict(self._loaded)

    def status(self) -> List[Dict[str, Any]]:
        """Return JSON-serialisable status for the health endpoint."""
        result = []
        for name, spec in self._specs.items():
            state = self._loaded.get(name)
            entry: Dict[str, Any] = {
                "name": name,
                "role": spec.role,
                "loaded": state is not None,
                "model_path": spec.model_path,
                "dtype": spec.dtype,
                "quantize": spec.quantize,
                "device": spec.device,
            }
            if state is not None:
                entry["loaded_at"] = state.loaded_at
                entry["is_nemotron"] = state.is_nemotron
                entry["audio_capable"] = state.audio_capable
                entry["supports_tools"] = state.supports_tools
                entry["native_agentic"] = state.native_agentic
                entry["is_omni"] = state.is_omni
                entry["is_janus"] = state.is_janus
                entry["adapter_count"] = len(state.adapter_registry)
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _free_vram_mb(self) -> float:
        """Return free VRAM in MB, or float('inf') if CUDA unavailable."""
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                return free_bytes / (1024 * 1024)
        except Exception:
            pass
        return float("inf")

    def _maybe_evict(self, exclude: str) -> None:
        """Evict the LRU non-primary slot if free VRAM is below threshold."""
        if self._free_vram_mb() >= self._vram_threshold_mb:
            return

        # Collect eviction candidates: loaded, non-primary, not the one being loaded
        candidates = [
            (name, state)
            for name, state in self._loaded.items()
            if state.spec.role != "primary" and name != exclude
        ]
        if not candidates:
            logger.warning(
                "SlotRegistry: VRAM below threshold (%d MB free < %d MB required) "
                "but no eviction candidates available.",
                int(self._free_vram_mb()),
                self._vram_threshold_mb,
            )
            return

        # Sort by loaded_at ascending (oldest first)
        candidates.sort(key=lambda t: t[1].loaded_at)
        victim_name, victim_state = candidates[0]
        logger.warning(
            "SlotRegistry: VRAM low — evicting slot %r (role=%s, loaded_at=%.0f)",
            victim_name,
            victim_state.spec.role,
            victim_state.loaded_at,
        )
        self._loaded.pop(victim_name)
        self._release(victim_name, victim_state)

    @staticmethod
    def _release(name: str, state: SlotState) -> None:
        """Free VRAM held by a SlotState."""
        try:
            import torch  # type: ignore
            del state.model
            del state.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SlotRegistry: slot %r unloaded and VRAM released", name)
        except Exception as exc:
            logger.warning("SlotRegistry: error releasing slot %r: %s", name, exc)
