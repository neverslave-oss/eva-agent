"""JSON-RPC request handlers for the model server.

Extracted from the original model_server.py monolith (refactor) — the final and
largest cut. This module owns the entire handler/inference region:

  - core chat:        _handle_infer, _handle_infer_plain, _handle_infer_draft
  - tool calling:     _run_two_stage_if_available, _nemotron_synthesize_answer,
                      _handle_infer_with_tools
  - multimodal:       _handle_infer_with_image, _handle_infer_with_audio,
                      _handle_infer_local, _cloud_multimodal_infer
  - ops/telemetry:    _handle_vram_free_mb, _resolve_loaded_model_name,
                      _handle_health, _handle_load_slot, _handle_unload_slot,
                      _handle_slot_status, _handle_unload, _handle_swap_model

Shared inference helpers (prompt building, vLLM processor, Omni text, config
wrappers, slot-globals sync) live in handlers_common.py so this module never
imports back into model_server.py (forbidden circular). model_server.py re-imports
and re-exports every handler so the thin entrypoint + HANDLERS dispatch table
keep working unchanged.
"""

import os
import traceback

from .. import state as server_state
from .. import vllm as _vllm
from ..handlers_common import (
    _build_chat_prompt,
    _get_vllm_processor,
    _omni_generate_text,
    _inference_cfg,
    _critique_cfg,
    _load_config,
    _sync_globals_from_slot,
    _mark_activity_start,
    _mark_activity_end,
    _DEBUG_TWO_STAGE,
)
from ..state import target_device as _target_device
from ..activity import mark_activity_start as _mark_activity_start_noarg, mark_activity_end as _mark_activity_end_noarg  # noqa: F401

# Slot registry — mirror the same lazy import pattern model_server.py uses;
# the _SLOTS_AVAILABLE flag is owned here (and re-exported by model_server.py).
try:
    from model_slots import SlotRegistry, SlotSpec, SlotState  # type: ignore
    _SLOTS_AVAILABLE = True
except ImportError:
    _SLOTS_AVAILABLE = False

# Re-exported backend leaves (HF loaders, LoRA adapters, slot lifecycle,
# model-name predicates, parsing, prompts, validation) so handlers can call
# them exactly as the old monolith did.
from ..loaders import (  # noqa: E402
    _load_hf_model,
    _load_nemotron,
    _load_janus,
    _janus_infer_text,
    _janus_infer_image,
    _nemotron_infer,
)
from ..adapters import (  # noqa: E402
    _load_adapter,
    _use_adapter,
)
from ..slots import (  # noqa: E402
    _ensure_model,
    _ensure_multimodal_slot,
    _ensure_tool_calling_slot,
)
from ..model_names import (  # noqa: E402
    is_nemotron_model as _is_nemotron_model,
    is_janus_model as _is_janus_model,
    friendly_model_name as _friendly_model_name,
)
from ..parsing import (  # noqa: E402
    _normalize_tool_args,
    _parse_function_eq_xml,
    _parse_tag_fallback,
)
from ..prompts import (  # noqa: E402
    build_failure_reprompt as _build_failure_reprompt,
    build_schema_repair_hint as _build_schema_repair_hint,
)
from ..validation import (  # noqa: E402
    tool_result_answers_query as _tool_result_answers_query,
)
from ..capabilities import apply_detect as _detect_capabilities

from .chat import _handle_infer, _handle_infer_plain  # noqa: F401 (re-exported for compatibility)
from .tools import (  # noqa: F401 (re-exported for compatibility)
    _run_two_stage_if_available, _nemotron_synthesize_answer, _handle_infer_with_tools,
)
from .multimodal import (  # noqa: F401 (re-exported for compatibility)
    _handle_infer_with_image, _handle_infer_with_audio, _handle_infer_local,
    _cloud_multimodal_infer,
)
from .ops import (  # noqa: F401 (re-exported for compatibility)
    _handle_vram_free_mb, _resolve_loaded_model_name, _handle_health,
)
from .slots import (  # noqa: F401 (re-exported for compatibility)
    _handle_load_slot, _handle_unload_slot, _handle_slot_status, _handle_unload,
    _handle_swap_model,
)
from .draft import (  # noqa: F401 (re-exported for compatibility)
    _ensure_drafter_only, _handle_infer_draft,
)

# ── Two-stage tool-calling pipeline (ADR-011) ─────────────────────────────
# Qwen3.5-0.8B handles tool loop; Nemotron handles conversation synthesis.

# ── Cloud multimodal routing for audio/vision ──────────────────────────────
# Exposed as config flags: providers.stt / providers.vision / providers.tts in config.yaml.
# When set to a cloud provider ("openrouter", "openai"), the handler calls the
# cloud API with base64-encoded content instead of the local Gemma 4 slot.

# ---------------------------------------------------------------------------
# Slot management handlers
# ---------------------------------------------------------------------------

