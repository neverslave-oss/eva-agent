"""Stateless model-capability detection helpers for the model server.

Extracted from the original model_server.py monolith (refactor). These helpers
only decide whether a model_name/model_path matches a family prefix; they carry
no shared state, so this is a leaf module — safe to import from anywhere without
circular-dependency risk.
"""

# Models that natively handle audio (Gemma 4, Gemma 3, Qwen Omni).
_AUDIO_CAPABLE_PREFIXES = ("google/gemma-4", "google/gemma-3", "qwen2.5-omni", "qwen2.5-omni")

# Models that can handle the FULL agentic flow (tool loop + synthesis) natively
# on their own — no Qwen two-stage detour, no Nemotron synthesis fallback.
# Any-to-any models like Qwen2.5-Omni are the canonical example.
_NATIVE_AGENTIC_PREFIXES = ("qwen2.5-omni", "qwen3-omni", "qwen3-omni-moe", "gemma-4", "gemma-3")

# Qwen2.5-Omni (any-to-any) — its generate() has a non-standard signature and
# needs generation_mode="text" for fast text-only output.
_OMNI_PREFIXES = ("qwen2.5-omni", "qwen3-omni", "qwen3-omni-moe")

# DeepSeek Janus / Janus-Pro — uses the custom `janus` package
# (MultiModalityCausalLM + VLChatProcessor), not standard transformers loading.
_JANUS_PREFIXES = ("deepseek-ai/janus", "janus-pro", "janus-1", "janus-7")

# Models that natively expose parse_response for tool calling (by name).
_TOOL_CAPABLE_PREFIXES = ("google/gemma-4",)


def _matches_any(model_name: str, model_path: str, prefixes) -> bool:
    """True if model_name starts with a prefix, or model_path contains its stem."""
    return any(
        model_name.lower().startswith(p.lower()) for p in prefixes
    ) or any(
        model_path.lower().replace("\\", "/").find(p.lower().split("/")[-1]) != -1
        for p in prefixes
    )


def detect(model_path: str, cfg: dict) -> dict:
    """Return capability flags derived from model_path + config.

    Pure: computes and returns a dict of booleans; does NOT touch module or
    module-level globals. Callers are responsible for applying the returned
    flags to their own state.
    """
    model_name = cfg.get("model", {}).get("name", "")
    return {
        "supports_tools": _matches_any(model_name, model_path, _TOOL_CAPABLE_PREFIXES),
        "audio_capable": _matches_any(model_name, model_path, _AUDIO_CAPABLE_PREFIXES),
        "native_agentic": _matches_any(model_name, model_path, _NATIVE_AGENTIC_PREFIXES),
        "is_omni": _matches_any(model_name, model_path, _OMNI_PREFIXES),
        "is_janus": _matches_any(model_name, model_path, _JANUS_PREFIXES),
    }
