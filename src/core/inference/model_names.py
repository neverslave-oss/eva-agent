"""Stateless model-name / predicate helpers for the model server.

Extracted from the original model_server.py monolith (refactor). Pure helpers
that classify a model path/name (Nemotron, Janus, PEFT/adapters) or turn a
path into a readable label. No shared state — a leaf module.
"""

from pathlib import Path


def is_nemotron_model(model_path: str) -> bool:
    """Return True if the model path/name is a Nemotron-Labs-Diffusion variant."""
    name = (model_path or "").lower()
    return "nemotron" in name or "nemotron-labs-diffusion" in name


def is_janus_model(model_path: str) -> bool:
    """Return True if the model path/name is a DeepSeek Janus/Janus-Pro variant."""
    name = (model_path or "").lower()
    return "janus" in name


def adapter_name(adapter_path: str) -> str:
    path = str(Path(adapter_path).expanduser())
    return f"adapter_{abs(hash(path))}"


def is_peft_model(model_obj) -> bool:
    try:
        from peft import PeftModel
        return isinstance(model_obj, PeftModel)
    except Exception:
        return False


def friendly_model_name(path: str) -> str:
    """Turn a local model path into a readable name.

    HF cache paths look like .../models--org--repo/snapshots/<hash>/ — recover
    'org/repo' from that structure instead of showing the meaningless hash
    (the snapshot dir's basename).
    """
    import re
    m = re.search(r"models--([^/\\]+)--([^/\\]+)", path or "")
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return str(path).rstrip("/\\").split("/")[-1]
