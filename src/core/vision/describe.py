"""
describe.py — Semantic scene description for the `look` tool.

Given a captured JPEG frame, produce a natural-language description of the
scene. Backends, in priority order:

1. **Local Gemma E2B (native)** — kernel-evolving's own multimodal slot via
   `model_client.infer_with_image` (JSON-RPC over the model-server Unix
   socket). Runs in parallel with cloud chat inference, no external deps.
2. **private-ai-server (:8005)** — vLLM Ollama drop-in (Ollama `/api/generate`
   with an image) as a fallback when the local model server is unavailable.

All endpoints/backends are config-driven (never hardcoded), so EVA stays
portable across installs.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default Ollama-compatible brain (private-ai-server, vLLM drop-in). Overridable
# via env / config so it stays portable.
DEFAULT_DESCRIBE_BASE = os.environ.get(
    "KERNEL_EVO_DESCRIBE_BASE", "http://localhost:8005"
)
DEFAULT_DESCRIBE_MODEL = os.environ.get(
    "KERNEL_EVO_DESCRIBE_MODEL", "google/gemma-4-26b-a4b-it"
)

# Default scene-description prompt (concise, grounded in the image).
DEFAULT_PROMPT = (
    "Describe this scene in 2-3 concise sentences. Mention the main objects, "
    "any people, notable text, and the overall setting. Be factual and grounded "
    "in what is actually visible."
)


def _local_gemma_describe(image_path: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    """Describe via kernel-evolving's own Gemma E2B multimodal slot (native)."""
    try:
        from core.inference import model_client

        if not model_client.is_server_running():
            logger.info("[describe] local model server not running — skipping local backend")
            return None
        out = model_client.infer_with_image(
            image_path, prompt, max_new_tokens=max_new_tokens
        )
        if not out or out.startswith("[model_server"):
            logger.warning("[describe] local Gemma returned: %r", out)
            return None
        return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[describe] local Gemma describe failed: %s", exc)
        return None


def _ollama_describe(base: str, model: str, image_path: str, prompt: str,
                     max_new_tokens: int) -> Optional[str]:
    """Describe via an Ollama-compatible brain (:8005, vLLM drop-in)."""
    try:
        import requests  # type: ignore

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }
        url = f"{base.rstrip('/')}/api/generate"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        return text or None
    except Exception as exc:  # pragma: no cover - network/parse errors
        logger.warning("[describe] Ollama describe failed (%s): %s", base, exc)
        return None


def describe_image(image_path: str, prompt: str = DEFAULT_PROMPT,
                   max_new_tokens: int = 1024,
                   base: Optional[str] = None,
                   model: Optional[str] = None) -> Dict[str, Any]:
    """Describe a scene from a JPEG frame.

    Returns a dict describing the backend used and the description text (or an
    error). Priority: local Gemma E2B native, then Ollama-compatible :8005.
    """
    base = base or DEFAULT_DESCRIBE_BASE
    model = model or DEFAULT_DESCRIBE_MODEL

    # 1) Local Gemma E2B (native, parallel with cloud chat).
    local = _local_gemma_describe(image_path, prompt, max_new_tokens)
    if local:
        return {"backend": "local_gemma_e2b", "description": local}

    # 2) Ollama-compatible brain (:8005).
    remote = _ollama_describe(base, model, image_path, prompt, max_new_tokens)
    if remote:
        return {"backend": "ollama", "description": remote}

    return {"backend": "none", "error": "no describe backend available"}
