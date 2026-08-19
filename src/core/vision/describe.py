"""
describe.py — Semantic scene description for the `look` tool.

Given a captured JPEG frame, produce a natural-language description of the
scene. Backends, in priority order:

1. **Local Gemma E2B (native)** — kernel-evolving's own multimodal slot via
   `model_client.infer_with_image` (JSON-RPC over the model-server Unix
   socket). This is the PRIMARY vision backend: it describes the scene with the
   onboard model, loaded on demand (model server spawned if not running).
2. **private-ai-server (:8005)** — vLLM Ollama drop-in (Ollama `/api/generate`
   with an image). This is the FALLBACK when the local Gemma E2B slot is
   unavailable/offline.

Design intent: inference runs on cloud (hf); vision is directed at the local
model in parallel. The native local Gemma E2B is the primary vision backend;
the Ollama brain at :8005 is the configured fallback.

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


def _ensure_local_server(timeout_s: float = 120.0) -> bool:
    """Ensure the local model server is running, spawning it on demand.

    Used by the native fallback: if the Ollama brain is offline and the local
    Gemma E2B slot is the configured fallback, start the model server (lazy) if
    it isn't already up, then wait for the socket.
    """
    try:
        from core.inference import model_client

        if model_client.is_server_running():
            return True

        import subprocess as _sp
        import sys as _sys
        import time as _time
        from pathlib import Path as _Path

        # Reuse the established spawn pattern (same as telegram_bot.py): launch
        # model_server.py with --lazy so the multimodal slot loads on first use.
        _cfg = os.environ.get(
            "KERNEL_EVO_CONFIG",
            str(_Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"),
        )
        logger.info("[describe] local model server not running — spawning on demand")
        _sp.Popen(
            [_sys.executable,
             str(_Path(__file__).resolve().parent.parent / "inference" / "model_server.py"),
             "--config", _cfg, "--lazy"],
            stdout=open("/tmp/kernel_evolving_model_server.log", "a"),
            stderr=_sp.STDOUT,
            start_new_session=True,
        )
        for _ in range(int(timeout_s)):
            _time.sleep(1)
            if model_client.is_server_running():
                return True
        logger.warning("[describe] local model server did not come up within %.0fs", timeout_s)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[describe] failed to ensure local model server: %s", exc)
        return False


def _local_gemma_describe(image_path: str, prompt: str, max_new_tokens: int) -> Optional[str]:
    """Describe via kernel-evolving's own Gemma E2B multimodal slot (native).

    Falls back to spawning the model server on demand if it isn't running.
    """
    try:
        from core.inference import model_client

        if not model_client.is_server_running() and not _ensure_local_server():
            logger.info("[describe] local model server unavailable — skipping local backend")
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
    error). Priority: local Gemma E2B native (PRIMARY), then Ollama-compatible
    :8005 (FALLBACK).
    """
    base = base or DEFAULT_DESCRIBE_BASE
    model = model or DEFAULT_DESCRIBE_MODEL

    # 1) Local Gemma E2B (native) — PRIMARY: describe with the onboard model,
    #    spawning the model server on demand if it isn't running.
    local = _local_gemma_describe(image_path, prompt, max_new_tokens)
    if local:
        return {"backend": "local_gemma_e2b", "description": local}

    # 2) Ollama-compatible brain (:8005) — FALLBACK when local Gemma is offline.
    remote = _ollama_describe(base, model, image_path, prompt, max_new_tokens)
    if remote:
        return {"backend": "ollama", "description": remote}

    return {"backend": "none", "error": "no describe backend available"}
