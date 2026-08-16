"""
embedding_client.py — Dual-backend embedding client.

backend="native"  → SentenceTransformer in-process (default for kernel-evolving)
backend="http"    → HTTP POST to ai-server :8770 (kernel base default)

Same public interface in both repos; backend is config-driven via `embedding_backend` key.
"""
import json
import logging
import math
import urllib.request
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover — present on target host, absent in CI
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)

import os as _os

# Default model path — resolved at import time, overridable via env.
# Users can set KERNEL_EMBEDDING_MODEL_PATH to point to their local snapshot.
_DEFAULT_MODEL_PATH = _os.environ.get(
    "KERNEL_EMBEDDING_MODEL_PATH"
) or _os.path.join(
    _os.environ.get("HF_HOME") or _os.environ.get("TRANSFORMERS_CACHE") or _os.path.expanduser("~/.cache/huggingface"),
    "hub_cache",
    "models--unsloth--embeddinggemma-300m-qat-q8_0-unquantized", "snapshots",
    "dc4294deb8cbaad174042a020037fb3a5b008976"
)
_DEFAULT_HTTP_URL = "http://localhost:8770/embeddings"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingClient:
    """
    Caching embedding client with native (in-process ST) and HTTP backends.

    Args:
        backend:        "native" | "http"
        embedding_url:  URL for the HTTP backend (ignored when native)
        model_path:     Local path for SentenceTransformer model (ignored when http)
    """

    def __init__(
        self,
        backend: str = "native",
        embedding_url: str = _DEFAULT_HTTP_URL,
        model_path: str = _DEFAULT_MODEL_PATH,
        # Legacy compat: kernel base passes url= positionally in some tests
        url: str = "",
    ):
        self.backend = backend
        self.embedding_url = url or embedding_url  # url= takes precedence if given
        self.model_path = model_path
        self._cache: dict[str, list[float]] = {}
        # Lazy ST model state (native backend only)
        self._st_model = None
        self._st_attempted = False

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "native":
            return self._embed_native(texts)
        if self.backend == "http":
            return self._embed_http(texts)
        raise RuntimeError(f"Unknown embedding backend: {self.backend!r}")

    def _embed_native(self, texts: list[str]) -> list[list[float]]:
        if not self._st_attempted:
            self._st_attempted = True
            try:
                if SentenceTransformer is None:
                    raise ImportError("sentence-transformers not installed")
                self._st_model = SentenceTransformer(self.model_path)
                logger.info(f"[EmbeddingClient] Loaded native model from {self.model_path}")
            except Exception as e:
                logger.warning(f"[EmbeddingClient] Failed to load native model: {e}")

        if self._st_model is None:
            raise RuntimeError("Native embedding model unavailable")

        vecs = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def _embed_http(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts}).encode()
        req = urllib.request.Request(
            self.embedding_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
        return result["data"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> Optional[list[float]]:
        """Get embedding for text, using cache. Returns None on failure."""
        if text in self._cache:
            return self._cache[text]
        try:
            vecs = self._embed([text])
            if vecs:
                self._cache[text] = vecs[0]
                return vecs[0]
        except Exception as e:
            logger.warning(f"[EmbeddingClient] embed failed: {e}")
        return None

    def get_batch(self, texts: list[str]) -> dict[str, list[float]]:
        """Embed multiple texts in one call. Returns dict text->vector for successful ones."""
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            try:
                vecs = self._embed(uncached)
                for text, vec in zip(uncached, vecs):
                    self._cache[text] = vec
            except Exception as e:
                logger.warning(f"[EmbeddingClient] batch embed failed: {e}")
        return {t: self._cache[t] for t in texts if t in self._cache}

    def similarity(self, a: str, b: str) -> Optional[float]:
        """Cosine similarity between two texts. Returns None if either embed fails."""
        va, vb = self.get(a), self.get(b)
        if va is None or vb is None:
            return None
        return cosine_similarity(va, vb)
