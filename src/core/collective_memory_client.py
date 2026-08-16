"""
Client for the multi-agent-collective-memory service.

Configured via config.yaml's `collective_memory.url` field.  Falls back to
an empty result (no subprocess, no hard dependency) when unconfigured or the
service is unreachable — collective memory is always optional.

Usage:
    from core.collective_memory_client import search as cm_search
    results = cm_search(query, url)   # url from config["collective_memory"]["url"]
"""

import time
from typing import Optional

_CACHE_TTL_SECONDS = 45        # avoid hammering the service on every message
_MAX_RESULT_CHARS = 600        # cap on what's injected into the system prompt
_REQUEST_TIMEOUT = 3.0         # don't block triage() waiting on a slow/down service

_cache: dict[str, tuple[float, str]] = {}


def _cache_key(query: str) -> str:
    return query[:100].lower().strip()


def search(query: str, url: str) -> str:
    """Search the collective-memory HTTP service for *query*.

    Returns up to _MAX_RESULT_CHARS of matching memory content, or "" on
    miss / error / timeout.  Results are cached per query key for
    _CACHE_TTL_SECONDS to avoid a service call on every triage() turn.
    """
    if not url:
        return ""

    key = _cache_key(query)
    now = time.monotonic()
    if key in _cache:
        ts, result = _cache[key]
        if now - ts < _CACHE_TTL_SECONDS:
            return result

    try:
        import requests as _req
        resp = _req.get(
            f"{url.rstrip('/')}/memory/search",
            params={"q": query[:200], "top_k": 3},
            timeout=_REQUEST_TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            items = data if isinstance(data, list) else data.get("results", [])
            parts = []
            for item in items[:3]:
                # Service returns content_text (primary) + optional content_images
                content = (item.get("content_text") or item.get("content")
                           or item.get("value") or "")
                if content:
                    agent = item.get("agent", "")
                    prefix = f"[{agent}] " if agent else ""
                    parts.append(f"{prefix}{str(content)[:200]}")
            result = "\n".join(parts)[:_MAX_RESULT_CHARS]
        else:
            result = ""
    except Exception as exc:
        print(f"[collective_memory] search failed ({exc})", flush=True)
        result = ""

    _cache[key] = (now, result)
    return result


def health_check(url: str) -> bool:
    """Return True if the collective-memory service's /health endpoint responds."""
    if not url:
        return False
    try:
        import requests as _req
        resp = _req.get(f"{url.rstrip('/')}/health", timeout=3.0)
        return resp.ok
    except Exception:
        return False
