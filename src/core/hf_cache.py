"""Single source of truth for the HuggingFace cache root.

MS4: telegram_bot.py's "is this model downloaded?" check and model_server.py's
from_pretrained()/AsyncEngineArgs() calls each resolved HF_HOME independently.
When a bare repo_id (not a local path) is passed to swap_model, huggingface_hub
resolves its own cache location inside the model_server process — if that
process's environment ever disagrees with the bot process's environment (e.g.
different launch contexts: systemd service vs interactive shell), the bot can
show a model as downloaded that the server can't find, or vice versa. Both
sides must call resolve_hf_cache_root()/ensure_hf_home_env() instead of
inlining their own env var lookups.
"""

import os
from pathlib import Path


def resolve_hf_cache_root() -> str:
    """Return the HF cache root, matching huggingface_hub's own fallback order."""
    return os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or str(Path.home() / ".cache" / "huggingface")


def resolve_hf_hub_dir() -> Path:
    """Return the 'hub' subdirectory under the resolved cache root, where
    snapshot_download()/from_pretrained() actually store models--org--repo dirs."""
    return Path(resolve_hf_cache_root()) / "hub"


def ensure_hf_home_env() -> str:
    """Pin os.environ['HF_HOME'] to the resolved cache root (no-op if already
    set) so huggingface_hub's internal resolution — used implicitly when
    from_pretrained()/AsyncEngineArgs() are given a bare repo_id rather than a
    local path — can't silently disagree with what this module reports as
    downloaded. Returns the resolved value.
    """
    resolved = resolve_hf_cache_root()
    os.environ.setdefault("HF_HOME", resolved)
    return resolved
