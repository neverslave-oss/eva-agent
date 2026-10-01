"""
model_helpers.py — self-contained local-model helpers for the Telegram bot.

Extracted from telegram_bot.py (issue #3 — decompose monolith). Contains the
XP7 VRAM-fit / curated-slot helpers used by the /provider local model picker.
Kept behavior-identical; telegram_bot.py re-imports these names, so all call
sites keep working unchanged.
"""
import json


def _estimate_model_gb(repo_id: str) -> float:
    """Rough model-size estimate in GB from the repo id (params → bytes).

    Heuristic: parse a known param-count marker in the repo id (e.g. 0.8B, 3B,
    7B, 30B). Falls back to 4GB when unknown. Multiply params by ~2 bytes/param
    (bf16) and add ~1GB overhead; quantized 4-bit models use far less, so this
    is intentionally conservative (a "fits" flag is safe).
    """
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)[bB]\b", repo_id)
    if not m:
        return 4.0
    params_b = float(m.group(1))
    gb = params_b * 2.0 + 1.0  # ~2 bytes/param (bf16) + overhead
    return round(gb, 1)


def _vram_fit_mark(est_gb: float, vram_free_mb: int) -> str:
    """Return a short emoji marker showing whether a model likely fits free VRAM."""
    if vram_free_mb <= 0:
        return ""
    free_gb = vram_free_mb / 1024.0
    if est_gb <= free_gb * 0.9:
        return "🟢"  # green — fits comfortably
    if est_gb <= free_gb * 1.5:
        return "🟡"  # yellow — tight / may need quantization
    return "🔴"      # red — likely too big for free VRAM


def _curated_slot_for_repo(repo_id: str) -> str:
    """Return the default model_slots entry for a curated repo id, or '' if none."""
    try:
        import urllib.request as _ur_c
        with _ur_c.urlopen(f"http://localhost:{8779}/models/curated", timeout=5) as _rc:
            curated = (json.loads(_rc.read()) or {}).get("curated", [])
        for cm in curated:
            if (cm.get("repo_id") or "").lower() == repo_id.lower():
                return cm.get("slot") or ""
    except Exception:
        pass
    return ""
