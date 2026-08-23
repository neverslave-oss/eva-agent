"""
expertise_field_bridge — kernel-side bridge to the expertise-field sidecar.

Integration seam (ADR-015):
    The expertise-field module lives as a SIDECAR at
    expansions/expertise-field/ (not edited into kernel source). This bridge
    is the only place the kernel touches it.

    ADR-022 (Unified Tool-First Pipeline) removed semantic skill matching from
    triage(). To stay compatible, expertise-field is wired as a CONTEXT
    PROVIDER, not a pre-triage interceptor:

        * inject_field_context(chat_id, text)
            -> routes text to hot field(s) and returns a compact progressive-
               disclosure block (field name + domain skills + KB pointers) for
               the caller to append into the system prompt.

        * biased_search_skills(query, matched_skills, chat_id)
            -> re-ranks search_skills results toward the active field's
               domain-tagged skills. "Feed, don't bypass" (decision #3): the
               model always calls search_skills/run_skill as tools; the field
               only biases what it finds.

    Every function is defensive: if the sidecar is missing or errors, it
    returns a safe no-op so the live kernel is never broken by this bridge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Locate the sidecar module (single source of truth for the path) ────────
_KERNEL_ROOT = Path(__file__).resolve().parents[3]  # .../repositories/kernel-evolving
_SIDECAR_DIR = _KERNEL_ROOT / "expansions" / "expertise-field"
_SIDECAR_SRC = _SIDECAR_DIR / "src"


def _import_sidecar():
    """Import the expertise_field package if present. Returns None if absent.

    Safe for boot: imports wrapped in try/except; never raises to the caller.
    """
    if not _SIDECAR_SRC.exists():
        return None
    sys.path.insert(0, str(_SIDECAR_SRC))
    try:
        import expertise_field  # type: ignore
        return expertise_field
    except Exception:
        # Log-and-degrade: a broken sidecar must not take down the kernel.
        try:
            import logging
            logging.getLogger("kernel.evo").exception(
                "expertise-field sidecar import failed; bridge degraded to no-op"
            )
        except Exception:
            pass
        return None


# ── Runtime handles (lazily resolved, cached) ──────────────────────────────
_registry = None
_state = None
_sidecar = None


def _ensure_loaded():
    global _registry, _state, _sidecar
    if _registry is not None:
        return True
    mod = _import_sidecar()
    if mod is None:
        return False
    try:
        _sidecar = mod
        _registry = mod.load_registry(_SIDECAR_DIR)
        hot_path = os.environ.get(
            "EXPERTISE_FIELD_HOT_STATE",
            str(_KERNEL_ROOT / "expansions" / "expertise-field" / "hot_fields.json"),
        )
        _state = mod.HotFieldState(hot_path)
        return True
    except Exception as e:
        _registry = None
        _state = None
        try:
            import logging
            logging.getLogger("kernel.evo").warning(
                "expertise-field bridge init failed (%s); degraded to no-op", e
            )
        except Exception:
            pass
        return False


def available() -> bool:
    """True if the sidecar loaded successfully (safe call)."""
    return _ensure_loaded()


# ── Context-provider seam ──────────────────────────────────────────────────
def inject_field_context(chat_id: str = "", text: str = "") -> str:
    """Return a progressive-disclosure block about the active field(s).

    The caller (build_system_prompt) appends this to the system prompt so the
    model knows which expertise domain it is in and which skills are most
    relevant. Kept COMPACT per ADR-022 progressive disclosure — pointers, not
    dumps.

    Returns "" (safe no-op) if the sidecar is unavailable or nothing matches.
    """
    if not _ensure_loaded():
        return ""
    try:
        hot_ids, _candidates = _sidecar.router.choose_fields(
            text, _registry, _state, chat_id=chat_id
        )
        if not hot_ids:
            return ""
        blocks = []
        for fid in hot_ids:
            field = _registry.field(fid)
            if not field:
                continue
            skills = _registry.skill_names_for(fid)
            kb = field.get("knowledge_base", [])
            kb_refs = []
            for entry in kb[:3]:
                if isinstance(entry, dict):
                    p = entry.get("path") or entry.get("namespace")
                    if p:
                        kb_refs.append(str(p))
            skill_str = ", ".join(skills[:8])
            kb_str = ", ".join(kb_refs)
            block = (
                f"- **{field.get('name', fid)}** (hot)\n"
                f"  - domain skills: {skill_str if skill_str else '—'}\n"
                f"  - kb: {kb_str if kb_str else '—'}"
            )
            blocks.append(block)
        header = "Active expertise fields (progressive disclosure):"
        return header + "\n" + "\n".join(blocks)
    except Exception as e:
        try:
            import logging
            logging.getLogger("kernel.evo").warning(
                "expertise-field context inject failed (%s); no-op", e
            )
        except Exception:
            pass
        return ""


# ── search_skills bias (feed, don't bypass) ────────────────────────────────
def bias_skill_names(query: str, chat_id: str = "") -> set[str] | None:
    """Return the active field's domain skill names to bias search_skills toward.

    Returns None (no bias) if the sidecar is unavailable. The caller should
    treat None as "apply no re-ranking".
    """
    if not _ensure_loaded():
        return None
    try:
        # Use a generic probe text so the route resolves via existing hot state
        # rather than requiring a task string. chat_id keeps it namespaced.
        hot_ids, _c = _sidecar.router.choose_fields(
            query, _registry, _state, chat_id=chat_id
        )
        names: set[str] = set()
        for fid in hot_ids:
            names.update(_registry.skill_names_for(fid))
        return names or None
    except Exception:
        return None


def reorder_matches(query: str, matches: list[dict], chat_id: str = "") -> list[dict]:
    """Re-rank skill search matches so active-field skills float to the top.

    Pure "feed": never drops non-field matches, only reorders. Returns the
    original list unchanged if no bias should apply.
    """
    bias = bias_skill_names(query, chat_id)
    if not bias or not matches:
        return matches
    def _score(s: dict) -> int:
        return 0 if s.get("name") in bias else 1
    return sorted(matches, key=_score)