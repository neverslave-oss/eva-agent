"""
router.py — Route an incoming task/text to the best-matching expertise field(s),
and narrow the candidate skill set accordingly.

Design decision #3 (SPEC 7.5): fields are a HIGHER-LEVEL abstraction that SELECT
skills. A field hit FEEDS skill selection — it narrows the candidate skill set
before semantic matching runs inside that subset. It does NOT bypass the
existing semantic skill matcher (kernel src/core/skills.py find_semantic).

Routing order:
    text --> trigger-term match (cheap, deterministic) --> narrow fields
         --> optional embedding fallback per-field description
         --> promote matched field(s) to hot (persist to JSON)
         --> return hot + candidate skill names

Skill organisation (decision #5): skills are tagged with a `field:` frontmatter
key and grouped in per-domain folders, so they are re-discoverable. This module
reads the `field:` key to map skills <-> fields where present.
"""
from __future__ import annotations

import re
from typing import Iterable

from .registry import Registry, HotFieldState

STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "for", "on", "in",
    "with", "how", "what", "why", "can", "you", "me", "please", "i", "my",
    "we", "this", "that", "it", "there", "give", "tell", "show", "do", "does",
    "about", "at", "by", "from", "be", "have", "has", "would", "could",
}


def _tokens(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9\-_]{1,}", text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def trigger_terms_for(field: dict) -> set[str]:
    """Expand a field's trigger_terms into a set of lowercase tokens/phrases."""
    terms = set()
    for t in field.get("trigger_terms", []):
        if not isinstance(t, str):
            continue
        terms.add(t.strip().lower())
        # multi-word triggers also add their individual words
        for w in t.lower().split():
            if len(w) > 1 and w not in STOPWORDS:
                terms.add(w)
    return terms


def match_by_triggers(text: str, registry: Registry, min_hits: int = 1) -> list[dict]:
    """Deterministic field matching against trigger terms. Returns fields whose
    trigger vocab overlaps the input tokens by >= min_hits distinct terms.

    Decision #1 (DOMAINS): matches at the domain level — returns the whole
    field (and its linked skills), not individual skills.
    """
    toks = _tokens(text)
    if not toks:
        return []
    scored: list[tuple[int, str]] = []
    for f in registry.all_fields():
        vocab = trigger_terms_for(f)
        hits = len(toks & vocab)
        if hits >= min_hits:
            scored.append((hits, f["id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [registry.field(fid) for _hits, fid in scored]


def narrow_skills(
    fields: Iterable[dict],
    registry: Registry,
    all_skill_names: list[str],
    field_key: str = "field",
) -> tuple[list[dict], list[str]]:
    """Map matched fields to candidate skill names.

    Merges two sources:
      * field['skills']      — explicit linked skills in the field definition
      * field-tagged skills  — skills already tagged with `field: <id>` in
                               frontmatter (decision #5)

    Returns (matched_fields, candidate_skill_names). This FEEDS downstream
    semantic matching; it does not execute anything.
    """
    tag_index = _index_skills_by_field(all_skill_names, field_key)
    candidates: list[str] = []
    matched: list[dict] = []
    for f in fields:
        fid = f["id"]
        matched.append(f)
        for s in f.get("skills", []):
            if isinstance(s, str) and s not in candidates:
                candidates.append(s)
        for s in tag_index.get(fid, []):
            if s not in candidates:
                candidates.append(s)
    return matched, candidates


def _index_skills_by_field(all_skill_names: list[str], field_key: str) -> dict[str, list[str]]:
    """Build {field_id: [skill_name,...]} from a list of 'skill:field' refs.

    Accepts two shapes in all_skill_names:
      * plain skill names (no field info) -> ignored here
      * 'name::field_id' entries -> indexed (used when skills carry field tags)
    """
    index: dict[str, list[str]] = {}
    for ref in all_skill_names:
        if "::" in ref:
            name, fid = ref.split("::", 1)
            index.setdefault(fid, []).append(name)
        # plain names are handled via field['skills'] in caller
    return index


def choose_fields(
    text: str,
    registry: Registry,
    state: HotFieldState,
    chat_id: str = "",
    all_skill_names: list[str] | None = None,
    min_hits: int = 1,
) -> tuple[list[str], list[str]]:
    """Full routing entry point.

    Returns (hot_field_ids, candidate_skill_names).

    Logic:
      1. If a task text is provided, run trigger matching (deterministic).
      2. Promote matched fields to hot in the per-chat JSON state (decision #2).
      3. If nothing matched, fall back to the chat's existing hot fields.
      4. Narrow candidate skills from the final field set (decision #3).
    """
    matched = []
    if text and text.strip():
        matched = match_by_triggers(text, registry, min_hits=min_hits)

    if chat_id:
        for f in matched:
            state.set_hot(chat_id, f["id"], save=True)

    # effective field set = matched (from this task) OR pre-existing hot fields
    if matched:
        effective = matched
    else:
        hot_ids = state.hot_fields(chat_id) if chat_id else []
        effective = [f for f in registry.all_fields() if f["id"] in hot_ids]

    hot_ids = [f["id"] for f in effective]
    _m, candidates = narrow_skills(effective, registry, all_skill_names or [])
    return hot_ids, candidates