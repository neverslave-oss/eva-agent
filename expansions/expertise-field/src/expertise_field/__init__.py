"""
expertise_field — modular capability-expansion layer for Kernel-Evo.
See SPEC.md (Part 1-7) for the full design.

Design decisions (SPEC 7.5):
  1. Granularity : DOMAINS (~10-20 fields, skills are tasks within)
  2. Hot state   : JSON file (cheap read/write, no dependency)
  3. Routing     : CARRIED-into-build — fields narrow the candidate skill set
                   before semantic matching (a field hit feeds, not bypasses,
                   skill selection). Skills tagged with `field:` frontmatter and
                   grouped in per-domain folders.
  4. Debug       : /debug/fields endpoint to inspect hot fields + registry
  5. Tagging     : synthesized skills get `field:` frontmatter key, grouped
                   per domain, re-discoverable.

Public API:
    Registry.load()            -> Registry
    Registry.field(id)         -> dict | None
    Registry.all_fields()      -> list[dict]
    Registry.is_active(id)     -> bool
    HotFieldState.read(path)  / .write()   -> per-chat hot-field state (JSON)
    router.choose_fields(text, reg, state) -> (hot_ids, candidate_skill_names)
"""
from .registry import Registry, HotFieldState, load_registry
from . import router

__all__ = ["Registry", "HotFieldState", "load_registry", "router"]
__version__ = "0.1.0"