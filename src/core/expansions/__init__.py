"""
expansions — kernel-side bridge layer for sidecar expansion modules.

Current bridge:
    expertise_field_bridge  ->  loads expansions/expertise-field/ sidecar
                               and exposes context-injection + search-bias hooks
                               (integration seam documented in ADR-015).

Each bridge is OPTIONAL at import-time: if the sidecar is absent, the bridge
degrades to a no-op so the live kernel never fails to boot because of an
expansion module.
"""