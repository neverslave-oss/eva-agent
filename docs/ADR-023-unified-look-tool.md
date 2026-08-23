# ADR-023: Unified Look Tool (Vision for EVA)

**Status:** Implemented
**Date:** 2026-08-19
**Author:** Fabio + Olly
**Related:** ADR-018 (Named Model Slots), ADR-022 (Unified Tool-First Pipeline)

---

## Problem

EVA (kernel-evolving) had no unified way to "see" the world. Vision was scattered:
ad-hoc image handling, no consistent registry of capture sources (camera, screen,
file), and no routing between capture backends and the multimodal model slot.
Each new vision source required bespoke plumbing.

## Decision

Add a **unified `look` tool** — a config-driven vision pipeline with four layers:

- **Registry** (`src/core/vision/registry.py`): declares available capture sources
  (camera, screen, file) and their config, so sources are discoverable rather than
  hardcoded.
- **Router** (`src/core/vision/router.py`): picks the right capture backend for a
  given request and dispatches to it.
- **Eyes** (`src/core/vision/eyes/`): pluggable capture implementations
  (e.g. `object_face`, `plant_health`) — each source is a discrete "eye".
- **Dispatch** (`src/core/tools.py`): the `look` tool entry point that ties
  registry → router → eyes together and returns the captured/described result.

The tool is registered in the system prompt (`context.py`) and micro-planner so the
model can invoke it natively. Vision runs through the multimodal model slot (Gemma 4
E2B), with a local fallback when cloud vision is unavailable.

## Consequences

- New vision sources are added by registering a new "eye" — no core plumbing changes.
- `look` is a first-class native tool (tool count grew with this addition).
- Multimodal inference reuses the named audio/vision slot (ADR-018), avoiding a
  separate text-only main-model load for vision tasks.

**Key files:** `src/core/vision/`, `src/core/tools.py`, `config.yaml`, `tests/test_look_tool.py`
