# ADR-026: Tools Call Loop Cap (30 → 60)

**Status:** Implemented
**Date:** 2026-08-23
**Author:** Fabio + Olly
**Related:** ADR-022 (Unified Tool-First Pipeline), ADR-011 (Micro-Planner Triage)

---

## Problem

Long, multi-step tasks (especially those chaining many tool calls — browser
automation, expertise-field triage, multi-source lookups) were hitting the **30-turn
tools loop cap** and getting cut off mid-task. The cap was originally 15, raised to
30, but still too low for longer agentic workflows.

## Decision

Raise the **maximum tools call loop steps from 30 to 60** to handle longer tasks.

The cap is applied consistently across the inference stack so the loop budget is
uniform regardless of which inference path is active:

- `src/core/agent.py`
- `src/core/inference/model.py`
- `src/core/inference/model_client.py`
- `src/core/inference/model_server.py`
- `src/core/inference/provider.py`
- `src/core/tools.py`

## Consequences

- Longer agentic tasks can complete without being truncated at the loop cap.
- The cap remains bounded (60) — not unlimited — preserving a safety ceiling against
  runaway tool loops.
- Budget is consistent across all inference providers/paths.

**Key files:** `src/core/agent.py`, `src/core/inference/*`, `src/core/tools.py`
