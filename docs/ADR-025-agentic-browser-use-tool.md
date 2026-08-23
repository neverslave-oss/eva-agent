# ADR-025: Agentic Browser-Use Tool (Multi-Step Web Tasks)

**Status:** Implemented
**Date:** 2026-08-18
**Author:** Fabio + Olly
**Related:** ADR-022 (Unified Tool-First Pipeline)

---

## Problem

The existing `web_search` tool is a quick lookup — it fetches a page but cannot
perform **multi-step, interactive web tasks** (navigate, click, fill forms, operate
web apps). EVA needed agentic browser automation without opening a window on the
host desktop.

## Decision

Integrate **browser-use** as a new native `browser_use` tool alongside `web_search`:

- Add `browser-use` + `playwright` deps to `requirements.txt`.
- New `browser` config section (`enabled` / `headless` / `max_steps` / `timeout` /
  `provider`).
- Native `browser_use` tool in `tools.py`: params `task` / `url` / `max_steps` /
  `save_screenshot`, with an async bridge via `asyncio.run()`.
- Reuses the `task_inference` provider (HF Router) for the agent loop.
- **Headless enforced** via `BROWSER_USE_HEADLESS` + `Browser(headless=...)` so no
  window opens on the host desktop.
- `max_steps` capped at **50** to prevent runaway loops.
- Registered in the system prompt (`context.py`), micro-planner, arg normalization
  (`tool_arg_utils.py`), and error markers (`memory.py`, `model_server.py`).

## Consequences

- EVA can execute real multi-step web interactions, not just page fetches.
- Browser automation is a first-class native tool (tool count grew to 12 with this).
- Headless default keeps the host desktop clean; step cap bounds resource use.

**Key files:** `src/core/tools.py`, `config.yaml`, `requirements.txt`, `tests/`
