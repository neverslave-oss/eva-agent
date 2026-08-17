# Plan: Browser-Use Integration + Telegram Tool-Call UI

**Created:** 2026-08-17
**Status:** in-progress
**Branch:** dev

## Overview

Two follow-up workstreams after Eva proved able to autonomously create/edit files
and use tools/skills on request:

1. **Browser-use integration** — replace/enhance the current simple web search/fetch
   with the full [browser-use](https://github.com/browser-use/browser-use) agentic
   browser, enabling multi-step web workflows and web-app usage.
2. **Telegram tool-call UI** — make the in-chat tool-call progress messages more
   informative (what tool was called and to do what), instead of the current
   compact `🔧 tool1 → tool2` line.

Both are planned now but **browser-use is the larger, later effort**; the Telegram
UI is a quick, independent win that can land first.

---

## Part 1 — Telegram Tool-Call UI (quick win)

### Current behaviour
`telegram_bot.py` `_step_cb` shows only a compact single-line progress indicator:
`🔧 tool1 → tool2 → tool3`. It does not show *what* the tool was called with or
*what* it did.

### Goal
Show informative, readable tool-call messages in Telegram:
- Tool name + a short human-readable summary of the call (e.g. `📄 read_file
  ~/docs/api.md`)
- On completion, a one-line result status (success / error)
- Keep it non-spammy: update a single message per step rather than posting many

### Approach
- Extend `_step_cb` to render a richer snippet using `tool_name`, `args`, and
  `result`:
  - Format args into a short label (e.g. path/query/filename).
  - Show a status emoji per tool (📄 read, ✍️ write, 🔍 search, ⚙️ exec, etc.).
  - Truncate long args/results.
- Add a per-tool emoji/verb map (small helper) reused by both the message and
  voice paths.
- Keep the existing "quiet mode" single-line summary as an option.

### Files Affected
- `src/services/channels/telegram_bot.py` — `_step_cb` (message path) and
  `_voice_step_cb` (voice path)

### Success Criteria
- [ ] Telegram shows which tool ran and what it did (args + result status)
- [ ] Long args/results truncated; message updated in place (not spammed)
- [ ] Voice path also shows the richer step info
- [ ] Existing tests pass; live smoke test on a tool-using query

---

## Part 2 — Browser-Use Integration (larger effort)

### Current behaviour
`web_search` tool (`src/core/tools.py`) calls the browser-automation skill's
`browse.py`, which does: Puppeteer/Chromium search (DDG → Bing fallback) or a
direct URL fetch. It returns text content — **no multi-step browser interaction**
(no clicking, form-filling, login, or web-app usage).

### Goal
Integrate [browser-use](https://github.com/browser-use/browser-use) so Eva can:
- Perform **agentic multi-step web tasks** (navigate, click, fill forms, extract,
  complete workflows)
- **Use web apps** (not just read pages)
- Return structured results (text + optionally screenshots/HTML)

### Approach
1. **Add dependency** — `browser-use` (+ `playwright` browser install) to
   `requirements.txt` (and Docker image).
2. **New tool `browser_use`** in `src/core/tools.py`:
   - Params: `task` (natural-language web task), `url` (optional start), `max_steps`,
     `save_screenshot` (optional).
   - Runs browser-use with the configured LLM (reuse the provider's `infer` /
     task_inference routing so it works in cloud mode).
   - Returns a concise summary of what was done + extracted data.
3. **Keep `web_search`** for quick search/fetch; `browser_use` is for interactive
   tasks. Update the tool description so the model picks the right one.
4. **Config** — add a `browser` section (headless mode, provider, max steps,
   timeout) to `config.yaml`.
5. **Safety** — browser-use runs in a sandboxed/headless Chromium; no `exec_shell`
   approval needed (it's a controlled tool), but log actions + cap steps.

### Files Affected
- `requirements.txt` (and `deploy/Dockerfile`) — add `browser-use`, `playwright`
- `src/core/tools.py` — add `browser_use` tool + dispatch
- `src/core/memory/context.py` — document the new tool in the system prompt
- `config.yaml` — add `browser` section
- `.specs/plans/` — this plan

### Success Criteria
- [ ] `browser_use` tool executes a multi-step task (e.g. "search X, open the top
  result, extract the title") via browser-use
- [ ] Works in cloud mode (uses task_inference provider for the browser LLM)
- [ ] `web_search` still works for quick lookups
- [ ] Step/action logging; actions capped to prevent runaway loops
- [ ] Tests pass; live smoke test on a real web task

---

## Priority / Sequencing

1. **Telegram tool-call UI** — independent, quick, high-visibility. Land first.
2. **Browser-use** — larger; depends on dependency install + tool wiring. Land
   second, after the UI work is merged.

## Risks / Open Questions
- browser-use needs a browser LLM; confirm it can reuse the HF/cloud provider
  routing (or needs its own key).
- Playwright browser binaries add install weight (Docker image size).
- Browser-use is async; ensure the tool call path (sync tool loop) wraps it
  correctly (asyncio event loop bridge).

## Status
[ ] Not started
