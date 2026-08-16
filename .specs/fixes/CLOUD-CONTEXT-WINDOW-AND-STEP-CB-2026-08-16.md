# Cloud Context Window + Step-Callback Bugs (2026-08-16)

## Overview

Two bugs surfaced after switching task_inference to a cloud provider (HF):

1. **Cloud conversations truncated at the local Nemotron 64k context limit.**
2. **`_step_cb() missing 1 required positional argument: 'result'`** when a
   routine (e.g. `deploy`) ran through the Telegram bot.

## Bug 1 — Cloud context capped at 64k

### Root cause
`_get_context_tokens()` in `src/core/agent.py` derived the context window from
the **loaded local model** (model server health → `model_context_lengths`
map). In cloud mode the model server is NOT running, so it fell back to
`config.model.name` = Nemotron → `model_context_lengths.nemotron` = **65536**.
Cloud models (e.g. DeepSeek-V4-Flash via HF) have ~1M context, so long cloud
conversations were truncated unnecessarily.

### Fix
`_get_context_tokens()` now checks the effective `task_inference` provider via
`get_provider("task_inference")`. When it is a cloud provider (not `local`),
the history is **unbounded** — no context cap is applied at all, because cloud
models have varied, model-specific context windows (e.g. DeepSeek-V4-Flash-0731
has 1M tokens) and capping at any fixed value either truncates large-window
conversations or risks overflow on small-window ones. The cloud model handles
its own window. Only local inference keeps the per-model context budget.

- `src/core/agent.py` — `_get_context_tokens()` returns `(ctx, unbounded)`;
  the history-trimming loop is skipped when `unbounded` is True.

## Bug 2 — `_step_cb()` missing 'result' arg

### Root cause
The Telegram bot's `_step_cb(n, tool_name, args, result)` required 4 positional
args. The routine executor (`src/core/routines.py`) calls
`step_callback(step_num, label, result)` with **3 args** (label is a string).
Running a routine through the Telegram bot raised
`missing 1 required positional argument: 'result'`.

### Fix
Made the Telegram bot's `_step_cb` (message path) and `_voice_step_cb` (voice
path) tolerant: `args=None, result=None` defaults, so both the inference path
(4 args) and the routine path (3 args) work.

- `src/services/channels/telegram_bot.py` — `_step_cb`, `_voice_step_cb`

## Verification
- Live: after restart, log shows `cloud context: UNBOUNDED (no history cap)`
  and `history=<N> turns ctx_tokens=unbounded` — no truncation on cloud.
- Live: `POST /message` returns a real reply (`{"reply":"context fix verified"}`).
- `py_compile` clean on agent.py + telegram_bot.py; config.yaml valid.

## Status
[x] Completed
