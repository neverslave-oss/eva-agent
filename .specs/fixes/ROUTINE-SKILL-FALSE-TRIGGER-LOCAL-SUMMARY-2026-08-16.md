# Routine/Skill False-Trigger + Local-Model Summary Bug (2026-08-16)

## Overview

A user message that merely *mentioned* the word "deploy" ("i'll deploy it
there") accidentally triggered the `deploy` routine. The routine then crashed
with `[model error] 'NoneType' object has no attribute 'apply_chat_template'`,
which was logged as a `skill_error` in failed_requests (id=80).

## Bug 1 — Routine false-positive trigger

### Root cause
`triage()` in `src/core/agent.py` matched routine names with a bare word-boundary
regex: `\bdeploy\b`. Any message containing the standalone word (even mid-sentence
like "i'll deploy it there") fired the routine — the user was just mentioning the
word, not requesting the routine.

### Fix
Bare word-boundary matching now only counts when the message is **short and
command-like** (≤ 4 words, e.g. "deploy"). Explicit request signals always match:
`/<name>`, "run <name>", "routine <name>", "run the <name> routine".

- `src/core/agent.py` — routine trigger matching

## Bug 2 — Routine/skill LLM summary used the local model

### Root cause
`triage()` passed the module-level `infer` (the LOCAL model wrapper) to
`run_routine()` and `run_skill()`. In cloud mode there is no model server, so the
local path fell back to in-process loading where `_processor` is `None` →
`'NoneType' object has no attribute 'apply_chat_template'`.

### Fix
Added `_provider_infer_fn()` — an infer callable routed through the configured
provider (cloud in cloud mode). Both `run_routine` and `run_skill` call sites
(`/run`, `/skill`, and the routine trigger) now use it.

- `src/core/agent.py` — `_provider_infer_fn()` + call sites

## Verification
- `py_compile` clean on agent.py.
- (Pending) restart + live test: a message containing "deploy" mid-sentence must
  NOT trigger the routine; `/run deploy` must run and summarize via cloud.

## Status
[x] Completed
