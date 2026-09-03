# Fix Plan — Auth-gate regression causing conversation loss & approval failures

**Date:** 2026-09-03
**Status:** In progress
**Branch (verify):** `debug/pre-auth-gate` (a60747a) confirmed good; fix applied on `main`.

## Problem (from user DEBUG NOTES)

1. During the tools loop, authorization requests are NOT delivered to the user for approval.
2. The agent loses the conversation (user messages are not persisted to history).
3. Root cause attributed to the `feat/exec-shell-approve-all` branch (commits `78d43ae`, `f33cac8`), merged afterward.

## Confirmed regression

- On pre-regression commit `a60747a` (branch `debug/pre-auth-gate`), the agent follows the conversation and messages persist (verified: user "HAPPY?" message id 1348 was found in `chat_history_evolving.db`, agent quoted it).
- On newer code, user messages were missing from `chat_history_evolving.db` (e.g. the user's DEBUG NOTES message — gap at message id 1338).

## Root cause

### Primary trigger — cross-process approval timeout
- `request_auth()` runs in the **model_server** process (tool loop); `resolve_auth()` runs in the **API/bot** process (Telegram callback).
- Each process has its OWN in-memory `_pending` dict, so a button tap in the bot process could never resolve the blocking request in the model_server process → **every `exec_shell` approval timed out**.
- **Already fixed** on `main` in `36c2718` / `beb5363` (persist pending auth requests to a tmp file so the bot process's decision reaches the model_server process).

### Secondary trigger — stop-on-3-denials aborts the tool loop mid-turn
- `f33cac8` added: after `MAX_CONSECUTIVE_DENIES` (3) non-approvals, `request_stop(chat_id)` sets a stop marker.
- With the cross-process timeout counting each timeout as a denial, after 3 timeouts the tool loop sees `is_stop_requested()` and **returns `"(stopped by user)"` immediately**, aborting the turn.
- This aborted/early-return path disrupts normal completion, which is when the conversation turn is persisted → **the turn (and the user message) is lost**.

### Latent robustness gap
- The conversation save in `agent.py` is conditional (`should_skip_persisting`) and the stop-abort path does not reliably persist the user message + assistant response as a complete turn pair.

## Fix steps

### 1. Cross-process approval resolution (DONE — `beb5363`)
- Persist each pending auth request to a tmp file (`_AUTH_DIR`); `resolve_auth()` writes the decision; `request_auth()` polls the file. Approvals no longer time out.

### 2. Preserve conversation on tool-loop abort (TO DO)
- Ensure the user message + assistant response are persisted to SQLite even when the tool loop aborts (stop request / 3 denials).
- Specifically: the `"(stopped by user)"` path must still complete the `agent.py` save of the user message, and should not be treated as a "dead-end" that skips persistence.

### 3. Guard against spurious stop aborts (TO DO / verify)
- Verify that a stop marker from a previous task is cleared at task start (`clear_task_state` already does this). Confirm no stale stop marker causes an immediate abort.

## Verification

- [ ] On `main` with fixes, run `./start.sh`, send an `exec_shell` command requiring approval → buttons appear in Telegram, tapping Allow lets it run.
- [ ] Send a multi-turn conversation (e.g. the "HAPPY?" test) → user messages persist in `chat_history_evolving.db` and the agent references them.
- [ ] Run test suite green (`tests/test_auth_gate.py`, `tests/test_all_native_tools.py`, `tests/test_context.py`, etc.).

## Additional fix (separate)

- Expand the `/status` slash command to report: agent status, system state, in-use modality (cloud vs local), current providers, and running model.
