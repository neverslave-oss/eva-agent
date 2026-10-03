# Plan: `exec_shell` — "Approve All" for the current task

**Created:** 2026-08-30
**Status:** implemented (2026-08-31)
**Branch:** `feat/exec-shell-approve-all` (see "Branch note" below)
**Author:** Fabio (pacificDev) + Kernel-Evo
**Scope:** Feature addition to the existing `exec_shell` approval flow: a third inline button
that auto-approves every subsequent `exec_shell` command until the current task completes
or the hard cap on function calls is hit — whichever comes first. Also adds two stop
controls: 3 consecutive Denies halt the tool loop, and `/stop` halts the agent + tool loop.

> ⚠️ **This is a design spec.** It pairs a concrete change set with the exact files and
> mechanisms already in the codebase. No code changes were made; implementation should
> follow the Approved/Success criteria below.

---

## 1. Goal

Today every non-"safe" `exec_shell` command triggers an inline-button approval prompt
(`✅ Allow` / `❌ Deny`) that blocks the tool loop until the user responds. For a lengthier
task that legitimately needs several shell commands, this forces the user to babysit the
chat and approve each one.

Add an **"Approve All"** (`✅ Approve all`) button to the prompt. Tapping it grants
approval not just for the current command but for **every `exec_shell` call in the current
task** — automatically and non-interactively — until either:

1. the task's tool loop ends (a final answer is produced), **or**
2. the tool loop hits the **hard cap on function calls** (`max_steps`, default 60).

Tapping **Deny** or the current **Allow** keeps the existing per-command behaviour unchanged.

## 2. Context / Current Implementation

### 2.1 The approval prompt — `src/core/auth_gate.py`
- `request_auth(chat_id, command, timeout=DEFAULT_TIMEOUT)`:
  - Returns `"allow"` immediately for commands in `SAFE_PREFIXES`.
  - For everything else allocates a UUID `request_id`, registers a `threading.Event`
    in the module-global `_pending` dict, sends a two-button inline message, then blocks
    on `event.wait(timeout)`.
  - Returns `"allow"` / `"deny"` / `"timeout"` based on how the user (or no one) resolved
    the request.
- `resolve_auth(request_id, approved)` — called by the Telegram callback handler; sets
  `entry["result"]` and fires the event.

### 2.2 The buttons — `auth_gate.py` (lines ~93–99)
```python
buttons = [
    [
        {"text": "✅ Allow", "callback_data": f"auth_allow_{request_id}"},
        {"text": "❌ Deny",  "callback_data": f"auth_deny_{request_id}"},
    ]
]
```

### 2.3 The callback router — `src/services/channels/telegram_bot.py` `handle_callback` (~line 755)
```python
if data.startswith("auth_allow_") or data.startswith("auth_deny_"):
    from core.auth_gate import resolve_auth
    approved = data.startswith("auth_allow_")
    request_id = data.split("_", 2)[-1]
    resolve_auth(request_id, approved)
    status = "✅ Approved" if approved else "❌ Denied"
    if message_id:
        edit_message(chat_id, message_id, f"*Shell command* — {status}")
    return
```

### 2.4 The gate consumer — `src/core/tools.py` `execute_tool` (exec_shell, ~lines 470–485)
```python
_chat_id = chat_id or _current_chat_id
if _chat_id:
    from core.auth_gate import request_auth
    auth_result = request_auth(_chat_id, cmd)
    if auth_result == "deny":
        return "(authorization denied — command blocked)"
    elif auth_result == "timeout":
        return "(authorization timed out — command blocked)"
    elif auth_result.startswith("deny"):
        return f"(authorization failed: {auth_result})"
    # auth_result == "allow" → proceed
```

### 2.5 The hard cap — tool loop in `src/core/inference/model.py` (and `provider.py`)
The tool loop runs `for step in range(max_steps)` where `max_steps` defaults to `60`
(`infer_with_tools(..., max_steps=60)` in `src/core/agent.py`). Returning a final answer
without a tool call ends the loop early. **This range is the "hard cap on function calls"
referenced in the feature request** — it bounds how many tool iterations a task may run.

## 3. Architecture Decision

### 3.1 Model: per-(chat, task) transient "auto-approve" flag
Introduce a **task-scoped auto-approve grant** keyed by the active chat/task, stored
in `auth_gate._pending` alongside the request registry. It is:

- **Transient** — lives only for the currently-executing tool loop; it is cleared when the
  loop ends or hits `max_steps`. It is *not* persisted to disk and never survives a restart.
- **Scoped to the initiating event** — the cleanest correct scope for "until task complete
  or hard cap" is the tool-loop call frame itself. The natural owner is the **current
  `request_auth` blocking session** or an explicit "approve-all handle" that the caller
  closes when the loop finishes.

> **Design choice (recommended): explicit session handle.** Rather than a bare flag keyed
> only on `chat_id` (which could leak across concurrent tasks in the same chat), have the
> *agent/tool-loop caller* open an **approve-all scope** and close it when the loop ends.
> Implementation options:
>
> - **(A) Simple, chat-scoped flag + auto-expiry** — set `_auto_allow[chat_id] = True` on
>   `auth_allow_all_*` and check it in `request_auth`. Because `_pending` is module-global
>   and single-task-per-chat is the current de-facto model, this is the least invasive and
>   matches how `max_steps` already bounds the loop. Recommend starting here.
> - **(B) Explicit `ApproveAllScope`** (context-manager) handed to the tool loop, closed in
>   `finally` — most correct under concurrency but touches the loop entry/exit points in
>   `model.py` / `provider.py` / `model_server.py`. Recommended only if concurrent
>   tool loops per chat become a requirement.
>
> **Adopt (A)** for this spec, with a documented follow-up to (B) if concurrency appears.
>
> **Implementation note (2026-08-31):** Option (A) was implemented with the grant cleared
> at the **start of each new task's tool loop** (in `model.py`, `model_server.py`
> `_handle_infer_with_tools`, and `model_server.py` `_run_two_stage_if_available`) rather
> than in a per-iteration `finally`. Rationale: a per-iteration `finally` would clear the
> grant after the *first* `exec_shell` call, defeating "approve all subsequent commands in
> the task"; clearing at the next task start keeps the grant alive for the whole current
> loop while still guaranteeing it cannot leak into an unrelated later task. The
> `AUTO_ALLOW_TTL` (10 min) remains as the hard safety net for any missed path.

### 3.2 Button layout
```python
buttons = [
    [
        {"text": "✅ Allow",        "callback_data": f"auth_allow_{request_id}"},
        {"text": "✅ Approve all",  "callback_data": f"auth_allow_all_{request_id}"},
        {"text": "❌ Deny",         "callback_data": f"auth_deny_{request_id}"},
    ]
]
```
`callback_data` stays within Telegram's 64-byte limit (`auth_allow_all_` + 8-hex id ≪ 64).

### 3.3 New result values
`request_auth` returns:
- `"allow"` — current command only (unchanged)
- `"allow_all"` — current command **and** every subsequent `exec_shell` in the task
- `"deny"` / `"timeout"` — unchanged

### 3.4 Routing changes
- **`auth_gate.resolve_auth(request_id, approved)`** — gains an `allow_all: bool = False`
  param. Sets `entry["result"]` to `"allow_all"` when `allow_all` is true.
- **`auth_gate.request_auth(...)`** — before hitting the gate, if an auto-approve grant is
  active for this chat/task, immediately return `"allow"` **without sending a prompt**
  (still honour `is_safe_command` and `BLOCKED_PATTERNS`).
- **`telegram_bot.handle_callback`** — handle the new `auth_allow_all_` prefix:
  ```python
  if data.startswith("auth_allow_all_"):
      from core.auth_gate import resolve_auth
      request_id = data.split("_", 3)[-1]
      resolve_auth(request_id, True, allow_all=True)
      if message_id:
          edit_message(chat_id, message_id, "✅ Shell command approved — *all* commands for this task will run automatically")
      return
  ```
  (Mirror the existing `auth_allow_`/`auth_deny_` branch immediately above.)

### 3.5 The auto-approve state + task lifetime
In `auth_gate.py` add a module-level dict (transient):
```python
# chat_id -> auto-approve for the current task loop
_auto_allow: dict[str, bool] = {}
```
- **On** `allow_all` resolution → `_auto_allow[chat_id] = True`.
- **Consumed** in `request_auth`: `if _auto_allow.get(chat_id): return "allow"` (after the
  safe/blocked checks).
- **Cleared** when the task loop finishes. Since the cleanest hook is the loop bounding:
  the tool loop in `model.py`/`provider.py` already knows when it's done (final answer
  returned) and when it's at `max_steps`. Add `clear_auto_allow(chat_id)` calls:
  - `finally`-style after the loop in the model_server/agent tool-loop entry, **and**
  - as a defensive bound in `clear_auto_allow` itself: also clear when `max_steps` is
    reached (the loop naturally ends, which is the same finally path).
  A TTL expiry (e.g. clear after `max_steps * per-step timeout`, or a hard wall-clock TTL
  like 10 min) is a recommended safety net so a grant can never linger if a caller path is
  missed.

> **Safety note:** the grant never bypasses `BLOCKED_PATTERNS` (`rm -rf /`, fork bombs, …).
> `is_safe_command`/`BLOCKED_PATTERNS` checks run *before* the `_auto_allow` early-return so
> dangerous commands still short-circuit to deny even under an active grant.

## 4. Files Affected

| File | Change |
|---|---|
| `src/core/auth_gate.py` | Add `_auto_allow` dict + `AUTO_ALLOW_TTL`; add `is_blocked_command()` (never auto-approved); add `allow_all` to `resolve_auth`; add third button; check grant in `request_auth`; add `clear_auto_allow()` + TTL expiry |
| `src/services/channels/telegram_bot.py` | Handle `auth_allow_all_` prefix in `handle_callback` |
| `src/core/tools.py` | Accept `"allow_all"` as a proceed result (treat like `"allow"` — explicit comment) |
| `src/core/inference/model.py` | Clear auto-approve at tool-loop start (in-process path) |
| `src/core/inference/model_server.py` | Clear auto-approve at tool-loop start in `_handle_infer_with_tools` and `_run_two_stage_if_available` |
| `tests/test_auth_gate.py` | Added `TestApproveAll` — grant set/consume, blocked-pattern short-circuit, TTL expiry, `clear_auto_allow`, `execute_tool` allow_all proceed |

## 5. Implementation Notes

1. **`execute_tool` (tools.py):** treat `"allow_all"` as proceed — the simplest change is
   the existing `if auth_result == "deny": … elif "timeout": … elif startswith("deny"): …`
   chain already falls through to execution for everything else, so **`"allow_all"` already
   proceeds unmodified**. Verify and add an explicit comment/guard so intent is clear.
2. **Concurrency:** since the grant is keyed on `chat_id` and the current architecture is
   single-task-per-chat, a chat-scoped flag is safe *today*. Document the follow-up to a
   proper `ApproveAllScope` (option B) before concurrency is introduced.
3. **TTL:** clear the grant if the tool loop does not explicitly close it within a sane
   bound (e.g. 10 minutes or `max_steps × step_timeout`), closing any missed-care-path leak.

## 6. Success Criteria

- [x] The approval prompt shows **three** buttons: `✅ Allow`, `✅ Approve all`, `❌ Deny`.
- [x] `✅ Allow` behaves exactly as today (single command only).
- [x] `✅ Approve all` approves the current command and **all subsequent `exec_shell`
      commands in the same task** with no further prompts.
- [x] Auto-approval **stops** when the task produces a final answer (loop ends).
- [x] Auto-approval **stops** when the hard cap on function calls (`max_steps`) is hit.
- [x] `BLOCKED_PATTERNS` and `is_safe_command` still short-circuit correctly even under an
      active grant (dangerous commands are never auto-run).
- [x] `❌ Deny` always blocks; `timeout` still blocks.
- [x] Grant cannot leak across unrelated later tasks in the same chat (cleared on task
      start + TTL).
- [x] New unit tests in `tests/test_auth_gate.py` pass; existing tests still pass.
- [ ] Live smoke test: a multi-shell-command task runs to completion on a single
      `✅ Approve all` tap. *(manual — pending)*

## 7. Out of Scope

- Persistence of the grant across restarts (intentionally transient).
- Cross-chat / global "never ask me again" toggles (separate feature).
- Changes to how `max_steps` itself is configured or surfaced.

## 8. Branch note

The currently checked-out branch is `fix/omni-vision-processor` (an unrelated vision fix).
Recommended: implement this feature on a dedicated branch
`feat/exec-shell-approve-all` cut from `dev` / the release line that owns `auth_gate.py`.

---

## 9. Stop controls (added 2026-08-31)

Two additional controls so the user can halt a runaway tool loop:

### 9.1 3 consecutive Denies → stop the tool loop

- `auth_gate.py` tracks a per-chat consecutive-deny counter (`_deny_count`).
- `request_auth` routes every **user Deny** and **timeout** result through `_record_denial()`,
  which increments the counter. Any approval (`allow`/`allow_all`) resets it.
- When the counter reaches `MAX_CONSECUTIVE_DENIES = 3`, `_record_denial()` calls
  `request_stop(chat_id)`.
- The tool loop checks `is_stop_requested(chat_id)` between steps and aborts.
- **Blocked patterns (`rm -rf /` etc.) are NOT counted** — they are auto-denied by the
  system, not user denials, and must not trip the stop rule.

### 9.2 `/stop` halts the agent + tool loop

- `_handle_stop(chat_id)` in `telegram_bot.py` now calls `request_stop(chat_id)` **before**
  killing the model server, so the running tool loop aborts at the next step boundary.
- The stop signal is **cross-process safe**: `request_stop` sets an in-process event AND
  writes a tmp marker file (`/tmp/kernel_evolving_stops/stop_{chat_id}`) so the separate
  model_server process (which runs the tool loop) observes it via `is_stop_requested()`.

### 9.3 Stop mechanism (`auth_gate.py`)

- `request_stop(chat_id)` — set in-process event + write marker file.
- `is_stop_requested(chat_id)` — check in-process event then marker file.
- `clear_stop(chat_id)` / `clear_task_state(chat_id)` — reset at task start so a stale
  stop can't leak into an unrelated later task.
- `tools.py` `execute_tool` also checks `is_stop_requested` before prompting, so no new
  auth prompt is sent after a stop.

### 9.4 Files affected (stop controls)

| File | Change |
|---|---|
| `src/core/auth_gate.py` | `_deny_count`, `MAX_CONSECUTIVE_DENIES`, `_record_denial`, `clear_deny_count`, `request_stop`, `is_stop_requested`, `clear_stop`, `clear_task_state` |
| `src/core/tools.py` | `execute_tool` checks `is_stop_requested` before prompting |
| `src/core/inference/model.py` | Stop check at each loop step; `clear_task_state` at task start |
| `src/core/inference/model_server.py` | Stop check at each loop step in `_handle_infer_with_tools` + `_run_two_stage_if_available`; `clear_task_state` at task start |
| `src/services/channels/telegram_bot.py` | `_handle_stop` calls `request_stop` before killing the server |
| `tests/test_auth_gate.py` | `TestStopSignal`, `TestConsecutiveDenyStop` |

### 9.5 Success criteria (stop controls)

- [x] 3 consecutive Denies request a stop; the tool loop aborts.
- [x] An approval between Denies resets the streak.
- [x] Blocked patterns do NOT count toward the deny-stop rule.
- [x] `/stop` requests a stop that the tool loop observes (cross-process).
- [x] `execute_tool` returns `"(stopped by user)"` instead of prompting after a stop.
- [x] `clear_task_state` prevents a stale stop from leaking into the next task.
- [x] New unit tests pass; existing tests still pass.