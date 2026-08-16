# Fix Spec — Identity Injection Conflicts with Self-Declared Utility Prompts

**Date:** 2026-08-06
**Status:** fixed 2026-08-06 (see tasks.md "New issues found" section for implementation/verification notes)
**Related tasks:** [tasks.md](../tasks.md) — T1, R2, R9

## Problem

`_handle_infer`'s system-prompt guard (and previously `_handle_infer_plain`'s, and the
tool-loop's) inserts a default Kernel-Evo identity system message whenever a plain `infer()`
call arrives without one:

```
"You are Kernel-Evo, the core AI agent of the Kernel-Evolving project. ..."
```

This is correct/desired for callers that want conversational identity by default (confirmed
by `model_client.infer()`'s own comment: callers explicitly send `tools=[]` "to avoid the
two-stage short-circuit ... that silently falls to plain Nemotron **without identity**" —
i.e. identity injection is a deliberate feature for some callers).

However, at least one caller — `thought_engine.py`'s fact extractor (~line 1078-1097) — sends
a fully self-contained utility prompt that already declares its own role:

```
"You are a fact extractor. Read the following messages from a user and extract ..."
```

Live log evidence (2026-08-06, from the T1 dispatcher fix smoke test) confirms both the
extractor's own role instruction AND the injected Kernel-Evo identity system message are sent
together — two conflicting "you are X" instructions in the same request. In this instance the
model still returned a valid `{}` JSON, so no visible failure, but the conflict is real and
could degrade extraction quality/reliability in other cases (e.g. persona bleeding into
extracted values, or refusal/confusion from contradictory role instructions).

## Root Cause

The system-prompt guard operates purely on "does a system message exist," not on "does the
caller already have a self-declared role." There's no way for a caller to opt out of identity
injection while still omitting a system role (the guard treats "no system message" as
universally "needs the default identity").

## Proposed Fix (not yet implemented)

Give callers an explicit opt-out, e.g.:
- A `params["skip_identity_guard"] = True` flag threaded through `model_client.infer()` /
  `_handle_infer()`, set by `thought_engine.py` for the fact-extractor call (and any other
  purely-utility prompt), or
- Treat any first message whose content starts with a `"You are ..."` role declaration as
  self-sufficient and skip the guard for that call.

The flag-based approach is more explicit and less fragile than text sniffing. Recommend
implementing as a small follow-up once T1-T4 land, not blocking current priority-0 work.

## Non-goals

- Not changing the guard's default-on behavior for callers that don't declare their own role
  (e.g. `thought_engine.py` lines 200/252) — those still need the fallback.

## Verification (once implemented)

- Fact extractor call sends `skip_identity_guard=True` (or equivalent) and the resulting
  Nemotron prompt contains only the fact-extractor's own system content — no Kernel-Evo
  persona line.
- Existing callers without an opt-out flag still get the fallback identity injected
  (no regression).
