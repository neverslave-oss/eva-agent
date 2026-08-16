# Fix Spec — Legacy Session Hint Leaks Cross-Chat History on Read

**Date:** 2026-08-06
**Status:** identified during verification-checklist testing, not fixed — needs a decision on scope/priority
**Related:** `src/core/memory/memory.py::load()`, `src/database/memory/chat_history.py::get_history_by_chat_id()`

## Problem

`memory.load(chat_id)` always merges in messages from a single, global "legacy session hint"
file (`.kernel_evolving_session_id`, one file per host/process, **not** scoped by chat_id) in
addition to the requested chat_id's own history:

```python
legacy_hint = ""
try:
    sf = _session_file()          # a single, global path — same for every chat_id
    if sf.exists():
        legacy_hint = sf.read_text().strip()
except Exception:
    pass
msgs = repo.get_history_by_chat_id(chat_id, legacy_session_hint=legacy_hint)
```

`get_history_by_chat_id()` (`chat_history.py`) then queries messages from **both**:
1. Sessions whose `chat_id` column matches the requested `chat_id`, **and**
2. The session whose `id` matches `legacy_session_hint` — unconditionally, regardless of what
   chat_id was requested.

## Evidence

Live-tested during today's verification pass (2026-08-06). Sent a message on a fabricated,
never-before-used chat_id (`BRAND_NEW_NEVER_USED_XYZ123`):

```
[agent] pre-save ok: chat_id='BRAND_NEW_NEVER_USED_XYZ123' user_msg='test isolation check'
[agent] memory saved: chat_id='BRAND_NEW_NEVER_USED_XYZ123' total_turns=30
```

`total_turns=30` after a single message on a chat_id that has never been used before. Multiple
other fabricated/test chat_ids (`SMOKE_TEST_T14`, `SMOKE_TEST_CONTINUITY`) independently showed
the exact same `history=28 turns` baseline before their first message — confirming they were
all merging in the same shared legacy session, not isolated per chat_id.

Confirmed the legacy hint file (`~/.kernel-evolving/workspace/memory/.kernel_evolving_session_id`)
contains a real, populated session UUID (`dae018ee-48c0-4148-86fe-42ed0eafad68`) — i.e. this is
not an edge case with an empty/harmless hint, it actively leaks a real conversation's content
into unrelated chat_id queries.

## Scope of the leak (read vs. write)

- **Read (`load()`): confirmed leaking.** Any chat_id — including ones that have never been
  used — receives the legacy session's history injected as if it were their own.
- **Write (`save()`): confirmed isolated, not leaking.** `save(messages, chat_id)` uses
  `session = chat_id` (when provided) and calls `repo.touch_session(session, chat_id)` — writes
  go to a session keyed by the caller's own chat_id, not the legacy session. Test messages sent
  during today's verification were **not** written into the legacy/real session.

Net effect: a one-directional **read** leak. Test/unrelated chat sessions can see (and
potentially have the model reference) the legacy session's real conversation content, but do
not corrupt it by writing into it.

## Root Cause

The legacy-session merge was almost certainly added to recover pre-chat_id-column history after
a schema migration (per the code comment: "the pre-migration UUID session (empty chat_id) ...
for pre-migration turns"). It's applied unconditionally to *every* `get_history_by_chat_id()`
call instead of being scoped to only the specific chat_id(s) that legitimately owned that
legacy session (e.g. the bot owner's real `ALLOWED_CHAT_ID`, or only when the caller passes no
chat_id at all).

## Proposed Fix (not yet implemented — needs a decision)

Options, roughly in order of safety:
1. **Only merge the legacy hint for the real bot chat_id.** Compare `chat_id` against
   `telegram_bot.ALLOWED_CHAT_ID` (or an equivalent "primary chat" config value) before including
   `legacy_session_hint` in the query — every other chat_id gets its own isolated history only.
2. **Only merge the legacy hint once, then migrate it.** On first load for the real primary
   chat_id, backfill the legacy session's messages into a session row with that chat_id set,
   then stop reading the legacy hint file for future calls (one-time migration instead of a
   permanent merge on every read).
3. **Drop the legacy hint merge entirely** if the pre-migration data is no longer valuable,
   accepting loss of that old history.

Recommend option 1 as the least invasive, most reversible fix.

## Non-goals

- Not investigating whether other read paths (long-term memory markdown mirror, embedding
  retrieval) have a similar leak — flagged only for the SQLite chat-history path found here.
- Not fixing without user confirmation — this affects real conversation data on a live,
  in-use agent; scope/priority should be confirmed before changing.

## Verification (once implemented)

- A fabricated/never-used chat_id should show 0 history turns on first load, not the legacy
  session's turn count.
- The real bot's primary chat_id should still see its full historical continuity (no regression
  for the intended use case).
