# Plan: Split `handle_message` (final slice of telegram_bot.py decomposition)

**Date:** 2026-10-02
**Issue:** #3 — decompose telegram_bot.py monolith (3865 lines)
**Branch:** `refactor/telegram-messaging-split`
**Status:** ✅ DONE — PR #11 merged to `dev` (squash `3b74fd6`), issue #3 CLOSED (2026-10-03).

## Context

`telegram_bot.py` is down to **2362 lines**, of which a single function —
`handle_message` (lines 65–2330) — spans ~2265 lines. It is the message-processing
core: media handling (photo/voice/document), ~30 slash-command handlers, and the
agent reply/triage loop.

Slices 1–9 established the proven extraction pattern:
- One module per concern under `src/services/channels/telegram_*.py`
- Behavior-identical body moves (no logic rewrites)
- Re-export imports in `telegram_bot.py` (`# noqa: F401 (re-exported for compatibility)`)
- **Bot-module seam:** shared mutable state and cross-cutting helpers stay on the
  `telegram_bot` module; extracted modules resolve them at call time via
  `sys.modules["services.channels.telegram_bot"]` so the test harness's
  `patch.object(bot, ...)` still applies
- Test gate after every slice:
  `pytest tests/test_inline_buttons.py tests/test_telegram_missing_coverage.py tests/test_telegram_poll_resilience.py`
  → must stay **52 passed, 1 xfailed**

## Target layout after this plan

| Module | Contents |
|---|---|
| `telegram_media.py` | `_handle_photo_message`, `_handle_voice_message`, `_handle_document_message` (extracted blocks) |
| `telegram_commands.py` | per-command handler functions `_handle_<name>(chat_id, text, ...)` (extracted blocks) |
| `telegram_handlers.py` | core reply/triage loop (agent triage, streaming, tool-step formatting, collective-memory write) |
| `telegram_bot.py` | thin dispatcher: `handle_message` header + media/command routing (~200 lines) |

## Slices

### Slice A — `telegram_media.py` (~300 lines) — LOW RISK, do first
Extract the three contiguous media blocks out of `handle_message`:
- Photo block (starts at `if photo_file_id:`)
- Voice block (starts at `if voice_file_id:`)
- Document block (starts at `if document_file_id:`)

Each becomes a module-level function taking the same params it currently closes
over (`chat_id`, `text`, `sender_name`, file ids, working message id, etc.).

Seam names used inside (resolve via `_bot_module()` at call time, or via
function-local aliases bound from `_bot_module()`):
`send_message`, `edit_message`, `send_typing`, `_clone_voice_reply`,
`TypingKeepAlive`, `_MultimodalActivity`, `download_file`, `_ensure_agent`.

Globals touched by media blocks (stay on bot module): `_active_voice_sample`
(voice block may set it from `set_voice_` flow), `_agent_ready`.

### Slice B — `telegram_commands.py` (~1400 lines, 3 sub-slices)
Extract the `if text == "/..."` command blocks into `_handle_<name>(...)` functions.

- **B1** — system/info: `/new /fresh /session /init /skills /routines /packages
  /status /stop /restart /update /rollback /verbose /thoughts` (~500 lines)
- **B2** — provider/model: `/local /cloud /models /providers /provider /replica
  /install /clone /search /private_repo` (~600 lines)
- **B3** — skill/routine/voice/run: `/skill_ /run_ /run /voices /voice-clone
  /evolve` (~300 lines)

Each sub-slice: test gate → commit → push.

### Slice C — `telegram_handlers.py` (~500 lines)
Extract the tail of `handle_message` (agent triage, streaming replies, tool-step
formatting, collective-memory write). Leaves `telegram_bot.py` as a thin
dispatcher. Final `telegram_bot.py` ~200 lines.

## Guardrails

- Pure extraction only: no logic rewrites, no behavior changes, no renames of
  user-visible commands/formatting.
- Shared mutable state (`_active_voice_sample`, `_agent_ready`, `_verbose_mode`,
  `_memory`) remains defined on the bot module; extracted modules read/write via
  the seam.
- Test gate (52 passed, 1 xfailed) green after every slice before commit.
- Commit + push after every slice — never leave work in-flight across an
  interruption.

## Out of scope (other issues)

- #2: `api.py` (2947) + `handlers.py` (2168) decomposition — separate plan.
- Desktop #13 / #15 — separate repos.