# AUDIT 2026-08-06 — Chat Session Splitting & `/fresh` Endpoint

**Branch:** `fix/qwen-tool-calling-sampling` (no changes to source, no commit)
**System:** Kernel-Evolving (port 8779)

---

## Table of Contents

1. [Audit C: Implementability — Split Conversations into Sessions](#audit-c-implementability--split-conversations-into-sessions)
2. [Audit D: Implementability — `/fresh` Endpoint with Factory Defaults](#audit-d-implementability--fresh-endpoint-with-factory-defaults)

---

## Audit C: Implementability — Split Conversations into Sessions

**Request:** Split long chat conversations into sessions. Pass the previous session's ID to the agent in the follow-up session so it can refer to the old conversation using memory retrieval (`recall_memory`).

### C1 — Current Session Architecture

**There is already a session concept, but it's one-per-chat, not rotating.**

- **`_session_id()`** (memory.py:42) — a UUID generated once per process run, persisted to `~/.kernel-evolving/workspace/memory/.kernel_evolving_session_id`. This is the *process* session, not a conversation session.
- **SQLite sessions table** (chat_history.py:42-51):
  ```sql
  CREATE TABLE IF NOT EXISTS sessions (
      id            TEXT PRIMARY KEY,        -- = chat_id when chat_id is given
      chat_id       TEXT NOT NULL DEFAULT '',
      created_at    TEXT NOT NULL,
      updated_at    TEXT NOT NULL,
      message_count INTEGER NOT NULL DEFAULT 0
  );
  ```
- **When `chat_id` is provided** (e.g., Telegram chat `"844251003"`), `save()` and `load()` both use `session = chat_id` — so the chat_id **doubles as the SQLite session_id**. One row in `sessions` per Telegram chat.
- **`get_history_by_chat_id`** (chat_history.py:125-180) queries for all sessions WHERE `chat_id = X` (plus a legacy UUID hint). Since each chat has exactly one session (chat_id = its own chat_id), this returns all messages in one pile.
- **`load(chat_id)`** returns ALL messages for that chat from SQLite (or JSON window as cache). Trimming is done by `_max_history_chars` (char-based proxy) in `agent.py` — oldest turns dropped from the end of the list.

**Summary:** Currently, a "session" for a Telegram chat lasts forever with no rotation. All messages accumulate in one SQLite session. `_max_history_chars` trims by character count, dropping the oldest without any session awareness.

### C2 — Database Schema Readiness

The existing `sessions` table is a good foundation but needs additions:

| Current column | What it has | What's needed for sessions |
|---|---|---|
| `id` (PK) | = chat_id currently | A new system: session_id = UUID per session rotation |
| `chat_id` | ✅ Present | Keep as-is |
| `created_at` | ✅ Present | Keep |
| `updated_at` | ✅ Present | Keep |
| `message_count` | ✅ Present | Keep |
| *(missing)* | — | **`previous_session_id TEXT`** — link to the closed session |
| *(missing)* | — | **`session_number INTEGER`** — sequential number per chat (1, 2, 3...) |
| *(missing)* | — | **`status TEXT DEFAULT 'active'`** — 'active' or 'closed' |

The `messages` table already has `session_id TEXT NOT NULL` — it can store messages from any session. No schema changes needed there.

**Legacy migration:** Existing chats have one session with `id = chat_id` and `status` implicit (active). The new schema should treat these as `session_number: 1` and `status: 'closed'` when a session split is triggered for that chat.

### C3 — How Session Splitting Would Work

#### Trigger Mechanisms

1. **Automatic — `MAX_TURNS` threshold** (memory.py:30 currently `MAX_TURNS = 300`). When session reaches N turns, auto-close and start new session.
2. **Automatic — Time gap** (e.g., no messages for >24h = new session)
3. **Explicit — `/session new` command** (user-initiated split)
4. **Explicit — API endpoint** (e.g., `POST /chat/session/new`)

#### Auto-Split Flow

```
triage() or save() is called
  → Check current session (the 'active' session for this chat_id)
    → If session.message_count >= MAX_TURNS:
      1. Close current session: UPDATE status='closed', updated_at=NOW
      2. Compute compact summary of the closed session (optional, see C4)
      3. Create new session: INSERT with previous_session_id=closed_id,
         session_number=prev+1, status='active'
      4. From now on, load() only loads the current (active) session
      5. Inject previous session ID + summary into the next system prompt
    → Else: resume normally
```

#### What `load()` Should Return After Splitting

Currently `load()` returns ALL messages from all sessions. After splitting:
- `load(chat_id)` should return **only the current (active) session's messages**.
- The previous session's data is accessible via `recall_memory(chat_id, session_id)` or the existing `history(limit, chat_id)` — although `history()` currently also returns all sessions for that chat_id.
- A new method: `get_history_by_session(session_id)` — targeted retrieval for memory recalls.

**Migration risk:** Changing `load()` to return only active-session messages would break existing single-session chats (they have one implicit session). Solution: `get_history_by_chat_id` should default to "all sessions" when there's only one session, and switch to "active only" after a split is triggered.

### C4 — Passing Previous Session ID to the Agent

The previous_session_id needs to reach the agent so it can use `recall_memory` to reference the old conversation. Two channels:

#### Option A — Inject into System Prompt

In `agent.py` `triage()`, after building the system prompt:

```python
if _has_previous_session:
    system_prompt += (
        f"\n\n## Previous conversation\n"
        f"There is a previous conversation in session `{prev_session_id}` "
        f"with {prev_message_count} turns (closed {prev_updated_at}). "
        f"Use `recall_memory(session='{prev_session_id}')` to retrieve "
        f"information from that conversation.\n"
    )
```

**Pros:** Simple, explicit, the model sees it directly.
**Cons:** Adds tokens to the system prompt (maybe ~200 chars). The previous session ID and metadata (~50 bytes) are negligible but if the user does `/session new` frequently, the injected blocks accumulate.

#### Option B — Auto-Load a Compact Session Summary

On session close, synthesize a 3-5 sentence summary of the session using the model (e.g., "Discussed deployment of olly-voice-server v1.4, finalized API docs, agreed on lazy loading"). Store it in the `sessions` table as a `summary TEXT` column. Inject the summary into the new session's system prompt.

**Pros:** Highly relevant context with minimal tokens. Even better than raw session ID — model understands what the old chat was about without needing to fire `recall_memory`.
**Cons:** Extra inference call on session close. Summary quality depends on the model.

#### Option C — Inject Previous Session ID + Rely on `recall_memory`

```python
system_prompt += (
    f"\n\n**Previous session:** `{prev_session_id}`. "
    f"Use `recall_memory('{prev_session_id}')` to search it.\n"
)
```

Minimal tokens (~80 chars). The model decides when to recall. Relies on the model being diligent about calling `recall_memory` when needed — which is consistent with the existing progressive-disclosure pattern (search for it when you need it).

**Recommendation:** Option C for minimalism, extended with a short summary if the session was long.

### C5 — Changes Required

#### Schema (`chat_history.py`)

| Change | Impact |
|---|---|
| Add `previous_session_id TEXT` to sessions table | Small — nullable, backward-compat |
| Add `session_number INTEGER` to sessions table | Small — nullable; treat NULL as session 1 |
| Add `status TEXT DEFAULT 'active'` to sessions table | Small — nullable; treat NULL as 'active' |
| Add `summary TEXT` to sessions table (optional, for Option B) | Small — nullable |
| Add `get_history_by_session(session_id)` method | New method, zero impact on existing |
| Modify `touch_session()` to set session_number automatically | Need to query MAX(session_number) for chat |
| Add `create_new_session(chat_id)` which closes the old one | New method |
| Add `get_active_session(chat_id)` — find status='active' for chat | New method |

#### `memory.py`

| Change | Impact |
|---|---|
| Modify `load(chat_id)` to only load active session | **Breaking** for single-session chats without active marker. Guard: "if only one session for chat, return all as before" |
| Modify `save()` to check `MAX_TURNS` and auto-split | New branching logic in the hot path |
| Add `active_session()` helper | New function |
| Add `session_history(session_id)` for targeted recall | New function |
| Modify `history()` to filter by session_id or chat_id | Optional param |

#### `agent.py` `triage()`

| Change | Impact |
|---|---|
| After loading history, detect previous session | New logic |
| Inject previous session ID (+ optionally summary) into system prompt | Small addition to prompt builder |
| Modify `_max_history_chars` to work within a session, not across all history | Changes context window logic |

#### `telegram_bot.py`

| Change | Impact |
|---|---|
| Add `/session new` command | New slash handler |
| Add `/session list` command to show past sessions | New slash handler |
| Update `/new` to also close the current session | Modify existing `/new` handler |

#### `api.py`

| Change | Impact |
|---|---|
| Add `POST /chat/session/new` — rotate to new session | New endpoint |
| Add `GET /chat/sessions` — list sessions for a chat | New endpoint |

### C6 — Risks and Open Questions

1. **`load()` backwards compatibility:** Changing `load()` to return only the active session would break all existing chats (they have one session without an 'active' marker). Must add a compat layer: "if only one session exists for this chat, behave as before (load all). After a split, only load the active one."

2. **`get_history_by_chat_id()` still joins across sessions:** This is the query used by `load()`. If session splitting is enabled, this JOIN would still return ALL sessions. Must change to filter by active session. But the fallback `legacy_session_hint` creates another complication.

3. **`clear_chat()` would break:** Currently `clear_chat(chat_id)` does `repo.clear_session(session)` where `session = chat_id`. With multiple sessions per chat, this only deletes one session. `clear_chat` would need a new `clear_all_for_chat(chat_id)` that deletes ALL sessions for that chat_id.

4. **`MAX_TURNS` vs `_max_history_chars`:** The char-based trim (`_max_history_chars`) and the turn-based session split (`MAX_TURNS = 300`) overlap. If sessions auto-split at 300 turns, the char-trim might only ever apply within a session. But `_max_history_chars` uses `_ctx_tokens - 4096 / 0.4` which might be 10,000+ tokens — ~25K chars — which could hold maybe 100 average turns. So the char-trim would fire before the 300-turn split. Need to coordinate: session split at M turns OR char-trim triggers archive.

5. **Char-trim = session close:** The most natural integration: when `_max_history_chars` drops old turns, mark those turns' session as 'closed' and start a new one. But `_max_history_chars` is recalculated every triage based on model context length — it's not a fixed boundary. Using `MAX_TURNS = 300` as a hard session boundary is simpler.

6. **`clear_session` only deletes by session_id = chat_id:** With session splitting, `session_id` would be UUIDs, not chat_id. `clear_session(session_id)` works as-is. But `clear_chat(chat_id)` needs a new implementation.

7. **Attachments inconsistent:** `record_attachment` stores `session_id = _session_id()` (the **process UUID**, not chat_id) but also has `chat_id = chat_id`. So attachments are linked to the process run, not the chat session. `recent_attachments(chat_id=chat_id)` queries by chat_id which works (chat_id column is set). But `clear_chat` doesn't clear attachments by chat_id. This is already a bug that session splitting would make more prominent.

8. **Auto-summary (Option B) cost:** Synthesizing a summary per session close costs one inference call. For a chat that generates 300 turns and gets auto-split, that's one extra inference. Acceptable. But if the model is unreliable (as R1 shows), the summary is noise.

9. **Ordering: session_number + created_at:** Multiple sessions for a chat need deterministic ordering. `session_number` (incremented per chat) is unambiguous. `created_at` is an alternative but UUIDs can't be sorted chronologically directly.

### C7 — Verdict

**Implementable: YES.** The foundation exists — sessions table, SQLite, `chat_id` namespacing. The changes are moderate in scope but well-contained:

- **~5 new database methods** in ChatHistoryRepository (each ~10-20 lines)
- **~3 new methods** in memory.py
- **~50 lines added** in agent.py (session detection + prompt injection)
- **~50 lines added** in telegram_bot.py (slash commands)
- **~30 lines added** in api.py (APIs)

**Timeline estimate:** 2-3 hours for a focused implementation session. Schema migration is the riskiest part (existing data).

---

## Audit D: Implementability — `/fresh` Endpoint with Factory Defaults

**Request:** Add a `fresh/` endpoint (HTTP API + slash command) that wipes the conversation history so the agent can restart to "factory defaults" with a clean conversation.

### D1 — What Already Exists

| Feature | What it does | Status |
|---|---|---|
| `/new` slash command (telegram_bot.py:1296) | `clear_chat(str(chat_id))` + `clear_chat_logs(str(chat_id))` | ✅ Exists |
| `/new` in `_BUILTIN_COMMANDS` (api.py:86) | Routes through `_run_builtin_command` | ✅ Exists |
| `/evolve fresh` (telegram_bot.py:2421) | Backup → dry-run → confirm → wipe evolution DBs only | ✅ Exists, different scope |
| `memory.clear_chat(chat_id)` | Deletes JSON window + SQLite session for chat_id | ✅ Exists |
| `memory.clear_all()` | Deletes everything (all chats, all sessions) | ✅ Exists |
| `prompt_logger.clear_chat_logs(chat_id)` | Deletes prompt logs for this chat | ✅ Exists |

**What's missing for a true `/fresh`:**
- No dedicated HTTP endpoint (`POST /fresh` or `POST /chat/fresh`)
- `clear_chat` doesn't clear attachment records for the chat
- `clear_chat` doesn't clear long-term memory entries for the chat
- `clear_chat` doesn't reset user profile (USER.md / user.json)
- No "backup → confirm → clear" pattern (Telegram `/evolve fresh` has this but conversation clearing doesn't)
- No confirmation step — `/new` fires immediately (good for quick restart, but risky)

### D2 — What a `/fresh` Should Clear

**Scope definition — conversation area only (not workspace):**

| Item | Currently cleared by `/new` | Should `/fresh` clear? |
|---|---|---|
| JSON hot window (`~/.kernel_evolving_memory_844251003.json`) | ✅ Yes (JSON unlink) | ✅ Yes |
| SQLite messages for this chat | ✅ Yes (clear_session by chat_id) | ✅ Yes |
| SQLite session row for this chat | ✅ Yes (clear_session) | ✅ Yes — but see issue D3 on safety |
| Prompt logs for this chat | ✅ Yes (clear_chat_logs) | ✅ Yes |
| Attachment records for this chat | ❌ No | ✅ Yes — delete WHERE chat_id = X |
| Long-term memory entries for this chat | ❌ No | ✅ Yes — delete by session_id + chat_id |
| Session-id file (`kernel_evolving_session_id`) | `clear()` removes it | ❌ No — that's the process UUID, shared across chats |
| USER.md / user.json | ❌ No | ❌ Optional — "factory" might mean anonymous state |
| Workspace notes (`notes/`) | ❌ No | ❌ Optional — these are long-lived artifacts |
| Workspace thoughts (`thoughts/`) | ❌ No | ❌ Same |
| Workspace files | ❌ No | ❌ Definitely no |

**Recommended scope:** Clear only conversation-history data for the specific chat: JSON window, SQLite messages + session, prompt logs, attachment records, long-term memory entries. Keep user profile, workspace files, notes, thoughts intact — they represent ongoing work, not conversation.

### D3 — Schema-Level Issues for Clearing

1. **`clear_chat(chat_id)` uses `session = chat_id` — fragile with session splitting.** Currently `clear_session(chat_id)` deletes `WHERE session_id = chat_id`. With session splitting (Audit C), each chat would have multiple session rows with UUID IDs. `clear_chat` would need to:
   ```sql
   DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE chat_id = ?)
   DELETE FROM sessions WHERE chat_id = ?
   ```
   A new `clear_chat_all(chat_id)` method in ChatHistoryRepository.

2. **Attachments use process UUID, not chat_id.** `record_attachment` sets `session_id = _session_id()` (the process UUID), not the chat_id. But it also stores `chat_id = chat_id`. So `DELETE FROM messages WHERE chat_id = ?` would delete attachments for the right chat, but the `session_id` column is the process UUID — meaning messages from OTHER chats in the same process would also have that session_id. However, attachments are filtered by `chat_id`, so `DELETE FROM attachments WHERE chat_id = ?` is correct. Currently `clear_chat` doesn't touch attachments → they survive. Add:
   ```python
   _get_repo().clear_attachments_by_chat(chat_id)
   ```

3. **Long-term memory is in a separate SQLite DB** — `long_term_memory.py` manages its own database (`memory/chat_history_{bot_name}.db`). `persist_messages` writes there. To clear it: call its `clear_by_chat(chat_id)` or similar. Currently has no per-chat clear.

4. **`clear()` (global, no args) also removes the session-id file** (`~/.kernel_evolving_session_id`). This is shared across all chats — clearing it for one chat would invalidate the process-wide session UUID. The `/fresh` endpoint must NOT call `clear()`. Only `clear_chat()`.

### D4 — Implementation Plan

#### New endpoint in `api.py`

```python
@app.post("/chat/fresh")
def chat_fresh(body: FreshBody):
    """Wipe conversation history for a chat. Factory-resets the conversation."""
    chat_id = body.chat_id or ""
    if not chat_id:
        return {"error": "chat_id is required"}

    # 1. Clear conversation memory
    import core.memory.memory as _mem
    _mem.clear_chat(chat_id)  # → modify to clear ALL sessions for chat_id

    # 2. Clear prompt logs
    import prompt_logger as _pl
    _pl.clear_chat_logs(chat_id)

    # 3. Clear attachment records
    import database.memory.chat_history as _ch
    _ch.ChatHistoryRepository(...).clear_attachments_by_chat(chat_id)
    # OR: extend clear_chat to include attachments

    # 4. Clear long-term memory entries
    from core.memory.long_term_memory import clear_by_chat
    clear_by_chat(chat_id)

    return {"status": "ok", "chat_id": chat_id, "cleared": "conversation_history"}
```

#### New slash command in `telegram_bot.py(simplified)`

```python
if text.startswith("/fresh"):
    # Optional: backup step (like /evolve fresh pattern, for safety)
    import core.memory.memory as _mem
    import prompt_logger as _pl
    _mem.clear_chat(str(chat_id))  # → extend
    _pl.clear_chat_logs(str(chat_id))
    send_message(chat_id, "✨ Fresh start — all conversation history cleared. Starting clean.")
    return
```

Currently `/new` already does most of this. `/fresh` can be its twin that also clears attachments + long-term memory, or `/new` can be enhanced to do the full clear.

#### New methods needed in `ChatHistoryRepository`

```python
def clear_attachments_by_chat(self, chat_id: str, bot=BOT_NAME) -> int:
    """Delete all attachment records for a chat. Returns count."""
    with self.connection() as conn:
        cur = conn.execute(
            "DELETE FROM attachments WHERE bot=? AND chat_id=?",
            (bot, chat_id or ""),
        )
        conn.commit()
        return cur.rowcount

def clear_all_for_chat(self, chat_id: str, bot=BOT_NAME) -> dict:
    """Wipe messages + session + attachments for a chat. Used by /fresh."""
    with self.connection() as conn:
        # Find all session IDs for this chat
        session_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM sessions WHERE chat_id=?", (chat_id,)
            ).fetchall()
        ]
        if not session_ids:
            return {"sessions": 0, "messages": 0, "attachments": 0}
        placeholders = ",".join("?" * len(session_ids))
        msg_count = conn.execute(
            f"DELETE FROM messages WHERE bot=? AND session_id IN ({placeholders})",
            (bot, *session_ids),
        ).rowcount
        att_count = conn.execute(
            "DELETE FROM attachments WHERE bot=? AND chat_id=?",
            (bot, chat_id),
        ).rowcount
        sess_count = conn.execute(
            f"DELETE FROM sessions WHERE id IN ({placeholders})",
            (*session_ids,),
        ).rowcount
        conn.commit()
        return {"sessions": sess_count, "messages": msg_count, "attachments": att_count}
```

#### Modification to `memory.clear_chat()`

Current:
```python
def clear_chat(chat_id: str) -> None:
    session = chat_id if chat_id else _session_id()
    _get_repo().clear_session(session)  # deletes only one session where id = chat_id
```

After session splitting (Audit C), `session` might be a UUID not equal to `chat_id`. Also there might be multiple sessions per chat. The current `clear_chat` only deletes one session. Fix:

```python
def clear_chat(chat_id: str) -> dict:
    """Wipe conversation history for a chat. Returns count of deleted items."""
    mem_file = _chat_memory_file(chat_id)
    if mem_file.exists():
        mem_file.unlink()
    result = _get_repo().clear_all_for_chat(chat_id)
    return result
```

### D5 — `/fresh` vs `/new` — Relationship

| Aspect | `/new` (current) | `/fresh` (proposed) |
|---|---|---|
| Clears JSON window | ✅ | ✅ |
| Clears SQLite messages | ✅ | ✅ |
| Clears SQLite session row | ✅ | ✅ |
| Clears prompt logs | ✅ (called externally) | ✅ (same) |
| Clears attachments | ❌ | ✅ |
| Clears LTM entries | ❌ | ✅ |
| Clears user profile | — | ❌ (optional, opt-in) |
| Confirmation step | ❌ (instant) | ❌ (instant) — could add optional confirm |
| Security | ❌ no auth — anyone can wipe a chat | Same — chat_id must match caller |
| Scope | Single chat | Single chat |

**Recommendation:** Enhance `/new` to do everything `/fresh` does. No need for two separate commands. The `_IDLE_BYPASS_PATHS` already routes `/new` through builtin commands. Add the missing clears (attachments + LTM) to the existing `/new` handler. The endpoint is already called "new" — it's the factory-reset button for a conversation.

If a separate `/fresh` is preferred, use it as a public-facing synonym that explicitly signals "I will destroy all conversational memory." `/new` could become an alias.

### D6 — API Layer — What Exists vs What's Needed

**Already in api.py:**
```python
# Line 82-86
"/new", "/voices", ...
```
`/new` is in `_BUILTIN_COMMANDS` → `_run_builtin_command` → `telegram_bot.handle_message(chat_id, "/new")` → works via API.

**Missing — needs to be added:**
```python
"/fresh",  # explicit synonym for /new if desired
```

**New API endpoint (alternative to routing through telegram_bot):**
```python
@app.post("/chat/fresh")
def chat_fresh(body: FreshBody):
    """POST /chat/fresh with {"chat_id": "844251003"} — wipes the chat."""
    # same logic as D4
```

**Note:** The existing `POST /message` with text="/new" already works via `_run_builtin_command`. But a dedicated `POST /chat/fresh` is cleaner for programmatic use (no parsing of text response).

### D7 — "Factory Defaults" — What Does It Mean?

The user wants to "restart the agent to factory defaults with a clean conversation." This implies:

1. **Clean slate** — no previous messages, no history the model can see or reference
2. **The model loses awareness of past interactions** — tool calls, decisions, agreements all gone
3. **User profile persists** — the agent still knows who you are (from USER.md, user.json)
4. **Workspace persists** — files, notes, thoughts, evolution state are untouched (they're not "conversation")
5. **Skills/routines persist** — not conversation-specific

This matches the proposed scope in D2: wipe conversation memory only. The agent will still be itself (with system prompt, tools, identity) but with zero chat history.

### D8 — Risks and Open Questions

1. **`clear_chat` mutex — what if the model is generating when `/fresh` fires?** The JSON window is externally deleted while the model_server process still holds a reference. SQLite is ACID — DELETE won't block a concurrent write. But the agent might save a turn after the clear → that turn reappears. Low risk in practice (user must type `/fresh` between messages).

2. **Session-id file shared across chats:** `clear()` (global, no args) also deletes `.kernel_evolving_session_id`. This is a process-level file shared by ALL chats. A `/fresh` for one chat must NOT call `clear()`. Must use `clear_chat()` which only touches the specific chat's files and SQLite rows.

3. **Attachment clearance safety:** Attachment records point to files on disk. Deleting the records doesn't delete the actual files. `clear_attachments_by_chat(chat_id)` should only delete the records, not unlink files — files may be needed by other tools. This is the safe default.

4. **Long-term memory scope:** LTM entries are tagged by `session_id` (process UUID) + `chat_id` (if provided). To clear by chat, we need to know all session_ids for that chat. The LTM module might need a `clear_by_chat(chat_id)` method that queries its own db by session_id.

5. **No confirmation pattern:** The current `/new` fires instantly. `/evolve fresh` has a backup → confirm → execute flow. For `/new`/`/fresh`, adding a mandatory confirmation (inline button) might be too much friction — the user clearly intends to wipe. But a safeguard (3-second undo window, or an undo session in a "recently deleted" table) could prevent accidental wipe.

### D9 — Verdict

**Implementable: YES, and already partially done.** The core logic (`clear_chat` + `clear_chat_logs`) exists in the `/new` command. What's missing:

| Missing piece | Effort |
|---|---|
| Extend `clear_chat` to clear attachments + LTM | 30 min |
| Add `clear_all_for_chat(chat_id)` to ChatHistoryRepository (for session-splitting compat) | 20 min |
| Add `POST /chat/fresh` endpoint in api.py | 20 min |
| Add `/fresh` slash command in telegram_bot.py (or alias to `/new`) | 5 min |
| Optionally: backup + confirm pattern | 30 min |
| **Total** | **~1.5 hours focused implementation** |

**The `/new` command already handles 80% of the `/fresh` use case.** The main change is extending `clear_chat` to cover attachments + LTM, and exposing a dedicated API endpoint.

---

## Cross-Cutting Dependencies Between Audits C and D

| Dependency | If C (session splitting) comes first | If D (fresh endpoint) comes first |
|---|---|---|
| `clear_chat(chat_id)` | Must be rewritten to `clear_all_for_chat` (multiple sessions per chat) | Must be written once, then reworked when C adds sessions |
| Attachments clearance | Needed by both — `clear_attachments_by_chat(chat_id)` can be written once and used by both | Same |
| Long-term memory clear | Needed by both — `clear_by_chat(chat_id)` reusable | Same |
| Schema changes | C adds columns to `sessions` table; D adds no new schema | If D comes first, the schema stays unchanged |

**Recommendation:** Implement D first (it's simpler, self-contained, and builds on existing `/new`). Then implement C (which needs changes to the same `clear_chat` — rewrite it for the session-splitting model).

Both can be implemented independently of the inference-pipeline fixes (R1, R2) — they operate on the data layer, not the model inference layer.