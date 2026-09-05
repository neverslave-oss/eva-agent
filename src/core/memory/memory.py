"""
memory.py — Two-tier chat persistence for Kernel-Evolving.

Tier 1 (hot):  ~/.kernel_evolving_memory.json        — sliding context window (last N turns, fast load)
Tier 2 (cold): ~/.kernel-evolving/workspace/memory/chat_history_evolving.db — SQLite, every turn, permanent

On load()  : return JSON window; if empty, seed from SQLite.
On save()  : append new messages to SQLite, refresh JSON window.
On clear() : wipe JSON window only (SQLite history is permanent).
"""
import json
import sqlite3
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path
from runtime_paths import MEMORY_DIR, CHAT_HISTORY_DB
from core.memory.long_term_memory import persist_messages
from database.memory import ChatHistoryRepository

# ── Paths ─────────────────────────────────────────────────────────────────────
# The hot-memory filename is scoped per chat id so different chats don't share
# a sliding context window. The chat id is read from the environment (set in
# `.env` via KERNEL_EVO_TELEGRAM_CHAT_ID); when unset/placeholder we fall back
# to a neutral "default" scope so the file name never leaks a real user id.
_CHAT_SCOPE = os.environ.get("KERNEL_EVO_TELEGRAM_CHAT_ID", "").strip()
if not _CHAT_SCOPE or _CHAT_SCOPE in ("your_chat_id_here", "0", "None"):
    _CHAT_SCOPE = "default"
MEMORY_FILE = MEMORY_DIR / f"hot_memory_{_CHAT_SCOPE}.json"
DB_DIR      = MEMORY_DIR
# Allow test isolation: set KERNEL_MEMORY_DB to redirect to a temp path
_db_override = os.environ.get("KERNEL_MEMORY_DB", "")
DB_FILE     = Path(_db_override) if _db_override else CHAT_HISTORY_DB
if not _db_override:
    DB_DIR.mkdir(parents=True, exist_ok=True)


def _session_file() -> Path:
    return DB_FILE.parent / ".kernel_evolving_session_id"

# ── Config ────────────────────────────────────────────────────────────────────
MAX_TURNS   = 300   # pairs kept in JSON window
BOT_NAME    = "kernel-evolving"

# ── Session ID ────────────────────────────────────────────────────────────────
# One session per process run — stable across the life of the bot instance.
_SESSION_ID: str | None = None

def _session_id() -> str:
    global _SESSION_ID
    if _SESSION_ID is None:
        try:
            session_file = _session_file()
            if session_file.exists():
                _SESSION_ID = session_file.read_text().strip()
        except Exception:
            _SESSION_ID = None
        if not _SESSION_ID:
            _SESSION_ID = str(uuid.uuid4())
            try:
                session_file = _session_file()
                session_file.parent.mkdir(parents=True, exist_ok=True)
                session_file.write_text(_SESSION_ID)
            except Exception:
                pass
    return _SESSION_ID


# ── SQLite via ChatHistoryRepository ─────────────────────────────────────────
# Module-level repo instance; re-created if DB_FILE is patched (test isolation)
_repo: ChatHistoryRepository | None = None
_repo_db_file: Path | None = None


_repo: "ChatHistoryRepository | None" = None
_repo_db_file: "Path | None" = None


def _get_conn() -> sqlite3.Connection:
    """Connection factory. Retained for backward-compat; tests may patch this symbol."""
    return ChatHistoryRepository(DB_FILE).connection()


def _get_repo() -> ChatHistoryRepository:
    global _repo, _repo_db_file
    if _repo is None or _repo_db_file != DB_FILE:
        _repo = ChatHistoryRepository(DB_FILE)
        _repo_db_file = DB_FILE
    return _repo


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """No-op: schema is ensured by ChatHistoryRepository.__init__."""
    pass


def _touch_session(conn: sqlite3.Connection, session_id: str, chat_id: str) -> None:
    _get_repo().touch_session(session_id, chat_id)


def _append_to_db(conn: sqlite3.Connection, messages: list[dict], session_id: str) -> None:
    """Insert only messages that aren't already stored (idempotent)."""
    _get_repo().append_messages(messages, session_id)


# ── Public API ────────────────────────────────────────────────────────────────

def _chat_memory_file(chat_id: str) -> Path:
    """Return per-chat JSON window path. Falls back to global file when chat_id is empty."""
    if not chat_id:
        return MEMORY_FILE
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in chat_id)[:64]
    return MEMORY_FILE.parent / f".kernel_evolving_memory_{safe}.json"


_ERROR_MARKERS = (
    "[model_server error]",
    "[model_client error]",
    "[model_server timeout]",
    "VRAM guard:",
    "(HF generate error:",
    # Tool execution errors from core/tools.py — these are never valid agent replies
    "(error: skill '",
    "(error: routine '",
    "(error: file not found:",
    "(error: write_file",
    "(error: web_search failed:",
    "(error: browser_use",
    "(error: exec_shell",
    "(vLLM error:",
    "(adapter error:",
    "(max steps reached)",
)


_USER_POLLUTION_MARKERS = (
    "[Previous step output]:",
    "[Previous attempt failed.",
    "[Previous attempt failed. Critic feedback:",
)


def _clean_user_content(content: str) -> str:
    """Truncate user content at the first tool-retry pollution marker.

    The tool-retry loop concatenates "[Previous step output]:" and
    "[Previous attempt failed. Critic feedback:" markers into the next
    user turn.  Strip everything from the first marker onward.
    """
    for marker in _USER_POLLUTION_MARKERS:
        idx = content.find(marker)
        if idx != -1:
            content = content[:idx]
    return content.rstrip()


def _sanitise(msgs: list) -> list:
    """Strip obviously broken turns that would corrupt context:
    - assistant turns containing model errors (VRAM guard, server errors)
    - user turns polluted by tool-retry critic feedback markers
    - consecutive user turns with no assistant reply between them
      (keeps only the LAST one in each run, which is most likely the intended message)
    """
    # 1. Remove error assistant turns
    cleaned = [
        m for m in msgs
        if not (
            m.get("role") == "assistant"
            and any(marker in str(m.get("content", "")) for marker in _ERROR_MARKERS)
        )
    ]

    # 1b. Clean user turns — strip critic-retry pollution markers
    cleaned2 = []
    for m in cleaned:
        if m.get("role") == "user":
            raw = str(m.get("content", ""))
            scrubbed = _clean_user_content(raw)
            if scrubbed == raw:
                # Not polluted — keep as-is
                cleaned2.append(m)
            elif len(scrubbed) < 3:
                # Became tiny after stripping — drop orphaned turn
                continue
            else:
                cleaned2.append({**m, "content": scrubbed})
        else:
            cleaned2.append(m)

    # 2. Collapse consecutive user turns — keep only last in each run
    result = []
    i = 0
    while i < len(cleaned2):
        m = cleaned2[i]
        if m.get("role") == "user":
            # Scan forward to find last consecutive user turn
            j = i
            while j + 1 < len(cleaned2) and cleaned2[j + 1].get("role") == "user":
                j += 1
            result.append(cleaned2[j])  # keep only the last user turn in the run
            i = j + 1
        else:
            result.append(m)
            i += 1

    return result


def load(chat_id: str = "") -> list[dict]:
    """Return last MAX_TURNS message pairs for the context window.

    Order of precedence:
    1. SQLite — full session history across ALL sessions sharing this chat_id
       (source of truth).  Uses get_history_by_chat_id which JOINs across
       current + legacy session rows to survive session-id migrations.
    2. JSON window for this chat_id (hot cache / recovery path)

    Sanitisation is applied before returning: model-error assistant turns and
    consecutive orphaned user turns are stripped so the model never sees a
    broken context that causes looping/stuck behaviour.
    """
    mem_file = _chat_memory_file(chat_id)
    session = chat_id if chat_id else _session_id()

    # Source of truth: full SQLite history across ALL sessions for this chat.
    # Uses get_history_by_chat_id which queries sessions by chat_id column
    # (surviving session-id migrations / UUID changes).
    try:
        repo = _get_repo()
        if chat_id:
            # Check for a rotated (session-split) chat: load only the active session.
            active = repo.get_active_session(chat_id)
            if active and active["id"] != chat_id and (active.get("session_number") or 1) > 1:
                # Multiple sessions exist — use only the active one (not the full history).
                msgs = repo.get_history_by_session(active["id"])
            else:
                # Single session or legacy layout: read legacy hint for pre-migration turns.
                legacy_hint = ""
                try:
                    sf = _session_file()
                    if sf.exists():
                        legacy_hint = sf.read_text().strip()
                except Exception:
                    pass
                msgs = repo.get_history_by_chat_id(chat_id, legacy_session_hint=legacy_hint)
        else:
            repo.touch_session(session, "")
            msgs = repo.get_history(session)
        if msgs:
            sanitised = _sanitise(msgs)
            # Warm up the JSON window with sanitised content
            mem_file.write_text(json.dumps({'messages': sanitised[-(MAX_TURNS * 2):]}, indent=2))
            return sanitised
        # msgs is empty or falsy — return empty
        return []
    except Exception:
        try:
            data = json.loads(mem_file.read_text())
            msgs = data.get("messages", [])
            return _sanitise(msgs)
        except Exception:
            return []


# ── Session rotation threshold ─────────────────────────────────────────────
SESSION_SPLIT_TURNS = 200  # auto-rotate after this many messages in one session


def rotate_session(chat_id: str, summary: str = "") -> dict:
    """Explicitly rotate to a new session for chat_id. Returns rotation info dict."""
    if not chat_id:
        return {"error": "chat_id required"}
    repo = _get_repo()
    new_id = str(uuid.uuid4())
    result = repo.rotate_session(chat_id, new_id, summary=summary)
    # Clear JSON hot window so next load() reads the fresh session
    mem_file = _chat_memory_file(chat_id)
    if mem_file.exists():
        try:
            mem_file.unlink()
        except Exception:
            pass
    print(f"[memory] rotate_session: chat_id={chat_id!r} → new={new_id} (#{result['session_number']})")
    return result


def get_previous_session_hint(chat_id: str) -> dict:
    """Return metadata about the most recent closed session for system prompt injection."""
    if not chat_id:
        return {}
    try:
        repo = _get_repo()
        active = repo.get_active_session(chat_id)
        if not active or not active.get("previous_session_id"):
            return {}
        return {
            "previous_session_id": active["previous_session_id"],
            "session_number": active.get("session_number", 1),
        }
    except Exception:
        return {}



def save(messages: list[dict], chat_id: str = "") -> None:
    """Persist messages. Appends new turns to SQLite; refreshes JSON window.
    Auto-rotates the session when SESSION_SPLIT_TURNS is reached.
    """
    trimmed = messages[-(MAX_TURNS * 2):]
    mem_file = _chat_memory_file(chat_id)
    session = chat_id if chat_id else _session_id()

    # JSON window (hot cache)
    mem_file.write_text(json.dumps({"messages": trimmed}, indent=2))

    # SQLite (permanent record) — use chat_id as session_id when provided
    try:
        repo = _get_repo()
        # For chat_id-keyed sessions, use the active session UUID (not chat_id directly
        # after the first rotation, since chat_id-as-session-id is the legacy single-session path)
        if chat_id:
            active = repo.get_active_session(chat_id)
            if active:
                session = active["id"]
                # Auto-rotate if this session has grown past the threshold
                if (active.get("message_count") or 0) >= SESSION_SPLIT_TURNS:
                    new_id = str(uuid.uuid4())
                    repo.rotate_session(chat_id, new_id)
                    session = new_id
                    if mem_file.exists():
                        mem_file.unlink()
                    print(f"[memory] auto-rotated session for chat_id={chat_id!r} → {new_id}")
            else:
                repo.touch_session(session, chat_id)
        else:
            repo.touch_session(session, chat_id)
        repo.append_messages(messages, session)
    except Exception as e:
        print(f"[memory] SQLite write failed: {e}")

    # Long-term memory mirror (markdown + sqlite index)
    try:
        persist_messages(messages, chat_id=chat_id, session_id=session)
    except Exception as e:
        print(f"[memory] long-term memory write failed: {e}")


def clear() -> None:
    """Clear the JSON window. SQLite history is preserved."""
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
    session_file = _session_file()
    if session_file.exists():
        try:
            session_file.unlink()
        except Exception:
            pass
    global _SESSION_ID
    _SESSION_ID = None


def clear_chat(chat_id: str) -> None:
    """Clear JSON window AND SQLite history for a specific chat session."""
    mem_file = _chat_memory_file(chat_id)
    session = chat_id if chat_id else _session_id()
    if mem_file.exists():
        mem_file.unlink()
    try:
        _get_repo().clear_session(session)
    except Exception as e:
        print(f"[memory] SQLite clear_chat failed: {e}")


def fresh_chat(chat_id: str) -> dict:
    """Factory-reset a chat: wipe ALL sessions, messages, and attachments for chat_id.

    Broader than clear_chat() — removes every session row tied to this chat_id
    (not just the current session), plus attachment records. Safe to call even if
    session splitting is in use (multiple session rows per chat).
    Returns a summary dict with counts.
    """
    if not chat_id:
        return {"error": "chat_id required"}
    mem_file = _chat_memory_file(chat_id)
    if mem_file.exists():
        try:
            mem_file.unlink()
        except Exception:
            pass
    repo = _get_repo()
    deleted_msgs = 0
    deleted_attachments = 0
    try:
        deleted_msgs = repo.clear_all_for_chat(chat_id)
    except Exception as e:
        print(f"[memory] fresh_chat: clear_all_for_chat failed: {e}")
    try:
        deleted_attachments = repo.clear_attachments_by_chat(chat_id)
    except Exception as e:
        print(f"[memory] fresh_chat: clear_attachments_by_chat failed: {e}")
    print(f"[memory] fresh_chat: chat_id={chat_id!r} cleared {deleted_msgs} messages, {deleted_attachments} attachments")
    return {"messages_deleted": deleted_msgs, "attachments_deleted": deleted_attachments}


def clear_all() -> None:
    """Wipe everything — JSON window AND SQLite history for this bot."""
    clear()
    # Also wipe all per-chat JSON windows
    try:
        import glob
        for f in glob.glob(str(MEMORY_FILE.parent / ".kernel_evolving_memory_*.json")):
            try:
                __import__('os').remove(f)
            except Exception:
                pass
    except Exception:
        pass
    try:
        _get_repo().clear_all()
    except Exception as e:
        print(f"[memory] SQLite clear failed: {e}")


def history(limit: int = 100, chat_id: str = "") -> list[dict]:
    """Return up to `limit` most recent turns from SQLite.

    When chat_id is provided, queries across ALL sessions sharing that chat_id
    (including legacy UUID sessions).  This survives session-id migrations.
    """
    try:
        repo = _get_repo()
        if chat_id:
            legacy_hint = ""
            try:
                sf = _session_file()
                if sf.exists():
                    legacy_hint = sf.read_text().strip()
            except Exception:
                pass
            return repo.get_history_by_chat_id(chat_id, legacy_session_hint=legacy_hint)[-limit:]
        return repo.get_recent_user_messages(limit=limit)
    except Exception:
        return []


# ── Attachment memory ────────────────────────────────────────────────────────

def record_attachment(
    *,
    kind: str,
    local_path: str,
    original_name: str = "",
    mime_type: str = "",
    caption: str = "",
    chat_id: str = "",
) -> int | None:
    """Persist one attachment record.  Returns the new row id or None on error."""
    return _get_repo().record_attachment(
        kind=kind, local_path=local_path, session_id=_session_id(),
        chat_id=chat_id, original_name=original_name,
        mime_type=mime_type, caption=caption,
    )


def recent_attachments(limit: int = 5, chat_id: str = "", max_age_seconds: int | None = None) -> list[dict]:
    """Return the most recent attachment records, optionally filtered by chat.

    max_age_seconds: if set, only attachments created within this window are
    returned. Stale attachments (e.g. a photo uploaded days ago) are excluded so
    they are never surfaced as "recent" context for a later turn.
    """
    try:
        atts = _get_repo().recent_attachments(chat_id=chat_id, limit=limit)
    except Exception as e:
        print(f"[memory] recent_attachments failed: {e}")
        return []
    if max_age_seconds is None:
        return atts
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - max_age_seconds
    filtered = []
    for a in atts:
        created = a.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created).timestamp()
        except Exception:
            # Unparseable timestamp: keep it (cannot prove staleness).
            filtered.append(a)
            continue
        if ts >= cutoff:
            filtered.append(a)
    return filtered


def attachment_context_block(chat_id: str = "", limit: int = 3, max_age_seconds: int | None = None) -> str:
    """Build a compact context string listing recent attachments for prompt injection.

    max_age_seconds: if set, only attachments created within this window are
    listed, so stale attachments are not injected as if freshly uploaded.
    """
    atts = recent_attachments(limit=limit, chat_id=chat_id, max_age_seconds=max_age_seconds)
    if not atts:
        return ""
    lines = ["Recent uploaded files (most recent first):"]
    for a in atts:
        name = a["original_name"] or os.path.basename(a["local_path"])
        exists = "✓" if os.path.exists(a["local_path"]) else "✗ (deleted)"
        lines.append(
            f"  [{a['kind']}] {name} → {a['local_path']} "
            f"({a['mime_type'] or 'unknown type'}) {exists}  @ {a['created_at'][:19]}"
        )
        if a["caption"]:
            lines.append(f"    Caption: {a['caption'][:120]}")
    return "\n".join(lines)


# ── Completion / critique guard ───────────────────────────────────────────────

_ATTACHMENT_KEYWORDS = (
    "pdf", "document", "file", "attachment", "uploaded", "sent you",
    "the file", "this file", "that file", "the doc", "the pdf",
)


def attachment_guard(user_text: str, reply_text: str, chat_id: str = "", max_age_seconds: int | None = None) -> dict:
    """Lightweight guard: checks whether the reply likely addressed a recent attachment.

    Returns:
        {"ok": True}  — either no attachment was implied, or the reply references it.
        {"ok": False, "reason": str, "recent": list}  — reply missed a relevant attachment.
    """
    user_lower = user_text.lower()
    # Only fire when the user message implies an attachment
    if not any(kw in user_lower for kw in _ATTACHMENT_KEYWORDS):
        return {"ok": True}

    atts = recent_attachments(limit=3, chat_id=chat_id, max_age_seconds=max_age_seconds)
    if not atts:
        return {"ok": True}

    reply_lower = reply_text.lower()
    for a in atts:
        name = (a["original_name"] or os.path.basename(a["local_path"])).lower()
        path_lower = a["local_path"].lower()
        # Check if reply mentions the file name or its path
        if name and name in reply_lower:
            return {"ok": True}
        if path_lower and path_lower in reply_lower:
            return {"ok": True}
        # Accept if reply just mentions the workspace documents folder
        if "documents" in reply_lower or "/workspace/" in reply_lower:
            return {"ok": True}

    return {
        "ok": False,
        "reason": "Reply does not appear to reference any recently uploaded attachment.",
        "recent": atts,
    }


# Keywords that suggest the user expects a file/artifact to be produced
_ARTIFACT_REQUEST_KEYWORDS = (
    "write", "create", "generate", "save", "export", "produce", "make",
    "summarise", "summarize", "extract", "convert", "output",
)


def artifact_check(
    user_text: str,
    reply_text: str,
    workspace: str = "",
    since_seconds: int = 90,
) -> dict:
    """Check whether the agent actually produced an artifact when one was requested.

    Returns:
        {"ok": True}  — either no artifact was requested, or one was found.
        {"ok": False, "reason": str}  — artifact requested but none found.
    """
    import glob
    import time

    lower = user_text.lower()
    if not any(kw in lower for kw in _ARTIFACT_REQUEST_KEYWORDS):
        return {"ok": True}  # no artifact requested

    ws = workspace or str(MEMORY_DIR.parent)
    now = time.time()
    recent = [
        f for f in glob.glob(f"{ws}/**", recursive=True)
        if os.path.isfile(f) and (now - os.path.getmtime(f)) < since_seconds
    ]
    if recent:
        return {"ok": True}

    # Check if reply mentions a file path or creation
    reply_lower = reply_text.lower()
    if any(kw in reply_lower for kw in ("saved to", "written to", "created at", "file:", "/workspace/")):
        return {"ok": True}  # agent claims it wrote something — trust it

    return {
        "ok": False,
        "reason": "User requested artifact production but no new workspace file was detected.",
    }


def show() -> str:
    """Human-readable summary of the current context window."""
    msgs = load()
    if not msgs:
        return "  No memory stored."
    lines = []
    for m in msgs:
        role = "You" if m["role"] == "user" else "Evo"
        content = m["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        lines.append(f"  \033[92m{role}:\033[0m {str(content)[:120]}")
    return "\n".join(lines)
