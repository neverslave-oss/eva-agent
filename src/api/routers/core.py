"""api/routers/core.py — health, chat/session, and message routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Route functions
resolve the live api module (app.state, agent/rep/mdl globals, _BUILTIN_COMMANDS)
through the _api() seam at call time so startup-set state is visible.
Behavior-identical; re-exported from api.__init__.
"""
import sys
import importlib
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

from ..schemas import MessageIn, NewSessionIn
from ..helpers import _conversations_repo, _run_builtin_command, _log_interaction

router = APIRouter()


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.get("/peers")
def get_peers():
    """Return all discovered peers with their status and capabilities. ADR-015."""
    _m = _api()
    discovery = getattr(_m.app.state, "discovery", None)
    if discovery is None:
        return {"peers": [], "discovery_enabled": False}
    return {
        "peers": discovery.get_peers(),
        "discovery_enabled": True,
        "peer_count": len(discovery.get_peers()),
    }

@router.get("/health")
def health():
    _m = _api()
    return {
        "status": "ok",
        "vram_free_mb": _m.mdl.vram_free_mb(),
        "active_replicas": len(_m.rep.active()),
        "skills": len(_m.agent._skills),
        "routines": len(_m.agent._routines),
    }

@router.get("/")
def root(request: Request):
    """Human-friendly API index for quick discovery at the base URL."""
    _m = _api()
    from infra.updater import get_current_version as _get_current_version

    base_url = str(request.base_url).rstrip("/")
    endpoints = []
    for route in _m.app.routes:
        if not isinstance(route, APIRoute):
            continue
        methods = sorted([m for m in route.methods if m not in {"HEAD", "OPTIONS"}])
        method_urls = {m: f"{base_url}{route.path}" for m in methods}
        endpoints.append({
            "path": route.path,
            "url": f"{base_url}{route.path}",
            "methods": methods,
            "method_urls": method_urls,
            "name": route.name,
        })

    endpoints.sort(key=lambda e: (e["path"], ",".join(e["methods"])))

    return {
        "name": _m.app.title,
        "status": "ok",
        "version": _get_current_version(),
        "base_url": base_url,
        "docs": f"{base_url}{_m.app.docs_url}",
        "openapi": f"{base_url}{_m.app.openapi_url}",
        "health": f"{base_url}/health",
        "peers": f"{base_url}/peers",
        "total_endpoints": len(endpoints),
        "endpoints": endpoints,
    }

@router.post("/chat/fresh")
def chat_fresh(body: MessageIn):
    """Factory-reset a chat: wipe all history, prompt logs, and attachments for chat_id."""
    _m = _api()
    chat_id = (body.chat_id or "").strip()
    if not chat_id:
        return JSONResponse({"error": "chat_id is required"}, status_code=400)
    import core.memory.memory as _mem
    import prompt_logger as _pl
    result = _mem.fresh_chat(chat_id)
    result["prompt_logs_deleted"] = _pl.clear_chat_logs(chat_id)
    return {"ok": True, "chat_id": chat_id, **result}

@router.post("/chat/session/new")
def chat_session_new(body: MessageIn):
    """Rotate to a new session for chat_id (keeps old history accessible via recall_memory)."""
    _m = _api()
    chat_id = (body.chat_id or "").strip()
    if not chat_id:
        return JSONResponse({"error": "chat_id is required"}, status_code=400)
    import core.memory.memory as _mem
    result = _mem.rotate_session(chat_id, summary=body.message or "")
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return {"ok": True, "chat_id": chat_id, **result}

@router.get("/api/sessions")
def api_sessions_list(limit: int = 100):
    """List persisted conversations (real chat-history sessions), newest first.
    _m = _api()

    Reads from the chat-history store (ChatHistoryRepository.list_sessions),
    which is backed by the sessions table that is written on every turn — not
    the ConversationsRepository metadata table, which nothing populates and
    stays empty.
    """
    try:
        from runtime_paths import CHAT_HISTORY_DB
        from database.memory import ChatHistoryRepository
        repo = ChatHistoryRepository(db_path=CHAT_HISTORY_DB)
        rows = repo.list_sessions(limit=limit)
    except Exception as exc:  # DB may be uninitialised on a fresh install
        return {"sessions": [], "error": str(exc)}
    return {"sessions": rows, "count": len(rows)}

@router.post("/api/sessions")
def api_sessions_create(body: NewSessionIn):
    """Create a new conversation record and return its id/chat_id."""
    _m = _api()
    conversation_id = str(uuid.uuid4())
    chat_id = (body.chat_id or "").strip() or f"agent-{uuid.uuid4().hex[:8]}"
    title = (body.title or "").strip() or "New session"
    repo = _conversations_repo()
    repo.upsert(conversation_id=conversation_id, chat_id=chat_id, title=title)
    created = repo.get(conversation_id)
    return {"ok": True, **created}

@router.post("/message")
def message(body: MessageIn):
    """Main entry point — triage and respond."""
    _m = _api()
    text = body.message.strip()
    if not text:
        return {"reply": ""}

    # ADR-019 Phase 2: pass user text to think-at-rest probe checker
    _think = getattr(_m.app.state, "think_at_rest", None)
    if _think is not None:
        _think.mark_active(user_text=text)

    cmd_word = text.split()[0].lower() if text.startswith("/") else ""
    # /skill_<slug> and /run_<slug> from command picker always route through bot handler
    is_picker_cmd = cmd_word.startswith("/skill_") or cmd_word.startswith("/run_")
    if (cmd_word and cmd_word in _m._BUILTIN_COMMANDS) or is_picker_cmd:
        result = _run_builtin_command(text, body.chat_id)
        _log_interaction(text, result.get("reply", ""), source="api-slash")
        return result

    # Everything else (free text + unknown slash commands) through _m.agent.triage()
    # This includes exec-backed skill commands like /markdown, /anonymize

    # Build a step_callback that streams routine step results to Telegram in real-time.
    # Each completed shell/skill step sends a short progress message so the user
    # doesn't wait in silence during long routines.
    _step_cb = None
    try:
        import services.channels.telegram_bot as _tb
        _tg_chat_id = _tb.ALLOWED_CHAT_ID
        if _tg_chat_id:
            def _step_cb(step_num: int, label: str, args_or_result=None, result: str = None):
                # Accepts both (step_num, label, result) and (step_num, tool_name, args, result_str)
                _result_val = result if result is not None else (args_or_result or "")
                if isinstance(_result_val, dict):
                    # args dict passed as 3rd arg — result is actually the 4th
                    _result_val = result or ""
                _short = str(_result_val).strip()[:1200] if _result_val else "(no output)"
                _msg = f"⚙️ *Step {step_num}*: `{label}`\n```\n{_short}\n```"
                try:
                    _tb.send_message(_tg_chat_id, _msg)
                except Exception:
                    pass
    except Exception:
        pass

    reply = _m.agent.triage(text, chat_id=body.chat_id, step_callback=_step_cb)
    _log_interaction(text, reply, source="api")

    # ADR-020: failure detection — log failed responses as evolution anchors
    try:
        import database.agent.failed_requests as _fr
        _failure_type = _fr.detect_failure(reply, text)
        if _failure_type:
            _fr.record(str(body.chat_id or "api"), text, reply, _failure_type)
    except Exception:
        pass

    return {"reply": reply}

@router.post("/message/stream")
def message_stream(body: MessageIn):
    """NDJSON streaming endpoint — same as /message but streams step events back to caller.
    _m = _api()

    Yields newline-delimited JSON lines (one per step + one final):
      {"event": "step", "step": N, "tool": "...", "result": "..."}  (per tool call)
      {"event": "done", "reply": "..."}                               (final reply)
      {"event": "error", "error": "..."}                              (on exception)

    Keeps the HTTP connection alive during long multi-step inference: triage() runs
    in a daemon thread and posts step events + the final reply into a queue;
    the sync generator drains the queue and yields each line immediately so the
    HTTP body starts flowing before inference is complete.  No asyncio conflict
    because this is a sync def (runs in uvicorn's thread-pool).
    """
    import queue as _queue
    import threading as _threading

    text = body.message.strip()
    if not text:
        return StreamingResponse(
            iter([json.dumps({"event": "done", "reply": ""}) + "\n"]),
            media_type="application/x-ndjson"
        )

    # ADR-019 Phase 2: pass user text to think-at-rest probe checker
    _think_s = getattr(_m.app.state, "think_at_rest", None)
    if _think_s is not None:
        _think_s.mark_active(user_text=text)

    cmd_word = text.split()[0].lower() if text.startswith("/") else ""
    is_picker_cmd = cmd_word.startswith("/skill_") or cmd_word.startswith("/run_")
    if (cmd_word and cmd_word in _m._BUILTIN_COMMANDS) or is_picker_cmd:
        result = _run_builtin_command(text, body.chat_id)
        _log_interaction(text, result.get("reply", ""), source="api-stream-slash")

        def _slash_generate():
            yield json.dumps({"event": "done", "reply": result.get("reply", ""), "buttons": result.get("buttons", [])}) + "\n"

        return StreamingResponse(_slash_generate(), media_type="application/x-ndjson")

    q: _queue.Queue = _queue.Queue()
    _DONE = object()

    def _step_cb(step_num: int, label: str, args_or_result=None, result: str = None):
        _result_val = result if result is not None else (args_or_result or "")
        if isinstance(_result_val, dict):
            _result_val = result or ""
        _short = str(_result_val).strip()[:1200] if _result_val else ""
        _args_val = args_or_result if result is not None else {}
        q.put({"event": "step", "step": step_num, "tool": label, "args": _args_val, "result": _short})
        try:
            import services.channels.telegram_bot as _tb
            _tg_chat_id = _tb.ALLOWED_CHAT_ID
            if _tg_chat_id:
                _msg = f"⚙️ *Step {step_num}*: `{label}`\n```\n{_short[:1200]}\n```"
                _tb.send_message(_tg_chat_id, _msg)
        except Exception:
            pass

    def _chunk_cb(chunk: str):
        if chunk:
            q.put({"event": "chunk", "text": str(chunk)})

    def _run():
        try:
            reply = _m.agent.triage(text, chat_id=body.chat_id, step_callback=_step_cb, chunk_callback=_chunk_cb)
            _log_interaction(text, reply, source="api-stream")
            q.put({"event": "done", "reply": reply})
        except Exception as exc:
            q.put({"event": "error", "error": str(exc)})
        finally:
            q.put(_DONE)

    _threading.Thread(target=_run, daemon=True).start()

    def _generate():
        while True:
            try:
                item = q.get(timeout=600)
            except _queue.Empty:
                yield json.dumps({"event": "error", "error": "inference timeout (600s)"}) + "\n"
                break
            if item is _DONE:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(_generate(), media_type="application/x-ndjson")
