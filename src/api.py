"""
api.py — FastAPI server. Local + mesh endpoints.
Port 8779 by default.
"""
from fastapi import FastAPI, Request, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.discovery import AgentDiscovery
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel
from typing import Optional
import yaml
import infra.bootstrap as _bootstrap
import asyncio
import os, json, time, uuid
import logging as _logging
from runtime_paths import LOGS_DIR, load_config as _load_config
from core.voice_activity import voice_activity
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        _logging.StreamHandler(),  # stdout (captured by Docker/uvicorn)
    ]
)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lazily initialized in startup() to avoid import-time circular failures.
agent = None
rep = None
mdl = None

app = FastAPI(title="Kernel Evolving", version="0.2.0")

# Allow requests from the desktop app (NativePHP serves on random localhost ports)
# and any LAN client. Credentials are not used so wildcard origin is safe here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_IDLE_BYPASS_PATHS = {
    "/evolution", "/evolution/state", "/evolution/stream",
    "/health", "/version", "/system", "/thoughts", "/thoughts/today",
    "/evolution/dashboard", "/", "/peers",
    "/skills", "/routines", "/replica/active", "/evolution/trajectories",
    "/debug/prompt-logs", "/debug/prompt-log", "/debug/trajectories", "/debug/chat-history",
    "/debug/fields",
    "/provider", "/provider/available",
    "/memory/files", "/memory/file", "/memory/stats", "/workspace/tree",
    "/memory/file/rename", "/memory/file/new",
    "/sqlite/tables", "/sqlite/table", "/sqlite/row", "/sqlite/table/data", "/sqlite/db/data",
    "/voice/status",
}

@app.middleware("http")
async def _activity_middleware(request: Request, call_next):
    """Mark the think-at-rest idle detector active on real user requests only.
    Monitoring/polling paths are excluded so idle threshold can be reached."""
    think = getattr(app.state, "think_at_rest", None)
    if think is not None and request.url.path not in _IDLE_BYPASS_PATHS:
        think.mark_active()
    return await call_next(request)

_cfg = {}

# ── Interaction logger ──────────────────────────────────────────────────────
_LOG_DIR  = str(LOGS_DIR)
_LOG_FILE = os.path.join(_LOG_DIR, "kernel_calls.jsonl")
os.makedirs(_LOG_DIR, exist_ok=True)

def _log_interaction(prompt: str, reply: str, source: str = "api", meta: dict | None = None) -> None:
    """Append one JSONL record to the interaction log (non-blocking best-effort)."""
    try:
        record = {
            "id":        str(uuid.uuid4()),
            "timestamp": time.time(),
            "source":    source,
            "prompt":    prompt,
            "reply":     reply,
            "meta":      meta or {},
        }
        with open(_LOG_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        print(f"[logger] warning: could not write interaction log: {exc}")
# ────────────────────────────────────────────────────────────────────────────


_BUILTIN_COMMANDS = {
    "/start", "/help", "/skills", "/routines", "/run", "/skill",
    "/status", "/verbose", "/packages", "/search", "/install",
    "/clone", "/private_repo", "/update", "/restart", "/rollback",
    "/replica", "/workspaces", "/system", "/version", "/provider",
    "/new", "/fresh", "/session", "/voices", "/models", "/thoughts", "/evolve",
    "/init",
}


def _run_builtin_command(text: str, chat_id: str) -> dict:
    """Execute a Telegram-style slash command and capture text/button output."""
    import services.channels.telegram_bot as _tb

    captured_texts: list[str] = []
    captured_buttons: list = []
    _orig_send = _tb.send_message
    _orig_send_buttons = _tb.send_buttons

    def _capture(_chat_id, msg, **kwargs):
        captured_texts.append(str(msg))

    def _capture_buttons(_chat_id, msg, buttons, **kwargs):
        captured_texts.append(str(msg))
        captured_buttons.append(buttons)

    _tb.send_message = _capture
    _tb.send_buttons = _capture_buttons
    try:
        _tb.handle_message(chat_id or _tb.ALLOWED_CHAT_ID or "api", text)
    finally:
        _tb.send_message = _orig_send
        _tb.send_buttons = _orig_send_buttons

    reply = "\n".join(captured_texts) if captured_texts else ""
    buttons = captured_buttons[-1] if captured_buttons else []
    return {"reply": reply, "buttons": buttons}


class MessageIn(BaseModel):
    message: str
    chat_id: str = ""  # optional session key for conversation isolation

class TaskIn(BaseModel):
    role: str
    task: str
    adapter_path: Optional[str] = None

class NamedReplicaIn(BaseModel):
    name: str
    role: str = "custom"
    brief_path: Optional[str] = None
    custom_prompt: Optional[str] = None
    workspace: Optional[str] = None
    tools_enabled: bool = False
    output_path: Optional[str] = None
    input_path: Optional[str] = None
    adapter_path: Optional[str] = None
    slot: Optional[str] = None  # named model slot (e.g. "audio"); None = primary

class PipelineStage(BaseModel):
    name: str
    role: str = "custom"
    brief: str
    task: str = "Begin your work."
    tools: bool = False
    workspace: Optional[str] = None
    input_from: Optional[str] = None   # name of previous stage whose output to inject
    output_path: Optional[str] = None  # write stage result to this path
    adapter_path: Optional[str] = None

class PipelineIn(BaseModel):
    pipeline: str = "default"
    stages: list[PipelineStage]

class ReplicaMessageIn(BaseModel):
    message: str


class BackupRequest(BaseModel):
    description: str = ""
    full: bool = True


class InitRequest(BaseModel):
    force: bool = False
    description: str = ""


class FreshRequest(BaseModel):
    description: str = ""
    keep_ecosystem: bool = True
    dry_run: bool = False
    # confirmation handled via inline buttons


@app.on_event("startup")
async def startup():
    global _cfg, agent, rep, mdl
    import core.agent as _agent
    import core.replica.replica as _rep
    import core.inference.model as _mdl
    import infra.setup as _setup
    from runtime_paths import ensure_runtime_dirs
    from infra.workspace_migration import migrate_runtime_workspace
    _setup.setup_workspace()          # no-op if already exists
    ensure_runtime_dirs()
    migrate_runtime_workspace()
    _setup.refresh_identity_files()   # always sync AGENTS.md + SOUL.md from repo

    # Inject persisted env overrides (e.g. provider API keys from the desktop app
    # via /config/env) into os.environ before the agent initializes.
    try:
        from core.env_overrides import load_env_overrides
        load_env_overrides()
    except Exception as _eoe:
        print(f"[api] WARNING: load_env_overrides failed: {_eoe}", flush=True)

    agent = _agent
    rep = _rep
    mdl = _mdl

    # Use the env-expanding loader so ${VAR} references in config.yaml resolve
    # from the environment (e.g. collective_memory.url, vision eye bases).
    _cfg = _load_config(os.path.join(_BASE, "config.yaml"))
    # Expand ~ in well-known path keys so config stays portable
    for _key in ("olly_workspace", "workspace", "skills_dir", "private_skills_dir",
                 "routines_dir", "embedding_model_path"):
        if _key in _cfg and isinstance(_cfg[_key], str):
            _cfg[_key] = os.path.expanduser(_cfg[_key])
    if "paths" in _cfg and isinstance(_cfg["paths"], dict):
        _cfg["paths"] = {k: os.path.expanduser(v) if isinstance(v, str) else v
                        for k, v in _cfg["paths"].items()}
    # Bootstrap skills and routines from ecosystem repos
    counts = _bootstrap.bootstrap(_cfg)
    if counts["skills"] or counts["routines"]:
        print(f"[bootstrap] Added {counts['skills']} skills, {counts['routines']} routines from ecosystem")

    _config_path = os.path.join(_BASE, "config.yaml")
    mdl.load(_config_path)
    agent.init(_config_path)
    print(f"[api] Kernel ready on :{_cfg['api']['port']}")
    # Start agent auto-discovery (ADR-015)
    _disc_cfg = _cfg.get("discovery", {})
    if _disc_cfg.get("enabled", True):
        discovery = AgentDiscovery(_cfg, self_port=_cfg.get("api", {}).get("port", 8779))
        discovery.start()
        app.state.discovery = discovery
        if _disc_cfg.get("mdns", False):
            discovery.start_mdns()
        print(f"[api] Agent discovery started (port-scan={_disc_cfg.get('port_scan', True)}, mdns={_disc_cfg.get('mdns', False)})", flush=True)
    else:
        app.state.discovery = None
        print("[api] Agent discovery disabled", flush=True)
    # Start goal discovery background thread if evolution is enabled
    if os.environ.get("EVOLUTION_ENABLED", "false").lower() == "true":
        import core.pipelines.goal_discovery as goal_discovery
        # FIX #1: inject infer_fn so background evolution paths use the capability verifier
        goal_discovery.set_infer_fn(mdl.infer)
        goal_discovery.start_discovery_thread(_cfg, skills_dir=os.path.expanduser(
            _cfg.get("skills_dir", "./skills")
        ))
        print("[api] Goal discovery thread started")
    # Start Telegram bot in-process (model already loaded above)
    import services.channels.telegram_bot as telegram_bot
    telegram_bot.start_bot_thread()
    # Start Think-at-Rest if enabled
    from services.thought_engine import EvolvingThinkAtRest
    _think = EvolvingThinkAtRest(config=_cfg)
    if _cfg.get("thinking", {}).get("enabled", False):
        _think.start()
    app.state.think_at_rest = _think


    # Resolve voice server URL: env > config > empty (disabled)
    global _VOICE_SERVER_URL
    if not _VOICE_SERVER_URL:
        _VOICE_SERVER_URL = (_cfg.get("services", {}).get("voice_server", {}).get("url") or "").strip()
    if _VOICE_SERVER_URL:
        print(f"[api] Voice server: {_VOICE_SERVER_URL}", flush=True)
    else:
        print("[api] Voice server: not configured (voice features disabled)", flush=True)


@app.get("/peers")
def get_peers():
    """Return all discovered peers with their status and capabilities. ADR-015."""
    discovery = getattr(app.state, "discovery", None)
    if discovery is None:
        return {"peers": [], "discovery_enabled": False}
    return {
        "peers": discovery.get_peers(),
        "discovery_enabled": True,
        "peer_count": len(discovery.get_peers()),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vram_free_mb": mdl.vram_free_mb(),
        "active_replicas": len(rep.active()),
        "skills": len(agent._skills),
        "routines": len(agent._routines),
    }


@app.get("/")
def root(request: Request):
    """Human-friendly API index for quick discovery at the base URL."""
    from infra.updater import get_current_version as _get_current_version

    base_url = str(request.base_url).rstrip("/")
    endpoints = []
    for route in app.routes:
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
        "name": app.title,
        "status": "ok",
        "version": _get_current_version(),
        "base_url": base_url,
        "docs": f"{base_url}{app.docs_url}",
        "openapi": f"{base_url}{app.openapi_url}",
        "health": f"{base_url}/health",
        "peers": f"{base_url}/peers",
        "total_endpoints": len(endpoints),
        "endpoints": endpoints,
    }


@app.post("/chat/fresh")
def chat_fresh(body: MessageIn):
    """Factory-reset a chat: wipe all history, prompt logs, and attachments for chat_id."""
    chat_id = (body.chat_id or "").strip()
    if not chat_id:
        return JSONResponse({"error": "chat_id is required"}, status_code=400)
    import core.memory.memory as _mem
    import prompt_logger as _pl
    result = _mem.fresh_chat(chat_id)
    result["prompt_logs_deleted"] = _pl.clear_chat_logs(chat_id)
    return {"ok": True, "chat_id": chat_id, **result}


@app.post("/chat/session/new")
def chat_session_new(body: MessageIn):
    """Rotate to a new session for chat_id (keeps old history accessible via recall_memory)."""
    chat_id = (body.chat_id or "").strip()
    if not chat_id:
        return JSONResponse({"error": "chat_id is required"}, status_code=400)
    import core.memory.memory as _mem
    result = _mem.rotate_session(chat_id, summary=body.message or "")
    if "error" in result:
        return JSONResponse(result, status_code=400)
    return {"ok": True, "chat_id": chat_id, **result}


@app.post("/message")
def message(body: MessageIn):
    """Main entry point — triage and respond."""
    text = body.message.strip()
    if not text:
        return {"reply": ""}

    # ADR-019 Phase 2: pass user text to think-at-rest probe checker
    _think = getattr(app.state, "think_at_rest", None)
    if _think is not None:
        _think.mark_active(user_text=text)

    cmd_word = text.split()[0].lower() if text.startswith("/") else ""
    # /skill_<slug> and /run_<slug> from command picker always route through bot handler
    is_picker_cmd = cmd_word.startswith("/skill_") or cmd_word.startswith("/run_")
    if (cmd_word and cmd_word in _BUILTIN_COMMANDS) or is_picker_cmd:
        result = _run_builtin_command(text, body.chat_id)
        _log_interaction(text, result.get("reply", ""), source="api-slash")
        return result

    # Everything else (free text + unknown slash commands) through agent.triage()
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

    reply = agent.triage(text, chat_id=body.chat_id, step_callback=_step_cb)
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


@app.post("/message/stream")
def message_stream(body: MessageIn):
    """NDJSON streaming endpoint — same as /message but streams step events back to caller.

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
    _think_s = getattr(app.state, "think_at_rest", None)
    if _think_s is not None:
        _think_s.mark_active(user_text=text)

    cmd_word = text.split()[0].lower() if text.startswith("/") else ""
    is_picker_cmd = cmd_word.startswith("/skill_") or cmd_word.startswith("/run_")
    if (cmd_word and cmd_word in _BUILTIN_COMMANDS) or is_picker_cmd:
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
            reply = agent.triage(text, chat_id=body.chat_id, step_callback=_step_cb, chunk_callback=_chunk_cb)
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


@app.get("/debug/prompt-logs")
def debug_prompt_logs(limit: int = 20, chat_id: str = ""):
    """Return recent prompt log entries for the UI inspector."""
    import prompt_logger as _pl
    return {"logs": _pl.query_recent(limit=limit, chat_id=chat_id)}


@app.get("/debug/prompt-log/{entry_id}")
def debug_prompt_log(entry_id: int):
    """Return one prompt log entry with the full prompt and history payload."""
    import prompt_logger as _pl

    def _model_messages(history: list | None, user_message: str = "") -> list[dict]:
        turns = []
        for item in (history or []):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "tool", "system"}:
                continue
            turns.append({"role": role, "content": str(item.get("content", ""))})

        user_message = str(user_message or "")
        if user_message:
            last = turns[-1] if turns else {}
            if not (last.get("role") == "user" and last.get("content") == user_message):
                turns.append({"role": "user", "content": user_message})
        return turns

    entry = _pl.get_entry(entry_id)
    if not entry:
        return JSONResponse({"error": "prompt log not found"}, status_code=404)
    entry["history"] = _model_messages(entry.get("history") or [], entry.get("user_message", ""))
    entry["source"] = "prompt-log"
    return entry


@app.get("/debug/trajectories")
def debug_trajectories(limit: int = 12, call_type: str = "task_inference"):
    """Return recent tool-call trajectories for the agent inspector."""
    import core.evolution.trajectory_collector as _tc
    return {"trajectories": _tc.recent_trajectories(limit=limit, call_type=call_type or None)}


@app.get("/debug/current-prompt")
def debug_current_prompt(chat_id: str = ""):
    """Return the currently assembled system prompt preview for the active UI session."""
    import core.memory.context as _ctx
    import core.memory.memory as _mem

    cfg = getattr(agent, "_config", {}) or _cfg or {}
    skills = getattr(agent, "_skills", []) or []
    routines = getattr(agent, "_routines", []) or []
    chat_id = chat_id or ""
    history = _mem.load(chat_id=chat_id)

    prompt = _ctx.build_system_prompt(
        cfg,
        skills,
        routines,
        vram_free_fn=mdl.vram_free_mb if mdl else None,
        chat_id=chat_id,
    )
    model_name = ""
    if isinstance(cfg, dict):
        model_name = cfg.get("model", {}).get("name", "")

    model_messages = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in history
        if isinstance(m, dict) and str(m.get("role", "")) in {"user", "assistant", "tool", "system"}
    ]
    return {
        "chat_id": chat_id,
        "prompt": prompt,
        "prompt_len": len(prompt),
        "history": model_messages,
        "user_message": "",
        "provider": "",
        "model": model_name,
        "source": "live-preview-memory",
        "persisted_turns": len(model_messages),
    }


@app.get("/debug/chat-history")
def debug_chat_history(limit: int = 80, chat_id: str = "", include_all: bool = True):
    """Return recent persisted chat turns for dashboard live conversation views.

    - include_all=true (default): returns turns across all sessions/channels.
    - include_all=false: returns only turns for `chat_id`.
    """
    import core.memory.memory as _mem

    safe_limit = max(1, min(int(limit or 80), 400))
    if include_all:
        rows = _mem.history(limit=safe_limit, chat_id="")
    else:
        rows = _mem.history(limit=safe_limit, chat_id=chat_id or "")

    return {
        "history": rows,
        "count": len(rows),
        "chat_id": chat_id or "",
        "include_all": bool(include_all),
    }


@app.get("/debug/fields")
def debug_fields(chat_id: str = "", query: str = ""):
    """(ADR-015, decision #4) Inspect the expertise-field module state.

    Exposes whether the sidecar is loaded, the active field registry, per-chat
    hot-field state, and optional routing results for a sample `query`.

    Degrades to a safe no-op JSON if the module is absent or errors.
    """
    try:
        from core.expansions.expertise_field_bridge import debug_snapshot
        return debug_snapshot(chat_id=chat_id, query=query)
    except Exception as e:
        return {
            "available": False,
            "reason": f"/debug/fields handler error: {e}",
            "registry": None,
            "hot_fields": {},
            "routed": None,
        }

# ── Memory & Workspace endpoints ────────────────────────────────────────────

from runtime_paths import WORKSPACE_ROOT as _WORKSPACE_ROOT
from datetime import datetime as _dt
from fastapi import Body as _Body

_MEMORY_ALLOWED_EXTS = {".md", ".txt", ".yaml", ".yml", ".json", ".sqlite", ".sqlite3", ".db", ".db3"}
_MEMORY_SKIP_DIRS = {".git", "__pycache__", ".db", ".db-journal"}


def _safe_workspace_path(relative: str) -> "os.PathLike | None":
    """Resolve a relative path inside WORKSPACE_ROOT. Returns None if unsafe."""
    import pathlib
    rel = relative.strip().lstrip("/")
    if not rel or ".." in pathlib.PurePosixPath(rel).parts:
        return None
    try:
        resolved = (_WORKSPACE_ROOT / rel).resolve()
        if not str(resolved).startswith(str(_WORKSPACE_ROOT.resolve())):
            return None
        return resolved
    except Exception:
        return None


@app.get("/memory/files")
def memory_files():
    """List all memory-relevant files in the workspace."""
    import pathlib
    results = []
    try:
        root = _WORKSPACE_ROOT.resolve()
        for dirpath, dirnames, filenames in os.walk(str(root)):
            # Skip hidden/unwanted dirs in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in _MEMORY_SKIP_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                ext = pathlib.Path(fname).suffix.lower()
                if ext not in _MEMORY_ALLOWED_EXTS:
                    continue
                full = pathlib.Path(dirpath) / fname
                try:
                    stat = full.stat()
                    rel = str(full.relative_to(root))
                    results.append({
                        "name": fname,
                        "path": rel,
                        "full_path": str(full),
                        "size": stat.st_size,
                        "modified": _dt.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    })
                except Exception:
                    pass
        results.sort(key=lambda x: x["path"])
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    return {"files": results, "count": len(results), "workspace": str(_WORKSPACE_ROOT)}


@app.get("/memory/file")
def memory_file_read(path: str = ""):
    """Read a workspace file by relative path."""
    import pathlib
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    if not fp.is_file():
        return JSONResponse({"error": "not a file"}, status_code=400)
    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
        stat = fp.stat()
        return {
            "name": fp.name,
            "path": path,
            "content": content,
            "size": stat.st_size,
            "modified": _dt.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.put("/memory/file")
def memory_file_write(body: dict = _Body(...)):
    """Write/update a workspace file by relative path."""
    import pathlib
    path = (body.get("path") or "").strip()
    content = body.get("content", "")
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "size": fp.stat().st_size}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/memory/file")
def memory_file_delete(path: str = ""):
    """Delete a workspace file by relative path (moves to trash if available)."""
    import pathlib
    import shutil
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=403)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    if not fp.is_file():
        return JSONResponse({"error": "not a file"}, status_code=400)
    try:
        # Prefer trash CLI if available
        import shutil as _shutil
        trash_cmd = _shutil.which("trash") or _shutil.which("trash-put")
        if trash_cmd:
            import subprocess
            subprocess.run([trash_cmd, str(fp)], check=True)
        else:
            os.remove(fp)
        return {"ok": True, "path": path}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/memory/file/rename")
def memory_file_rename(body: dict = _Body(...)):
    """Rename a workspace file. Body: {from: str, to: str}"""
    import pathlib
    src_rel = (body.get("from") or "").strip()
    dst_rel = (body.get("to") or "").strip()
    if not src_rel or not dst_rel:
        return JSONResponse({"error": "'from' and 'to' are required"}, status_code=400)
    src_full = _safe_workspace_path(src_rel)
    dst_full = _safe_workspace_path(dst_rel)
    if src_full is None:
        return JSONResponse({"error": "invalid or unsafe 'from' path"}, status_code=403)
    if dst_full is None:
        return JSONResponse({"error": "invalid or unsafe 'to' path"}, status_code=403)
    src = pathlib.Path(src_full)
    dst = pathlib.Path(dst_full)
    if not src.exists():
        return JSONResponse({"error": "source file not found"}, status_code=404)
    if dst.exists():
        return JSONResponse({"error": "destination already exists"}, status_code=400)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {"ok": True, "from": src_rel, "to": dst_rel}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/memory/file/new")
def memory_file_new(body: dict = _Body(...)):
    """Create a new workspace file. Body: {path: str, content: str}"""
    import pathlib
    path = (body.get("path") or "").strip()
    content = body.get("content", "")
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=403)
    fp = pathlib.Path(full)
    if fp.exists():
        return JSONResponse({"error": "file already exists"}, status_code=400)
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/memory/stats")
def memory_stats():
    """Return aggregated statistics about workspace memory files and databases."""
    import pathlib
    file_count = 0
    total_size = 0
    try:
        root = _WORKSPACE_ROOT.resolve()
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [
                d for d in dirnames
                if d not in _MEMORY_SKIP_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                ext = pathlib.Path(fname).suffix.lower()
                if ext not in _MEMORY_ALLOWED_EXTS:
                    continue
                try:
                    full = pathlib.Path(dirpath) / fname
                    total_size += full.stat().st_size
                    file_count += 1
                except Exception:
                    pass
    except Exception:
        pass

    # Chat sessions (DB rows count)
    chat_sessions = 0
    try:
        from runtime_paths import CHAT_HISTORY_DB
        import sqlite3 as _sqlite3
        if pathlib.Path(str(CHAT_HISTORY_DB)).exists():
            with _sqlite3.connect(str(CHAT_HISTORY_DB), timeout=2) as conn:
                row = conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages").fetchone()
                if row:
                    chat_sessions = row[0]
    except Exception:
        pass

    # DB size (chat_history)
    db_size = 0
    try:
        from runtime_paths import CHAT_HISTORY_DB
        p = pathlib.Path(str(CHAT_HISTORY_DB))
        if p.exists():
            db_size = p.stat().st_size
    except Exception:
        pass

    return {
        "files": file_count,
        "total_size": total_size,
        "chat_sessions": chat_sessions,
        "db_size": db_size,
        "workspace": str(_WORKSPACE_ROOT),
    }


@app.get("/workspace/tree")
def workspace_tree(path: str = "/", depth: int = 2):
    """Directory tree listing relative to workspace root."""
    import pathlib
    depth = max(1, min(int(depth), 5))

    if path in ("/", "", "."):
        base = _WORKSPACE_ROOT.resolve()
    else:
        resolved = _safe_workspace_path(path.strip().lstrip("/"))
        if resolved is None:
            return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
        base = pathlib.Path(resolved)

    if not base.exists():
        return JSONResponse({"error": "path not found"}, status_code=404)

    def _build(p: pathlib.Path, d: int) -> dict:
        node: dict = {"name": p.name or "/", "type": "dir", "children": []}
        if d <= 0:
            return node
        try:
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return node
        for entry in entries:
            if entry.name in _MEMORY_SKIP_DIRS or entry.name.startswith("."):
                continue
            if entry.is_dir():
                node["children"].append(_build(entry, d - 1))
            else:
                try:
                    node["children"].append({
                        "name": entry.name,
                        "type": "file",
                        "size": entry.stat().st_size,
                        "ext": entry.suffix.lower(),
                        "modified": _dt.fromtimestamp(entry.stat().st_mtime).isoformat(timespec="seconds"),
                    })
                except Exception:
                    node["children"].append({"name": entry.name, "type": "file"})
        return node

    return _build(base, depth)

# ── SQLite viewer/editor endpoints ──────────────────────────────────────────


@app.get("/sqlite/tables")
def sqlite_tables(path: str = ""):
    """List all tables in a SQLite database file."""
    import pathlib, sqlite3
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    ext = fp.suffix.lower()
    if ext not in (".sqlite", ".sqlite3", ".db", ".db3"):
        return JSONResponse({"error": "not a SQLite file"}, status_code=400)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        conn.close()
        return {"tables": table_names, "count": len(table_names), "path": path}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/sqlite/table")
def sqlite_table(path: str = "", table: str = "", page: int = 1, per_page: int = 100):
    """Get table schema and paginated data from a SQLite database."""
    import pathlib, sqlite3
    if not path or not table:
        return JSONResponse({"error": "path and table are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        # Verify table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        if not exists:
            conn.close()
            return JSONResponse({"error": f"table '{table}' not found"}, status_code=404)
        # Schema
        schema = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        columns = [{"cid": r["cid"], "name": r["name"], "type": r["type"],
                     "notnull": bool(r["notnull"]), "pk": bool(r["pk"])} for r in schema]
        # Row count
        count_row = conn.execute(f"SELECT COUNT(*) as cnt FROM '{table}'").fetchone()
        total_rows = count_row["cnt"] if count_row else 0
        # Paginated data
        offset = (max(1, page) - 1) * per_page
        rows_raw = conn.execute(
            f"SELECT * FROM '{table}' LIMIT ? OFFSET ?", (per_page, offset)
        ).fetchall()
        col_names = [c["name"] for c in schema]
        rows_data = []
        for r in rows_raw:
            row_dict = {}
            for cn in col_names:
                val = r[cn]
                # Convert bytes to hex string for display
                if isinstance(val, bytes):
                    val = "<bytes>"
                row_dict[cn] = val
            rows_data.append(row_dict)
        conn.close()
        return {
            "columns": columns,
            "column_names": col_names,
            "rows": rows_data,
            "total_rows": total_rows,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total_rows + per_page - 1) // per_page),
            "path": path,
            "table": table,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.put("/sqlite/row")
def sqlite_row_update(body: dict = _Body(...)):
    """Update a single row in a SQLite table. Body: {path, table, pk_column, pk_value, updates: {col: val}}"""
    import pathlib, sqlite3
    path = (body.get("path") or "").strip()
    table = (body.get("table") or "").strip()
    pk_column = (body.get("pk_column") or "").strip()
    pk_value = body.get("pk_value")
    updates = body.get("updates") or {}
    if not path or not table or not pk_column or pk_value is None or not updates:
        return JSONResponse({"error": "path, table, pk_column, pk_value, and updates are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        set_clauses = []
        params = []
        for col, val in updates.items():
            set_clauses.append(f'"{col}" = ?')
            params.append(val)
        params.append(pk_value)
        sql = f'UPDATE "{table}" SET {", ".join(set_clauses)} WHERE "{pk_column}" = ?'
        conn.execute(sql, params)
        conn.commit()
        conn.close()
        return {"ok": True, "rows_affected": conn.total_changes}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/sqlite/row")
def sqlite_row_delete(path: str = "", table: str = "", pk_column: str = "", pk_value: str = ""):
    """Delete a single row from a SQLite table."""
    import pathlib, sqlite3
    if not path or not table or not pk_column or not pk_value:
        return JSONResponse({"error": "path, table, pk_column, and pk_value are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.execute(f'DELETE FROM "{table}" WHERE "{pk_column}" = ?', (pk_value,))
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return {"ok": True, "rows_deleted": affected}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/sqlite/table/data")
def sqlite_table_clear(path: str = "", table: str = ""):
    """Delete all data from a table while keeping its structure (DELETE FROM)."""
    import pathlib, sqlite3
    if not path or not table:
        return JSONResponse({"error": "path and table are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.execute(f'DELETE FROM "{table}"')
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return {"ok": True, "rows_deleted": affected, "table": table}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.delete("/sqlite/db/data")
def sqlite_db_clear(path: str = ""):
    """Delete all data from all tables in a SQLite database while keeping structure."""
    import pathlib, sqlite3
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        total = 0
        cleared = []
        for t in tables:
            tname = t[0]
            affected = conn.execute(f'DELETE FROM "{tname}"').rowcount
            total += affected
            cleared.append(tname)
        conn.commit()
        conn.close()
        return {"ok": True, "tables_cleared": cleared, "total_rows_deleted": total}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/sqlite/row")
def sqlite_row_insert(body: dict = _Body(...)):
    """Insert a new row into a SQLite table. Body: {path, table, row: {col: val}}"""
    import pathlib, sqlite3
    path = (body.get("path") or "").strip()
    table = (body.get("table") or "").strip()
    row_data = body.get("row") or {}
    if not path or not table or not row_data:
        return JSONResponse({"error": "path, table, and row are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        cols = ", ".join(f'"{c}"' for c in row_data.keys())
        placeholders = ", ".join("?" for _ in row_data)
        vals = list(row_data.values())
        sql = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
        conn.execute(sql, vals)
        conn.commit()
        conn.close()
        return {"ok": True, "inserted": True}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ── End Memory & Workspace endpoints ────────────────────────────────────────


@app.get("/skills")
def list_skills():
    return [{"name": s["name"], "description": s["description"], "commands": s.get("commands", []), "is_core": s.get("is_core", False)} for s in agent._skills]


@app.get("/routines")
def list_routines():
    return [{"name": r["name"], "description": r["description"], "trigger": r["trigger"]} for r in agent._routines]


@app.post("/replica/spawn")
def spawn_replica(body: TaskIn):
    """Spawn a specialist replica if VRAM allows."""
    r = rep.spawn(body.role, body.task, adapter_path=body.adapter_path)
    if r is None:
        return {"status": "rejected", "reason": "VRAM limit or replica cap reached"}
    return {"status": "spawned", "role": r.role, "name": r.name}


@app.get("/replica/active")
def active_replicas():
    return [{
        "name": r.name,
        "role": r.role,
        "task": r.task[:80],
        "done": r.done,
        "persistent": r.persistent,
        "adapter_path": r.adapter_path,
    } for r in rep.active()]


@app.post("/replica/named")
def spawn_named_replica(body: NamedReplicaIn):
    """Spawn a named persistent replica with optional brief injection."""
    r = rep.spawn_named(
        name=body.name,
        role=body.role,
        brief_path=body.brief_path,
        custom_prompt=body.custom_prompt,
        workspace=body.workspace,
        tools_enabled=body.tools_enabled,
        output_path=body.output_path,
        input_path=body.input_path,
        adapter_path=body.adapter_path,
        slot=body.slot,
    )
    if r is None:
        return {"status": "rejected", "reason": "VRAM limit or replica cap reached"}
    return {"status": "spawned", "name": r.name, "role": r.role,
            "persistent": r.persistent, "tools_enabled": r.tools_enabled,
            "adapter_path": r.adapter_path}


@app.post("/replica/pipeline")
def replica_pipeline(body: PipelineIn):
    """ADR-008 Change 2: Run a multi-stage writer→critic pipeline.

    Stages run sequentially. Each stage's output is passed as context
    to the next stage (via input_from). Results are written to output_path
    when specified. Replicas are cleaned up after the pipeline completes.
    """
    import logging
    logger = logging.getLogger(__name__)
    results: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}

    for stage in body.stages:
        # Build message — inject previous stage output if input_from specified
        task_text = stage.task
        if stage.input_from and stage.input_from in stage_outputs:
            task_text = (
                f"{stage.task}\n\n"
                f"## Output from '{stage.input_from}' stage\n"
                f"{stage_outputs[stage.input_from]}"
            )
        # Also inject from output_path on disk if input_from stage wrote one
        input_path = None
        if stage.input_from:
            prev = next((s for s in body.stages if s.name == stage.input_from), None)
            if prev and prev.output_path:
                input_path = prev.output_path

        # Ensure no stale replica with this name
        rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

        r = rep.spawn_named(
            name=f"pipeline-{body.pipeline}-{stage.name}",
            role=stage.role,
            custom_prompt=stage.brief,
            workspace=stage.workspace,
            tools_enabled=stage.tools,
            output_path=stage.output_path,
            input_path=input_path,
            adapter_path=stage.adapter_path,
        )
        if r is None:
            return JSONResponse(
                {"status": "error", "stage": stage.name,
                 "reason": "VRAM limit or replica cap reached"},
                status_code=503
            )

        reply = r.message(task_text)
        results[stage.name] = reply
        stage_outputs[stage.name] = reply
        logger.info(f"[pipeline:{body.pipeline}] stage '{stage.name}' done ({len(reply)} chars)")

        # Clean up stage replica immediately to free capacity for next stage
        rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

    return {"status": "complete", "pipeline": body.pipeline, "results": results}


# ── ADR-012: Async pipeline execution ──────────────────────────────────────
_pipeline_jobs: dict[str, dict] = {}  # job_id → job state dict


class PipelineJobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | complete | error | cancelled
    stage: Optional[str] = None
    results: dict = {}
    error: Optional[str] = None
    pipeline: str = "default"
    created_at: float = 0.0
    completed_at: Optional[float] = None


def _prune_pipeline_jobs():
    """Remove completed jobs older than TTL."""
    ttl = _cfg.get("pipeline", {}).get("job_ttl_s", 3600)
    cutoff = time.time() - ttl
    to_delete = [
        jid for jid, j in _pipeline_jobs.items()
        if j.get("completed_at", time.time()) < cutoff
    ]
    for jid in to_delete:
        del _pipeline_jobs[jid]


async def _run_pipeline_job(job_id: str, body: PipelineIn):
    """Background task: execute pipeline stages, update job state."""
    import logging as _log
    _logger = _log.getLogger(__name__)
    job = _pipeline_jobs.get(job_id)
    if not job:
        return
    job["status"] = "running"
    results: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}

    try:
        for stage in body.stages:
            if job.get("status") == "cancelled":


                break
            job["stage"] = stage.name

            # Build task text with input injection
            task_text = stage.task
            if stage.input_from and stage.input_from in stage_outputs:
                task_text = (
                    f"{stage.task}\n\n"
                    f"## Output from '{stage.input_from}' stage\n"
                    f"{stage_outputs[stage.input_from]}"
                )

            rep.stop(f"pipeline-{body.pipeline}-{stage.name}")
            r = rep.spawn_named(
                name=f"pipeline-{body.pipeline}-{stage.name}",
                role=stage.role,
                custom_prompt=stage.brief,
                workspace=stage.workspace,
                tools_enabled=stage.tools,
                output_path=stage.output_path,
                adapter_path=stage.adapter_path,
            )
            if r is None:
                raise RuntimeError(f"Stage '{stage.name}': could not spawn replica (VRAM/cap)")

            reply = r.message(task_text)
            results[stage.name] = reply
            stage_outputs[stage.name] = reply
            job["results"] = dict(results)
            _logger.info(f"[pipeline-async:{body.pipeline}:{job_id[:8]}] stage '{stage.name}' done")
            rep.stop(f"pipeline-{body.pipeline}-{stage.name}")

        if job.get("status") != "cancelled":
            job["status"] = "complete"
        job["completed_at"] = time.time()

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["completed_at"] = time.time()
        _logger.error(f"[pipeline-async:{job_id[:8]}] error: {e}")


from fastapi import BackgroundTasks


@app.post("/replica/pipeline/async")
async def replica_pipeline_async(body: PipelineIn, background_tasks: BackgroundTasks):
    """ADR-012: Async pipeline — returns job_id immediately, runs in background."""
    _prune_pipeline_jobs()
    job_id = str(uuid.uuid4())
    _pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": None,
        "results": {},
        "error": None,
        "pipeline": body.pipeline,
        "created_at": time.time(),
        "completed_at": None,
    }
    background_tasks.add_task(_run_pipeline_job, job_id, body)
    return {"job_id": job_id, "status": "queued"}


@app.get("/replica/pipeline/{job_id}")
async def get_pipeline_job(job_id: str):
    """ADR-012: Get current job state."""
    job = _pipeline_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return job


@app.get("/replica/pipeline/{job_id}/stream")
async def stream_pipeline_job(job_id: str):
    """ADR-012: SSE stream of stage completion events."""
    import asyncio

    async def event_generator():
        last_stage = None
        last_result_count = 0
        while True:
            job = _pipeline_jobs.get(job_id)
            if not job:
                yield f"data: {{\"event\": \"error\", \"message\": \"job not found\"}}\n\n"
                break

            current_stage = job.get("stage")
            results = job.get("results", {})

            # Emit stage_start when stage changes
            if current_stage and current_stage != last_stage:
                yield f"data: {json.dumps({'event': 'stage_start', 'stage': current_stage, 'job_id': job_id})}\n\n"
                last_stage = current_stage

            # Emit stage_complete for newly completed stages
            if len(results) > last_result_count:
                new_stages = list(results.keys())[last_result_count:]
                for sname in new_stages:
                    yield f"data: {json.dumps({'event': 'stage_complete', 'stage': sname, 'result': results[sname][:200], 'job_id': job_id})}\n\n"
                last_result_count = len(results)

            status = job.get("status")
            if status == "complete":
                total = (job.get("completed_at") or time.time()) - job.get("created_at", time.time())
                yield f"data: {json.dumps({'event': 'pipeline_complete', 'job_id': job_id, 'total_elapsed_s': round(total, 1)})}\n\n"
                break
            elif status in ("error", "cancelled"):
                yield f"data: {json.dumps({'event': 'pipeline_error', 'job_id': job_id, 'error': job.get('error', status)})}\n\n"
                break

            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/replica/pipeline/{job_id}")
async def cancel_pipeline_job(job_id: str):
    """ADR-012: Cancel a queued or running job."""
    job = _pipeline_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job["status"] = "cancelled"
    job["completed_at"] = time.time()
    return {"job_id": job_id, "status": "cancelled"}
# ── End ADR-012 ─────────────────────────────────────────────────────────────


@app.post("/replica/{name}/message")
def message_replica(name: str, body: ReplicaMessageIn):
    """Send a message to a named persistent replica."""
    r = rep.get(name)
    if r is None:
        return JSONResponse({"error": f"Replica '{name}' not found"}, status_code=404)
    if not r.persistent:
        return JSONResponse({"error": "Replica is not in persistent mode"}, status_code=400)
    reply = r.message(body.message)
    return {"name": name, "reply": reply}


@app.delete("/replica/{name}")
def stop_replica(name: str):
    """Stop and remove a named replica."""
    if rep.stop(name):
        return {"status": "stopped", "name": name}
    return JSONResponse({"error": f"Replica '{name}' not found"}, status_code=404)


@app.get("/replica/{name}/status")
def replica_status(name: str):
    """Get status of a named replica."""
    r = rep.get(name)
    if r is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "name": r.name,
        "role": r.role,
        "persistent": r.persistent,
        "done": r.done,
        "brief_path": r.brief_path,
        "history_turns": len(r.history),
        "adapter_path": r.adapter_path,
    }


@app.get("/system")
def system_info():
    return {
        "vram_free_mb": mdl.vram_free_mb(),
        "can_spawn": rep.can_spawn(),
        "max_replicas": rep.MAX_REPLICAS,
        "context_budget_mb": rep.CONTEXT_BUDGET_MB,
    }


@app.post("/evolve/backup")
def evolve_backup(body: BackupRequest):
    """Create a timestamped backup archive."""
    import infra.backup
    try:
        result = backup.create_backup(description=body.description, full=body.full)
        return {"status": "created", "backup": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/evolve/backups")
def list_backups():
    """List existing backups."""
    import infra.backup
    backups = backup.list_backups()
    return {"backups": backups, "count": len(backups)}


@app.post("/evolve/init")
def evolve_init(body: InitRequest):
    """Initialize kernel-evolving workspace and databases."""
    import init
    try:
        result = init.initialize(force=body.force)
        return {"status": result["status"], "result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/evolve/init")
def evolve_init_status(force: bool = False):
    """Return workspace initialization status using the same init mechanism."""
    import init
    try:
        result = init.initialize(force=force)
        return {"status": result.get("status", "unknown"), "result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/evolve/fresh")
def evolve_fresh(body: FreshRequest):
    """Reset to fresh state after creating backup. Requires confirmation."""
    import fresh
    try:
        if body.dry_run:
            result = fresh.dry_run_fresh()
        else:
            result = fresh.execute_fresh()
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



@app.get("/version")
def get_version():
    from infra.updater import get_current_version as _get_current_version
    return {"version": _get_current_version()}


@app.get("/workspaces")
def list_workspaces_endpoint():
    from core.workspaces import list_workspaces as _list
    return _list()


# ── Evolution monitoring endpoints ─────────────────────────────────────────

@app.get("/evolution/state")
def evolution_state_endpoint():
    """Return current evolution state machine status."""
    import core.evolution.evolution_state as _evo_state
    return _evo_state.get_status()


class EvolutionControlRequest(BaseModel):
    action: str          # start | pause | resume | stop | reset
    cap: Optional[int] = None
    reason: Optional[str] = None


@app.post("/evolution/control")
def evolution_control(body: EvolutionControlRequest):
    """Control evolution state: start, pause, resume, stop, reset."""
    import core.evolution.evolution_state as _evo_state
    action = body.action.lower().strip()
    if action == "start":
        return _evo_state.start(cap=body.cap)
    elif action == "pause":
        return _evo_state.pause(reason=body.reason or "manual")
    elif action == "resume":
        return _evo_state.resume()
    elif action == "stop":
        return _evo_state.stop()
    elif action == "reset":
        return _evo_state.reset(cap=body.cap)
    else:
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)


class EvolutionTriggerRequest(BaseModel):
    task: str
    cap: Optional[int] = None


@app.post("/evolution/trigger")
def evolution_trigger(body: EvolutionTriggerRequest):
    """Manually trigger an evolution cycle for a given task."""
    import core.evolution.evolution_state as _evo_state
    import core.evolution.evolution_hook
    import yaml as _yaml

    # Temporarily allow one cycle even if paused (manual trigger overrides cap but not stop)
    status = _evo_state.get_status()
    if status["state"] == "stopped":
        return JSONResponse({"error": "Evolution is stopped. Use /evolution/control to start first."}, status_code=400)

    if body.cap:
        _evo_state.start(cap=body.cap)

    with open(os.path.join(_BASE, "config.yaml")) as f:
        cfg = _yaml.safe_load(f)
    skills_dir = os.path.expanduser(cfg.get("skills_dir", "./skills"))

    # Manual trigger bypasses EVOLUTION_ENABLED flag — explicit user action
    old_flag = evolution_hook.EVOLUTION_ENABLED
    evolution_hook.EVOLUTION_ENABLED = True
    try:
        result = evolution_hook.maybe_evolve(body.task, cfg, skills_dir, infer_fn=mdl.infer)
    finally:
        evolution_hook.EVOLUTION_ENABLED = old_flag
    if result is None:
        return {
            "triggered": False,
            "reason": "Evolution is paused or cap reached. Resume or increase cap first.",
            "state": _evo_state.get_status(),
        }
    return {
        "triggered": True,
        "task": body.task,
        "result": {
            "found":                result.found,
            "installed":            result.installed,
            "confidence":           result.confidence,
            "escalated":            result.escalated,
            "provider":             result.provider_used,
            "gap":                  result.gap,
            "verification_result":  result.verification_result,
            "verification_reasoning": result.verification_reasoning,
            "recommendations":          getattr(result, "recommendations", []),
        },
        "state": _evo_state.get_status(),
    }


@app.get("/evolution")
def evolution_status():
    """Return evolution history, gaps, stats, and chart data for the dashboard."""
    from core.evolution.evolution_log import EvolutionLog
    import core.skills as _skills_mod
    import core.routines as _routines_mod
    log = EvolutionLog()
    history_full = log.get_history(limit=1000)
    gaps = log.get_gaps()
    evolution_events = [e for e in history_full if (e.get("event_type") in (None, "evolution"))]

    # Synthesised = escalated AND found (Tier 2 success)
    synthesised = len([e for e in evolution_events if e.get("escalated") and e.get("found")])

    # sim_stats — derive from timestamps in history
    SIM_BOUNDS = [
        ('Sim 1', None,                    '2026-05-06T10:28:56'),
        ('Sim 2', '2026-05-06T10:28:56',   '2026-05-06T10:38:40'),
        ('Sim 3', '2026-05-06T10:38:40',   '2026-05-06T11:36:51'),
        ('Sim 4', '2026-05-06T11:36:51',   '2026-05-06T12:38:22'),
        ('Sim 5', '2026-05-06T12:38:22',   '2026-05-06T12:47:45'),
        ('Free',  '2026-05-06T12:47:45',   None),
    ]
    sim_stats = []
    for label, start, end in SIM_BOUNDS:
        rows = [e for e in evolution_events if
                (start is None or e.get('ts','') >= start) and
                (end   is None or e.get('ts','') <  end)]
        sim_stats.append({
            'label':     label,
            'total':     len(rows),
            'tier1':     sum(1 for e in rows if e.get('found') and not e.get('escalated')),
            'tier2':     sum(1 for e in rows if e.get('escalated')),
            'false_pos': 6 if label == 'Sim 3' else 0,
        })

    # skill_counts — actual live count from ecosystem
    try:
        skill_count_now = len(agent._skills)
    except Exception:
        skill_count_now = 53
    skill_counts = [
        {'label': 'Pre-Sim1',  'count': 16},
        {'label': 'Post-Sim1', 'count': 22},
        {'label': 'Post-Sim4', 'count': 27},
        {'label': 'Post-Sim5', 'count': 28},
        {'label': 'Now',       'count': skill_count_now},
    ]

    return {
        "history":     log.get_history(limit=50),
        "gaps":        gaps,
        "sim_stats":   sim_stats,
        "skill_counts": skill_counts,
        "stats": {
            "total_events":    len(evolution_events),
            "resolved":        len([e for e in evolution_events if e.get("found")]),
            "synthesised":     synthesised,
            "unresolved_gaps": len(gaps),
            "gaps":            len(gaps),
            "providers_used":  list(set(
                e.get("provider_used") for e in evolution_events
                if e.get("provider_used")
            )),
        }
    }


@app.get("/evolution/dashboard")
def evolution_dashboard():
    """D3 force-graph + timeline evolution monitoring dashboard."""
    from fastapi.responses import HTMLResponse
    from core.evolution.evolution_log import EvolutionLog
    log = EvolutionLog()
    history = log.get_history(limit=500)
    gaps = log.get_gaps()
    history_full = log.get_history(limit=1000)

    stats = {
        "total": len(history_full),
        "resolved": len([e for e in history_full if e.get("found")]),
        "synthesised": len([e for e in history_full if e.get("escalated") and e.get("found")]),
        "gaps": len(gaps),
        "providers": list(set(e.get("provider_used") for e in history_full if e.get("provider_used"))),
    }

    events_json = json.dumps(history)
    gaps_json   = json.dumps(gaps)
    stats_json  = json.dumps(stats)

    # Load dashboard HTML and inject data
    _tpl = os.path.join(os.path.dirname(__file__), 'views', 'evolution_dashboard.html')
    tpl = open(_tpl).read()
    html = (tpl
        .replace('__EVENTS__', events_json)
        .replace('__GAPS__',   gaps_json)
        .replace('__STATS__',  stats_json)
    )
    return HTMLResponse(html)


@app.get("/evolution/stream")
def evolution_stream():
    """Server-sent events stream for live evolution monitoring."""
    from core.evolution.evolution_log import EvolutionLog
    import time as _time

    def event_generator():
        log = EvolutionLog()
        last_count = 0
        while True:
            history = log.get_history(limit=1000)
            if len(history) > last_count:
                new_events = history[last_count:]
                for event in new_events:
                    yield f"data: {json.dumps(event)}\n\n"
                last_count = len(history)
            _time.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/sim/mode")
def set_sim_mode(body: dict):
    """Enable/disable SIM_MODE — bypasses exec_shell approval gate for trajectory collection."""
    import os as _os
    enabled = bool(body.get("enabled", False))
    _os.environ["SIM_MODE"] = "true" if enabled else ""
    return {"sim_mode": enabled}


@app.get("/sim/mode")
def get_sim_mode():
    import os as _os
    return {"sim_mode": _os.environ.get("SIM_MODE") == "true"}


@app.get("/provider")

def get_provider_routing():
    """ADR-013: Return current provider routing table + availability."""
    from core.inference.provider import get_provider as _gp
    p = _gp()  # get existing singleton
    routing = {}
    for call_type in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
        prov = p.get_provider(call_type)
        model = p.get_model(prov, call_type)
        stream = p._streaming.get(call_type, False)
        routing[call_type] = {"provider": prov, "model": model, "streaming": stream}
    providers_cfg = _cfg.get("providers", {}) if isinstance(_cfg, dict) else {}
    return {
        "routing": routing,
        "collect_trajectories": _cfg.get("providers", {}).get("collect_trajectories", False),
        "hf_provider": providers_cfg.get("hf_provider") or getattr(p, "_cfg", {}).get("hf_provider", "deepinfra"),
        "model_catalog": providers_cfg.get("models", {}),
        "model_overrides": providers_cfg.get("model_overrides", {}),
        "call_types": ["task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"],
        "providers": ["local", "openai", "anthropic", "hf", "copilot", "openrouter", "google"],
    }


@app.post("/config/env")
def set_config_env(body: dict):
    """Set provider/API keys in kernel-evolving's environment and persist to .env.

    Body: {"keys": {"OPENAI_API_KEY": "sk-...", "HF_TOKEN": "hf_...", ...}}

    Keys are applied to the running process immediately (os.environ) and
    non-empty values are persisted to a dedicated JSON override store
    (workspace data/env_overrides.json), which is loaded into os.environ at
    startup. This avoids rewriting the base .env (no shell-parsing fragility,
    survives repo updates) while keeping overrides effective across restarts.
    Only known env-var names are accepted; empty values are ignored (never
    clobber an existing key with an empty one).
    """
    from core.env_overrides import save_env_overrides
    keys = body.get("keys", {})
    if not isinstance(keys, dict) or not keys:
        return JSONResponse({"error": "keys object is required"}, status_code=400)

    # Apply to the running process immediately and persist to the JSON store.
    applied = save_env_overrides(keys)
    for name, value in applied.items():
        os.environ[name] = value

    return {"ok": True, "applied": applied, "persisted_to_env_store": True}


@app.post("/provider/set")
def set_provider_routing(body: dict):
    """ADR-013: Hot-swap provider routing for one or more call types.
    Body: {\"task_inference\": \"openai\", \"model_override\": {\"task_inference\": \"gpt-5.4\"}, \"persist\": false}
    Session-scoped by default; set persist=true to write back to config.yaml.

    GPU safety rules (automatic, no user action needed):
    - Switching task_inference local → cloud: unloads model from VRAM first
    - Switching task_inference cloud → local: ensures model server is running
    """
    import yaml as _yaml
    from core.inference.provider import get_provider as _gp
    from core.inference.model_client import is_server_running
    p = _gp()  # get existing singleton — do not pass _cfg (would recreate)
    valid_call_types = {"task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"}
    valid_providers  = {"local", "openai", "anthropic", "hf", "copilot", "openrouter"}
    changed = {}
    vram_actions = []  # messages about GPU actions taken

    # Snapshot current task_inference provider before applying changes
    prev_task = p._routing.get("task_inference", "local")

    for key, val in body.items():
        if key in valid_call_types and val in valid_providers:
            p._routing[key] = val
            changed[key] = val
        elif key == "model_override" and isinstance(val, dict):
            for ct, m in val.items():
                if ct in valid_call_types:
                    p._cfg.setdefault("model_overrides", {})[ct] = m
                    changed[f"model_override.{ct}"] = m
        elif key == "collect_trajectories" and isinstance(val, bool):
            _cfg.setdefault("providers", {})["collect_trajectories"] = val
            changed["collect_trajectories"] = val
        elif key == "hf_provider" and isinstance(val, str) and val.strip():
            # HF Router inference provider (providers.hf_provider / HF_ROUTER_PROVIDER).
            p._cfg["hf_provider"] = val.strip()
            _cfg.setdefault("providers", {})["hf_provider"] = val.strip()
            changed["hf_provider"] = val.strip()

    new_task = p._routing.get("task_inference", "local")

    # ── GPU safety: unload when leaving local ───────────────────────────────────
    # Switching away from local → unload model to free VRAM before cloud takes over
    if prev_task == "local" and new_task != "local" and "task_inference" in changed:
        if is_server_running():
            try:
                from core.inference.model_client import unload as _unload
                result = _unload()
                freed = result.get("freed_mb", 0)
                vram_actions.append(f"unloaded local model (~{freed}MB freed)")
                print(f"[provider/set] GPU unload on switch local→{new_task}: {result}", flush=True)
            except Exception as _ue:
                print(f"[provider/set] WARNING: unload failed: {_ue}", flush=True)
                vram_actions.append(f"unload attempted (error: {_ue})")

    # ── GPU safety: ensure model server alive when switching to local ────────────
    # Switching to local → model server must be running for inference to work
    if new_task == "local" and prev_task != "local" and "task_inference" in changed:
        if not is_server_running():
            vram_actions.append("WARNING: model server not running — start it via /models reload or restart start.sh")
            print("[provider/set] WARNING: switched to local but model server not running", flush=True)
        else:
            vram_actions.append("model server running — ready for local inference")

    if body.get("persist"):
        cfg_path = os.path.join(_BASE, "config.yaml")
        with open(cfg_path) as f:
            on_disk = _yaml.safe_load(f)
        providers_block = on_disk.setdefault("providers", {})
        providers_block.update({k: v for k, v in changed.items() if k in valid_call_types})

        # Persist collect toggle when present.
        if "collect_trajectories" in changed:
            providers_block["collect_trajectories"] = changed["collect_trajectories"]

        # Persist HF Router provider when present.
        if "hf_provider" in changed:
            providers_block["hf_provider"] = changed["hf_provider"]

        # Persist model overrides when present (keys look like model_override.<call_type>).
        override_changes = {
            k.split(".", 1)[1]: v
            for k, v in changed.items()
            if k.startswith("model_override.")
        }
        if override_changes:
            providers_block.setdefault("model_overrides", {}).update(override_changes)

        with open(cfg_path, "w") as f:
            _yaml.dump(on_disk, f, default_flow_style=False, allow_unicode=True)
    return {"changed": changed, "persisted": bool(body.get("persist")), "vram_actions": vram_actions}


@app.get("/provider/available")
def get_provider_availability():
    """ADR-013: Check which providers are ready (API keys set)."""
    import os as _os
    from core.inference.model_client import is_server_running
    return {
        "local":     {"ready": is_server_running(), "reason": "model_server socket" if is_server_running() else "model_server not running"},
        "openai":    {"ready": bool(_os.environ.get("OPENAI_API_KEY") or _os.environ.get("TMP_OPEN_AI_API_KEY")), "reason": "OPENAI_API_KEY set" if _os.environ.get("OPENAI_API_KEY") or _os.environ.get("TMP_OPEN_AI_API_KEY") else "OPENAI_API_KEY not set"},
        "anthropic": {"ready": bool(_os.environ.get("ANTHROPIC_API_KEY")), "reason": "ANTHROPIC_API_KEY set" if _os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY not set"},
        "hf":        {"ready": bool(_os.environ.get("HF_TOKEN")),           "reason": "HF_TOKEN set" if _os.environ.get("HF_TOKEN") else "HF_TOKEN not set"},
        "copilot":   {"ready": bool(_os.environ.get("GITHUB_COPILOT_TOKEN") or _os.environ.get("GITHUB_TOKEN")), "reason": "GITHUB token set" if _os.environ.get("GITHUB_COPILOT_TOKEN") or _os.environ.get("GITHUB_TOKEN") else "GITHUB token not set"},
        "openrouter": {"ready": bool(_os.environ.get("OPENROUTER_API_KEY")), "reason": "OPENROUTER_API_KEY set" if _os.environ.get("OPENROUTER_API_KEY") else "OPENROUTER_API_KEY not set"},
        "google":    {"ready": bool(_os.environ.get("GOOGLE_AI_API_KEY")),  "reason": "GOOGLE_AI_API_KEY set" if _os.environ.get("GOOGLE_AI_API_KEY") else "GOOGLE_AI_API_KEY not set"},
    }


# ── Dynamic model catalog ─────────────────────────────────────────────────
# Cached fetch from each provider's model API, filtered by capability.
# Replaces the hardcoded _CLOUD_MODEL_CATALOG in telegram_bot.py.

_MODEL_CACHE: dict = {}
_MODEL_CACHE_TTL = 3600  # 1 hour


def _fetch_openrouter_models() -> dict:
    """Fetch all models from OpenRouter, grouped by capability."""
    import urllib.request, json as _j
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers=headers, method="GET"
        )
        resp = urllib.request.urlopen(req, timeout=15)
        data = _j.loads(resp.read())
        models = data.get("data", [])
        text_models, vision_models, audio_models = [], [], []
        for m in models:
            mid = m.get("id", "")
            inputs = m.get("architecture", {}).get("input_modalities", [])
            if not inputs:
                inputs = ["text"]
            has_text = "text" in inputs or not inputs
            has_image = "image" in inputs
            has_audio = "audio" in inputs
            if has_image:
                vision_models.append(mid)
            if has_audio:
                audio_models.append(mid)
            if has_text and not has_image and not has_audio:
                text_models.append(mid)
        return {
            "text": sorted(text_models),
            "vision": sorted(vision_models),
            "audio": sorted(audio_models),
        }
    except Exception as e:
        print(f"[api] OpenRouter model fetch failed: {e}", flush=True)
        return {}


# Fallback model lists for providers without a dynamic model API
_FALLBACK_MODELS = {
    "openai": {
        "text": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "o3-mini", "o4-mini"],
        "vision": ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-pro", "gpt-4.1", "gpt-4o"],
        "audio": ["gpt-audio", "gpt-audio-mini"],
    },
    "anthropic": {
        "text": ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5", "claude-sonnet-4", "claude-opus-4"],
        "vision": ["claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-3-5", "claude-sonnet-4", "claude-opus-4"],
        "audio": [],
    },
    "hf": {
        "text": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-32B", "mistralai/Mixtral-8x22B"],
        "vision": ["Qwen/Qwen2.5-VL-72B-Instruct"],
        "audio": [],
    },
    "copilot": {
        "text": ["claude-sonnet-4.5", "gpt-5.4"],
        "vision": ["claude-sonnet-4.5", "gpt-5.4"],
        "audio": [],
    },
}


@app.get("/provider/models")
def get_provider_models(provider: str = "", capability: str = "text"):
    """Return available models per provider and capability.
    Query params:
      provider (optional): filter to one provider
      capability (optional): "text" | "vision" | "audio" (default "text")
    """
    import time as _time
    provider = provider.strip()
    capability = capability.strip()

    # Refresh OpenRouter cache if stale
    global _MODEL_CACHE
    now = _time.time()
    if "openrouter" not in _MODEL_CACHE or now - _MODEL_CACHE.get("_ts", 0) > _MODEL_CACHE_TTL:
        or_models = _fetch_openrouter_models()
        if or_models:
            _MODEL_CACHE["openrouter"] = or_models
            _MODEL_CACHE["_ts"] = now

    # Build response: map all providers with their model lists for the requested capability
    result = {}
    providers_to_include = [provider] if provider else list(_FALLBACK_MODELS.keys()) + ["openrouter"]
    for prov in providers_to_include:
        if prov == "openrouter":
            mlist = _MODEL_CACHE.get("openrouter", {}).get(capability, [])
        else:
            mlist = _FALLBACK_MODELS.get(prov, {}).get(capability, [])
        result[prov] = mlist

    return {"capability": capability, "models": result}


@app.get("/evolution/trajectories")
def evolution_trajectories():
    """Return recent synthesis trajectories for fine-tuning monitoring."""
    from core.evolution.evolution_log import EvolutionLog
    log = EvolutionLog()
    return {"trajectories": log.get_trajectories(limit=100)}


@app.get("/trajectories/task")
def get_task_trajectories():
    """Return recent task trajectories (ADR-013)."""
    from core.evolution.trajectory_collector import get_collector
    col = get_collector(_cfg)
    with col._conn() as conn:
        rows = conn.execute(
            "SELECT id,ts,task,provider,model_name,call_type,critic_score,critic_verdict,artifacts,elapsed_s "
            "FROM task_trajectories ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return {"trajectories": [dict(r) for r in rows]}


@app.post("/trajectories/export")
def export_trajectories(min_score: float = 0.7, call_type: str = None):
    """Export task trajectories as JSONL (ADR-013). Returns path + count."""
    from core.evolution.trajectory_collector import get_collector
    col = get_collector(_cfg)
    path, count = col.export_jsonl(min_score=min_score, call_type=call_type or None)
    return {"path": path, "count": count, "min_score": min_score}


@app.get("/thoughts")
def get_thoughts():
    """Return last 20 thoughts from recent journals (today + yesterday)."""
    from core.memory.thought_journal import ThoughtJournal
    journal = ThoughtJournal(journal_dir=_cfg.get("thinking", {}).get("journal_dir"))
    thoughts = journal.read_recent(days=2)
    return thoughts[-20:] if len(thoughts) > 20 else thoughts


@app.get("/thoughts/today")
def get_thoughts_today():
    """Return recent thought journals as markdown text (today + yesterday)."""
    import datetime
    import glob
    from core.memory.thought_journal import ThoughtJournal
    journal = ThoughtJournal(journal_dir=_cfg.get("thinking", {}).get("journal_dir"))
    journal_dir = journal.journal_dir

    today = datetime.date.today()
    dates = [today, today - datetime.timedelta(days=1)]

    sections: list[str] = []
    included_dates: list[str] = []
    entries = 0
    ideas_count = 0

    ideas_dir = os.path.join(journal_dir, "ideas")
    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        path = os.path.join(journal_dir, date_str + ".md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                block = f.read().strip()
            if block:
                sections.append(f"# {date_str}\n\n{block}")
                included_dates.append(date_str)
                entries += block.count("## ")

        if os.path.isdir(ideas_dir):
            ideas_count += len(glob.glob(os.path.join(ideas_dir, f"{date_str}-*.md")))

    content = "\n\n".join(sections)
    return {
        "date": str(today),
        "included_dates": included_dates,
        "content": content,
        "entries": entries,
        "ideas_count": ideas_count,
    }


@app.post("/think/trigger")
def trigger_think():
    """Manually trigger one Think-at-Rest cycle. For testing."""
    think = getattr(app.state, "think_at_rest", None)
    if think is None:
        return {"status": "error", "message": "think_at_rest not initialized"}
    if not getattr(think, "_enabled", False):
        return {"status": "error", "message": "thinking disabled in config"}
    import threading
    t = threading.Thread(target=think._run_think_cycle, daemon=True)
    t.start()
    return {"status": "triggered", "message": "think cycle started in background"}


# -- Voice dashboard endpoints -----------------------------------------------
# /transcribe  -- STT via Gemma 4 E2B native multimodal (infer_with_audio)
# /voice/chat  -- text -> agent.triage() -> TTS via _clone_voice_reply -> WAV
# /voice/history/{session_id} DELETE -- clear voice session

# Voice server URL: env > config.yaml services.voice_server.url > localhost default.
# Empty string disables voice features without error.
_VOICE_SERVER_URL = os.environ.get("VOICE_SERVER_URL", "")
_VOICE_CHAT_TRIAGE_TIMEOUT_S = float(os.environ.get("VOICE_CHAT_TRIAGE_TIMEOUT_S", "60"))
_VOICE_CHAT_TTS_TIMEOUT_S = float(os.environ.get("VOICE_CHAT_TTS_TIMEOUT_S", "180"))


async def _build_voice_audio_response(reply: str, session_id: str = ""):
    """Try voice clone TTS and return audio Response or JSON fallback payload."""
    from fastapi.responses import Response
    from fastapi.concurrency import run_in_threadpool
    import services.channels.telegram_bot as _tb

    try:
        wav_path = await asyncio.wait_for(
            run_in_threadpool(_tb._clone_voice_reply, reply),
            timeout=_VOICE_CHAT_TTS_TIMEOUT_S,
        )
        if wav_path and os.path.isfile(wav_path):
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            os.unlink(wav_path)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Assistant-Reply": reply[:2000],
                    "X-Voice-Session": session_id[:120],
                    "X-Voice-Clone": "ok",
                },
            )
        raise RuntimeError("clone returned no file")
    except asyncio.TimeoutError:
        return JSONResponse({
            "reply": reply,
            "tts_error": "timeout",
            "session_id": session_id,
        })
    except Exception as e:
        print(f"[voice/chat] TTS unavailable: {e}")
        return JSONResponse({"reply": reply, "tts_error": str(e), "session_id": session_id})


@app.get("/voice/status")
def voice_status():
    """Check whether the voice server (olly-voice-server) is configured and reachable."""
    import urllib.request
    cfg_voice = (_cfg.get("services", {}).get("voice_server", {}) if isinstance(_cfg, dict) else {})
    url = _VOICE_SERVER_URL
    install_hint = cfg_voice.get("install_hint", "")

    if not url:
        return {
            "available": False,
            "configured": False,
            "url": None,
            "message": "Voice server not configured. Set services.voice_server.url in config.yaml or VOICE_SERVER_URL env var.",
            "install_hint": install_hint,
        }

    try:
        req = urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3)
        reachable = req.status == 200
    except Exception:
        reachable = False

    return {
        "available": reachable,
        "configured": True,
        "url": url,
        "message": "Voice server reachable" if reachable else f"Voice server configured at {url} but not reachable. Start it first.",
        "install_hint": install_hint if not reachable else "",
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """STT via Gemma 4 E2B -- saves upload to temp file, runs infer_with_audio."""
    import tempfile
    from fastapi.concurrency import run_in_threadpool
    print(f"[transcribe] Request: {file.filename} ({file.content_type})", flush=True)
    data = await file.read()
    print(f"[transcribe] Payload: {len(data)} bytes", flush=True)
    if not data or len(data) < 256:
        return JSONResponse({"error": "STT failed: empty or too-short audio payload"}, status_code=400)

    # Prefer extension from filename; fallback from content-type for browser recordings.
    filename = file.filename or ""
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    else:
        ctype = (file.content_type or "").lower()
        if "ogg" in ctype:
            suffix = ".ogg"
        elif "wav" in ctype:
            suffix = ".wav"
        elif "mp3" in ctype or "mpeg" in ctype:
            suffix = ".mp3"
        else:
            suffix = ".webm"

    if suffix not in {".webm", ".ogg", ".wav", ".mp3", ".m4a", ".aac", ".flac"}:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    print(f"[transcribe] Saved to: {tmp_path} ({suffix})", flush=True)
    try:
        with voice_activity("stt"):
            text = await run_in_threadpool(
                mdl.infer_with_audio,
                tmp_path,
                "Transcribe exactly what is said in this audio. Output only the spoken words, nothing else.",
                256,
                None,  # history
                "stt",  # mode - explicitly force STT mode
            )
        print(f"[transcribe] Result: {repr(text[:120])}", flush=True)
        if isinstance(text, str) and text.startswith("[model_server error]"):
            return JSONResponse({"error": f"STT failed: {text}"}, status_code=500)
        if isinstance(text, str) and text.startswith("[model_client error]"):
            return JSONResponse({"error": f"STT failed: {text}"}, status_code=500)
        return {"text": text or ""}
    except Exception as e:
        print(f"[transcribe] Exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/voice/chat")
async def voice_chat(body: dict):
    """text -> agent.triage() -> _clone_voice_reply() -> WAV response.
    Returns WAV bytes with X-Assistant-Reply header.
    Falls back to JSON {reply} if TTS unavailable.
    """
    from fastapi.concurrency import run_in_threadpool
    user_text = (body.get("text") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    print(f"[voice/chat] User: {repr(user_text[:100])}", flush=True)
    if not user_text:
        return JSONResponse({"error": "missing text"}, status_code=400)

    # Full triage -- history, context, tools, skills
    try:
        print(f"[voice/chat] Starting triage (timeout={_VOICE_CHAT_TRIAGE_TIMEOUT_S}s)...", flush=True)
        reply = await asyncio.wait_for(
            run_in_threadpool(agent.triage, user_text, None, False, session_id or "voice-dashboard"),
            timeout=_VOICE_CHAT_TRIAGE_TIMEOUT_S,
        )
        print(f"[voice/chat] Triage complete: {repr(reply[:100])}", flush=True)
    except asyncio.TimeoutError:
        print(f"[voice/chat] Triage TIMEOUT after {_VOICE_CHAT_TRIAGE_TIMEOUT_S}s", flush=True)
        fallback = "I got your message, but processing timed out. Please try again."
        return JSONResponse({
            "reply": fallback,
            "timeout": "triage",
            "session_id": session_id,
        })
    except Exception as e:
        print(f"[voice/chat] Triage ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        fallback = f"Processing error: {e}"
        return JSONResponse({
            "reply": fallback,
            "error": str(e),
            "session_id": session_id,
        })

    if not reply or not str(reply).strip():
        print(f"[voice/chat] Empty reply from triage", flush=True)
        return JSONResponse({
            "reply": "I could not generate a response right now. Please try again.",
            "error": "empty_reply",
            "session_id": session_id,
        })

    reply = str(reply).strip()
    print(f"[voice/chat] Starting TTS clone...", flush=True)
    _log_interaction(user_text, reply, source="voice-dashboard")

    # TTS via _clone_voice_reply -- same function used by telegram_bot
    return await _build_voice_audio_response(reply, session_id=session_id)


@app.post("/voice/tts")
async def voice_tts(body: dict):
    """Generate cloned voice audio from text for dashboard retry/playback."""
    text = (body.get("text") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    if not text:
        return JSONResponse({"error": "missing text"}, status_code=400)
    with voice_activity("tts"):
        return await _build_voice_audio_response(text, session_id=session_id)


@app.delete("/voice/history/{session_id}")
async def delete_voice_history(session_id: str):
    """Clear voice session -- wipes JSON memory window (SQLite preserved)."""
    import core.memory.memory as _mem
    try:
        _mem.clear_chat(session_id)
    except Exception:
        pass
    return {"status": "cleared", "session_id": session_id}
