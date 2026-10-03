"""api/routers/debug_memory.py — debug, memory, and workspace routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Route functions
resolve live api state (app.state, _cfg, mdl/rep/agent) through the _api() seam;
workspace helpers come from api.helpers. Behavior-identical; re-exported from
api.__init__.
"""
import sys
import importlib
import os
from datetime import datetime as _dt

from fastapi import APIRouter, Body as _Body
from fastapi.responses import JSONResponse

from ..helpers import _safe_workspace_path
from runtime_paths import WORKSPACE_ROOT as _WORKSPACE_ROOT

# Memory file filters (moved from api.__init__ — only used by these routes)
_MEMORY_ALLOWED_EXTS = {".md", ".txt", ".yaml", ".yml", ".json", ".sqlite", ".sqlite3", ".db", ".db3"}
_MEMORY_SKIP_DIRS = {".git", "__pycache__", ".db", ".db-journal"}

router = APIRouter()


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.get("/debug/prompt-logs")
def debug_prompt_logs(limit: int = 20, chat_id: str = ""):
    """Return recent prompt log entries for the UI inspector."""
    _m = _api()
    import prompt_logger as _pl
    return {"logs": _pl.query_recent(limit=limit, chat_id=chat_id)}

@router.get("/debug/prompt-log/{entry_id}")
def debug_prompt_log(entry_id: int):
    """Return one prompt log entry with the full prompt and history payload."""
    _m = _api()
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

@router.get("/debug/trajectories")
def debug_trajectories(limit: int = 12, call_type: str = "task_inference"):
    """Return recent tool-call trajectories for the agent inspector."""
    _m = _api()
    import core.evolution.trajectory_collector as _tc
    return {"trajectories": _tc.recent_trajectories(limit=limit, call_type=call_type or None)}

@router.get("/debug/current-prompt")
def debug_current_prompt(chat_id: str = ""):
    """Return the currently assembled system prompt preview for the active UI session."""
    _m = _api()
    import core.memory.context as _ctx
    import core.memory.memory as _mem

    cfg = getattr(_m.agent, "_config", {}) or _m._cfg or {}
    skills = getattr(_m.agent, "_skills", []) or []
    routines = getattr(_m.agent, "_routines", []) or []
    chat_id = chat_id or ""
    history = _mem.load(chat_id=chat_id)

    prompt = _ctx.build_system_prompt(
        cfg,
        skills,
        routines,
        vram_free_fn=_m.mdl.vram_free_mb if _m.mdl else None,
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

@router.get("/debug/chat-history")
def debug_chat_history(limit: int = 80, chat_id: str = "", include_all: bool = True):
    """Return recent persisted chat turns for dashboard live conversation views.
    _m = _api()

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

@router.get("/debug/fields")
def debug_fields(chat_id: str = "", query: str = ""):
    """(ADR-015, decision #4) Inspect the expertise-field module state.
    _m = _api()

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

@router.get("/debug/computer")
def debug_computer(chat_id: str = "", query: str = ""):
    """(Phase 4) Inspect the computer-use expansion state.
    _m = _api()

    Exposes whether the sidecar is loaded, the driver/registry config, and a
    debug snapshot. Degrades to a safe no-op JSON if the module is absent or
    errors, so the live kernel never breaks.
    """
    try:
        from core.expansions.computer_use_bridge import debug_snapshot
        return debug_snapshot(chat_id=chat_id, query=query)
    except Exception as e:
        return {
            "available": False,
            "reason": f"/debug/computer handler error: {e}",
            "chat_id": chat_id,
            "query": query,
        }

@router.get("/memory/files")
def memory_files():
    """List all memory-relevant files in the workspace."""
    _m = _api()
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

@router.get("/memory/file")
def memory_file_read(path: str = ""):
    """Read a workspace file by relative path."""
    _m = _api()
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

@router.put("/memory/file")
def memory_file_write(body: dict = _Body(...)):
    """Write/update a workspace file by relative path."""
    _m = _api()
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

@router.delete("/memory/file")
def memory_file_delete(path: str = ""):
    """Delete a workspace file by relative path (moves to trash if available)."""
    _m = _api()
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

@router.post("/memory/file/rename")
def memory_file_rename(body: dict = _Body(...)):
    """Rename a workspace file. Body: {from: str, to: str}"""
    _m = _api()
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

@router.post("/memory/file/new")
def memory_file_new(body: dict = _Body(...)):
    """Create a new workspace file. Body: {path: str, content: str}"""
    _m = _api()
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

@router.get("/memory/stats")
def memory_stats():
    """Return aggregated statistics about workspace memory files and databases."""
    _m = _api()
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

@router.get("/workspace/tree")
def workspace_tree(path: str = "/", depth: int = 2):
    """Directory tree listing relative to workspace root."""
    _m = _api()
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
