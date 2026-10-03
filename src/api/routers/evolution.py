"""api/routers/evolution.py — evolution, trajectories, and thoughts routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Route fns
resolve live api state through the _api() seam; schemas from api.schemas.
Behavior-identical; re-exported from api.__init__.
"""
import sys
import importlib
import json
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

from ..helpers import _resolve_config_path

from ..schemas import (
    BackupRequest, InitRequest, FreshRequest,
    EvolutionControlRequest, EvolutionTriggerRequest,
)

router = APIRouter()


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.post("/evolve/backup")
def evolve_backup(body: BackupRequest):
    """Create a timestamped backup archive."""
    _m = _api()
    import infra.backup
    try:
        result = backup.create_backup(description=body.description, full=body.full)
        return {"status": "created", "backup": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/evolve/backups")
def list_backups():
    """List existing backups."""
    _m = _api()
    import infra.backup
    backups = backup.list_backups()
    return {"backups": backups, "count": len(backups)}

@router.post("/evolve/init")
def evolve_init(body: InitRequest):
    """Initialize kernel-evolving workspace and databases."""
    _m = _api()
    import init
    try:
        result = init.initialize(force=body.force)
        return {"status": result["status"], "result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/evolve/init")
def evolve_init_status(force: bool = False):
    """Return workspace initialization status using the same init mechanism."""
    _m = _api()
    import init
    try:
        result = init.initialize(force=force)
        return {"status": result.get("status", "unknown"), "result": result}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.post("/evolve/fresh")
def evolve_fresh(body: FreshRequest):
    """Reset to fresh state after creating backup. Requires confirmation."""
    _m = _api()
    import fresh
    try:
        if body.dry_run:
            result = fresh.dry_run_fresh()
        else:
            result = fresh.execute_fresh()
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@router.get("/evolution/state")
def evolution_state_endpoint():
    """Return current evolution state machine status."""
    _m = _api()
    import core.evolution.evolution_state as _evo_state
    return _evo_state.get_status()

@router.post("/evolution/control")
def evolution_control(body: EvolutionControlRequest):
    """Control evolution state: start, pause, resume, stop, reset."""
    _m = _api()
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

@router.post("/evolution/trigger")
def evolution_trigger(body: EvolutionTriggerRequest):
    """Manually trigger an evolution cycle for a given task."""
    _m = _api()
    import core.evolution.evolution_state as _evo_state
    import core.evolution.evolution_hook
    import yaml as _yaml

    # Temporarily allow one cycle even if paused (manual trigger overrides cap but not stop)
    status = _evo_state.get_status()
    if status["state"] == "stopped":
        return JSONResponse({"error": "Evolution is stopped. Use /evolution/control to start first."}, status_code=400)

    if body.cap:
        _evo_state.start(cap=body.cap)

    with open(_resolve_config_path()) as f:
        cfg = _yaml.safe_load(f)
    skills_dir = os.path.expanduser(cfg.get("skills_dir", "./skills"))

    # Manual trigger bypasses EVOLUTION_ENABLED flag — explicit user action
    old_flag = evolution_hook.EVOLUTION_ENABLED
    evolution_hook.EVOLUTION_ENABLED = True
    try:
        result = evolution_hook.maybe_evolve(body.task, cfg, skills_dir, infer_fn=_m.mdl.infer)
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

@router.get("/evolution")
def evolution_status():
    """Return evolution history, gaps, stats, and chart data for the dashboard."""
    _m = _api()
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
        skill_count_now = len(_m.agent._skills)
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

@router.get("/evolution/dashboard")
def evolution_dashboard():
    """D3 force-graph + timeline evolution monitoring dashboard."""
    _m = _api()
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

@router.get("/evolution/stream")
def evolution_stream():
    """Server-sent events stream for live evolution monitoring."""
    _m = _api()
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

@router.get("/evolution/trajectories")
def evolution_trajectories():
    """Return recent synthesis trajectories for fine-tuning monitoring."""
    _m = _api()
    from core.evolution.evolution_log import EvolutionLog
    log = EvolutionLog()
    return {"trajectories": log.get_trajectories(limit=100)}

@router.get("/trajectories/task")
def get_task_trajectories():
    """Return recent task trajectories (ADR-013)."""
    _m = _api()
    from core.evolution.trajectory_collector import get_collector
    col = get_collector(_m._cfg)
    with col._conn() as conn:
        rows = conn.execute(
            "SELECT id,ts,task,provider,model_name,call_type,critic_score,critic_verdict,artifacts,elapsed_s "
            "FROM task_trajectories ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return {"trajectories": [dict(r) for r in rows]}

@router.post("/trajectories/export")
def export_trajectories(min_score: float = 0.7, call_type: str = None):
    """Export task trajectories as JSONL (ADR-013). Returns path + count."""
    _m = _api()
    from core.evolution.trajectory_collector import get_collector
    col = get_collector(_m._cfg)
    path, count = col.export_jsonl(min_score=min_score, call_type=call_type or None)
    return {"path": path, "count": count, "min_score": min_score}

@router.get("/thoughts")
def get_thoughts():
    """Return last 20 thoughts from recent journals (today + yesterday)."""
    _m = _api()
    from core.memory.thought_journal import ThoughtJournal
    journal = ThoughtJournal(journal_dir=_m._cfg.get("thinking", {}).get("journal_dir"))
    thoughts = journal.read_recent(days=2)
    return thoughts[-20:] if len(thoughts) > 20 else thoughts

@router.get("/thoughts/today")
def get_thoughts_today():
    """Return recent thought journals as markdown text (today + yesterday)."""
    _m = _api()
    import datetime
    import glob
    from core.memory.thought_journal import ThoughtJournal
    journal = ThoughtJournal(journal_dir=_m._cfg.get("thinking", {}).get("journal_dir"))
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

@router.post("/think/trigger")
def trigger_think():
    """Manually trigger one Think-at-Rest cycle. For testing."""
    _m = _api()
    think = getattr(_m.app.state, "think_at_rest", None)
    if think is None:
        return {"status": "error", "message": "think_at_rest not initialized"}
    if not getattr(think, "_enabled", False):
        return {"status": "error", "message": "thinking disabled in config"}
    import threading
    t = threading.Thread(target=think._run_think_cycle, daemon=True)
    t.start()
    return {"status": "triggered", "message": "think cycle started in background"}
