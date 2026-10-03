"""api/routers/system.py — system/version/workspaces, computer, and provider routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Route fns
resolve live api state through the _api() seam. Behavior-identical;
re-exported from api.__init__.
"""
import sys
import importlib
import json

from fastapi import APIRouter
from ..helpers import _resolve_config_path

from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter()


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.get("/system")
def system_info():
    _m = _api()
    return {
        "vram_free_mb": _m.mdl.vram_free_mb(),
        "can_spawn": _m.rep.can_spawn(),
        "max_replicas": _m.rep.MAX_REPLICAS,
        "context_budget_mb": _m.rep.CONTEXT_BUDGET_MB,
    }

@router.get("/version")
def get_version():
    _m = _api()
    from infra.updater import get_current_version as _get_current_version
    return {"version": _get_current_version()}

@router.get("/workspaces")
def list_workspaces_endpoint():
    _m = _api()
    from core.workspaces import list_workspaces as _list
    return _list()

@router.get("/computer/stream")
def computer_stream():
    """Server-sent events stream of the live computer-use screen (Desktop/Mobile/Dashboard).
    _m = _api()

    Each frame is the same caption + screenshot the Telegram watch streams; a
    broadcast hub in computer_use_bridge.publish_watch fans it out to every
    subscribed surface so the computer-use run is visible everywhere, not just
    Telegram. Screenshot is a base64 data-URI (or empty).
    """
    import time as _time
    from core.expansions.computer_use_bridge import subscribe_watch, unsubscribe_watch

    sub = subscribe_watch()

    def event_generator():
        last_beat = _time.time()
        try:
            while sub.active:
                try:
                    frame = sub.queue.get(timeout=15)
                    payload = {"caption": frame.get("caption", ""), "screenshot": frame.get("screenshot", None) or ""}
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_beat = _time.time()
                except Exception:
                    # Keepalive comment so proxies don't kill an idle connection.
                    if _time.time() - last_beat > 15:
                        yield ": ping\n\n"
                        last_beat = _time.time()
        finally:
            sub.active = False
            unsubscribe_watch(sub)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/computer/publish")
def computer_publish(body: dict):
    """Cross-process bridge: model_server POSTs each live frame here so the
    _m = _api()
    uvicorn process (which owns the SSE subscribers) fans it out to every
    /computer/stream consumer. The endpoint calls publish_watch directly, which
    fans out to in-process subscribers only — it never re-forwards back over
    HTTP, so there is no loop between the two processes.
    """
    from core.expansions.computer_use_bridge import publish_watch

    publish_watch(body.get("caption", ""), body.get("screenshot", None))
    return {"ok": True}

@router.post("/sim/mode")
def set_sim_mode(body: dict):
    """Enable/disable SIM_MODE — bypasses exec_shell approval gate for trajectory collection."""
    _m = _api()
    import os as _os
    enabled = bool(body.get("enabled", False))
    _os.environ["SIM_MODE"] = "true" if enabled else ""
    return {"sim_mode": enabled}

@router.get("/sim/mode")
def get_sim_mode():
    _m = _api()
    import os as _os
    return {"sim_mode": _os.environ.get("SIM_MODE") == "true"}

@router.get("/provider")

def get_provider_routing():
    """ADR-013: Return current provider routing table + availability."""
    _m = _api()
    from core.inference.provider import get_provider as _gp
    p = _gp()  # get existing singleton
    routing = {}
    for call_type in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
        prov = p.get_provider(call_type)
        model = p.get_model(prov, call_type)
        stream = p._streaming.get(call_type, False)
        routing[call_type] = {"provider": prov, "model": model, "streaming": stream}
    providers_m._cfg = _m._cfg.get("providers", {}) if isinstance(_m._cfg, dict) else {}
    return {
        "routing": routing,
        "collect_trajectories": _m._cfg.get("providers", {}).get("collect_trajectories", False),
        "hf_provider": providers_m._cfg.get("hf_provider") or getattr(p, "_m._cfg", {}).get("hf_provider", "deepinfra"),
        "model_catalog": providers_m._cfg.get("models", {}),
        "model_overrides": providers_m._cfg.get("model_overrides", {}),
        "call_types": ["task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"],
        "providers": ["local", "openai", "anthropic", "hf", "copilot", "openrouter", "google", "doubleword"],
    }

@router.post("/config/env")
def set_config_env(body: dict):
    """Set provider/API keys in kernel-evolving's environment and persist to .env.
    _m = _api()

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

@router.post("/provider/set")
def set_provider_routing(body: dict):
    """ADR-013: Hot-swap provider routing for one or more call types.
    _m = _api()
    Body: {\"task_inference\": \"openai\", \"model_override\": {\"task_inference\": \"gpt-5.4\"}, \"persist\": false}
    Session-scoped by default; set persist=true to write back to config.yaml.

    GPU safety rules (automatic, no user action needed):
    - Switching task_inference local → cloud: unloads model from VRAM first
    - Switching task_inference cloud → local: ensures model server is running
    """
    import yaml as _yaml
    from core.inference.provider import get_provider as _gp
    from core.inference.model_client import is_server_running
    p = _gp()  # get existing singleton — do not pass _m._cfg (would recreate)
    valid_call_types = {"task_inference", "synthesis", "critic", "planning", "trajectory_teacher", "vision", "stt", "tts"}
    valid_providers  = {"local", "openai", "anthropic", "hf", "copilot", "openrouter", "doubleword"}
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
                    p._m._cfg.setdefault("model_overrides", {})[ct] = m
                    changed[f"model_override.{ct}"] = m
        elif key == "collect_trajectories" and isinstance(val, bool):
            _m._cfg.setdefault("providers", {})["collect_trajectories"] = val
            changed["collect_trajectories"] = val
        elif key == "hf_provider" and isinstance(val, str) and val.strip():
            # HF Router inference provider (providers.hf_provider / HF_ROUTER_PROVIDER).
            p._m._cfg["hf_provider"] = val.strip()
            _m._cfg.setdefault("providers", {})["hf_provider"] = val.strip()
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
        cfg_path = _resolve_config_path()
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

@router.get("/provider/available")
def get_provider_availability():
    """ADR-013: Check which providers are ready (API keys set)."""
    _m = _api()
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
