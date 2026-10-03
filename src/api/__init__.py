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
from .schemas import (
    MessageIn, TaskIn, NamedReplicaIn, PipelineStage, PipelineIn,
    ReplicaMessageIn, BackupRequest, InitRequest, FreshRequest,
    NewSessionIn, PipelineJobStatus, EvolutionControlRequest, EvolutionTriggerRequest,
)
from .helpers import (
    _resolve_config_path, _log_interaction, _run_builtin_command,
    _conversations_repo, _safe_workspace_path,
)
from .routers.core import router as _core_router
from .routers.core import get_peers, health, root, chat_fresh, chat_session_new, api_sessions_list, api_sessions_create, message, message_stream  # noqa: F401 (re-exported for compatibility)
from .routers.debug_memory import router as _debug_router
from .routers.debug_memory import (  # noqa: F401 (re-exported for compatibility)
    debug_prompt_logs, debug_prompt_log, debug_trajectories, debug_current_prompt,
    debug_chat_history, debug_fields, debug_computer, memory_files, memory_file_read,
    memory_file_write, memory_file_delete, memory_file_rename, memory_file_new,
    memory_stats, workspace_tree,
)
from .routers.sqlite import router as _sqlite_router
from .routers.sqlite import (  # noqa: F401 (re-exported for compatibility)
    sqlite_tables, sqlite_table, sqlite_row_update, sqlite_row_delete,
    sqlite_table_clear, sqlite_db_clear, sqlite_row_insert,
)
from .routers.replicas import router as _replicas_router
from .routers.replicas import (  # noqa: F401 (re-exported for compatibility)
    _pipeline_jobs,
    list_skills, list_routines, spawn_replica, active_replicas, spawn_named_replica,
    replica_pipeline, _prune_pipeline_jobs, _run_pipeline_job, replica_pipeline_async,
    get_pipeline_job, stream_pipeline_job, cancel_pipeline_job, message_replica,
    stop_replica, replica_status,
)
from .routers.evolution import router as _evolution_router
from .routers.evolution import (  # noqa: F401 (re-exported for compatibility)
    evolve_backup, list_backups, evolve_init, evolve_init_status, evolve_fresh,
    evolution_state_endpoint, evolution_control, evolution_trigger, evolution_status,
    evolution_dashboard, evolution_stream, evolution_trajectories, get_task_trajectories,
    export_trajectories, get_thoughts, get_thoughts_today, trigger_think,
)
from .routers.system import router as _system_router
from .routers.system import (  # noqa: F401 (re-exported for compatibility)
    system_info, get_version, list_workspaces_endpoint, computer_stream, computer_publish,
    set_sim_mode, get_sim_mode, get_provider_routing, set_config_env, set_provider_routing,
    get_provider_availability,
)
from .routers.models_voice import router as _models_voice_router
from .routers.models_voice import (  # noqa: F401 (re-exported for compatibility)
    _fetch_openrouter_models, get_provider_models, _curated_local_models, _local_model_scan,
    _local_model_size, _repo_downloaded, list_models, curated_models, hub_search, pull_model,
    get_pull_job, assign_model_to_slot, get_thought_slot, set_thought_slot,
    _build_voice_audio_response, voice_status, transcribe_audio, voice_chat, voice_tts,
    delete_voice_history,
)

from typing import Optional
import yaml
import infra.bootstrap as _bootstrap
import asyncio
import os, json, re, time, uuid
import logging as _logging
from runtime_paths import LOGS_DIR, load_config as _load_config
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        _logging.StreamHandler(),  # stdout (captured by Docker/uvicorn)
    ]
)

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

from .routers.core import router as _core_router_inc
app.include_router(_core_router_inc)
app.include_router(_debug_router)
app.include_router(_sqlite_router)
app.include_router(_replicas_router)
app.include_router(_evolution_router)
app.include_router(_system_router)
app.include_router(_models_voice_router)

_IDLE_BYPASS_PATHS = {
    "/evolution", "/evolution/state", "/evolution/stream",
    "/health", "/version", "/system", "/thoughts", "/thoughts/today",
    "/evolution/dashboard", "/", "/peers",
    "/skills", "/routines", "/replica/active", "/evolution/trajectories",
    "/debug/prompt-logs", "/debug/prompt-log", "/debug/trajectories", "/debug/chat-history",
    "/debug/fields",
    "/provider", "/provider/available",
    "/models", "/models/curated", "/models/assign", "/pull", "/hub/search", "/jobs",
    "/config/thought-slot",
    "/memory/files", "/memory/file", "/memory/stats", "/workspace/tree",
    "/memory/file/rename", "/memory/file/new",
    "/sqlite/tables", "/sqlite/table", "/sqlite/row", "/sqlite/table/data", "/sqlite/db/data",
    "/voice/status",
    "/api/sessions",
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

# ────────────────────────────────────────────────────────────────────────────

_BUILTIN_COMMANDS = {
    "/start", "/help", "/skills", "/routines", "/run", "/skill",
    "/status", "/verbose", "/packages", "/search", "/install",
    "/clone", "/private_repo", "/update", "/restart", "/rollback",
    "/replica", "/workspaces", "/system", "/version", "/provider",
    "/new", "/fresh", "/session", "/voices", "/models", "/thoughts", "/evolve",
    "/init",
}

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
    _cfg = _load_config(_resolve_config_path())
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

    _config_path = _resolve_config_path()
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

# ── Memory & Workspace endpoints ────────────────────────────────────────────


# ── SQLite viewer/editor endpoints ──────────────────────────────────────────

# ── End Memory & Workspace endpoints ────────────────────────────────────────

# ── ADR-012: Async pipeline execution ──────────────────────────────────────

from fastapi import BackgroundTasks
# ── Local model management (Priority #7) ───────────────────────────────────
# Mirrors the pull/list/search/jobs mechanism from ai-server-py
# (src/routes/models.py), adapted for kernel-evolving's slot registry.
# Local models are stored in the HF hub-cache layout under HF_HOME/hub.

# Curated local-model catalog: model_slots defaults + multimodal candidates.
# Extends config.yaml `model_catalog` with slots + omni candidates.

# -- Voice dashboard endpoints -----------------------------------------------
# /transcribe  -- STT via Gemma 4 E2B native multimodal (infer_with_audio)
# /voice/chat  -- text -> agent.triage() -> TTS via _clone_voice_reply -> WAV
# /voice/history/{session_id} DELETE -- clear voice session

# Voice server URL: env > config.yaml services.voice_server.url > localhost default.
# Empty string disables voice features without error.
_VOICE_SERVER_URL = os.environ.get("VOICE_SERVER_URL", "")

