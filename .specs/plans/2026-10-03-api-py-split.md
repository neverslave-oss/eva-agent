# Plan: Split `api.py` (issue #2 — api.py scope)

**Date:** 2026-10-03
**Issue:** #2 — split oversized api.py (2947), handlers.py (2168), thought_engine.py (1641)
**Scope:** `src/api.py` ONLY (2947 lines, 101 top-level defs/classes, 86 route decorators, 13 Pydantic classes). `src/core/inference/handlers.py` and `src/services/thought_engine.py` are separate parts of the issue — NOT in this plan.
**Branch:** `refactor/api-split` (based on latest `origin/dev`, which includes merged #3/#11)
**Status:** ✅ DONE — PR #12 merged to `dev` (squash `8574906f`, 2026-10-03). `src/api.py` (2947) → package: 13 one-class-per-file schemas + helpers + 7 routers; `__init__.py` = 298 lines. Review caught + fixed missing `_pipeline_jobs` re-export (test_async_pipeline regression); final gate 95 passed. api.py scope of issue #2 COMPLETE.

## Hard rules (Fabio, non-negotiable)

1. **ONE CLASS PER FILE.** Every Pydantic model class gets its own file. No exceptions, no grouping.
2. Pure extraction only — no logic rewrites, no route/path/param/behavior changes, no renamed public endpoints.
3. Dedicated branch, chunked slices, test gate green + commit + push after EVERY slice. Never leave work uncommitted across an interruption.
4. API compat preserved via re-export imports (`# noqa: F401` pattern) so `from api import app` and any external importer keeps resolving.
5. Watch the latent-bug class that bit the telegram split: moved code referencing module-level globals that stayed behind — import them properly or route through the module seam; run an AST undefined-name scan on each new module before committing.

## Inventory (src/api.py)

**Classes (13) — each → own file:**
`MessageIn` (149), `TaskIn` (153), `NamedReplicaIn` (158), `PipelineStage` (170), `PipelineIn` (181), `ReplicaMessageIn` (185), `BackupRequest` (189), `InitRequest` (194), `FreshRequest` (199), `NewSessionIn` (387), `PipelineJobStatus` (1369), `EvolutionControlRequest` (1661), `EvolutionTriggerRequest` (1686)

**Shared helpers:** `_resolve_config_path` (30), `_log_interaction` (92), `_run_builtin_command` (120), `_conversations_repo` (380), `_safe_workspace_path` (734), `_prune_pipeline_jobs` (1380)

**Route groups (86 routes):**
- Health/core: `get_peers`, `health`, `root`
- Chat/sessions: `chat_fresh`, `chat_session_new`, `api_sessions_list`, `api_sessions_create`, `message`, `message_stream`
- Debug: `debug_prompt_logs/log/current/trajectories/chat_history/fields/computer`
- Memory/workspace: `memory_files/read/write/delete/rename/new/stats`, `workspace_tree`
- Sqlite: `sqlite_tables/table/row_update/row_delete/table_clear/db_clear/row_insert`
- Skills/routines/replicas: `list_skills`, `list_routines`, `spawn_replica`, `active_replicas`, `spawn_named_replica`, `replica_pipeline`, `message_replica`, `stop_replica`, `replica_status`
- Evolution: `evolve_backup/list_backups/init/init_status/fresh`, `evolution_state/control/trigger/status/dashboard/stream`
- System: `system_info`, `get_version`, `list_workspaces_endpoint`

## Target layout

```
src/api.py                     → thin app assembly (imports app, routers, models; ~150 lines)
src/api/__init__.py            → re-exports `app` + everything external code imports from api.py
src/api/schemas/               → 13 files (one class each)
    message_in.py, task_in.py, named_replica_in.py, pipeline_stage.py, pipeline_in.py,
    replica_message_in.py, backup_request.py, init_request.py, fresh_request.py,
    new_session_in.py, pipeline_job_status.py, evolution_control_request.py,
    evolution_trigger_request.py
src/api/helpers.py             → _resolve_config_path, _log_interaction, _run_builtin_command,
                                 _conversations_repo, _safe_workspace_path, _prune_pipeline_jobs
src/api/routers/
    core.py        → health/root/peers + chat/sessions/message (incl. stream)
    debug.py       → debug_* + memory/workspace
    sqlite.py      → sqlite_*
    replicas.py    → skills/routines/replicas/pipeline
    evolution.py   → evolve_* + evolution_*
    system.py      → system_info/version/workspaces
```

FastAPI `APIRouter` per module, `app.include_router(...)` in api.py with the same paths/prefixes.

## Slices

- **S1 — schemas:** 13 one-class files + `src/api/__init__.py`; re-export models from api.py. Gate: test_api green.
- **S2 — helpers:** `src/api/helpers.py` (shared helpers; route seam for module globals).
- **S3 — core router:** health/chat/sessions/message/message_stream.
- **S4 — debug + memory router.**
- **S5 — sqlite router.**
- **S6 — replicas router.**
- **S7 — evolution router.**
- **S8 — system router + api.py thin assembly; final full-gate run.**

Each slice: syntax + `pytest tests/test_api.py -q` (36 tests) green → commit → push. Final: also run the telegram gate (56 passed, 1 xfailed).

## Test gate
- `python3 -m pytest tests/test_api.py -q` after every slice (36 tests).
- Full: `python3 -m pytest tests/test_api.py tests/test_inline_buttons.py tests/test_telegram_missing_coverage.py tests/test_telegram_poll_resilience.py -q` → 36 + 56 + 1 xfailed.

## Out of scope
- `handlers.py` (2168) and `thought_engine.py` (1641) — separate follow-up plans for issue #2.