# kernel-evolving — Project Structure
> Generated: 2026-06-02 | Author: Olly

---

## Overview

kernel-evolving is a self-evolving BDI (Belief-Desire-Intention) AI agent. **Primary inference model: Nemotron-Labs-Diffusion-3B** (local GPU, 64k context, 16GB VRAM). Gemma 4 E2B-it runs as a secondary audio/multimodal slot. Cloud fallback via OpenAI, Anthropic, GitHub Copilot, and OpenRouter. The agent exposes a FastAPI HTTP interface, a Telegram bot, and a self-improvement loop that synthesises new skills and routines from user trajectories.

The codebase currently lives almost entirely in `src/` as a flat module collection (~20K lines). Fabio has started drafting a sub-folder structure (`core/`, `services/`, `views/`, `logs/`) — this document maps what exists today, proposes where each module belongs, and flags what's missing.

---

## Current `src/` File Inventory

### Flat modules (all in `src/` root today)

| File | Lines | Layer | Responsibility |
|------|------:|-------|----------------|
| `agent.py` | 639 | Core / Orchestration | Main triage loop, tool dispatch, BDI routing |
| `api.py` | 1,584 | Interface | FastAPI app, all HTTP endpoints |
| `api.py.bak` | — | — | ⚠️ Stale backup — safe to delete |
| `audio.py` | — | Service | Audio I/O (voice input/output helpers) |
| `backup.py` | 266 | Infra | Workspace backup/restore |
| `bootstrap.py` | — | Infra | First-run environment setup |
| `capability_verifier.py` | — | Evolution | Validates new skills before promotion |
| `code_synthesizer.py` | 436 | Evolution | LLM-driven skill/routine code generation |
| `context.py` | 708 | Core / Memory | Conversation context window builder |
| `critique_layer.py` | 249 | Evolution / Layers | Critic scoring of agent outputs |
| `discovery.py` | 347 | Infra | Peer agent discovery (port-scan + mDNS) |
| `embedding_client.py` | — | Core / Memory | Embedding API client (semantic search) |
| `evo_routine_executor.py` | 275 | Evolution | Executes routines as part of the evo loop |
| `evolution_dashboard.html` | — | Views | ⚠️ HTML file — belongs in `views/` |
| `evolution_hook.py` | 523 | Evolution | Hooks that trigger evolution from chat |
| `evolution_log.py` | 364 | Evolution | Structured evolution event log |
| `evolution_state.py` | 203 | Evolution | Persists evolution loop state (FSM) |
| `evolver.py` | 516 | Evolution | Main evolution orchestrator |
| `failed_requests.py` | — | Infra | Queue/retry for failed LLM requests |
| `fresh.py` | — | Infra | DB/state reset ("fresh start") |
| `goal_discovery.py` | 428 | Core | Infers latent goals from chat history |
| `init.py` | — | Infra | Module init / lazy loader |
| `long_term_memory.py` | — | Core / Memory | Long-term memory store (separate from STM) |
| `main.py` | — | Entry | Process entry point |
| `mcp_client.py` | — | Service | MCP (Model Context Protocol) client |
| `memory.py` | 585 | Core / Memory | Short-term conversation memory + SQLite |
| `micro_planner.py` | 350 | Core | Step-level task planner (BDI intentions) |
| `model.py` | 467 | Core / Inference | Model abstraction + provider routing shim |
| `model_client.py` | 365 | Core / Inference | HTTP client for local model_server |
| `model_server.py` | 2,331 | Core / Inference | GPU inference server (vLLM / HF backend) |
| `model_slots.py` | 229 | Core / Inference | Multi-slot model hot-swap manager |
| `observer_layer.py` | 366 | Core / Layers | Observes agent behaviour, emits signals |
| `pending_synthesis.py` | — | Evolution | Queues skills/routines pending synthesis |
| `probe_store.py` | — | Evolution | Stores evaluation probes for skill testing |
| `prompt_logger.py` | — | Infra | Raw prompt/response logging |
| `provider.py` | 610 | Core / Inference | Multi-provider routing (local/cloud/fallback) |
| `recommender.py` | — | Evolution | Recommends skills to synthesise next |
| `replica.py` | 322 | Core | Replica agent spawning + management |
| `routines.py` | 348 | Core | Routine loading, scheduling, execution |
| `runtime_paths.py` | — | Infra | Centralised runtime path resolution |
| `setup.py` | 214 | Infra | Install/setup helpers |
| `skills.py` | 204 | Core | Skill discovery, loading, dispatch |
| `telegram_bot.py` | 2,689 | Interface | Telegram bot (commands, inline buttons, callbacks) |
| `thought_engine.py` | 1,253 | Core | Think-at-rest proactive reasoning (ADR-005) |
| `thought_journal.py` | — | Core / Memory | Persists thought_engine outputs |
| `tools.py` | 596 | Core | Tool registry + execution harness |
| `trajectory_collector.py` | 310 | Evolution | Collects scored trajectories for fine-tune |
| `updater.py` | — | Infra | Self-update (git pull + restart) |
| `version.py` | — | Infra | Version string |
| `workspace_migration.py` | — | Infra | Workspace schema migrations |
| `workspaces.py` | — | Core | Multi-workspace management |

---

## Drafted Sub-folder Structure (`src/core/` et al.)

Fabio started this — current state (all empty/scaffold):

```
src/
├── core/
│   ├── evolution/        # empty
│   ├── inference/        # empty
│   ├── layers/           # empty
│   ├── memory/           # empty
│   ├── pipelines/        # empty
│   ├── replica/          # empty
│   └── tar/              # drafted for think-at-rest logic → drop; use services/thought_engine.py
├── logs/                 # empty
├── services/
│   └── channels/         # empty
└── views/                # empty
```

---

## Proposed Target Structure

Based on the actual module responsibilities, architecture (BDI + self-evolution pipeline), and the folder names Fabio drafted:

```
src/
│
├── core/                         # Agent brain — pure logic, no I/O
│   ├── __init__.py
│   ├── agent.py                  # BDI triage + orchestration
│   ├── tools.py                  # Tool registry + execution
│   ├── routines.py               # Routine loader + executor
│   ├── skills.py                 # Skill discovery + dispatch
│   ├── workspaces.py             # Multi-workspace management
│   │
│   ├── memory/                   # Memory subsystem
│   │   ├── __init__.py
│   │   ├── memory.py             # STM — conversation memory + SQLite
│   │   ├── long_term_memory.py   # LTM — persistent knowledge store
│   │   ├── context.py            # Context window builder
│   │   ├── embedding_client.py   # Embedding API client
│   │   └── thought_journal.py    # Thought engine output store
│   │
│   ├── inference/                # Model + provider layer
│   │   ├── __init__.py
│   │   ├── provider.py           # Multi-provider routing (ADR-013)
│   │   ├── model.py              # Model abstraction shim
│   │   ├── model_server.py       # GPU inference server (vLLM/HF)
│   │   ├── model_client.py       # HTTP client → model_server
│   │   └── model_slots.py        # Multi-slot hot-swap manager
│   │
│   ├── layers/                   # Interceptor / middleware layers
│   │   ├── __init__.py
│   │   ├── observer_layer.py     # Behaviour observer → signals
│   │   └── critique_layer.py     # Critic scoring layer
│   │
│   ├── pipelines/                # Multi-step execution pipelines
│   │   ├── __init__.py
│   │   ├── micro_planner.py      # Step-level BDI planner
│   │   └── goal_discovery.py     # Latent goal inference
│   │
│   ├── replica/                  # Replica / parallel agent support
│   │   ├── __init__.py
│   │   └── replica.py
│   │
│   └── evolution/                # Self-improvement loop
│       ├── __init__.py
│       ├── evolver.py            # Evolution orchestrator
│       ├── evolution_hook.py     # Chat → evolution triggers
│       ├── evolution_state.py    # FSM state persistence
│       ├── evolution_log.py      # Structured event log
│       ├── code_synthesizer.py   # LLM skill/routine codegen
│       ├── capability_verifier.py# Skill validation before promotion
│       ├── evo_routine_executor.py
│       ├── trajectory_collector.py
│       ├── pending_synthesis.py  # Synthesis queue
│       ├── probe_store.py        # Evaluation probes
│       └── recommender.py        # Next-to-synthesise recommender
│
├── services/                     # External I/O adapters
│   ├── __init__.py
│   ├── discovery.py              # Peer agent discovery
│   ├── audio.py                  # Voice I/O helpers
│   ├── mcp_client.py             # MCP protocol client
│   ├── thought_engine.py         # Think-at-rest proactive engine (ADR-005)
│   └── channels/                 # Per-channel delivery adapters
│       ├── __init__.py
│       └── telegram_bot.py       # Telegram bot (commands, callbacks)
│
├── views/                        # UI assets / rendered outputs
│   ├── __init__.py
│   └── evolution_dashboard.html  # Evolution status dashboard
│
├── logs/                         # Runtime log sinks (structured)
│   └── .gitkeep
│
├── database/                     # ⚠️ MISSING — suggest adding
│   ├── __init__.py               # Exports BaseRepository, get_conn
│   ├── base.py                   # BaseRepository ABC + connection factory
│   ├── memory/                   # Memory-domain databases
│   │   ├── __init__.py
│   │   ├── chat_history.py       # sessions + messages + attachments (chat_history_evolving.db)
│   │   └── long_term_memory.py   # memory_entries + memory_session_files (memory_store.db)
│   ├── evolution/                # Evolution-domain databases
│   │   ├── __init__.py
│   │   ├── evolution_log.py      # evolution_events + synthesis_trajectories (evolution.db)
│   │   ├── trajectory_store.py   # task_trajectories (evolution.db)
│   │   ├── probe_store.py        # probes (probes.db)
│   │   └── recommendation_store.py # recommendation_hits (evolution.db)
│   ├── agent/                    # Agent-runtime databases
│   │   ├── __init__.py
│   │   ├── promoted_signals.py   # promoted_signals (promoted_signals.db)
│   │   ├── failed_requests.py    # failed_requests (failed_requests.db)
│   │   ├── conversations.py      # conversations (conversations.db)
│   │   └── prompt_log.py         # prompt_log (system_debug_log.db)
│   └── migrations/               # Versioned schema migrations
│       ├── __init__.py           # Migration runner
│       ├── memory/
│       │   ├── 001_initial_schema.sql
│       │   └── 002_add_attachments.sql
│       ├── evolution/
│       │   ├── 001_initial_schema.sql
│       │   ├── 002_add_critic_fields.sql    # was ADR-010 inline ALTER TABLE
│       │   └── 003_add_synthesis_trajectories.sql
│       └── agent/
│           ├── 001_initial_schema.sql
│           └── 002_add_prompt_log_columns.sql
│
├── infra/                        # ⚠️ MISSING — suggest adding
│   ├── __init__.py
│   ├── runtime_paths.py          # Centralised path resolution
│   ├── backup.py                 # Backup/restore
│   ├── bootstrap.py              # First-run setup
│   ├── fresh.py                  # State reset
│   ├── updater.py                # Self-update
│   ├── version.py                # Version string
│   ├── workspace_migration.py    # File-level workspace migrations (legacy → canonical paths)
│   └── setup.py                  # Install helpers
│
├── api.py                        # FastAPI app + all HTTP routes (entry/interface layer)
├── main.py                       # Process entry point
└── __init__.py
```

---

## Notes & Observations

---

## Database Layer — Detailed Inventory

### Current state: 8 SQLite databases, schema scattered across 10+ modules

| Database file | Location key | Schema defined in | Owns tables |
|---|---|---|---|
| `chat_history_evolving.db` | `CHAT_HISTORY_DB` | `memory.py`, `init.py` | `sessions`, `messages`, `attachments` |
| `memory_store.db` | `LONG_TERM_MEMORY_DB` | `long_term_memory.py` | `memory_entries`, `memory_session_files` |
| `evolution.db` | `EVOLUTION_DB` | `evolution_log.py`, `evolver.py`, `trajectory_collector.py` | `evolution_events`, `synthesis_trajectories`, `recommendation_hits`, `task_trajectories` |
| `promoted_signals.db` | `PROMOTED_SIGNALS_DB` | `goal_discovery.py`, `init.py` ⚠️ | `promoted_signals` |
| `failed_requests.db` | `FAILED_REQUESTS_DB` | `failed_requests.py` | `failed_requests` |
| `probes.db` | `PROBES_DB` | `probe_store.py` | `probes` |
| `system_debug_log.db` | `SYSTEM_DEBUG_LOG_DB` | `prompt_logger.py` | `prompt_log` |
| `conversations.db` | `CONVERSATIONS_DB` | *(referenced in runtime_paths + workspace_migration only — schema TBD)* | *(unknown)* |

### Problems with current approach

1. **Schema fragmented across modules** — `promoted_signals` DDL appears in *both* `goal_discovery.py` and `init.py` (duplication risk, divergence over time)
2. **Inline migrations via `ALTER TABLE`** — ADR-006/007/010 migrations are `try/except ALTER TABLE` loops scattered inside `_conn()` methods — no version tracking, no rollback
3. **No connection factory** — every module calls `sqlite3.connect()` directly with its own pragma/row_factory setup, inconsistent WAL mode usage
4. **evolution.db shared by 3 modules** — `evolution_log.py`, `evolver.py`, `trajectory_collector.py` all open the same file independently — potential write contention
5. **`failed_requests.py` not OOP** — only non-class db module; uses module-level `_conn()` function
6. **`init.py::ensure_database_schema()`** — bootstraps schemas already owned by other modules (duplicates DDL from `memory.py` and `goal_discovery.py`)

### Proposed `database/` layer design

#### `database/base.py` — `BaseRepository` ABC

```python
class BaseRepository:
    """Abstract base for all SQLite repositories.
    Subclasses declare db_path and implement _ensure_schema().
    """
    db_path: Path

    def connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @abstractmethod
    def _ensure_schema(self, conn: sqlite3.Connection) -> None: ...

    def migrate(self, migration_dir: Path) -> None:
        """Run numbered .sql files in order, skip already-applied."""
        ...
```

#### Migration convention

- Migrations live in `database/migrations/<scope>/NNN_description.sql`
- A `_schema_version` table tracks applied migrations per database
- `BaseRepository.migrate()` runs the runner at startup — idempotent
- `ALTER TABLE` inline hacks are replaced by numbered SQL files
- Scope folders mirror the database sub-folders: `memory/`, `evolution/`, `agent/`

#### Connection ownership

| Database | Owner repository | Shared access |
|---|---|---|
| `chat_history_evolving.db` | `ChatHistoryRepository` | `context.py`, `observer_layer.py` (read-only) |
| `memory_store.db` | `LongTermMemoryRepository` | `embedding_client.py` (read-only) |
| `evolution.db` | `EvolutionLogRepository` | `TrajectoryStore`, `RecommendationStore` share same file but via **same repo instance** (passed as dep) |
| `promoted_signals.db` | `PromotedSignalsRepository` | `goal_discovery.py` (was defining DDL inline — moves here) |
| `failed_requests.db` | `FailedRequestsRepository` | None |
| `probes.db` | `ProbeRepository` | `capability_verifier.py` (read) |
| `system_debug_log.db` | `PromptLogRepository` | `api.py` (read for `/logs` endpoint) |
| `conversations.db` | `ConversationsRepository` | *(schema to define)* |

---

### ✅ What's well-structured already
- Clear separation between **inference** (model_server, model_client, provider) and **agent logic** (agent, tools, skills)
- Evolution subsystem is coherent and isolatable — all files have clear single responsibilities
- `trajectory_collector` → `code_synthesizer` → `capability_verifier` is a clean pipeline

### ⚠️ Issues to address during migration
| Issue | File | Suggestion |
|-------|------|-----------|
| HTML asset in src root | `evolution_dashboard.html` | Move to `views/` |
| Stale backup file | `api.py.bak` | Delete |
| `core/tar/` — unclear intent | `src/core/tar/` | Rename to `infra/archive/` or remove if unused |
| `telegram_bot.py` is 2,689 lines | Too large for one file | Split: `bot_core.py` (dispatch loop) + `bot_commands.py` (handlers) + `bot_callbacks.py` (inline button handlers) |
| `thought_engine.py` (1,253 lines) in root | Belongs in `services/` or `core/` | Move to `services/thought_engine.py` |
| `model_server.py` (2,331 lines) | Monolithic | Consider splitting: `model_server.py` (entry) + `inference/backends/` (vllm.py, hf.py, local.py) |
| No `infra/` folder | All infra files in root | Fabio didn't draft this — recommend adding it |
| No `database/` folder | DB schema in 10+ modules | Extract to `database/` with `BaseRepository` + versioned migrations |
| `promoted_signals` DDL duplicated | `goal_discovery.py` + `init.py` | Single `PromotedSignalsRepository` in `database/agent/` |
| Inline `ALTER TABLE` migrations | `evolution_log.py`, `prompt_logger.py` | Replace with numbered `.sql` migration files |
| `evolution.db` opened by 3 modules | `evolution_log.py`, `evolver.py`, `trajectory_collector.py` | Single `EvolutionLogRepository` owns the file; others receive it as a constructor dependency |
| `conversations.db` — no schema | `runtime_paths.py` defines path, nothing creates schema | Define `ConversationsRepository` to own it |

### 💡 Architecture pattern notes
- The layered design (observer → critique → agent → provider) maps well to a **middleware pipeline pattern** — `core/layers/` is the right home for this
- `core/pipelines/` could also house an explicit `Pipeline` base class so `micro_planner` and `goal_discovery` share a common interface
- `services/channels/` is the right abstraction for multi-channel delivery — Telegram today, could be Discord/Signal/API tomorrow; worth defining a `BaseChannel` interface there
- Keep `api.py` at `src/` root (not inside any subfolder) — it's the HTTP entrypoint and imports from all layers; burying it causes circular import risk

### 📦 Migration approach (safe, incremental)

#### Phase 0 — Foundations ✅ (PR #82, commit 99547df)
1. ✅ Create `database/base.py` with `BaseRepository` ABC + `MigrationRunner`
2. ✅ Create empty `database/migrations/<scope>/` directories
3. ✅ Extract existing DDL into numbered `.sql` files (no code changes yet)

#### Phase 1 — Database layer ✅ (PR #83, commit 8c80e0b)
4. ✅ Implement `FailedRequestsRepository`
5. ✅ Implement `ProbeRepository`
6. ✅ Implement `PromotedSignalsRepository` — deduplicated DDL
7. ✅ Tests pass

#### Phase 2 — Memory repositories ✅ (PR #84, commit 1554082)
8. ✅ `ChatHistoryRepository` — wraps `memory.py` schema
9. ✅ `LongTermMemoryRepository` — wraps `long_term_memory.py`

#### Phase 3 — Evolution repositories ✅ (PR #85, commit 79d5fff)
10. ✅ `EvolutionLogRepository` — single owner of `evolution.db`
11. ✅ Replace all `ALTER TABLE` inline hacks with migration files

#### Phase 4 — Folder restructure ✅ (PR #86)
12. ✅ Create `infra/` — moved `backup`, `bootstrap`, `fresh`, `updater`, `version`, `workspace_migration`, `setup`
13. ✅ Backward-compat shims at old paths
14. ✅ Removed empty `src/core/tar/` directory
15. ✅ Full test suite: 140 passed

#### Invariants (never break these)
- `api.py`, `main.py` stay at `src/` root until all dependencies settled
- `runtime_paths.py` is the last thing to move (everything imports it)
- Every phase ends with a green test run before the next begins
- No squash merges — keep atomic commits per file moved for clean `git bisect`

---

## Project Root Restructure

### Current root inventory

| File/Dir | Type | Keep at root? | Proposed location |
|---|---|---|---|
| `src/` | Source code | ✅ yes | — |
| `tests/` | Test suite | ✅ yes | — |
| `docs/` | ADRs + protocol docs | ✅ yes | — |
| `scripts/` | ML/fine-tune scripts | ✅ yes (rename consideration) | `scripts/` or `tools/` |
| `simulations/` | Simulation runs (sim6–sim17) | ⚠️ maybe | `experiments/simulations/` |
| `start.sh` | Main process launcher | ✅ yes | — |
| `start_bot.sh` | Bot-only launcher | ⚠️ secondary | `bin/start_bot.sh` |
| `install.sh` | Installer | ✅ yes | — |
| `entrypoint.sh` | Docker entrypoint | ✅ yes (Docker context) | — |
| `eva` | CLI binary (EVA persona) | ✅ yes | — |
| `kernel` | CLI binary (Kernel persona) | ✅ yes | — |
| `pyproject.toml` | Python project config | ✅ yes | — |
| `requirements.txt` | Deps | ✅ yes | — |
| `conftest.py` | pytest fixtures | ✅ yes | — |
| `config.yaml` | Runtime config | ✅ yes | — |
| `config.container.yaml` | Docker config variant | ⚠️ | `deploy/config.container.yaml` |
| `config.yaml.bak` | Stale backup | ❌ delete | — |
| `docker-compose.yml` | Compose file | ✅ yes (or `deploy/`) | `deploy/docker-compose.yml` |
| `Dockerfile` | Production image | ✅ yes (or `deploy/`) | `deploy/Dockerfile` |
| `Dockerfile.sandbox` | Sandbox image | ⚠️ | `deploy/Dockerfile.sandbox` |
| `Dockerfile.sandbox-lite` | Lite sandbox | ⚠️ | `deploy/Dockerfile.sandbox-lite` |
| `Dockerfile.test` | Test image | ⚠️ | `deploy/Dockerfile.test` |
| `kernel-evolving.service` | systemd unit | ⚠️ | `deploy/kernel-evolving.service` |
| `live_evolution_test.py` | Manual E2E test | ⚠️ | `tests/manual/live_evolution_test.py` |
| `startup.log` | Runtime log | ❌ gitignore | — |
| `.specs/` | Plans/specs | ✅ yes | — |
| `AGENTS.md` | Agent context | ✅ yes | — |
| `CHANGELOG.md` | Release history | ✅ yes | — |
| `CONTRIBUTING.md` | Contribution guide | ✅ yes | — |
| `README.md` | Project docs | ✅ yes | — |

### Proposed root after restructure

```
kernel-evolving/
├── src/                        # Source code (layered, see above)
├── tests/                      # Automated test suite
│   └── manual/                 # ⬅ move live_evolution_test.py here
├── docs/                       # ADRs, protocol docs, diagrams
├── scripts/                    # Fine-tune + ML tooling scripts
├── deploy/                     # ⬅ NEW — all deployment artefacts
│   ├── Dockerfile
│   ├── Dockerfile.sandbox
│   ├── Dockerfile.sandbox-lite
│   ├── Dockerfile.test
│   ├── docker-compose.yml
│   ├── config.container.yaml
│   └── kernel-evolving.service
├── experiments/                # ⬅ NEW — simulation runs, research
│   └── simulations/            # ⬅ move simulations/ contents here
├── bin/                        # ⬅ NEW — secondary launchers
│   └── start_bot.sh            # ⬅ move start_bot.sh here
├── .specs/                     # Plans, specs, ADRs in progress
├── eva                         # CLI entry point (EVA persona)
├── kernel                      # CLI entry point (Kernel persona)
├── start.sh                    # Primary process launcher (stays at root — referenced by systemd)
├── install.sh                  # Installer
├── entrypoint.sh               # Docker entrypoint (stays at root — referenced in Dockerfiles)
├── config.yaml                 # Runtime config
├── conftest.py                 # pytest root config
├── pyproject.toml
├── requirements.txt
├── AGENTS.md
├── README.md
├── CHANGELOG.md
└── CONTRIBUTING.md
```

### Files to delete
- `config.yaml.bak` — stale backup
- `startup.log` — runtime log, add to `.gitignore`

### Notes
- `start.sh` stays at root — the systemd unit and user muscle-memory reference it directly
- `entrypoint.sh` stays at root — Docker `ENTRYPOINT` references it by name
- `deploy/Dockerfile` move means updating the `docker-compose.yml` `build.dockerfile` path and any CI references
- `simulations/` are numbered experiment runs (sim6–sim17); they're research artefacts, not source — `experiments/` is the right semantic home
- `kernel-evolving.service` also has a stale description (`Gemma 4 E2B-it`) — fix the `Description=` line when moving

### Phase 5a — core/ sub-folders ✅ (issue #89, PR #92)
### Phase 5b — services/ sub-folders ✅ (issue #90, PR #93)
### Phase 5c — views/, logs/, cleanup ✅ (issue #91, PR #94)
### Phase 6 — Root restructure ✅ (PR #97, v1.23.0)
- Open issue #89
- Create `deploy/`, `experiments/simulations/`, `bin/`, `tests/manual/`
- Move files per table above
- Update `.gitignore` to include `startup.log`
- Update CI workflow paths if Dockerfile moves
- Run `pytest tests/ -x -q`

---

*Generated by Olly — kernel-evolving v1.22.0*
