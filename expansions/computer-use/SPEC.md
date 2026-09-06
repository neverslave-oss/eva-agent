# computer_use — Modular AI Computer-Use Expansion (Spec v0.1)

> **Author:** Olly + Fabio
> **Date:** 2026-09-06
> **Status:** Implemented baseline v0.1 (desktop-first + dry-run default)
> **Target repo:** `repositories/kernel-evolving`
> **Expansion root:** `expansions/computer-use/`

---

## 1) Goal

Add a first-party expansion that gives Kernel-Evolving a **modular computer-use stack** for web + desktop tasks, while keeping integration compatible with current kernel design patterns:

- **Sidecar expansion pattern** (like `expertise-field`)
- **Kernel bridge pattern** (`src/core/expansions/*_bridge.py`)
- **No-op degradation** when expansion is unavailable
- **Tool-first execution** (assist, bias, execute safely; never bypass safety gates)

This module must be reusable by both:
1. Kernel-Evolving runtime, and
2. OpenClaw-side orchestration flows.

---

## 2) Design principles (pattern-aligned)

1. **Separation of concerns**
   - Kernel core stays stable.
   - Expansion owns computer-use logic as sidecar.

2. **Adapter/Bridge boundary**
   - Kernel talks to expansion only via a bridge.
   - Expansion internals can evolve without kernel refactors.

3. **Strategy pattern for executors**
   - One action DSL, multiple drivers (`playwright`, `pyautogui`, `pywinauto`, later OpenClaw bridge).

4. **Pipeline pattern for runtime loop**
   - `observe -> plan -> act -> verify -> checkpoint`.

5. **Policy-first safety**
   - Destructive/external-sensitive steps require explicit policy + confirmation.

6. **Progressive disclosure in prompts**
   - Inject concise, actionable context; do not dump full module state into prompt.

---

## 3) Folder structure (initial)

```text
expansions/computer-use/
├── README.md
├── SPEC.md
├── AGENTS.md
├── computer_use_registry.json
├── policies/
│   ├── default.policy.json
│   ├── browser-allowlist.example.json
│   └── app-allowlist.example.json
├── docs/
│   ├── architecture.md
│   ├── dsl-actions.md
│   ├── safety-model.md
│   └── driver-contract.md
├── src/computer_use/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── schema.py
│   ├── router.py
│   ├── planner.py
│   ├── verifier.py
│   ├── safety.py
│   ├── state_store.py
│   ├── telemetry.py
│   ├── debug.py
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── screen.py
│   │   ├── dom.py
│   │   ├── ocr.py
│   │   └── matcher.py
│   └── drivers/
│       ├── __init__.py
│       ├── base.py
│       ├── playwright_driver.py
│       ├── pyautogui_driver.py
│       ├── pywinauto_driver.py
│       └── openclaw_driver.py
├── tests/
│   ├── test_schema.py
│   ├── test_router.py
│   ├── test_orchestrator.py
│   ├── test_safety.py
│   ├── test_state_store.py
│   └── test_driver_contracts.py
└── tmp/
```

---

## 4) Module responsibilities

### 4.1 `schema.py` (Typed DSL)
Defines strict Pydantic models for:
- Actions: `click`, `double_click`, `type`, `hotkey`, `navigate`, `scroll`, `wait`, `assert_text`, `assert_url`, `upload`, `submit`, `done`, `abort`
- Observation snapshots
- Verification targets
- Run constraints (max steps, timeout, retries)

**Pattern:** DTO + contract-first validation.

### 4.2 `orchestrator.py` (Runtime coordinator)
Owns the step loop:
1. get observation
2. ask planner for next action batch
3. validate actions against schema + safety policy
4. execute via selected driver
5. verify post-conditions
6. checkpoint state

**Pattern:** Pipeline + coordinator.

### 4.3 `router.py` (Driver + context routing)
Selects driver strategy based on target:
- browser/web -> playwright
- desktop generic -> pyautogui
- windows-native app -> pywinauto
- remote/OpenClaw tool runtime -> openclaw driver

**Pattern:** Strategy selection.

### 4.4 `planner.py` (LLM planning boundary)
Turns high-level goal + latest observation into 1..N atomic actions.
Returns schema-valid action proposals only.

**Pattern:** Pure planner adapter (no direct execution).

### 4.5 `verifier.py`
Checks expected outcomes after each action/batch.
Supports DOM assertions, OCR text checks, URL/window focus, and semantic “task complete” markers.

**Pattern:** Guard/quality gate.

### 4.6 `safety.py`
Central policy engine:
- allow/deny by action type
- domain allowlist for navigation
- app allowlist for desktop targeting
- sensitive-field masking in logs
- irreversible action checkpoint hooks

**Pattern:** Policy enforcement layer.

### 4.7 `state_store.py`
Persistent per-run state:
- current step index
- observation hash
- last successful checkpoint
- retry counters
- recovery hints

Backed by JSON initially; swappable later.

**Pattern:** Repository/state store.

### 4.8 `telemetry.py`
Structured events and metrics for each stage:
- step latency
- retries
- verification pass/fail
- abort reasons

**Pattern:** Observability decorator.

### 4.9 `perception/*`
- `screen.py`: screenshot capture (mss/Pillow)
- `dom.py`: browser DOM extraction hooks
- `ocr.py`: OCR (easyocr)
- `matcher.py`: template/feature matching (opencv)

**Pattern:** Perception providers with unified output model.

### 4.10 `drivers/*`
All drivers implement a common contract in `base.py`:
- `observe()`
- `execute(actions)`
- `verify(expectations)` (optional override)
- `recover(error)`

**Pattern:** Strategy + interface contract.

---

## 5) Kernel integration seam

Add new bridge:
- `src/core/expansions/computer_use_bridge.py`

Bridge API (proposed):
- `available() -> bool`
- `inject_computer_use_context(chat_id: str, text: str) -> str`
- `helper_plan_hint(chat_id: str, text: str) -> str`
- `run_computer_task(chat_id: str, goal: str, target: dict) -> dict`
- `debug_snapshot(chat_id: str, query: str) -> dict`

Rules:
- **Never crash kernel boot** if sidecar missing.
- Return safe no-op/default values on import/runtime failure.
- Log warnings via kernel logger only.

---

## 6) Registry + policies

### 6.1 `computer_use_registry.json`
Tracks:
- enabled drivers
- default driver preference order
- feature flags (OCR, DOM, recovery mode)
- limits (step budget, timeout)

### 6.2 Policy files (`policies/*.json`)
- global defaults
- environment overrides later
- deny-by-default for risky actions until explicitly enabled

---

## 7) Dependency starter kit (modern baseline)

Baseline:
- `pydantic>=2.13`
- `tenacity>=9.1`
- `structlog>=26.1`
- `httpx>=0.28`
- `Pillow>=12.3`
- `mss>=10.2`
- `opencv-python>=5.0`
- `easyocr>=1.7`

Execution drivers:
- `playwright>=1.62` (optional)
- `pyautogui>=0.9.54` (desktop default)
- `pynput>=1.8.2`
- `pywinauto>=0.6.9` (Windows optional)

Documentation + version audit is tracked in `docs/dependencies.md` and must be refreshed before release tags.

Note: Keep driver imports lazy to avoid hard dependency failures on unsupported hosts.

---

## 8) Test strategy (must exist from day one)

- **Unit tests:** schema validation, policy checks, router selection
- **Contract tests:** every driver conforms to `base.py`
- **Loop tests:** orchestrator happy path + recoverable failure path
- **Safety tests:** blocked navigation, blocked destructive actions, mask coverage
- **No live external side effects** in default suite

All tests isolated; no production state touched.

---

## 9) Milestone plan

### M0 — Foundation (completed)
- create folder structure
- define schema and interfaces
- add no-op-safe bridge in kernel core

### M1 — Desktop-first path (locked)
- implement pyautogui driver baseline
- implement screenshot/OCR verification baseline
- persist checkpoints in per-chat state store

### M2 — Browser interoperability
- keep compatibility with kernel native browser-use tool
- implement playwright driver only where module-driven browser steps are needed
- implement DOM + URL verifier for shared action DSL parity

### M3 — Windows-native refinement
- pywinauto driver
- window-aware selectors + focused-app policy

### M4 — OpenClaw bridge driver
- execute DSL via OpenClaw tool actions where available
- unify result format with local drivers

---

## 10) Locked decisions

Locked by user:
1. **Default target is desktop-first** (kernel already has native browser-use tooling).
2. **State retention is per-chat** (with per-run checkpoints under each chat state).
3. **Dependency policy:** avoid legacy dependencies; prefer current stable releases validated against official docs.
4. **Execution default:** `run_computer_task` uses `dry_run=true` by default in production until explicitly overridden by policy/intent.

---

## 11) Acceptance criteria for this spec stage

- Folder/module architecture documented and pattern-aligned.
- Bridge seam defined and no-op behavior explicitly required.
- Safety and testing sections present before implementation.
- Milestones clear enough for planner/builder subagents.

---

## 12) Completed tasks checklist

- [x] Spec created and aligned with sidecar + bridge architecture
- [x] Expansion folder structure created under `expansions/computer-use/`
- [x] Core modules implemented (`schema`, `planner`, `orchestrator`, `router`, `safety`, `state_store`, `verifier`, `debug`, `telemetry`)
- [x] Driver contracts and baseline drivers implemented (`pyautogui`, `pywinauto`, `playwright`, `openclaw`)
- [x] Perception adapters implemented (`screen`, `dom`, `ocr`, `matcher`)
- [x] Kernel bridge implemented at `src/core/expansions/computer_use_bridge.py`
- [x] Desktop-first default configured
- [x] Browser interoperability path retained
- [x] Per-chat state retention implemented with per-run checkpoints
- [x] Dry-run default implemented (`dry_run=true`)
- [x] Dependency policy documented with official docs + latest-version audit (`docs/dependencies.md`)
- [x] Happy-path + edge-case tests added and passing

---

## 13) Roadmap

### Phase 1 (current baseline complete)
- Deterministic planning pipeline
- Policy validation gates
- Driver routing and bridge execution

### Phase 2 (next)
- Real screenshot capture in driver path with OCR-backed assertions
- Rich expectation model (`assert_element`, `assert_window_focus`, retry windows)
- Persistent run resume API from per-chat cursor

### Phase 3
- Playwright-driven browser parity for module-driven browser tasks
- Domain allowlist enforcement + navigation policy hardening
- Unified action result schema across desktop + browser

### Phase 4
- OpenClaw native tool driver execution path (`browser`/`computer` action mapping)
- Recovery strategies for stale refs, focus loss, and partial failures
- Telemetry export for post-run analytics and evaluation

