# Plan: Reactivate Think-at-Rest (thought engine starving since June 24)

**Created:** 2026-08-22
**Status:** implemented 2026-08-24 (branch `feature/think-at-rest-reactivation`; desktop on `feature/local-model-selection`) — pending user review
**Branch:** `fix/cloud-audio-vision-402-fallback` (issues live on the currently checked-out branch)
**Related:** `ADR-005` (Think-at-Rest), `ADR-012`, `ADR-019` (Observer/Critique), src/services/thought_engine.py

> ⚠️ **Pair-programming required.** This is a design spec for a shared session — no code
> changes were made. Do not implement from this doc alone.

## Problem

The Think-at-Rest engine has journaled **no new thought since June 24** (~2 months). The
daemon is enabled (`thinking.enabled: true`) and running (port 8779, pid 237295), but the
idle cycle never produces output — the bot is effectively "paused" at the thought layer.

## Root Cause (from audit, 2026-08-22, refined 2026-08-24)

The engine is **starved by design**, not crashed. The 2026-06-18 restructure made
curiosity generation conditional on real outward data, and **three gates compound** to
block nearly every cycle. However, in the current deployed config (`providers.task_inference: hf`,
cloud provider), **gate #3 is the decisive, first gate** — it alone guarantees starvation
in cloud mode, because the resident primary model is never loaded there.

### Gate ordering in practice (cloud mode)

In `_run_think_cycle()` (thought_engine.py), the residency check runs **first**:

```python
if not self._main_model_loaded():
    logger.debug("[ThinkAtRest] model not loaded — skipping curiosity generation")
    return
```

`_main_model_loaded()` reads `health()["main_model_loaded"]`, which is
`_vllm_enabled or _model is not None` (model_server.py `_handle_health`). In cloud mode
the primary text path routes to the HF Router and the local primary model is **never
loaded** — so this returns `False` on every idle cycle and the other two gates are never
even reached.

1. **`_main_model_loaded()` gate — PRIMARY in cloud mode** (thought_engine.py ~L635):
   the resident primary model is absent when `task_inference` is a cloud provider. This is
   the single decisive blocker in the current deployment.
2. **`_run_think_cycle()`** → `_has_exploration_changed()` (thought_engine.py ~L622):
   even with a model present, only proceeds if exploration data (git log / system health)
   hashed differently since the last cycle. Idle box with stable repos → hash never changes → skip.
3. **`ThoughtGenerator.generate(exploration_summary="")`** (thought_engine.py ~L141):
   returns `[]` when `exploration_summary` is empty:
   `if not exploration_summary: logger.debug(... no exploration data — skipping synthetic generation); return []`.

Net effect: idle time yields zero thoughts unless **both** a resident model exists **and**
 the environment changed. In cloud mode the first condition is never met.

### Contributing observations (not the cause)

- `model_activity.json` is healthy (`in_flight: 0`, `last_end_ts` today) → idle detection
  is **not** blocked.
- Last journaled entries (June 23–24) were a repeated low-value self-improvement stub
  ("11 active probes accumulating") — not genuine curiosity.
- The bot's `read` error (`Errno 21 Is a directory` on `.../workspace/tmp/`) is a
  separate, cosmetic tooling issue — unrelated to the pause.

## Design Direction (to settle during pairing)

Goal: keep the "real data first, no synthetic gap-filling" philosophy (which the 06-18
restructure correctly enforced) while removing the accidental starvation. Priorities are
layered (each layer is a prerequisite for the next):

### Priority #0 — Local thought slot (the enabler; fixes the primary cause in cloud mode)

> ✅ **Implemented 2026-08-24** (branch `feature/think-at-rest-reactivation`):
> - `model_client.infer_local(slot="audio", ...)` — local-slot text inference path.
> - `model_server._handle_infer_local` + dispatch — lazy-loads the multimodal slot.
> - `thought_engine` — `_local_slot_available()` replaces the hard `_main_model_loaded()`
>   gate; `ThoughtGenerator.generate()` and `ThoughtEvaluator.evaluate()` now use
>   `infer_local` (local Gemma slot) instead of cloud-routed `infer`.
> Tests: `tests/test_thought_engine.py` (11), `test_think_at_rest_memory.py`,
> `test_observer_layer.py`, `test_critique_layer.py` (40 total) — all green.

Give Think-at-Rest a **resident-or-lazy-loadable local model** independent of the cloud
text provider, so the residency gate can pass even when `task_inference` is a cloud
provider. Reuse the proven `_ensure_multimodal_slot()` pattern (model_server.py) that
already lazy-loads the **Gemma 4 E2B-it** slot for vision/STT.

**Design (Option B — native local thoughts on Gemma 4 E2B-it):**
- Reuse the existing `audio` slot (Gemma 4 E2B-it, `model_slots.audio`) as the thought
  model. It is text-capable, already wired into the SlotRegistry, and zero extra VRAM when
  already warm from an audio/vision request.
- Add a **local-slot inference path** for the generator/evaluator, e.g.
  `model_client.infer_local(slot="audio", ...)` that targets the multimodal slot directly
  instead of the cloud-routed primary. This is required: `ThoughtGenerator.generate()`
  currently calls `infer_draft`/`infer`, which in cloud mode route to the cloud provider.
- Replace the hard `_main_model_loaded()` gate with a *local-slot-available* check that
  **lazy-loads** Gemma on first idle cycle (mirroring `_ensure_multimodal_slot`), rather
  than skipping.
- **Option A (use/load resident primary when available)** remains valid for local-only
  deployments: keep primary resident or lazy-load it on first think cycle. Note the primary
  is a diffusion model (Nemotron-Labs-Diffusion-3B) — not ideal for text curiosity; Option B
  is preferred.
- **Tier 2 stays cloud:** if a thought surfaces and triggers evolution, the currently
  configured cloud model/provider (`providers.synthesis: hf`) handles synthesis — exactly
  as today. The *thought* is local; the *evolution* is cloud.

**VRAM note:** lazy-loading Gemma for thoughts contends with the `audio` slot's LRU
eviction. During an active think cycle, mark the thought slot non-evictable (or accept a
load on demand).

1. **Relax the exploration-change gate (`_has_exploration_changed`).**
   - Option A: add a max-staleness cap — if no thought in N hours, run a cycle anyway
     (with whatever data exists), instead of skipping indefinitely on an unchanged hash.
   - Option B: lower the bar to "recent real data exists" rather than "data hash changed".
   - Keep the hash as an *optimisation* (avoid repeating the identical thought), not as a
     hard gate.

2. **Decouple generation from an always-nonempty summary.**
   - `ThoughtGenerator.generate` should still refuse *synthetic-from-vacuum* generation,
     but should accept a real-but-sparse summary (e.g. system health alone, or a single
     changed repo) rather than requiring a rich tokenized summary.

3. **Revisit the residency gate — now superseded by Priority #0.**
   - The old `_main_model_loaded()` check is replaced by the local-slot lazy-load path. Keep
     a graceful log of *why* a cycle was skipped so silence is diagnosable, not invisible.

4. **Add signal diversity** so idle time isn't pinned to one repeated stub.
   - Refresh/cap the "active probes accumulating" self-improvement signal (it fired
     identically for 2 days and added no value).
   - Consider additional cheap real signals (git log on active repos, dependency-release
     checks, GPU/temp trends) that change over time and feed genuine curiosity thoughts.

5. **Observability.**
   - Log every skipped cycle with the *specific* gate that blocked it (no local slot / no
     change / no data / generator empty). Right now a skipped cycle is near-silent, which
     made this 2-month pause invisible.

### Priority #6 — Event-triggered cadence + state-change suppression (the quality layer)

> ✅ **Implemented 2026-08-24** (branch `feature/think-at-rest-reactivation`):
> - `_gather_performance_signals()` — state-change suppression: `stuck_probes` only
>   re-fires when the count changes or crosses the >10 threshold.
> - `_on_idle()` — delta-aware cadence: changed perf signals trigger a cycle (subject to
>   a min-interval floor), clock becomes a floor for slow-drift catch.
> Tests: `tests/test_thought_engine.py` still green.

Once unblocked, prevent the pipeline from degenerating into a **uniform clock heartbeat**
that re-fires the same observation every cycle. The current trigger is a pure clock
(`_on_idle` → `elapsed >= _thought_interval_s`, 1800s), and the anomaly detector
(`_gather_performance_signals`) re-reports a *confirmed* anomaly every cycle instead of
going quiet until the state changes. This is the same "event-driven vs time-sliced"
tension observed in the Think-at-Rest journal (rigid ~30-min cadence; `stuck_probes`
fired ~12×/day at the same score).

**Design:**
- **State-change suppression for anomaly signals.** `_gather_performance_signals()` should
  only emit `stuck_probes` when the underlying count *changes* (or crosses a threshold
  boundary), not when it merely stays >10. Track the last-emitted value per metric and only
  re-fire on delta. This is the single highest-leverage fix for the noise.
- **Delta-aware cadence.** Treat *changed* signals as the primary trigger and the clock as a
  **floor** (guarantee a slow-drift catch, e.g. RAM creep) rather than the primary driver.
  The outer `IdleDetector` is already event-driven; make the inner cadence gate fire on
  state change with a minimum-interval floor and a maximum-staleness cap.
- **Reframe Priority #1 Option A.** "Run a cycle after N hours" should not mean "run the
  same cycle on a clock" — it should mean "run a *minimal* cycle that reports only *what
  changed since the last cycle*." Preserves the no-synthetic-gap philosophy while avoiding
  the heartbeat.
- **Anomaly detection as the trigger.** The k-means expertise-field clustering (expansions/
  expertise-field) is the upstream inspiration: use delta/anomaly detection on real signals
  (sensor readings, service health, active probe count, VRAM, GPU temp) as the *event*
  that wakes a thought, rather than the clock. Real anomalies → fire; calm → go quiet.

### Priority #7 — Local model selection via the desktop app (pull + curated list + slot)

> ✅ **Implemented 2026-08-24** (branch `feature/think-at-rest-reactivation`):
> - **kernel-evolving** `src/api.py`: `GET /models`, `GET /models/curated`, `GET /hub/search`,
>   `POST /pull` (background job), `GET /jobs/{id}`, `POST /models/assign`. Mirrors
>   ai-server-py, adapted for the slot registry. New paths added to `_IDLE_BYPASS_PATHS`.
> - **kernel-desktop-v1** (branch `feature/local-model-selection`, merged to `main`):
>   `Settings.php` + `settings.blade.php` — "Local Model Management" section in the Model
>   Storage tab with downloaded-models list, curated catalog (Pull / assign-to-slot), and
>   Hub search.
> - **Telegram bot** `telegram_bot.py`: unified `/models` + `/provider` flow — `/models`
>   shows a provider selector; `/models provider <name>` lists that provider's models
>   (local = pulled/curated with Use/Pull buttons; cloud = model list routed to
>   `task_inference` with optional `model_override`). Short opaque callback tokens
>   (`lm0`, `lm1`) keep buttons under Telegram's 64-byte `callback_data` limit.
> - **Dedicated models folder + portability (final cleanup):**
>   - `runtime_paths.MODELS_DIR = WORKSPACE_ROOT/models` (added to `MANAGED_DIRS`).
>   - `POST /pull` now downloads into `MODELS_DIR` (not the shared HF cache), so
>     kernel-evolving's own pulled models are stored separately and incompatible
>     shared-cache models (FLUX, TTS voices, etc.) never appear in the local list.
>   - `_local_model_scan()` scans only `MODELS_DIR` (+ its `hub` subdir) and configured
>     `model_slots` paths — NOT the whole shared HF cache.
>   - `GET /models?with_size=true` computes sizes lazily via a portable,
>     bounded `os.scandir` walk (no `du -sb`, works on Linux/macOS/Windows).
> - **Deployment fix — lazy model server in cloud mode (`start.sh`, 2026-08-24):**
>   - Originally `start.sh` only started the model server when `task_inference == local`,
>     so in cloud mode (`task_inference: hf`) the local thought slot was never available
>     and Priority #0 was inert (no thoughts generated).
>   - Now in cloud mode the model server starts in **lazy mode** (`--lazy`) — the Gemma
>     thought slot lazy-loads on first idle cycle instead of reserving VRAM for the
>     cloud-only primary. `KERNEL_EVO_SKIP_MODEL_SERVER=1` still opts out entirely.
>   - Verified end-to-end: `infer_local(slot="audio")` lazy-loads Gemma and returns text
>     while `task_inference: hf`.
> Tests: `tests/test_api.py` (33) + related thought tests (51) + full suite
> (707 passed) — all green.

Make local model management usable from the **desktop app** (kernel-desktop-v1) by adding
REST endpoints to **kernel-evolving** that mirror the proven pull mechanism in the
`ai-server-py` project (`src/routes/models.py`). Today kernel-evolving has **no** local
model pull/list endpoints — local models are configured only by hand-editing
`config.yaml` `model_slots`. The desktop app already proxies to kernel-evolving on port
8779 (`Settings.php` → `http://127.0.0.1:8779/...`) and already browses the HF hub-cache
layout (`scanModels()`), so the missing piece is the pull/assign API.

**Mirror from `ai-server-py` (adapt, don't copy blindly):**
- `POST /pull` → background job (`snapshot_download`) + job DB. Returns `{status, job_id}`.
- `GET /models` → list local + cached models (manifest + HF snapshot scan + GGUF).
- `GET /hub/search?q=` → `HfApi.list_models(search=...)` for the curated/research list.
- `GET /jobs/{id}` → poll background pull progress.

**Kernel-evolving-specific adaptations:**
- **Target the slot registry, not a single cache.** ai-server loads one model into a cache;
  kernel-evolving uses named `model_slots` (`primary`, `audio`, `tool_calling`, + a future
  `thought` slot). After a pull, the endpoint should be able to **assign the pulled model to
  a slot** (write `config.yaml` `model_slots.<name>.model_path`), so the desktop UI can
  select "pull Qwen2.5-Omni-3B → assign to `audio` slot".
- **Reuse the SlotRegistry lazy-load path.** Pulling should populate the HF cache; the
  slot remains lazy-loaded on first use (mirroring `_ensure_multimodal_slot`), not
  force-loaded.
- **Curated list.** Expose a small curated catalog (the `model_slots` defaults + the
  multimodal candidates: `google/gemma-4-E2B-it`, `Qwen/Qwen2.5-Omni-3B`, and optionally
  `deepseek-ai/Janus-Pro-7B`) alongside the live `/hub/search` results, so the desktop app
  can offer both "pick from curated" and "search the Hub".
- **Env/VRAM safety.** Reuse the existing GPU-safety rules from `/provider/set` (unload
  when leaving local; ensure model server alive when switching to local).

**Desktop app (kernel-desktop-v1) work:**
- Add a **Models** surface (or extend `Settings.php`) that calls the new kernel endpoints
  through a Laravel service (`KernelEvolvingService`-style proxy), reusing the existing
  `scanModels()` UI for browsing, plus new controls to pull from the curated list / Hub
  search and assign to a slot.
- Poll `GET /jobs/{id}` for pull progress and surface it in the UI.

**Rationale:** This closes the loop for Priority #0 — once a user can pull + assign a
local model (e.g. Gemma 4 E2B-it to the `thought`/`audio` slot) from the desktop app, the
Think-at-Rest local thought slot becomes manageable without hand-editing config, and the
cloud-mode starvation fix is actually operable by a non-terminal user.

## Open Questions for the pairing session

- **Which local slot serves thoughts?** Prefer the existing `audio` slot (Gemma 4 E2B-it)
  vs a dedicated thought slot. (Leaning: reuse `audio` — zero extra VRAM when warm.)
- **Should a cycle ever run with *zero* real data**, or always require at least one real
  signal (health OR git)? (Leaning: always require ≥1 real signal, but don't require it to
  have *changed*.)
- **What cadence/rate-limit feels right once unblocked?** (`thought_interval_s` is 1800s.)
  With event-triggering, what min-interval floor and max-staleness cap?
- **Does the same starvation affect `observer`/`critique` layers**, or only the curiosity
  generator? (Audit suggested observer is fine.)
- **VRAM policy**: is the Gemma thought slot allowed to evict/be-evicted by the `audio`
  slot's LRU, or should it be pinned during an active think cycle?

## Acceptance Criteria

- Idle periods produce fresh journaled thoughts again (visible in
  `~/.kernel-evolving/workspace/thoughts/YYYY-MM-DD.md`), **including in cloud mode**
  (`task_inference: hf`) via the local Gemma thought slot.
- No regression to the "no synthetic gap-filling" intent: thoughts must come from real
  signals (git/health/metrics), not generated-from-vacuum.
- Skipped cycles are explicitly logged with the blocking gate.
- **The same signal must NOT be re-journaled on consecutive cycles unless its underlying
  state changed** (state-change suppression — the testable form of the event-triggered
  layer).
- Evolution (Tier 2) still routes to the configured cloud provider; only the *thought*
  generation is local.
- Existing tests (tests/test_thought_engine.py, test_think_at_rest_memory.py,
  test_observer_layer.py, test_critique_layer.py) stay green.

## Priority #7 Acceptance Criteria (desktop local-model selection)

- kernel-evolving exposes `POST /pull`, `GET /models`, `GET /hub/search`, `GET /jobs/{id}`
  (mirroring ai-server-py), and a slot-assignment capability.
- The desktop app lets the user pull a model from a curated list or Hub search and assign
  it to a named `model_slots` entry, with pull progress surfaced via job polling.
- Pulled models are lazy-loaded by the SlotRegistry on first use (no forced load), and
  the GPU-safety rules from `/provider/set` are respected.
- Picking `local` as a provider in the desktop Settings resolves against the newly pulled
  local models.
