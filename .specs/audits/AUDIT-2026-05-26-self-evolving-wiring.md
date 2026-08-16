# Audit — kernel-evolving self-evolving wiring (2026-05-26)

## Summary

- **Goal discovery IS wired end-to-end** but `infer_fn=None` is passed from `goal_discovery.py` into `maybe_evolve`, so the `capability_verifier` always **fails open** (returns `True`) in background goal-discovery cycles — skills are acquired without LLM verification.
- **Deferred queue drains correctly** (one pattern per cycle when idle), but `_is_system_idle()` silently returns `False` during `SIM_MODE`, meaning deferred patterns never drain in simulation.
- **Trajectory collector records to SQLite** but nothing downstream consumes the stored rows automatically — no fine-tune trigger, no feedback into the evolver's scoring. It's a populated database with a manual export endpoint only.
- **Discovered peers are not used** for cooperation, registry sync, or delegation (delegation.enabled = false); peers are only logged.
- **`eval_tasks.py` does not exist** — referenced in the task scope but absent from the codebase; `recommender.py` output also does not feed back into evolver scoring.

---

## Findings

### 1. Goal discovery → Evolver

- **Status:** ✅ wired (background thread calls evolver) but ⚠️ capability verifier bypassed
- **Evidence:**
  - `goal_discovery.py:214-215` calls `maybe_evolve(pattern, config, skills_dir=skills_dir)` — **no `infer_fn` passed**
  - `evolution_hook.py:34` passes `infer_fn` to `Evolver.__init__`
  - `evolver.py:282-292` skips verifier loop when `self._infer_fn is None`, setting `best = candidate` immediately on first match
- **Detail:** When evolution is triggered from background discovery (not from `agent.triage`), the capability verifier is skipped. Every ADR-006 `verify()` call is bypassed; skills are acquired based on keyword/semantic score alone. The `/evolution/trigger` API endpoint does pass `infer_fn=mdl.infer` (api.py:~548), so manual triggers use verification correctly.
- **Risk:** medium — skills may be installed that cannot handle the target task; false-positives inflate the evolution log
- **Recommendation:** Pass `infer_fn` from `goal_discovery.py`; expose model infer via a singleton accessor to avoid circular imports.

### 2. Evolver internals

- **Status:** ✅ `_score_match_semantic` + `_combined_score` reachable; ⚠️ keyword fallback can shadow semantic in low-embedding scenarios
- **Evidence:**
  - `evolver.py:114-128` `_combined_score` returns semantic score if `> 0.0`, else falls back to normalised keyword score
  - `agent.py:136-149` uses `find_skill_semantic` (skills.py) **before** falling through to `maybe_evolve`; if semantic skill match fires, evolver never runs — correct
  - `evolver.py:202` `search_ecosystem` calls `_combined_score` for every candidate; results sorted descending
  - Install path: `evolver.run()` → `self.acquire(best)` → `subprocess.run(["git", "clone", ...], dest)` for remote; `shutil.copytree` for ecosystem sub-dir. Locally-found skills skip clone. Chain is complete.
- **Detail:** If embedding server (`8770`) is down, `EmbeddingClient.similarity()` returns `None` → semantic score `0.0` → falls back to keyword. Keyword max is ~20, normalised to `0-1`; a skill scoring 7/20 (keyword) = 0.35 just meets `min_candidate_score`. Not a bug but a degraded path worth monitoring.
- **Risk:** low
- **Recommendation:** Log a warning when embedding fallback fires so degraded scoring is visible in the evolution dashboard.

### 3. Capability verifier

- **Status:** ⚠️ wired in `evolver.run()` but bypassed when `infer_fn=None` (all background discovery calls)
- **Evidence:**
  - `evolver.py:278-295`: `if self._infer_fn is not None: can_handle, reasoning = capability_verifier.verify(...)` — guarded by `None` check
  - `capability_verifier.py:28`: `fails open` on inference error (explicit comment + return `True`)
  - `evolution_hook.py:34`: `Evolver(config=config, skills_dir=skills_dir, infer_fn=infer_fn)` — `infer_fn` comes from caller
  - `goal_discovery.py:214-215`: calls `maybe_evolve(pattern, config, skills_dir=skills_dir)` — `infer_fn` absent → `None`
- **Detail:** The verifier is only active for manual `/evolution/trigger` calls (api.py passes `mdl.infer`) and direct `triage`-path calls. All automated background discovery runs skip it. `fails open` design means even when verifier IS called with an error, it passes the skill through.
- **Risk:** high (for automated discovery) / low (for manual trigger)
- **Recommendation:** Pass `infer_fn` from `goal_discovery.py`; consider a stricter fail-closed option behind a config flag for production.

### 4. Thought engine / ThinkAtRest

- **Status:** ✅ wired; output feeds back into memory, goals, and skills via multiple paths
- **Evidence:**
  - `thought_engine.py:270-295` `_on_thought_accepted()`: score ≥ promote_threshold → `_promote_to_idea()` → writes idea file + calls `goal_discovery.record_promoted(weight=3)` (thought_engine.py:315)
  - `thought_engine.py:298-305` `category == "gap_reflection"` → `_maybe_trigger_evolution()` → `evolution_hook.maybe_evolve()`
  - `thought_engine.py:308-311` `category == "self_improvement"` → appends to `todos.md`
  - `thought_engine.py:314-323` score ≥ 0.85 → proactive Telegram message
  - `EvolvingThinkAtRest._on_thought_accepted()` (thought_engine.py:473-480): `gap_reflection` at score ≥ 0.6 also feeds `goal_discovery.record_unhandled()`
  - `_run_memory_seeker()` extracts user facts → `user.json` every cycle
- **Detail:** Well wired. The loop is: idle → generate seeds → evaluate → journal → promote → goal_discovery → evolver. The only weak link is that `_maybe_trigger_evolution()` also calls `maybe_evolve` without `infer_fn` (thought_engine.py:300-310 imports `evolution_hook` directly, no infer_fn passed).
- **Risk:** low (functionally complete); medium for the same verifier-bypass issue
- **Recommendation:** Same as #1 and #3 — inject `infer_fn` via singleton.

### 5. Trajectory collector

- **Status:** ⚠️ wired-but-no-consumer (writes JSONL to SQLite; no automatic downstream consumption)
- **Evidence:**
  - `agent.py:~295-340`: `collect_trajectories: true` → spawns background thread → `TrajectoryCollector.record()` → inserts to `task_trajectories` table in `~/.kernel-evolving/workspace/evolution.db`
  - `trajectory_collector.py:62-107`: `export_jsonl()` writes HF-compatible JSONL
  - `api.py:GET /trajectories/task` and `POST /trajectories/export`: manual query/export endpoints only
  - `config.yaml:hf_dataset_repo: PacificDev/kernel-evo-trajectories` — config key defined but **no code reads or pushes to this repo automatically**
  - `evolution.finetune_gate.enabled: false` — the autonomous fine-tune gate is disabled
- **Detail:** The trajectory pipeline is "collection-ready" but the consumption end is entirely manual. No code auto-triggers fine-tuning, pushes to HF, or feeds trajectory scores back into `evolver._combined_score`. The `hf_dataset_repo` config key is orphaned.
- **Risk:** low (collection works), medium (value never realized without manual intervention)
- **Recommendation:** Implement an auto-push job or enable `finetune_gate.enabled: true`; remove or implement `hf_dataset_repo` auto-push.

### 6. Async pipeline + replicas

- **Status:** ✅ wired; replicas are on-demand (spawned per request, not pre-spawned)
- **Evidence:**
  - `replica.py:12`: `MAX_REPLICAS = 4` — capacity cap, not pre-spawned count
  - `replica.py:151`: `can_spawn()` checks `len(_replicas) < MAX_REPLICAS and vram_ok` — VRAM-gated
  - `api.py:/health` returns `active_replicas: len(rep.active())` — reflects live count; at idle this is `0`
  - Replicas are spawned per `/replica/spawn`, `/replica/named`, or `pipeline` endpoints; cleaned up by `rep.stop()` after each pipeline stage
  - No auto-spawn at startup; `active_replicas: 4` in health response means 4 concurrent pipeline stages were running at snapshot time
- **Detail:** No race condition in spawning; the `_lock` in `replica.py` guards `_replicas`. The `/replica/pipeline` endpoint (sync version, api.py:~460) calls `rep.stop()` per stage but `_run_pipeline_job` (async) also has a duplicate definition (api.py has two `_run_pipeline_job` and two `_prune_pipeline_jobs` definitions — dead code duplication from ADR-012 refactor).
- **Risk:** low (functional); medium for duplicate function definitions — last one wins in Python, may silently drop stage output injection
- **Recommendation:** Remove the first `_run_pipeline_job` / `_prune_pipeline_jobs` definitions (sync versions, ~api.py:500-540) — they are shadowed by the async versions defined later.

### 7. Discovery / peer agents

- **Status:** ❌ peers discovered but not used (delegation.enabled = false)
- **Evidence:**
  - `discovery.py`: scans ports, registers peers in `_peers` dict, exposes via `/peers`
  - `agent.py:206-226`: delegation hook reads `_discovery.get_peers_by_role("coding")` and logs intent — comment reads "actual delegation (POST to peer URL) is future work"
  - `config.yaml:delegation.enabled: false` — master switch off
  - `api.py:startup`: starts `AgentDiscovery`, stored at `app.state.discovery`; peers available at runtime
  - No other code path reads `_discovery` or sends messages to peer URLs
- **Detail:** Discovery runs, peers (including `openclaw:18789`) are registered, but zero cooperation happens. The delegation hook in `agent.py:220` explicitly documents this as passive/future work.
- **Risk:** low (by design); medium if user expects peer cooperation
- **Recommendation:** Enable delegation for coding peers as the next ADR when stability is confirmed; until then, document that discovery is monitoring-only.

### 8. Memory plumbing

- **Status:** ✅ all writers/readers consistent; minor orphan risk with per-chat JSON files
- **Evidence:**
  - `memory.py`: two-tier: JSON window (`~/.kernel_evolving_memory*.json`) + SQLite (`chat_history_evolving.db`)
  - `memory.load()` → JSON first, SQLite cold-start seed
  - `memory.save()` → writes both JSON and SQLite atomically
  - `memory.record_attachment()` + `recent_attachments()` + `attachment_context_block()` all use same SQLite DB
  - `telegram_bot.py:378-486`: `_search_collective_memory()` (reads `~/.openclaw/workspace/collective-memory/scripts/search.py`) and `_write_collective_memory()` (writes to `~/.openclaw/workspace/collective-memory/entries/`) — separate store, not backed by kernel-evolving SQLite
  - `memory.clear()` wipes JSON window only; SQLite preserved — documented
  - Per-chat JSON files (`~/.kernel_evolving_memory_<chat_id>.json`) accumulate on disk with no TTL or cleanup
- **Detail:** No orphan stores within kernel-evolving itself. Collective-memory bridge is a separate flat-file store; inconsistency only arises if search.py is missing (graceful fallback). Per-chat JSON accumulation could grow unbounded.
- **Risk:** low; medium long-term (disk accumulation)
- **Recommendation:** Add a periodic cleanup job for per-chat JSON files older than N days; document collective-memory bridge dependency.

### 9. Eval tasks → micro-planner → recommender feedback loop

- **Status:** ❌ `eval_tasks.py` does not exist; recommender output is logged but does NOT feed back into evolver scoring
- **Evidence:**
  - `ls src/eval_tasks.py` → FILE NOT FOUND
  - `recommender.py:recommend()` returns `list[SkillRecommendation]` — consumed only in `evolver.py:344-369` where names are added to `EvolutionResult.recommendations` field
  - `evolution_log.py` records recommendations in the DB; no downstream code reads them to influence future `_combined_score` or `_min_candidate_score`
  - `micro_planner.py:PlanExecutor` uses a critic to verify individual steps; results are journalled but not fed back to the evolver
  - `trajectory_collector.py` collects task trajectories independently of recommender/evolver; no cross-reference
- **Detail:** These three components are independent silos. Recommender surfaces useful names into `EvolutionResult` and logs them, but the loop does not close — evolver never reads prior recommendations to boost future candidate scores. Micro-planner critic verifies step completion but produces no signal usable by the evolver.
- **Risk:** medium — the recommendation layer adds logging overhead without closing the feedback loop it was designed for
- **Recommendation:** Implement a recommender feedback store (e.g. increment a `hit_count` column in evolution_log when a recommended skill is later successfully used); use that count as a score bonus in `_combined_score`.

### 10. Skills vs Routines dispatch + exec field path

- **Status:** ✅ unified in `agent.triage()`; no separate dispatch.py; ⚠️ `exec`-field skills bypass semantic routing but NOT the verifier (verifier only applies to evolver acquisition, not runtime dispatch)
- **Evidence:**
  - `agent.py:triage()`: single function handles slash commands, routine triggers, skill commands/intents/name, semantic fallback, micro-planner, evolution hook, and `infer_with_tools` — no separate dispatch module
  - `agent.py:130-149`: semantic skill fallback explicitly **excludes** skills with `exec` field: `semantic_candidates = [s for s in _skills if not s.get("command_only") and not s.get("exec")]`
  - `agent.py:112-127`: exec-backed skills only run via explicit slash command, intent match, or name match — cannot be triggered by freeform semantic routing
  - `tools.py:execute_tool("exec_shell")`: no approval gate, no SIM_MODE check — runs `subprocess.run(cmd, shell=True)` directly
  - `api.py:/sim/mode POST`: sets `SIM_MODE=true` env var; used only by `goal_discovery._is_system_idle()` — does NOT gate `exec_shell`
- **Detail:** The `capability_verifier` gates skill *acquisition* (evolver), not skill *runtime dispatch*. Once a skill is installed, it runs without re-verification. `exec_shell` in tools.py is fully unrestricted at runtime — no approval gate, no sandbox, no SIM_MODE guard. Any skill with `exec: script.sh` runs the script on dispatch.
- **Risk:** medium — exec-backed skills run without verification at runtime; a badly synthesised skill that passed the critic gate could execute arbitrary shell commands
- **Recommendation:** Add an `exec_shell` rate-limit or a prompt-approval flag for newly synthesised skills before their first runtime execution.

### 11. Config drift check

- **Status:** ⚠️ minor drift — one orphaned config key, two code reads with no config backing
- **Evidence and details:**
  | Key | Status |
  |---|---|
  | `providers.hf_dataset_repo` (config.yaml) | Defined in config; **never read in code** — orphaned |
  | `evolution.min_candidate_score` | Read in `evolver.py:119` via `config.get("evolution", {}).get("min_candidate_score", 0.35)` — **not present in config.yaml** (defaults to 0.35 silently) |
  | `goal_discovery.boot_pattern_cap` | Read in `goal_discovery.py:268`; present in `config.yaml:goal_discovery.boot_pattern_cap: 3` ✅ |
  | `goal_discovery.deferred_drain_per_cycle` | Read in `goal_discovery.py:329`; present in config ✅ |
  | `memory.max_replicas` (config.yaml) | Present in config but `replica.py` uses module-level `MAX_REPLICAS = 4` constant, never reads from config |
  | `cluster.*` (config.yaml) | Defined but **no code reads** the `cluster` key |
  | `inference.speculative_decoding` (config.yaml) | Present; referenced in model_server.py (not audited in detail) |
- **Risk:** low
- **Recommendation:** Add `evolution.min_candidate_score: 0.35` to config.yaml; remove or implement `providers.hf_dataset_repo`; either wire `memory.max_replicas` to `replica.MAX_REPLICAS` or remove from config.

### 12. Boot order

- **Status:** ✅ well-ordered; ⚠️ minor race: model server readiness check has 60s timeout but API startup calls `agent.init()` which does NOT wait for model to load into GPU (model is lazy)
- **Evidence:**
  - `start.sh:L40-80`: model server socket readiness polled up to 60s before starting API — correct
  - `api.py:startup()`: calls `mdl.load(config_path)` (model_server socket connected), then `agent.init()`, then `telegram_bot.start_bot_thread()`, then `EvolvingThinkAtRest.start()`
  - `model_server.py` starts with `--lazy` flag — model weights load on first inference, not at socket connect time
  - `start.sh` polls `/health` for 30s after API start — but health endpoint (`api.py:/health`) calls `mdl.vram_free_mb()` which does NOT require model to be loaded
  - `ThinkAtRest` starts before the 60s initial idle delay fires — safe
  - `goal_discovery.start_discovery_thread()` started only when `EVOLUTION_ENABLED=true`; first action is `seed_from_chat_history_and_defer()` which does NOT call the model — safe
  - `telegram_bot.start_bot_thread()` has no model dependency at start — safe
  - No `join()` / `await` on model loading thread; first user message will stall on model load (expected for lazy load)
- **Detail:** Boot order is correct. The only "race" is cosmetic: `/health` returns `ok` before the model is loaded into VRAM. This matches lazy-load intent and is documented in comments. No components are started but never joined.
- **Risk:** low
- **Recommendation:** Document in README that `status=ok` from `/health` does not guarantee model is loaded; consider a `model_ready` boolean in the health response.

---

## Cross-cutting issues

1. **`api.py` duplicate definitions:** `_pipeline_jobs`, `_prune_pipeline_jobs`, and `_run_pipeline_job` are each defined **twice** (sync then async). Python silently uses the last definition. The first sync versions (~lines 500-540) are dead code from an incomplete ADR-012 migration.

2. **`evolution_hook.py` module docstring says "NOT imported by agent.py yet"** — this is stale. `agent.py:triage()` imports and calls `maybe_evolve` directly. Docstring needs updating.

3. **`EVOLUTION_ENABLED` env var defaults to `"false"`** in both `goal_discovery.py:L22` and `evolution_hook.py:L10`. It is set to `true` only in `start.sh:L28`. Docker/bare-metal deployments that don't use `start.sh` will have evolution silently disabled with no warning.

4. **`memory.py:_append_to_db`** uses a count-based idempotency check (`existing = count WHERE bot=? AND session_id=?`) rather than content hashing. If messages are partially saved (e.g. crash mid-save), the offset can be wrong and duplicate or skip turns on the next save.

5. **`_try_tier2_legacy` has a double docstring** (two `"""..."""` strings back-to-back at `evolution_hook.py:~200`) — harmless but signals dead code from a merge.

6. **No `dispatch.py`** — routing is monolithic in `agent.triage()`. As the skill count grows past 60+, the linear scan for intents/commands becomes a performance concern.

---

## Suggested next actions (ranked)

1. **[High / Quick]** Pass `infer_fn` to `maybe_evolve` from `goal_discovery.py` and `thought_engine._maybe_trigger_evolution()` so the capability verifier is active in all evolution paths — not just manual trigger.

2. **[High / Quick]** Remove duplicate `_pipeline_jobs`, `_prune_pipeline_jobs`, `_run_pipeline_job` definitions in `api.py` — the sync versions are dead code and create silent bug risk.

3. **[Medium]** Add `evolution.min_candidate_score: 0.35` to `config.yaml`; wire `memory.max_replicas` to `replica.MAX_REPLICAS` or remove from config; remove orphaned `providers.hf_dataset_repo`.

4. **[Medium]** Implement the recommender feedback loop: increment a usage counter in `evolution_log` when a recommended skill is later triggered; weight counter into `_combined_score`.

5. **[Medium]** Enable `finetune_gate.enabled: true` or add a scheduled JSONL export+push to `hf_dataset_repo` — the trajectory collection pipeline currently produces data nobody consumes.

6. **[Low]** Update the `evolution_hook.py` module docstring (stale "NOT imported by agent.py yet" comment).

7. **[Low]** Add a `model_ready` boolean to the `/health` response to distinguish socket-alive from model-loaded; add a per-chat JSON file TTL cleanup job.
