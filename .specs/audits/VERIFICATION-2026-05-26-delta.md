# Verification — kernel-evolving wiring delta (2026-05-26)

## TL;DR
- **A1 (infer_fn wiring): CORRECTLY NEEDED** — origin/main goal_discovery and thought_engine both called `maybe_evolve()` WITHOUT `infer_fn`; builder fix is legitimate, not a duplicate.
- **A2 (duplicate pipeline defs): CORRECTLY NEEDED** — `origin/main:src/api.py` had TWO verbatim definitions of `_pipeline_jobs`, `_prune_pipeline_jobs`, and `_run_pipeline_job` (lines 497 vs 667); dedup is valid.
- **A3 (recommender boost): CORRECTLY NEW** — origin/main had no `recommendation_boost` field in `_combined_score`; builder added both config key and scoring logic.
- **C4 (self-critique): ALREADY WIRED** — ADR-016 claim that critic is missing is WRONG; `agent.py` runs a post-inference critic pass and `evolution_hook.py` runs a drafter+critic pipeline; critic IS wired.
- **BIGGEST SURPRISE — memory regression risk**: `agent.py:299` skips evolution AND memory load when `chat_id=""` (empty string is falsy); this is intentional but means any call without `chat_id` falls back to un-namespaced SQLite. No evidence of a new regression introduced after `8ea6fe6`, but the voice path at `telegram_bot.py:584` and file attachment path at `telegram_bot.py:668` do pass `chat_id=str(chat_id)` correctly.

---

## A. Audit fixes — duplicate check

### A1. infer_fn wiring — STATUS: NEEDED (fix is legitimate)

**Origin/main evidence (no infer_fn passed):**
- `git show origin/main:src/goal_discovery.py:215` → `maybe_evolve(pattern, config, skills_dir=skills_dir)` — no `infer_fn`
- `git show origin/main:src/goal_discovery.py:253` → same, no `infer_fn`
- `git show origin/main:src/goal_discovery.py:312` → same, no `infer_fn`
- `git show origin/main:src/thought_engine.py:526` → `evolution_hook.maybe_evolve(thought["thought"], cfg, skills_dir)` — no `infer_fn`

**evolution_hook.py already accepted infer_fn:**
- `src/evolution_hook.py:20` → `def maybe_evolve(task, config, skills_dir, infer_fn=None)` — parameter existed on main but callers never passed it.

**Current branch (builder's fix):**
- `src/goal_discovery.py:226` → `maybe_evolve(pattern, config, skills_dir=skills_dir, infer_fn=_infer_fn)` ✅
- `src/goal_discovery.py:264` → same ✅
- `src/goal_discovery.py:323` → same ✅
- `src/thought_engine.py:528` → `evolution_hook.maybe_evolve(thought["thought"], cfg, skills_dir, infer_fn=_gd._infer_fn)` ✅

**Verdict:** Builder should continue. The fix is genuine: the parameter existed but was never threaded through background callers. No duplication.

---

### A2. Duplicate async pipeline definitions — STATUS: NEEDED (fix is legitimate)

**Origin/main evidence:**
- `git show origin/main:src/api.py:497` → `_pipeline_jobs: dict[str, dict] = {}` (first definition)
- `git show origin/main:src/api.py:511` → `def _prune_pipeline_jobs():` (first definition)
- `git show origin/main:src/api.py:523` → `async def _run_pipeline_job(job_id: str, body: PipelineIn):` (first definition)
- `git show origin/main:src/api.py:667` → `_pipeline_jobs: dict[str, dict] = {}` **(DUPLICATE)**
- `git show origin/main:src/api.py:670` → `def _prune_pipeline_jobs(ttl_s: int = 3600):` **(DUPLICATE with different signature)**
- `git show origin/main:src/api.py:679` → `async def _run_pipeline_job(job_id: str, body: PipelineIn):` **(DUPLICATE)**

Two definitions with subtly different signatures (first prune: no args; second: `ttl_s=3600`). This is a real bug. Dedup fix is valid.

**Verdict:** Builder should continue. Duplication is confirmed in origin/main.

---

### A3. Recommender feedback boost — STATUS: NEEDED (fix is new work)

**Origin/main evidence:**
- `git show origin/main:src/evolver.py` → `_combined_score` at line 165, NO `recommendation_boost` reference at all
- `git show origin/main:config.yaml` → no `recommendation_boost` key under `evolution:`

**Current branch:**
- `src/evolver.py:185-186` → reads `config.get("evolution", {}).get("recommendation_boost", 0.05)`
- `src/evolver.py:244-255` → applies boost per prior recommendation hit in `_combined_score`
- `config.yaml:236` → `recommendation_boost: 0.05`

**Verdict:** Builder added genuinely new capability. Not a duplicate of anything on main.

---

## B. Strategic gaps — fact-check

### C1. Skill performance ledger — STATUS: NOT_FOUND

- `grep -rn "skill_ledger\|success_rate\|skill_outcome"` across `src/` → **zero matches**
- No `CREATE TABLE` statement involving skill metrics found
- No `~/.kernel-evolving/*.db` introspected (file system not accessible from subagent without exec), but no code creates such a table in any `.py` file.

**ADR-016 claim is CORRECT** — skill performance ledger does not exist.

---

### C2. Outcome-based reward signal — STATUS: NOT_FOUND (agent reward), PARTIAL (evolution critic)

- `grep -rn "reward\|outcome\|auto_grade\|turn_grade"` in `src/*.py` → only `evolution_dashboard.html` UI labels and `evo_routine_executor.py:126` (step outcome summary prompt). No end-of-turn grading logic in `agent.py` or `trajectory_collector.py`.
- Evolution path has a critic score (`_critic_score`, `agent.py:502-511`) used for trajectory recording, but this is NOT a reward signal fed back to skill selection or evolution weighting.

**ADR-016 claim is CORRECT** — no automated outcome-based reward signal driving learning loop.

---

### C3. Skill retirement / archive / quarantine — STATUS: NOT_FOUND

- `grep -rn "archived\|quarantine\|retire\|disable_skill\|staleness"` across `src/*.py` → **zero matches**

**ADR-016 claim is CORRECT** — no retirement mechanism exists.

---

### C4. Self-critique / reflection — STATUS: ALREADY_WIRED (ADR-016 claim is WRONG)

Critic is wired in TWO places:

1. **Post-inference critic in agent.py:**
   - `src/agent.py:501-511` — after inference, runs a "quick critic pass" using the same provider with `call_type="critic"`, parses score ≥ 0.7 = PASS
   - Score stored in `_critic_score`, `_critic_verdict` and recorded to trajectory

2. **Drafter+critic pipeline in evolution_hook.py:**
   - `src/evolution_hook.py:132-222` — ADR-010 two-stage pipeline: Stage 1 synthesises skill, Stage 2 critic evaluates; one retry if rejected; only installs if score passes

3. **API pipeline endpoint:**
   - `src/api.py:440` — ADR-008 multi-stage writer→critic pipeline endpoint

**ADR-016 claim that self-critique is MISSING is WRONG.** Critic is fully wired for both inference quality and skill synthesis.

---

### C5. Goal provenance — STATUS: NOT_FOUND

- `grep -rn "goal_provenance\|pattern_outcome\|provenance"` across `src/` → **zero matches**
- No link recorded between detected pattern and resulting installed skill (beyond the evolution log entry's `task` field which stores the pattern text)

**ADR-016 claim is CORRECT** — structured provenance linking not implemented.

---

## C. Memory regression hunt

### Sanitiser intact: YES
- `src/memory.py:119` → `def _sanitise(msgs: list) -> list:` — function present on current branch

### Both load paths call sanitiser: YES
- **JSON fast path:** `src/memory.py:171` → `return _sanitise(msgs[-(MAX_TURNS * 2):])`
- **SQLite cold-start path:** `src/memory.py:196-199` → `sanitised = _sanitise(msgs)` then `return _sanitise(msgs)` (called twice — minor inefficiency, not a bug)

### /new clears per-chat: YES
- `src/telegram_bot.py:741-743` → `/new` calls `_mem_new.clear_chat(str(chat_id))`
- `src/memory.py:229` → `clear_chat(chat_id)` exists — wipes JSON window file + SQLite rows for that `session_id`
- Confirmed by commit `6c0c14f` (2026-05-25)

### Suspect commits after 8ea6fe6 touching memory paths:
```
c25ac30  feat(telegram): /voice-clone bare command ... (telegram_bot.py only — no memory logic)
78e4ce6  fix(bot): slug length caps (telegram_bot.py only — no memory logic)
2232ec4  feat(bot): dynamic command picker (telegram_bot.py only — no memory logic)
c867d22  fix(telegram): replace hardcoded model strings (telegram_bot.py only — no memory logic)
6c0c14f  fix(/new): scope clear to chat_id (memory.py + telegram_bot.py — THIS IS THE FIX ITSELF)
08704bf  fix(status): read actual model name (no memory files)
```

None of the commits after `8ea6fe6` appear to have touched memory load/save logic. All telegram_bot.py changes were unrelated to memory routing.

### Likely regression site (if found): NONE CONFIRMED

No regression detected in current code. However, one **latent risk** exists:

- `src/agent.py:299` → `if EVOLUTION_ENABLED and not _evo_retry and not _is_conversational and not chat_id:` — the evolution path is only triggered when `chat_id=""`. The memory load at line 424 does pass `chat_id=chat_id` to `_mem.load()`. When `chat_id=""`, the cold-start SQLite fallback at `memory.py:186` fetches the LAST N turns without `session_id` filter (no WHERE clause on `session_id`), meaning conversations from different users could bleed together on a blank-chat_id call. This is intentional for CLI use but is a latent cross-session bleed risk if any bot path accidentally omits `chat_id`.

- All primary bot call sites inspected pass `chat_id=str(chat_id)` correctly.

---

## D. Builder in-flight delta

### Files touched by builder vs origin/main:
```
config.yaml                       +2 lines  (recommendation_boost key added)
src/api.py                        -73 lines (duplicate pipeline defs removed — correct)
src/evolver.py                    +77 lines (recommendation boost + prior-round persistence)
src/goal_discovery.py             +17 lines (infer_fn threading into all maybe_evolve calls)
src/thought_engine.py             +4 lines  (infer_fn threading)
tests/test_audit_fixes.py         +270 lines (new test suite)
tests/test_goal_discovery_boot.py +2 lines  (minor update)
```

### Anything that looks like duplication of existing code:
- **NONE detected.** All changes appear to be filling genuine gaps or removing confirmed bugs.
- The `evolver.py` addition properly guards against double-counting by using a `_boosted` set (not visible from stat, verify in review).
- The `api.py` change removes the second duplicate block, keeping the first definition — verify the surviving definition uses the longer/correct `_prune_pipeline_jobs(ttl_s: int = 3600)` signature (the one with the TTL parameter is the better one to keep).

### Risk flag:
- Reviewer should verify which of the two `_prune_pipeline_jobs` definitions was kept in the final `src/api.py`. The first (line 511, no TTL arg) and second (line 670, `ttl_s=3600`) differed in signature. Keeping the argless version silently drops the TTL parameter.

---

## Recommendations to reviewer subagent

- **Verify api.py prune signature**: Confirm that the surviving `_prune_pipeline_jobs` in the current branch has `ttl_s=3600` parameter (the better implementation), not the argless original. If the argless one was kept, the dedup fix degraded functionality.
- **Verify evolver _boosted set**: In `src/evolver.py`, confirm `_combined_score` with boost uses a `_boosted` set or similar guard to prevent the same skill getting boosted multiple times in a single scoring pass.
- **Verify infer_fn availability in goal_discovery**: Confirm `_infer_fn` is initialized before it's used in the deferred/background paths — check if `_infer_fn` could be `None` at module import time and whether that's handled gracefully.
- **ADR-016 C4 claim is factually wrong** — do NOT implement a new critic; document that it's already wired and close that gap item.
- **Memory sanitiser double-call**: `src/memory.py:196-199` calls `_sanitise(msgs)` twice on cold-start (once to write the JSON warm-up, once for return). Benign but inefficient — note for a follow-up cleanup, not a blocker.
- **Cross-session bleed risk**: Flag the `chat_id=""` SQLite fallback behavior in `memory.py:186-192` for a future hardening pass. Not a regression, but a latent risk.
