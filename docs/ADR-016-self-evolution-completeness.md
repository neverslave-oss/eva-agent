# ADR-016 — Self-Evolution Completeness

**Status:** PROPOSED
**Date:** 2026-05-26
**Author:** Olly (audit + proposal), reviewed with Fabio
**Supersedes / extends:** ADR-006 (verification), ADR-007 (recommendation), ADR-010 (evolution pipeline), ADR-013 (trajectory fine-tuning), ADR-015 (auto-discovery)

---

## Context

A wiring audit on 2026-05-26 (`.specs/audits/AUDIT-2026-05-26-self-evolving-wiring.md`) identified that kernel-evolving has the **scaffolding** of a self-improving agent — goal discovery, evolver, capability verifier, recommender, trajectory collector, ThinkAtRest, agent discovery — but the loops are **not closed**. The agent acquires capability; it doesn't measure, reflect on outcomes, or compound improvements over time.

This ADR records:

1. **Audit fixes already approved** (being implemented by builder subagent, branch `fix/audit-2026-05-26-evolver-wiring`).
2. **ThinkAtRest deep-dive** — what's there, what's not.
3. **Strategic gaps** worth filling next, ranked by leverage.

The trajectory → fine-tune loop is intentionally **parked** pending the Nemotron-Labs-Diffusion-3B swap. Gemma 4 + v4 adapter made the system too slow to iterate on.

---

## Decision

### Part A — Wiring fixes (approved, in flight)

| # | Fix | File(s) | Status |
|---|-----|---------|--------|
| 1 | Pass `infer_fn` from background `maybe_evolve` callers so capability verifier actually runs in autonomous evolution loops | `src/goal_discovery.py`, `src/thought_engine.py`, possibly `src/evolver.py` constructor | builder in progress |
| 5 | Remove duplicate sync definitions of `_pipeline_jobs`, `_prune_pipeline_jobs`, `_run_pipeline_job` left over from incomplete ADR-012 migration | `src/api.py` | builder in progress |
| 4 | Persist `EvolutionResult.recommendations` and apply a small additive boost (+0.05, config-driven) to those candidates on the next `search_ecosystem` round | `src/evolver.py`, `src/recommender.py`, `config.yaml` | builder in progress |
| 2 | Trajectory feedback into evolver scoring | `src/trajectory_collector.py`, `src/evolver.py` | **PARKED** until Nemotron-Diffusion-3B is default. Marker: `# TODO(nemotron-swap)` comments at consumer sites. |

Acceptance: 334+ tests pass on the branch; reviewer subagent verifies and approves before merge.

### Part B — ThinkAtRest audit (correction to initial audit)

Initial audit understated ThinkAtRest. On re-read of `src/thought_engine.py` (784 lines), it's actually one of the more wired subsystems:

**What works (✅):**

- **System 1 / System 2 split.** `ThoughtGenerator` produces cheap seeds; `ThoughtEvaluator` uses the main model to score+classify (`retrospective | gap_reflection | self_improvement | curiosity`), discards below `min_score=0.4`.
- **Idle gating.** `IdleDetector` only fires after `idle_threshold=300s`. Doesn't compete with active user turns. `_model_is_active()` + `_main_model_loaded()` further guard inference.
- **Category-aware downstream actions:**
  - `score ≥ promote_threshold` → `_promote_to_idea()` writes markdown + feeds `goal_discovery.record_promoted(weight=3)` so the desire counts toward `RECURRENCE_THRESHOLD` and persists across restarts.
  - `category == "gap_reflection"` → `_maybe_trigger_evolution()` → `evolution_hook.maybe_evolve(...)`. (After Fix #1, this will pass `infer_fn` so the verifier runs.)
  - `category == "self_improvement"` → appends to `~/.kernel-evolving/workspace/todos.md`.
  - `score ≥ 0.85` → proactive Telegram delivery with inline action buttons (capped per day).
- **Memory seeker.** Every idle cycle, scans last 40 user turns, asks the model to extract personal facts, merges into `user.json`. Cheap passive memory build-up.
- **Deferred drafts.** When the model is busy, seeds are persisted to disk and replayed next cycle. Survives restarts.
- **EvolvingThinkAtRest subclass.** Overrides `_get_recent_interactions / _get_recent_gaps / _get_recent_evolution_context` to feed kernel-evolving-specific signal back into prompts (loop-closing for the evolver, conceptually).

**What's weak / missing (⚠️):**

| Issue | Detail | Risk |
|-------|--------|------|
| Single eval pass | `ThoughtEvaluator` parses one JSON blob. If the model returns malformed JSON twice in a row, an idle cycle silently produces zero thoughts. No retry/repair. | low — degradation only |
| `min_score=0.4` is uncalibrated | No empirical basis for the threshold. A weekly histogram of evaluator scores would tell us if 0.4 is too loose or too tight. | low |
| Promotion → Telegram has no de-dup | Two near-identical "gap_reflection" thoughts on consecutive cycles each trigger a Telegram message. | low UX irritant |
| `_maybe_trigger_evolution` swallows `ImportError` silently | If `evolution_hook` import fails (rename, syntax error during refactor), evolution is silently disabled. | medium — would mask a real regression |
| Thought outcomes not tracked | A promoted idea becomes a `goal_discovery` signal, but there's no record of "this idea resulted in skill X / no skill / abandoned". Same blind spot as `recommender`. | medium — blocks measuring whether ThinkAtRest is *useful* |
| Eval threshold not adjustable at runtime | Only via `config.yaml` reload. No `/think_threshold 0.5` command. | low |
| No "compounding" link to skill ledger | Idle thoughts never read the (not-yet-existing) skill performance ledger to reflect on *which* skills underperformed. | high — biggest unrealised value |

**No changes proposed in this ADR for ThinkAtRest itself** beyond Fix #1's verifier wiring. Items above are recorded as follow-ups; we'll spec them after Nemotron lands and we know what model behaviour to tune against.

### Part C — Strategic gaps (proposed; not yet approved for build)

These are the high-leverage missing pieces. **Ranked by leverage × shippability.**

#### C1. Skill Performance Ledger 📊 — *do this first after Nemotron*

**Problem.** Evolver ranks by similarity. A skill that's failed 9 of 10 times ranks the same as a perfect one. ThinkAtRest can't reflect on what's broken because there's no record.

**Proposal.** SQLite table `skill_ledger`:

```
skill_id   TEXT
chat_id    TEXT
ts         TIMESTAMP
outcome    TEXT  -- "success" | "fail" | "abort"
latency_ms INTEGER
error      TEXT  -- nullable
```

- Write from `agent.py` after every skill invocation (one wrapper).
- `evolver._combined_score()` multiplies by `success_rate(skill_id)` clamped to `[0.5, 1.2]` (so a fresh skill isn't punished, a winning one gets a small boost).
- ThinkAtRest's `_get_recent_gaps()` reads worst performers and feeds them into prompts → reflection becomes data-driven.

**Effort:** <500 LOC, one new table, one wrapper, one scoring tweak. Single PR. No architecture change.

#### C2. Outcome-based reward signal 🧠

**Problem.** Trajectories are turn-by-turn dumps without "did this answer the user?" labels.

**Proposal.** End-of-turn auto-grader:
- Rule-based first pass: presence of error in tool calls, retry count, explicit user "thanks"/"no that's wrong" tokens in next turn.
- Optional LLM second pass when idle: re-read (user, assistant) and label `solved | partial | failed | ambiguous`.
- Reward column added to `task_trajectories`.
- This is the missing piece that makes trajectories into training data the day Nemotron lands.

**Effort:** ~300 LOC. Couples cleanly with C1 (ledger outcome ↔ trajectory reward).

#### C3. Skill retirement / quarantine 🗑️

**Problem.** Ecosystem grows monotonically. 63 skills today → 200 in 3 months → context bloat, picker noise, already hit the Telegram 100-command cap once (commit `78e4ce6`).

**Proposal.**
- Skills with 0 invocations in 30d → `archived` flag (still on disk, excluded from matching).
- Skills with `success_rate < 0.3` over `n ≥ 10` invocations → `quarantined` (excluded; user-visible alert).
- `/skill_status <slug>` to inspect; `/skill_restore <slug>` to un-archive.
- Depends on C1.

**Effort:** ~200 LOC after C1 lands.

#### C4. Self-critique / reflection step 🪞 — **ALREADY WIRED (correction)**

**Correction (2026-05-26 verification pass):** A post-inference critic IS already wired. Confirmed at `src/agent.py:501-511` — after every tool-using turn, a critic prompt scores 0.0-1.0 → PASS if ≥0.7 else FAIL. Also referenced by ADR-008 (critic replica) and ADR-010 (evolution pipeline critic), wired in `src/evolution_hook.py:132-222` for the drafter+critic synthesis pipeline.

**Remaining gap (smaller scope):** the critic SCORES turns but does not feed back a REVISE pass (re-run the model with critic feedback to fix the answer in-line). That refinement step is the only piece missing. Tracked as a follow-up; not urgent.

#### C5. Goal provenance + outcome tracking 🔗

**Problem.** `goal_discovery` fires → `evolver` installs → ??? No link back saying *"this skill exists because pattern X was detected"*.

**Proposal.** `goal_provenance` table:

```
pattern        TEXT
first_seen     TIMESTAMP
n_recurrences  INTEGER
skill_id       TEXT  -- nullable
outcome        TEXT  -- "skill_installed" | "skill_created" | "no_match" | "abandoned"
```

Lets us answer "is the evolving agent actually evolving in the right direction?"

**Effort:** ~150 LOC.

#### C6. Honourable mentions (not yet specced)

- Skill versioning + rollback (already have `/rollback` for core; extend to skills).
- Adversarial prompt-injection probe in `capability_verifier`.
- Cross-chat learning (memory is per-chat_id; siloed).
- Shadow query against discovered peer (kernel base) for free eval signal.
- Per-turn tool budget (cost/latency cap, not just call-count).

### Part D — ThinkAtRest follow-ups (defer to post-Nemotron)

| Item | Notes |
|------|-------|
| Re-prompt on malformed evaluator JSON | One retry with stricter "JSON only" reminder before giving up |
| De-dup promoted ideas | Compare new thought against last N idea filenames, skip if cosine ≥ 0.85 (uses the embedding client that already exists) |
| Surface `evolution_hook` import failures | Log at ERROR not silent pass; emit one Telegram alert per process lifetime |
| Read skill ledger from ThinkAtRest | Once C1 lands, feed worst-performing skills into reflection prompts |

---

## Consequences

**Positive:**

- Audit fixes close the verifier gap → no more autonomous skill acquisition without LLM sanity check.
- Recommender feedback closes a real silo (small effort, real learning compounding).
- Roadmap (C1–C5) gives kernel-evolving the *measurement layer* it currently lacks. Without that, "self-evolving" is aspirational.

**Negative / risks:**

- New SQLite tables increase write volume (mitigated: indexed, async).
- Reflection step doubles latency unless gated.
- Quarantine could hide a skill the user wants — needs visible UX (`/skill_status`).

**Non-goals (explicit):**

- Cross-host federation (peer delegation per ADR-015 stays "future work").
- Full RLHF loop. Trajectories + reward signal is the *prep* for RLHF, not RLHF itself.
- Modifying core architecture (model server / API / Telegram bot boundaries unchanged).

---

## Execution plan

**Now (this PR, audit fixes branch):**

1. Builder ships Fix #1, #5, #4 with tests → reviewer verifies → Olly pushes, tags `v1.19.2`, releases.
2. This ADR ships in the same PR for traceability.

**Post-Nemotron-Diffusion-3B swap (in order):**

3. Wire trajectories → evolver (Fix #2, the parked one).
4. Build C1 (skill ledger) — foundation for everything else.
5. Build C2 (reward signal) — pairs with C1.
6. Build C3 (retirement) — depends on C1.
7. Build C5 (provenance) — depends on goal_discovery.
8. C4 (self-critique) — independent, can slot anywhere.

Each of C1–C5 = its own ADR (017+) when picked up. This ADR is the index.

---

## Decision required from Fabio

- [ ] Approve Part A wiring fixes as currently scoped (builder is already executing — say STOP if not).
- [ ] Approve Part C ranking as the post-Nemotron roadmap (no build action yet, just direction).
- [ ] Confirm Part D ThinkAtRest follow-ups stay deferred until after Nemotron lands.

---

*End of ADR-016.*
