# Plan: Reactivate Think-at-Rest (thought engine starving since June 24)

**Created:** 2026-08-22
**Status:** scoped, not implemented (pair-programming session required)
**Branch:** `fix/cloud-audio-vision-402-fallback` (issues live on the currently checked-out branch)
**Related:** `ADR-005` (Think-at-Rest), `ADR-012`, `ADR-019` (Observer/Critique), src/services/thought_engine.py

> ⚠️ **Pair-programming required.** This is a design spec for a shared session — no code
> changes were made. Do not implement from this doc alone.

## Problem

The Think-at-Rest engine has journaled **no new thought since June 24** (~2 months). The
daemon is enabled (`thinking.enabled: true`) and running (port 8779, pid 237295), but the
idle cycle never produces output — the bot is effectively "paused" at the thought layer.

## Root Cause (from audit, 2026-08-22)

The engine is **starved by design**, not crashed. The 2026-06-18 restructure made
curiosity generation conditional on real outward data, and three gates compound to block
nearly every cycle:

1. **`_run_think_cycle()`** → `_has_exploration_changed()` (thought_engine.py ~L622):
   only proceeds if exploration data (git log / system health) hashed differently since
   the last cycle. Idle box with stable repos → hash never changes → skip.
2. **`ThoughtGenerator.generate(exploration_summary="")`** (thought_engine.py ~L141):
   explicitly returns `[]` when `exploration_summary` is empty:
   `if not exploration_summary: logger.debug(... no exploration data — skipping synthetic generation); return []`.
3. **`_main_model_loaded()` gate** (in `_run_think_cycle`, thought_engine.py ~L635):
   curiosity is skipped entirely unless the main model is already resident. In a lazy-load /
   spawned-on-demand server this can frequently be `False`.

Net effect: idle time yields zero thoughts unless **both** the environment changed **and**
the main model was resident. Nothing met that bar since late June.

### Contributing observations (not the cause)

- `model_activity.json` is healthy (`in_flight: 0`, `last_end_ts` today) → idle detection
  is **not** blocked.
- Last journaled entries (June 23–24) were a repeated low-value self-improvement stub
  ("11 active probes accumulating") — not genuine curiosity.
- The bot's `read` error (`Errno 21 Is a directory` on `.../workspace/tmp/`) is a
  separate, cosmetic tooling issue — unrelated to the pause.

## Design Direction (to settle during pairing)

Goal: keep the "real data first, no synthetic gap-filling" philosophy (which the 06-18
restructure correctly enforced) while removing the accidental starvation. Priorities, in
order:

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

3. **Revisit the `_main_model_loaded()` residency gate.**
   - Determine the intended lifecycle: for a lazy/spawned model server, either (a) let
     Think-at-Rest trigger a load when idle and safely run, or (b) skip gracefully but
     log *why*, so the silence is diagnosable instead of invisible.

4. **Add signal diversity** so idle time isn't pinned to one repeated stub.
   - Refresh/cap the "active probes accumulating" self-improvement signal (it fired
     identically for 2 days and added no value).
   - Consider additional cheap real signals (git log on active repos, dependency-release
     checks, GPU/temp trends) that change over time and feed genuine curiosity thoughts.

5. **Observability.**
   - Log every skipped cycle with the *specific* gate that blocked it (no change / no
     model / no data / generator empty). Right now a skipped cycle is near-silent, which
     made this 2-month pause invisible.

## Open Questions for the pairing session

- Should a cycle ever run with **zero** real data, or always require at least one real
  signal (health OR git)? (Leaning: always require ≥1 real signal, but don't require it
  to have *changed*.)
- What cadence/rate-limit feels right once unblocked? (`thought_interval_s` is 1800s now.)
- Does the same starvation affect `observer`/`critique` layers, or only the curiosity
  generator? (Audit suggested observer is fine.)

## Acceptance Criteria

- Idle periods produce fresh journaled thoughts again (visible in
  `~/.kernel-evolving/workspace/thoughts/YYYY-MM-DD.md`).
- No regression to the "no synthetic gap-filling" intent: thoughts must come from real
  signals (git/health/metrics), not generated-from-vacuum.
- Skipped cycles are explicitly logged with the blocking gate.
- Existing tests (tests/test_thought_engine.py, test_think_at_rest_memory.py,
  test_observer_layer.py, test_critique_layer.py) stay green.
