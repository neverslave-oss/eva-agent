# ADR-009: Goal Discovery Boot Cap + Lazy Deferred Seeding

**Status:** Approved — implement  
**Date:** 2026-05-11  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving), ADR-008 (Critic Replica)

---

## Problem

`goal_discovery.seed_from_chat_history()` is called at startup and replays up to 200 historical messages into `_interaction_log`. With 116 messages in the DB (as of May 11 2026), this immediately triggers 10–20 consecutive `_run_discovery_cycle()` calls — each loading embedding weights (~300 ms) and running Gemma 4 inference for ADR-006 verification. This consumes the model server socket for **3–8 minutes after every boot**, blocking all pipeline and message calls during that window.

Observed impact:
- Sim 6: all 10 tasks timed out — goal_discovery queue consumed the model server
- Sim 7: pipeline 503s and curl timeouts despite endpoint being functional
- Every restart creates a burst that makes the agent unresponsive for minutes

---

## Decision

### Part 1 — Boot cap: top-3 patterns only, scored and de-duped

At boot, `seed_from_chat_history()` runs pattern detection immediately but **caps execution to the top-3 highest-weight patterns** before handing back the model server. Scoring uses the existing `_extract_intent_keywords()` count — higher-frequency patterns run first.

```python
BOOT_PATTERN_CAP = int(os.environ.get("DISCOVERY_BOOT_CAP", "3"))
```

Implementation:
1. After seeding `_interaction_log`, detect all patterns via `_discover_patterns()`
2. Sort by frequency descending
3. Run `_run_discovery_cycle()` for top-N only (N = `BOOT_PATTERN_CAP`)
4. Mark the remaining patterns as **deferred** — store in a module-level `_deferred_patterns: list[str]`
5. Return immediately — discovery thread proceeds to its normal interval loop

### Part 2 — Lazy deferred processing on first idle

The discovery thread already runs on an interval. Extend it to drain `_deferred_patterns` **one per cycle** when the system is idle (no active replicas, no active message calls):

```python
# In _discovery_loop, after normal pattern scan:
if _deferred_patterns and _is_system_idle():
    pattern = _deferred_patterns.pop(0)
    _run_single_pattern(pattern, config, skills_dir)
```

`_is_system_idle()` checks:
- `replica.active()` → empty
- No recent model server call in last 30s (tracked via a module-level `_last_model_call_ts`)

### Part 3 — De-duplication before install

Before `_run_discovery_cycle()` calls `maybe_evolve()`, check if the resolved skill is already installed in the ecosystem. If already present at `private/skills/<name>` or `community/skills/<name>`, mark as resolved without cloning/validating again.

```python
def _already_installed(skill_name: str, skills_dir: str) -> bool:
    from pathlib import Path
    base = Path(skills_dir).expanduser()
    return any((base / tier / skill_name).exists() 
               for tier in ["private/skills", "community/skills", "third-party/skills"])
```

---

## Configuration

```yaml
# config.yaml additions
goal_discovery:
  boot_pattern_cap: 3        # max patterns resolved immediately at boot
  deferred_drain_per_cycle: 1  # deferred patterns processed per idle cycle
```

---

## Expected impact

- Boot-to-ready time for pipeline calls: from 3–8 min → **< 30s**
- Deferred patterns drained within 1–2 idle cycles (5–10 min post-boot)
- No change to steady-state discovery behaviour
- De-duplication eliminates re-installing already-present skills on every restart

---

## Drafter role

The drafter (Gemma 4 MTP, 180 MB) is **already used** for ADR-006 verification via `infer_fn` passed to the evolver. The boot cap does not change this — it simply limits how many patterns trigger that path at startup.

Future: the drafter could independently pre-score patterns during idle (System 1 pass) before the main model validates (System 2 pass), reducing main-model calls for low-confidence patterns.
