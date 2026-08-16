# ADR-011: Micro-Planner in Triage Path

**Status:** Approved — implement  
**Date:** 2026-05-11  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving), ADR-005 (Think-at-Rest), ADR-008 (Critic Replica)

---

## Problem

`agent.triage()` is single-step: one message → one skill/routine/LLM call. When a user request requires multiple tools or skills in sequence (e.g. "fetch this URL, extract the table, save as JSON, and email me a summary"), triage routes it to `infer_with_tools()` which gives Gemma 4 the full tool-calling loop. In practice Gemma 4 E2B-it struggles with multi-step planning: it either stops after the first tool call, loses track of intermediate results, or generates a plan it can't execute because it doesn't know which skills exist.

The result: multi-step tasks either silently fail, produce partial results, or timeout in the tool-calling loop.

---

## Decision

Insert a **micro-planner** step between triage and `infer_with_tools` for requests that exceed a complexity threshold.

### Architecture

```
triage(text)
    │
    ├─ single-step (skill/routine match) → run directly ✓ unchanged
    │
    └─ complex (no match, multi-step signals) 
           │
           ▼
    MicroPlanner.plan(text)          ← NEW
           │
    returns: Plan(steps=[...]) | None
           │
    ├─ None (simple) → infer_with_tools() as before
    └─ Plan → PlanExecutor.run(plan)  ← NEW
               │
               for each step:
                 → triage(step.task)  # recursive, single-step each
               │
               → synthesise final reply from step results
```

### MicroPlanner

Uses the **drafter** (System 1) to generate a plan cheaply, then the main model (System 2) validates it:

```python
class MicroPlanner:
    COMPLEXITY_SIGNALS = [
        "then", "after", "next", "first", "second", "finally",
        "and then", "also", "additionally", "step", "extract.*save",
        "fetch.*summarize", "download.*convert",
    ]
    MAX_STEPS = 5

    def should_plan(self, text: str) -> bool:
        """Quick heuristic: is this request multi-step?"""
        lower = text.lower()
        signal_count = sum(1 for s in self.COMPLEXITY_SIGNALS if s in lower)
        word_count = len(text.split())
        return signal_count >= 2 or word_count > 30

    def plan(self, text: str, available_skills: list, available_routines: list) -> Plan | None:
        """
        Stage 1 (drafter — fast): generate a raw step list
        Stage 2 (main model): validate steps against available skills, rewrite if needed
        Returns None if request is actually single-step after validation.
        """
```

### Plan structure

```python
@dataclass
class PlanStep:
    order: int
    task: str              # natural language sub-task
    skill_hint: str        # suggested skill name (from planner, not enforced)
    depends_on: list[int]  # step indices whose output this step needs

@dataclass  
class Plan:
    original: str
    steps: list[PlanStep]
    rationale: str         # planner's one-line reasoning
```

### PlanExecutor

```python
class PlanExecutor:
    def run(self, plan: Plan, triage_fn, step_callback=None) -> str:
        results = {}
        for step in plan.steps:
            # Inject dependency outputs into step task
            task = step.task
            for dep in step.depends_on:
                if dep in results:
                    task += f"\n\n[Output from step {dep}]: {results[dep]}"
            results[step.order] = triage_fn(task, step_callback=step_callback)
        
        # Final synthesis: summarise all step results into one reply
        return self._synthesise(plan.original, results)
    
    def _synthesise(self, original: str, results: dict) -> str:
        # Uses infer() with a concise synthesis prompt
        # Drafter is suitable here — synthesis from structured inputs is a low-creativity task
        ...
```

### Drafter role in micro-planning

Three drafter uses in this ADR:

1. **Plan generation (Stage 1):** Drafter generates the raw step list — fast, cheap. Main model only validates/rewrites if drafter output is structurally invalid or contains non-existent skills.

2. **Step-level skill hint matching:** Drafter scores `(step.task, skill.name+description)` similarity quickly across all 53 skills, returning top-3 hints per step. Main model only runs ADR-006 verify on the top candidate. This pre-filters the skill match space cheaply.

3. **Result synthesis:** After all steps complete, drafter assembles the final reply from step results. Since inputs are structured (ordered key-value results), the drafter's associative generation is sufficient — no System 2 verification needed.

Formally: drafter = System 1 (generate fast, tolerate noise) / main model = System 2 (verify, correct, commit).

---

## Triage integration

```python
# In agent.triage(), before infer_with_tools fallback:

from micro_planner import MicroPlanner, PlanExecutor
_planner = MicroPlanner()

if _planner.should_plan(text):
    plan = _planner.plan(text, _skills, _routines)
    if plan and len(plan.steps) > 1:
        print(f"[agent] MicroPlanner: {len(plan.steps)} steps — {plan.rationale}")
        executor = PlanExecutor()
        return executor.run(plan, triage_fn=triage, step_callback=step_callback)

# Fallback: single-step infer_with_tools as before
```

---

## Config

```yaml
planning:
  enabled: true
  complexity_threshold: 2      # min signal count to trigger planning
  max_steps: 5
  draft_plan: true             # use drafter for Stage 1 plan generation
  synthesise_with_draft: true  # use drafter for final result synthesis
```

---

## Expected impact

- Multi-step tasks (fetch+extract+save, review+annotate+summarise) complete correctly
- Each step uses the full triage path — skills, routines, evolution all available per step
- Drafter reduces plan generation + synthesis overhead to ~200 ms per step vs ~8 s main model
- Falls back gracefully to `infer_with_tools` if planner returns None or plan has 1 step

---

## Tests

- `test_should_plan_simple` — short single-action request → False
- `test_should_plan_complex` — "fetch URL, extract table, save JSON" → True
- `test_plan_generation` — mock drafter+main model, verify Plan structure
- `test_plan_executor_sequential` — 3 steps, verify dependency injection
- `test_plan_executor_synthesis` — verify final reply includes all step outputs
- `test_triage_delegates_to_planner` — integration: complex request → planner activated
- `test_triage_simple_bypasses_planner` — simple request → planner not called
