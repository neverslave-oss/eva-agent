# ADR-010: Evolution Loop via Critic Replica Pipeline (ADR-008 Integration)

**Status:** Approved — implement  
**Date:** 2026-05-11  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving), ADR-006 (Verification), ADR-008 (Critic Replica)

---

## Problem

The current Tier 2 code synthesis path (`code_synthesizer.py`) generates a SKILL.md + script, runs a basic validation pass, and commits. The validation is shallow — it checks that the file exists and has the right structure, but does not evaluate whether the synthesised skill actually solves the original task. False positives from Tier 2 are silent: a skill gets installed, `retry=True` is returned, triage runs the skill, and it silently fails or returns garbage.

The ADR-008 writer→critic pipeline already exists and works. Applying it to the evolution loop gives Tier 2 synthesis a quality gate without adding new infrastructure.

---

## Decision

After Tier 2 synthesis produces a candidate skill, run it through a **two-stage evolution pipeline** before committing:

```
Tier 2 synthesis
    ↓
Stage 1 — Synthesiser replica (writer)
  brief: "You are a skill synthesiser for a local AI agent. Write a complete SKILL.md..."
  task:  <original gap task>
  output_path: ~/.kernel-evolving/workspace/tmp/synth_<ts>/SKILL.md

Stage 2 — Critic replica
  brief: "You are a quality critic for AI agent skills. You receive a synthesised SKILL.md
          and the original task. Evaluate: (1) does the skill name/description match the task
          semantically? (2) are the commands realistic and non-ambiguous? (3) is there a clear
          exec path or instructions block? Return JSON: {verdict: PASS|FAIL, score: 0-1,
          issues: [...], suggested_name: '...'}"
  task:  "Original task: <gap>\n\nSKILL.md:\n<content>"
  input_from: synthesiser

Stage 3 (conditional) — if critic score < 0.6:
  Re-synthesise with critic issues injected as context (one retry only)
```

### Integration point

In `evolution_hook._try_tier2()`, replace the current `CodeSynthesizer.synthesize()` call with `_run_evolution_pipeline()`:

```python
def _run_evolution_pipeline(task: str, gap: str, config: dict) -> EvolutionResult:
    import rep, json
    from pathlib import Path
    
    tmp_dir = Path("~/.kernel-evolving/workspace/tmp").expanduser()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(tmp_dir / f"synth_{int(time.time())}.md")
    
    result = rep.pipeline([
        PipelineStage(name="synthesiser", role="custom",
                      brief=SYNTH_PROMPT, task=gap,
                      tools=False, output_path=output_path),
        PipelineStage(name="critic", role="custom",
                      brief=CRITIC_PROMPT, task=f"Original task: {task}\n\nSKILL.md:\n{{synthesiser}}",
                      tools=False, input_from="synthesiser"),
    ])
    
    critic_out = result.get("critic", "")
    verdict = _parse_critic_verdict(critic_out)
    
    if verdict["verdict"] == "FAIL" and verdict["score"] < 0.6:
        # One retry with critic feedback injected
        result = rep.pipeline([
            PipelineStage(name="synthesiser", role="custom",
                          brief=SYNTH_PROMPT,
                          task=f"{gap}\n\nPrevious attempt issues:\n{verdict['issues']}",
                          tools=False, output_path=output_path),
            PipelineStage(name="critic", role="custom",
                          brief=CRITIC_PROMPT, task=f"...",
                          tools=False, input_from="synthesiser"),
        ])
        critic_out = result.get("critic", "")
        verdict = _parse_critic_verdict(critic_out)
    
    if verdict["verdict"] == "PASS":
        # Write skill to ecosystem, install
        ...
    else:
        # Log failure, create tracker todo, return not found
        ...
```

### `rep.pipeline()` — programmatic API

Add a programmatic `pipeline(stages: list[PipelineStage]) -> dict` function to `replica.py` (not just the HTTP endpoint) so `evolution_hook.py` can call it directly without HTTP round-trips.

---

## Drafter role

The drafter's speed advantage applies here. The synthesiser stage can use a **drafter-first draft** approach:
- Drafter generates the SKILL.md skeleton cheaply (fast, 180 MB model)
- Main model reviews and expands only sections below quality threshold
- This halves the synthesis latency for the first stage

Add `draft_first: bool = True` to `PipelineStage` — when set, calls `infer_draft()` (drafter-only) instead of full `infer()` for the writer stage. The critic always uses the full model.

```python
def infer_draft(prompt: str, max_new_tokens: int = 512) -> str:
    """Fast draft using MTP drafter only — no main model pass."""
    # Direct drafter call via model_server RPC: method='infer_draft'
```

---

## Expected impact

- Tier 2 synthesis quality gate: critic PASS/FAIL before any skill is installed
- Suggested name from critic fed back into `skill_name` — improves Tier 1 retrieval (addresses Sim 5 finding on naming quality)
- One retry loop on critic failure catches ~60% of fixable synthesis issues
- Drafter-first synthesis: estimated 40–60% reduction in Stage 1 latency

---

## Tests to add

- `test_evolution_pipeline_pass_path` — mock pipeline returns PASS, skill installed
- `test_evolution_pipeline_fail_retry` — first critic FAIL triggers retry, second PASS installs
- `test_evolution_pipeline_double_fail` — both critic attempts FAIL, skill NOT installed, todo created
- `test_critic_verdict_parse` — JSON parse with fallback on malformed critic output
- `test_pipeline_programmatic_api` — `rep.pipeline()` returns results dict without HTTP
