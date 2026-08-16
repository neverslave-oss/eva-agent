# ADR-020: User-Request-Anchored Evolution with Human Confirmation Reward

**Status:** Proposed  
**Date:** 2026-05-29  
**Author:** Fabio Pacifici + Olly  
**Supersedes:** ADR-019 (partial — extends Observer/Critique intent)

---

## Context

ADR-019 introduced Observer and Critique layers to gate evolution quality. In practice the implementation diverged from intent:

- **Observer** measures topic frequency in chat history, not whether a user request was fulfilled or failed
- **Thought seeds** are idle self-reflections paraphrased from recent chat topics — not grounded in actual user failures
- **Critique** verifies the synthesised skill's description matches the thought text — not whether it would satisfy the original user request
- **RESOLVED** is declared by the model unilaterally — no human confirmation, no reward signal

The result: evolution installs plausible-sounding but useless skills synthesised from abstract self-reflection, while real user failures go unaddressed.

**The design intent** (Fabio, 2026-05-29):
> *"Given a user request, can the agent evolve to complete a task it wasn't designed to complete or trained on? The agent thinks at rest — during idle it scans user requests and verifies if those were fulfilled, or evolves until the task is complete, keeping the user in the loop. After critique calls it resolved it returns back to the user with the requested output and inline buttons to confirm if actually resolved — and only then finally marks it resolved. That confirmation also works as a reward mechanism."*

---

## Decision

### 1. Failure Logging — anchor evolution to real user requests

When `infer_with_tools` returns a fallback, refusal, or partial answer, the **response handler writes a `failed_request` record** immediately:

```
failed_requests table:
  id, chat_id, ts, user_message, agent_response, failure_type, resolved, resolved_at, reward
```

**`failure_type`** values:
- `no_skill` — no skill matched, agent said it couldn't help
- `skill_error` — skill ran but produced an error or empty output
- `refusal` — agent declined (capability gap, not policy)
- `partial` — agent gave a degraded answer with a caveat

Detection heuristics (checked on every `infer_with_tools` response):
- Response contains: *"I can't", "I don't have access", "not available", "skill not found", "no tool for"*
- Response is a skill error trace
- Response length < 80 tokens on a request that expected substantive output

---

### 2. ThinkAtRest — idle scans failed requests, not abstract thoughts

`EvolvingThinkAtRest._get_recent_gaps()` is replaced with a query against `failed_requests` where `resolved=0`, ordered by recency. These are passed to the generator as **concrete task anchors**, not topics to reflect on.

The generator prompt changes:

```
# OLD — abstract reflection
"Requests you couldn't fulfil: {recent_gaps}"

# NEW — concrete anchor
"Unresolved user requests (try to complete these, not reflect on them):
{failed_requests}"
```

Thoughts generated from these anchors are tagged `_source_request_id` pointing back to the original `failed_request.id`. Thoughts without a source anchor remain valid (curiosity, retrospective) but **cannot trigger evolution**.

---

### 3. Observer — binary failure check, not frequency count

`ObserverLayer.evaluate()` gains a new evidence path for request-anchored thoughts:

```python
if thought.get("_source_request_id"):
    # Direct evidence: the original user request failed
    classification = "EVIDENCED"
    evolution_path = "full_evolution"
    evidence_summary = f"user_request:{thought['_source_request_id']}"
```

Frequency-based evidence (`_search_chat_history`) is **demoted** — it can only produce `SPECULATIVE` classification, never `EVIDENCED`. Only a concrete `_source_request_id` or a prior `NO-verification` evolution event produces `EVIDENCED`.

---

### 4. Evolution — runs against original user message

`_run_evolution_with_critique()` receives the original user message alongside the thought:

```python
# task passed to maybe_evolve = original user request, not paraphrased thought
task = thought.get("_source_user_message") or thought_text
```

The synthesised skill is tested against the original user message text, not the thought.

---

### 5. Critique — verifies original request is satisfied

`CritiqueLayer.assess()` prompt changes:

```
# OLD
"Does the installed skill's output address this gap: {gap}"

# NEW  
"A user sent this exact request: '{original_user_message}'
The agent previously failed to complete it.
The skill '{skill_name}' was synthesised and produced this output:
'{output}'
Does this output satisfy the user's original request?
Answer RESOLVED, PARTIAL, or FAILED. One word, then explanation."
```

This is the key change: critique now holds the skill to the original user standard, not to the abstract gap description.

---

### 6. Human confirmation gate + reward signal

When critique returns `RESOLVED`:

1. **Do NOT mark resolved yet**
2. Send the skill's output to the user's Telegram chat:

```
🧠 *Evolution result*
I encountered a gap when you asked: _"{original_user_message[:100]}"_

I synthesised a new skill and here's what it produces now:

{skill_output}

─────
Did this resolve your request?
```

With inline buttons:
- `✅ Yes, resolved` → `callback_data: evo_confirm_resolved:{request_id}`
- `❌ No, still broken` → `callback_data: evo_confirm_failed:{request_id}`

3. **On `evo_confirm_resolved`:**
   - Mark `failed_requests.resolved = 1`, `reward = 1`
   - Mark evolution event `resolved = 1`
   - Resolve matching probes in probe store
   - Write trajectory to `synthesis_trajectories` (positive, for fine-tuning)
   - Write routing hint to SKILL.md
   - Log: `[Evolution] user confirmed RESOLVED — reward=1`

4. **On `evo_confirm_failed`:**
   - Increment retry counter
   - If retry < max_iterations: loop back to `maybe_evolve()` with failure notes
   - If max reached: mark escalated, add tracker todo, notify user
   - Write trajectory to `synthesis_trajectories` (negative, reward=0)
   - Log: `[Evolution] user rejected — reward=0, retry={attempt}`

5. **Timeout (48h no response):** treat as unconfirmed, do not mark resolved, do not retry automatically.

---

### 7. Thought-triggered evolution (non-request path)

Curiosity and self_improvement thoughts **can still trigger evolution** if they reference a concrete, testable action — not introspection. Gate:

```python
# Only evolve on thoughts that have an observable output
EVOLVABLE_CATEGORIES = {"gap_reflection"}  # self_improvement and curiosity: no evolution
```

`gap_reflection` from idle thinking (no `_source_request_id`) must pass a stricter observer gate: requires ≥1 `NO-verification` evolution event (a prior actual failure), not just frequency.

---

## Implementation Plan

**Phase 1 — Failure logging** (telegram_bot.py + new `failed_requests.py`)
- Detect fallback responses in `handle_message` after `infer_with_tools` returns
- Write `failed_request` record with original message + agent response
- No behaviour change for user

**Phase 2 — ThinkAtRest anchoring** (thought_engine.py)
- `_get_recent_gaps()` queries `failed_requests WHERE resolved=0`
- Generator prompt updated to pass concrete tasks not topics
- Thoughts from anchored requests tagged with `_source_request_id` + `_source_user_message`

**Phase 3 — Observer + Critique update** (observer_layer.py, critique_layer.py)
- Observer: short-circuit to EVIDENCED on `_source_request_id`
- Critique: pass original user message to assess prompt

**Phase 4 — Human confirmation flow** (telegram_bot.py callback handler)
- `evo_confirm_resolved` / `evo_confirm_failed` callback handlers
- Reward write to synthesis_trajectories
- Routing hint on confirmed resolve

---

## Consequences

**Positive:**
- Evolution only fires on real user failures — no more junk skills from idle reflection
- Critique verifies the actual user need, not a model-generated description match
- Human confirmation gives ground truth for reward signal + fine-tuning trajectories
- User stays in the loop — transparency about what EVA is trying to learn

**Negative:**
- Evolution frequency drops significantly (intentional — quality over quantity)
- Requires user engagement for confirmation — unanswered confirmations leave requests in limbo
- Failure detection heuristics will have false positives/negatives — needs tuning over time

---

## Metrics

- `failed_requests` table growth rate (how many real gaps per day)
- Resolution rate: confirmed_resolved / total_evolved
- Reward rate: reward=1 / total_confirmed
- Junk skill rate: skills installed but never confirmed (should approach 0)

---

### Amendment 2026-06-18

The failed_requests → evolution path is intact and simplified. The restructure
(ADR-005 amendment) removed the synthetic gap_reflection generator step that
competed with failed_requests for evolution attention. Now:
- `gap_reflection` ONLY comes from failed_requests
- `EvolvingThinkAtRest._run_think_cycle()` Phase 1 is the sole evolution trigger
- Max 3 retries per request, then abandoned
- No thought-generation step can produce gap_reflection entries
