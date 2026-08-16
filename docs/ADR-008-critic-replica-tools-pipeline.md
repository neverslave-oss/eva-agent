# ADR-008: Autonomous Critic Replica with Tool Access + Inter-Replica Messaging

**Status:** Fully implemented ✅  
**Date:** 2026-05-07  
**Updated:** 2026-05-07 (session memory & closed-loop feedback wired)  
**Motivation:** The writer→critic validation loop requires replicas that (1) can use skills/tools to gather context and verify facts autonomously, and (2) can communicate results to each other without human orchestration. Currently replicas use `infer()` directly with no tool access, and have no inter-replica messaging channel.

---

## Implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| Session memory (SQLite) | ✅ Done | `chat_history_evolving.db` |
| goal_discovery seeding from history | ✅ Done | `seed_from_chat_history()` |
| Promoted signals persistence | ✅ Done | `promoted_signals.db` |
| ThoughtGenerator grounded in user history | ✅ Done | `{recent_interactions}` slot |
| context.py recent topics injection | ✅ Done | `_recent_conversation_topics()` |
| `_promote_to_idea()` → goal_discovery feedback | ✅ Done | `record_promoted(weight=3)` |
| Tool-access replicas (`infer_with_tools`) | ✅ Done | `replica.py` — `tools_enabled`, `output_path`, `input_path` |
| Inter-replica messaging / pipeline endpoint | ✅ Done | `POST /replica/pipeline` in `api.py` |

---

## Implemented: Session Memory & Closed-Loop Feedback

Commits `4bd3a57` → `536a73f` (v1.7.0-evolving).

### Why this was needed

Before this work, three feedback paths were broken:

1. **`_interaction_log` reset on every restart** — goal_discovery's ring buffer was in-memory only. RECURRENCE_THRESHOLD of 3 was effectively never reached in practice because it required 3 identical messages in a single process run.
2. **Think-at-Rest was context-free** — `_GENERATOR_TEMPLATE` had no signal about what the user actually asks. Evo generated 20 near-identical generic thoughts because Gemma 4 had nothing to ground on.
3. **Promoted ideas were dead ends** — `_promote_to_idea()` wrote a markdown file that no downstream subsystem ever read. High-score thoughts (0.9) produced a `.md` nobody acted on.

### `src/memory.py` — Two-tier chat persistence

**Tier 1 (hot):** `~/.kernel_evolving_memory.json` — sliding window, last 20 turns, fast load on every inference.

**Tier 2 (cold):** `~/.kernel/workspace/chat_history_evolving.db` — SQLite, every turn, permanent.

```sql
CREATE TABLE messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot        TEXT    NOT NULL DEFAULT 'kernel-evolving',
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
```

`load()` falls back to SQLite when JSON is empty — context survives restarts. `history(limit)` is the cross-session query API used by all three consumers.

### `src/goal_discovery.py` — Persistent pattern detection

**`seed_from_chat_history(limit=200)`** — called at `start_discovery_thread()` startup. Reads user messages from `chat_history_evolving.db` AND replays `promoted_signals.db` rows at their original weight into `_interaction_log`. Pattern counts survive restarts.

**`record_promoted(text, weight=3, category, score)`** — called when a thought is promoted. Inserts `weight` copies into `_interaction_log` AND persists to `promoted_signals.db`.

```sql
CREATE TABLE promoted_signals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    weight     INTEGER NOT NULL DEFAULT 1,
    category   TEXT,
    score      REAL,
    created_at TEXT    NOT NULL
);
```

**Why weight=3:** RECURRENCE_THRESHOLD=3. A promoted idea (already validated by the evaluator at ≥0.7) is a higher-quality signal than a raw user message. One promoted idea + one matching user message immediately hits threshold.

### `src/thought_engine.py` — Grounded Think-at-Rest

**`EvolvingThinkAtRest._get_recent_interactions()`** — reads last 6 user messages from SQLite via `memory.history()`.

**`_GENERATOR_TEMPLATE` extended** with `{recent_interactions}` slot:
```
- Recent interactions with your user: {recent_interactions}
```

**`EvolvingThinkAtRest._on_thought_accepted()` override:**
- Calls `super()` (handles evolution trigger, ideas, todos, Telegram)
- If `gap_reflection` AND score ≥0.6: calls `goal_discovery.record_unhandled(thought)` — feeds thought into pattern detection

**`_promote_to_idea()` extended:** After writing `.md`, calls `goal_discovery.record_promoted(weight=3)` — closes the previously dead-end path.

### `src/context.py` — Recent topics in system prompt

**`_recent_conversation_topics(limit=5)`** — reads last 5 user messages from SQLite.

Injected into `build_system_prompt()` under `## Session continuity`:
```
• Recent conversation topics (from prior sessions):
  • use the browser automation to read fabiopacifici.com
  • do you know how react useEffect works?
```

Also fixed: `_last_interaction()` and `_memory_stats()` now correctly point to `~/.kernel_evolving_memory.json`.

### Signal flow (implemented)

```
User message
    ↓
chat_history_evolving.db (permanent)
    ↓
memory.history() → 3 consumers:
    1. ThoughtGenerator ({recent_interactions})
    2. goal_discovery.seed_from_chat_history() [on startup]
    3. context._recent_conversation_topics() [every inference]

Think cycle (idle)
    ↓
gap_reflection (score ≥0.6) → goal_discovery.record_unhandled()

Promoted idea (score ≥0.7)
    ↓                         ↓
ideas/*.md              goal_discovery.record_promoted(weight=3)
                              ↓
                         _interaction_log (3× weight)
                              ↓
                         promoted_signals.db (persistent)
                              ↓ [on next startup]
                         seed_from_chat_history() replays at weight

Pattern count ≥ RECURRENCE_THRESHOLD
    ↓
_run_discovery_cycle() → evolution_hook.maybe_evolve() → new skill
```

### Databases introduced

| File | Purpose | Reset policy |
|------|---------|-------------|
| `~/.kernel/workspace/chat_history_evolving.db` | Every conversation turn | Never |
| `~/.kernel/workspace/promoted_signals.db` | Promoted idea weights | Never (accumulates) |

---

## Implemented: Change 1 — Tool-access replicas

Add `tools_enabled: bool = False` field to `Replica`. When True, `message()` calls `infer_with_tools()` instead of `infer()` — giving the replica access to exec_shell, read_file, write_file, http_get, run_skill, run_routine.

The critic replica gets `tools_enabled=True`. It can then:
- `run_skill("web-scrape-summarize", url)` to verify claims against live sources
- `run_skill("github", "check repo")` to validate code references
- `read_file(draft_path)` to load the writer's draft
- `write_file(annotations_path, content)` to save structured review

### Modify `replica.py`

1. Add `tools_enabled: bool = False` to `Replica` dataclass
2. In `message()`, branch on `tools_enabled`:
   ```python
   if self.tools_enabled:
       from agent import triage
       reply = triage(user_text)  # full tool-access path
   else:
       reply = infer(messages, max_new_tokens=512)
   ```
3. Add `output_path: Optional[str]` — replica writes final response to this path
4. Add `input_path: Optional[str]` — replica reads this before starting

---

## Implemented: Change 2 — Inter-replica messaging via shared result files

Simple, reliable approach — no new IPC layer needed:
- Writer replica writes draft to `~/.kernel/workspace/docs/drafts/<task>.md`
- A `replica_done` callback posts to `/replica/pipeline/{name}/next`
- Critic replica reads the draft, annotates, writes to `~/.kernel/workspace/docs/reviews/<task>.md`
- Coordinator collects both and returns structured result to the caller

### Pipeline definition

```json
{
  "pipeline": "write-review",
  "tasks": [
    {
      "name": "writer",
      "role": "custom",
      "brief": "You are a technical blog writer...",
      "tools": false,
      "output_path": "~/.kernel/workspace/docs/drafts/post6.md"
    },
    {
      "name": "critic",
      "role": "custom",
      "brief": "You are an autonomous quality critic. Read the draft, verify facts, annotate issues.",
      "tools": true,
      "input_from": "writer",
      "output_path": "~/.kernel/workspace/docs/reviews/post6.md"
    }
  ]
}
```

### New endpoint: `POST /replica/pipeline`

```python
@app.post("/replica/pipeline")
def replica_pipeline(body: PipelineRequest):
    results = {}
    prev_output = None
    for stage in body.tasks:
        brief = stage.brief
        if prev_output:
            brief += f"\n\n## Input from previous stage\n{prev_output}"
        r = replica.spawn_named(
            name=stage.name, role="custom",
            custom_prompt=brief, tools_enabled=stage.get("tools", False),
        )
        result = r.message(stage.get("task", "Begin your work."))
        results[stage.name] = result
        if stage.get("output_path"):
            Path(stage["output_path"]).expanduser().write_text(result)
        prev_output = result
    return {"status": "complete", "results": results}
```

---

## Tests

### Implemented (session memory & feedback loop)
- `test_memory_save_writes_to_sqlite`
- `test_memory_load_cold_start_seeds_from_sqlite`
- `test_memory_history_returns_cross_session_messages`
- `test_seed_from_chat_history_loads_user_messages`
- `test_seed_from_chat_history_replays_promoted_signals`
- `test_record_promoted_inserts_weight_copies_to_ring_buffer`
- `test_record_promoted_persists_to_db`
- `test_thought_generator_includes_recent_interactions`
- `test_evolving_on_thought_accepted_feeds_gap_reflection`
- `test_promote_to_idea_calls_record_promoted`
- `test_context_includes_recent_topics`

### Implemented (critic replica pipeline)
- `test_replica_with_tools_calls_infer_with_tools`
- `test_replica_without_tools_calls_infer`
- `test_pipeline_runs_stages_in_sequence`
- `test_pipeline_passes_output_between_stages`
- `test_critic_replica_can_use_run_skill`

---

## Think-at-Rest integration (updated)

The session memory work completes the passive feedback side. Once the critic replica pipeline is implemented, the full loop closes:

- Critic annotations → `record_promoted()` → high-weight pattern signals
- Think-at-Rest reads critic's annotations during idle → generates `gap_reflection` thoughts → triggers evolution for recurring annotation patterns
- Proactively messages the user if quality score falls below threshold

## Future work
- Base kernel parity — mirror session memory changes to base kernel (todo in `~/.kernel/workspace/todos.md`)
- Cross-session topic clustering — cluster user intents over full history, inject summaries instead of raw messages
- Promoted signal decay — time-based decay so old ideas don't permanently dominate pattern counts
