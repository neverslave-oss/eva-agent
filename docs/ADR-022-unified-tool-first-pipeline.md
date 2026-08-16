# ADR-022: Unified Tool-First Pipeline

**Status:** Proposed  
**Date:** 2026-07-04  
**Author:** Fabio + Olly  
**Related:** ADR-004 (Self-Evolving), ADR-008 (Critic Replica), ADR-011 (Micro-Planner)

---

## Problem

The current `agent.triage()` flow has a **routing-first** architecture that causes tool-call failures:

```
User message
    ↓
triage()
    ├─ Slash commands?        → handle directly
    ├─ Routine trigger?      → run_routine()
    ├─ Skill command match?  → run_skill() → infer() ← NO TOOLS
    ├─ Skill intent match?   → run_skill() → infer() ← NO TOOLS
    ├─ Skill name match?     → run_skill() → infer() ← NO TOOLS
    ├─ Semantic skill match? → run_skill() → infer() ← NO TOOLS  ← INTERCEPTS
    ├─ Status query?         → return status string
    ├─ Micro-Planner?        → plan → triage each step (recursive)
    ├─ Evolution hook?       → maybe_evolve → retry
    └─ infer_with_tools()    ← ONLY path with tools
```

### Root causes of tool-call failure

1. **Semantic skill matching intercepts before tool pipeline.** The embedding similarity step routes "read_file(USER.md)" to the `send-workspace-user-md` skill, which calls `infer_fn(messages)` — plain Nemotron inference with **no tools passed**. The model says "I can't access files" because it literally has no tools in that context.

2. **Skill execution is tool-blind.** `run_skill()` passes `infer` (plain) as the inference function, not `infer_with_tools`. Skills that need file access, shell execution, or HTTP calls cannot use the model's native tools — they rely entirely on the skill's SKILL.md instructions being understood by a model with no tool context.

3. **Poisoned history amplification.** When messages do reach `infer_with_tools`, the two-stage pipeline (Qwen tool-call loop → Nemotron synthesis) injects all accumulated history turns into Nemotron's context. Bad examples ("I can't access files", "these tools aren't real") reinforce the behavior.

4. **Session learnings teach wrong behavior.** AGENTS.md contained "user preferred simulated file analysis" and "simulated audio file analysis for non-existent files" — literally instructing the model to fake results instead of calling tools.

### Immediate fixes applied (2026-07-03)

- Cleared poisoned chat history (54 turns of tool-denial examples)
- Cleared SQLite chat history (83 rows)
- Rewrote AGENTS.md with corrective learnings ("ALWAYS call tools directly")
- Added missing `PROMPT_LOG_DB` to `runtime_paths.py`

These are band-aids. The architecture itself is the problem.

---

## Decision

Replace the routing-first triage with a **tool-first pipeline** where the model always has access to tools and decides what to call — including skill and routine invocation via tool calls.

### New architecture

```
User message
    ↓
1. Input received (Telegram webhook / API / voice)
    ↓
2. System prompt built
    ├─ Live context (services, VRAM, date, user profile)
    ├─ Corrective memory (session learnings, recent context)
    ├─ Progressive skill/routine disclosure (names + commands only)
    └─ Tool definitions (11 native + 3 meta: run_skill, run_routine, list_routines)
    ↓
3. Full message context assembled
    ├─ System prompt
    ├─ History window (token-budgeted, recent-first trim)
    ├─ Semantic retrieval (relevant past turns)
    └─ Current user message
    ↓
4. Two-stage inference with tools
    ├─ Stage 1: Qwen tool-calling loop (max_steps=15)
    │   ├─ Qwen decides: call tool, call skill, call routine, or answer directly
    │   └─ Each tool call executes and result feeds back
    └─ Stage 2: Nemotron synthesis
        ├─ Receives: system prompt + history + tool-call trajectory
        ├─ Synthesizes final answer from all tool results
        └─ NO re-injection of old history — only current turn trajectory
    ↓
5. Response streamed to Telegram
    ├─ Step callbacks send progress messages
    └─ Final reply delivered
    ↓
6. Memory persisted
    ├─ Turn saved to JSON hot-window + SQLite
    ├─ Trajectory collected if tool calls were made (ADR-013)
    └─ Failed request detection (ADR-020)
```

### Key changes

#### 1. Skills and routines become tool-callable

Instead of triage routing to skills before the model sees the message, skills and routines are exposed as **tools** that the model calls during the tool-calling loop:

```python
# In core/tools.py — add meta-tools
META_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Execute an installed skill by exact name. Use search_skills first to find the right skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Exact skill name"},
                    "input": {"type": "string", "description": "Input text for the skill"}
                },
                "required": ["skill_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": "Find skills matching a keyword query. Returns name, description, and commands for each match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_routine",
            "description": "Execute a named routine. Use list_routines first to see what's available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "routine_name": {"type": "string", "description": "Exact routine name"},
                    "input": {"type": "string", "description": "Optional input for the routine"}
                },
                "required": ["routine_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_routines",
            "description": "List all available routines with names, descriptions, and triggers.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

TOOLS = NATIVE_TOOLS + META_TOOLS  # 11 native + 4 meta = 15 total
```

#### 2. Triage becomes a thin pre-filter

The triage function is simplified to handle **only** slash commands and status queries — everything else goes straight to `infer_with_tools`:

```python
def triage(text, step_callback=None, chat_id="", chunk_callback=None):
    lower = text.lower().strip()
    
    # ── Slash commands (exact match only) ──
    if lower.startswith("/"):
        cmd = lower.split()[0]
        if cmd in BUILTIN_COMMANDS:
            return handle_builtin(cmd, text, chat_id)
        # /skill, /run, /routines, /skills — explicit dispatch
        if cmd.startswith("/skill_") or cmd.startswith("/run_"):
            return handle_picker_command(cmd, text, chat_id)
    
    # ── Everything else → tool-first pipeline ──
    # The model decides whether to call a skill, routine, or native tool
    # based on the tool definitions in the system prompt.
    return infer_with_tools_pipeline(text, chat_id, step_callback, chunk_callback)
```

#### 3. Progressive disclosure in system prompt

Instead of injecting full skill instructions (which bloats context), the system prompt lists skill names and one-line descriptions. The model calls `search_skills` to discover details, then `run_skill` to execute:

```
## Tools and capabilities

You have 15 tools available. Use them proactively.

| Tool | Description |
|------|-------------|
| `exec_shell(command)` | Run shell commands |
| `read_file(path)` | Read any local file |
| `write_file(path, content)` | Write files |
| `http_get(url)` | HTTP GET request |
| `web_search(query)` | Search the web |
| `send_file(path, caption)` | Send file via Telegram |
| `recall_memory(query)` | Search past conversations |
| `search_skills(query)` | Find skills by keyword |
| `run_skill(skill_name, input)` | Execute a skill by name |
| `run_routine(routine_name)` | Execute a routine |
| `list_routines()` | List available routines |

> Call `search_skills(query)` to discover what skills are available before using `run_skill`.
```

#### 4. History pruning: Nemotron synthesis receives only current turn

The two-stage pipeline currently injects all history turns into Nemotron's synthesis. The fix:

```python
# Stage 2: Nemotron synthesis receives ONLY the current turn trajectory
# (system prompt + current user message + tool call results)
# NOT the full accumulated history.
synthesis_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_message},
    *[msg for step in tool_steps for msg in step.messages],
]
```

History is still available to Qwen in Stage 1 for context, but Nemotron only sees the current exchange. This prevents bad examples from poisoning synthesis.

#### 5. Slash commands remain direct

Slash commands (`/start`, `/status`, `/skills`, `/routines`, `/run`, `/skill_`) continue to bypass the model and execute directly — they're deterministic UI actions, not inference tasks.

---

## Migration plan

### Phase 1: Immediate fixes (done 2026-07-03)
- [x] Clear poisoned chat history
- [x] Clear poisoned session learnings
- [x] Fix PROMPT_LOG_DB ImportError
- [ ] Remove `send-workspace-user-md` skill (auto-evolved, causes false semantic match)

### Phase 2: Tool-first pipeline (this ADR)
- [ ] Add `run_skill`, `search_skills`, `run_routine`, `list_routines` as tool definitions in `core/tools.py`
- [ ] Implement tool execution handlers for meta-tools in the tool loop
- [ ] Simplify `triage()` to handle only slash commands and status queries
- [ ] Move all skill/routine routing to model decision (via tool calls)
- [ ] Cap Nemotron synthesis history to current turn only

### Phase 3: Context management
- [ ] Progressive skill disclosure: names only in system prompt, details via `search_skills` tool call
- [ ] Smart history window: keep recent turns + semantically relevant turns, drop the rest
- [ ] Session learnings guard: never write "prefer simulated" or "cannot access" learnings

### Phase 4: Validation
- [ ] Tool calling test suite: verify each of the 15 tools is callable and returns correct results
- [ ] Regression test: "read_file(path)" reaches tool pipeline, not semantic skill match
- [ ] History poisoning test: 50-turn bad history does not prevent tool calls
- [ ] End-to-end Telegram test: message → tool call → response delivered

---

## Why this works

| Problem | Old behavior | New behavior |
|---------|-------------|--------------|
| "read_file(USER.md)" | Semantic match → `send-workspace-user-md` skill → plain `infer()` → "I can't access files" | Falls through to `infer_with_tools` → model calls `read_file` tool → actual file contents returned |
| Skill needs shell access | Skill runs via plain inference, no tool access | Skill runs via `run_skill` tool call → tool loop has `exec_shell` available |
| Poisoned history | Nemotron sees 54 turns of "I can't access files" | Nemotron sees only current turn trajectory |
| Wrong skill match | Embedding similarity routes to wrong skill | Model decides via `search_skills` → `run_skill` based on full context |
| Micro-Planner for multi-step | Separate planner step before tools | Model chains tool calls naturally in the tool loop |

---

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Model may not call `search_skills` before `run_skill` | System prompt instruction: "Always call search_skills first to find the right skill" |
| Model may call wrong skill name | `search_skills` returns exact names + descriptions; fuzzy matching in `run_skill` handler |
| Latency increase: tool call → skill execution → tool call | Skill execution is fast (subprocess); total latency similar to current semantic matching |
| Slash command habits broken | Slash commands remain direct — no change to `/status`, `/skills`, etc. |
| Nemotron synthesis quality without full history | Current turn + tool results provide sufficient context; full history available in Qwen Stage 1 |

---

## Tests

- `test_tool_first_pipeline_simple` — "what's 2+2" → direct answer, no tool calls
- `test_tool_first_pipeline_read_file` — "read ~/.kernel-evolving/workspace/USER.md" → `read_file` tool called
- `test_tool_first_pipeline_skill_discovery` — "search for browser skills" → `search_skills` called → results returned
- `test_tool_first_pipeline_routine` — "run morning briefing" → `run_routine` called
- `test_nemotron_synthesis_current_turn_only` — 50-turn bad history does not poison synthesis
- `test_progressive_disclosure` — system prompt contains skill names, not full instructions
- `test_slash_commands_unchanged` — `/status`, `/skills`, `/run` still work directly
- `test_semantic_skill_no_intercept` — "read_file(path)" reaches tool pipeline, not skill match