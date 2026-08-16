# ADR-014: Perceptive Context Pipeline — Adaptive History, Embedding Retrieval, and Skill-Driven Context Providers

**Status:** Implemented ✅  
**Date:** 2026-05-25  
**Supersedes:** Partial context work in ADR-008 (session memory, recent topics injection)

---

## Problem

The agent's context window was static and blind. Three independent failure modes:

1. **Fixed history window** — `_history[-80:]` hardcoded regardless of model. Gemma 4 at 32k tokens and Nemotron at 256k native were both capped at the same arbitrary 80 messages. No relationship between model capability and how much context it actually receives.

2. **No semantic memory retrieval** — The EmbeddingClient was initialised in `agent.py` for skill matching but never used for memory. When Fabio referenced something from a week ago, the agent had no way to surface it unless it happened to fall inside the linear 80-message window.

3. **Hardcoded collective memory path** — The only cross-agent knowledge injection was a hardcoded `subprocess` call to a specific filesystem path (`olly_workspace/collective-memory/scripts/search.py`). If the skill moved, wasn't installed, or the path changed, it silently failed. Other agent workspaces (Marty, Lawy, etc.) contributed entries that were never automatically surfaced.

---

## Decision

Redesign the context pipeline as three composable layers, each independently useful:

### Layer 1 — Token-aware history window

Replace the `[-80:]` hardcoded slice with a model-aware budget:

- Read `model_context_lengths` map from `config.yaml` (keyed by model name substring)
- Reserve 4096 tokens for system prompt + current message + tool definitions
- Convert remaining budget to a char-based proxy (`chars = tokens / 0.4`)
- Hard cap: 120 pairs (safety floor regardless of context size)
- Trim oldest turns first until total chars fit

```yaml
model_context_lengths:
  gemma-4-e2b:  32768   # 32k — VRAM-limited on 16GB
  gemma-4-e4b:  32768
  qwen3-vl:     32768
  nemotron:     65536   # 64k — native 256k, practical with 4-bit + 16GB VRAM
```

This means Nemotron can use 2× the history depth of Gemma 4 automatically when loaded.

### Layer 2 — Embedding-based memory retrieval

On every inference, before building the message list:

1. Embed the current user message using the loaded `EmbeddingClient`
2. Score all past turns (user + assistant) from full history by cosine similarity
3. Inject top-5 turns above similarity threshold 0.55 as a `## Relevant past context` block in the system prompt

This gives the model access to semantically relevant prior turns even when they're outside the linear history window. A question about a decision made 300 turns ago surfaces automatically if the current query is related.

```python
_query_emb = _embedding_client.embed(text)
_scored = [(cosine_similarity(_query_emb, _embedding_client.embed(m["content"])), m) for m in _all_history]
_top = top 5 above 0.55
```

### Layer 3 — Skill-driven context providers (self-discovering)

Replace the hardcoded collective memory subprocess with a discovery loop:

**SKILL.md frontmatter extension:**
```yaml
context_provider: true
context_search_cmd: "python3 /path/to/search.py {query} --top 5"
```

**Agent discovery loop (runs before every inference):**
```python
_ctx_skills = [s for s in _skills if s.get("context_provider")]
for skill in _ctx_skills:
    cmd = skill["context_search_cmd"].replace("{query}", shlex.quote(text[:200]))
    result = subprocess.run(cmd, shell=True, timeout=6)
    # inject result into system prompt
```

Any skill can opt in by setting `context_provider: true`. No path assumptions, no agent-specific code. New providers are discovered automatically as skills are installed. The collective-memory skill is the first provider; others can follow (e.g. a workspace-tracker skill, a web-search skill for live facts).

---

## Nemotron context window

Nemotron-Labs-Diffusion-3B has a **256k token native context** (`max_position_embeddings: 262144`). Loading it at full context would OOM on a 16GB laptop GPU. `_load_nemotron()` applies the `max_context_length` from config as an `AutoConfig` override before loading:

```python
_model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
_model_cfg.max_position_embeddings = int(max_ctx)  # 65536 from config
```

The `model_context_lengths` map then ensures the agent sends proportionally more history when Nemotron is loaded vs Gemma 4.

---

## Collective memory — multi-agent shared store

The collective memory at `workspace/collective-memory/` is shared across all agent workspaces:

| Workspace | Agent | collective-memory skill | context_provider |
|-----------|-------|------------------------|-----------------|
| workspace/ | Olly | ✅ | ✅ |
| workspace-marketing/ | Marty | ✅ | ✅ |
| workspace-legal/ | Lawy | ✅ | ✅ |
| workspace-invest/ | Invest agent | ✅ | ✅ |
| workspace-hack/ | Hack agent | ✅ | ✅ |
| workspace-opencode/ | Opencode agent | ✅ | ✅ |
| workspace-client/ | Client agent | ✅ | ✅ |
| kernel-evolving (this repo) | Kernel-evo | ✅ (via ecosystem) | ✅ |

All agents write to the same `entries/` directory and the same `memory.db`. Entries from any agent are visible to all others at inference time.

**Quality gate on auto-write** — `telegram_bot.py`'s `_write_collective_memory()` previously wrote every plain-text reply > 200 chars, including greetings, status checks, and error messages (309 files, 198 unindexed junk). The gate now requires:
- Message not a greeting/shorthand (regex pattern match)
- Reply > 300 chars
- Reply contains at least one substantive marker: URL, file path, version number, date, code block, currency symbol, commit/PR reference

This cut noise entries significantly and keeps the store signal-dense.

**Embedding fallback** — `search.py` now falls back to SentenceTransformer in-process if the HTTP embedding server at `:8770` is unreachable, so collective memory search never silently returns zero results due to a down dependency.

---

## Streaming output

Added `chunk_callback` to the inference pipeline to support live streaming of model output to Telegram:

- `provider.py _openai_tool_loop`: when `chunk_callback` is set, uses `stream=True` on the OpenAI/Copilot API. SSE lines are parsed, text deltas fire `chunk_callback(delta)`, tool call deltas are assembled from streamed fragments.
- `agent.py triage()`: accepts and forwards `chunk_callback`
- `telegram_bot.py`: creates `_chunk_cb` that edits the working message every 40 chars — Fabio sees the model writing in real time, tool steps interleaved with streamed text.

---

## Consequences

**Positive:**
- Model uses its full context capacity — Nemotron 64k vs Gemma 32k, automatically
- Semantically relevant memory surfaces without being in the linear window
- Any skill can contribute context — collective memory, workspace tracker, web search, etc.
- kernel-evolving shares Olly's cross-agent knowledge base at every inference
- Junk entries stop polluting collective memory
- Search works even when embedding server is down

**Negative / trade-offs:**
- Embedding retrieval adds ~100-200ms per inference (EmbeddingClient is in-process, cached)
- Context provider skill commands add ~50-100ms each (subprocess, bounded by timeout=6s)
- System prompt grows with retrieved context — need to monitor total token usage

**Not changed:**
- History is still linear (no summarisation). Cross-session topic clustering remains future work (noted in ADR-008).
- Context providers run synchronously before inference — no async pre-fetching yet.

---

## Files changed

| File | Change |
|------|--------|
| `config.yaml` | `model_context_lengths` map, Nemotron `max_context_length: 65536` |
| `src/agent.py` | Token-aware history, embedding retrieval, context provider loop |
| `src/skills.py` | Parse `context_provider` + `context_search_cmd` from frontmatter |
| `src/model_server.py` | `_load_nemotron`: apply `max_context_length` via AutoConfig |
| `src/provider.py` | `chunk_callback` + `stream=True` in `_openai_tool_loop` |
| `src/telegram_bot.py` | `_chunk_cb` live streaming, quality gate on `_write_collective_memory`, skill-based `_search_collective_memory` |
| `collective-memory/scripts/search.py` | Embedding HTTP fallback to SentenceTransformer |
| `~/.kernel-evolving/ecosystem/.../collective-memory/SKILL.md` | `context_provider: true` + `context_search_cmd` |
| All other workspace collective-memory SKILL.mds | Same flag propagated to all 7 agent workspaces |

---

## Related ADRs

- ADR-004: Self-evolving agent loop (tier 2 escalation)
- ADR-005: Think-at-Rest (idle reflection)
- ADR-008: Session memory, goal_discovery seeding, critic replica pipeline
