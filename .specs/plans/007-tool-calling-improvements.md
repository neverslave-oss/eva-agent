# Plan 007 — Tool Calling Improvements

**Created:** 2026-06-04
**Status:** in-progress
**Branch:** feat/plan-007-tool-calling-loop

## Problem

kernel-evolving models aren't calling tools effectively. After comparing with LangChain's agent harness architecture, we identified 4 gaps.

## Goals

1. **Structured failure re-prompts** — Use `classify_tool_result()` metadata to re-prompt the model with actionable guidance when tools fail
2. **Leaner system prompt** — Move heavy reference material out of the prompt, keep it focused
3. **Middleware hooks** — Add `before_model` / `after_tool` hook points for composable behavior
4. **Subagent isolation** — Give replicas filtered tool lists + lean system prompts

## Priority

Implement #1 and #2 first (highest impact ÷ effort ratio). Plan #3 and #4 for follow-up.

## Files Affected

- `src/core/inference/model_server.py` — tool loop with failure re-prompts
- `src/core/inference/model.py` — in-process tool loop (same fix)
- `src/core/memory/context.py` — system prompt builder
- `src/core/inference/model_client.py` — socket client (pass-through)
- `src/core/tools.py` — classify_tool_result already exists, just needs to be consumed

## Success Criteria

- [ ] Failed tool calls get structured re-prompts referencing failure reason + suggesting alternatives
- [ ] System prompt is under 1500 chars (down from 3000+)
- [ ] Heavy reference material accessible via recall_memory
- [ ] Test suite passes (487+ tests, no new regressions)
- [ ] Live smoke test on @kernel_evo_agi_bot with a tool-requiring query