# Plan: Split `handlers.py` (issue #2 — handlers.py scope)

**Date:** 2026-10-03
**Issue:** #2 — split oversized api.py (2947, DONE via PR #12), handlers.py (2168), thought_engine.py (1641, DONE via PR #9)
**Scope:** `src/core/inference/handlers.py` ONLY (2168 lines, 20 handler functions, no classes).
**Branch:** `refactor/handlers-split` (based on latest `origin/dev`)
**Status:** In progress

## Context

`handlers.py` is already a leaf module from the original model_server.py refactor. It owns the entire JSON-RPC handler/inference region. Shared helpers live in `handlers_common.py`; `model_server.py` re-imports every handler via `from .handlers import *` + an explicit 20-name import and the `HANDLERS` dispatch table.

Issue #2 asks to extract handler submodules.

## Hard rules (Fabio, non-negotiable)

1. **ONE CLASS PER FILE** — no classes exist here yet, but any new ones must follow. (This file is pure functions; the rule still governs.)
2. Pure extraction — no logic rewrites, no signature changes, no renamed handlers/methods.
3. Dedicated branch, chunked slices, gate green + commit + push after EVERY slice. Never leave work uncommitted across an interruption.
4. **Public-surface contract preserved**: `model_server.py` imports `_handle_infer...` etc. by name from `.handlers` — the new `handlers/__init__.py` MUST re-export all 20 handlers (+ `_detect_capabilities`, which model_server also imports from `.handlers`).
5. Watch the latent-bug class (proven twice: telegram + api splits): missing imports/constants in moved code → run the AST undefined-name scan on every new module before committing.
6. Cross-module calls (from the call graph) must import from the leaf module, with **no circular imports**: tools → chat (fallback), multimodal → (self), slots → ops (`_resolve_loaded_model_name`), draft → (self). Verify with the scan.

## Call graph (verified 2026-10-03)

```
_handle_infer                      L 88-184    -> (none)
_handle_infer_plain                L 187-284   -> (none)
_run_two_stage_if_available        L 291-560   -> _nemotron_synthesize_answer
_nemotron_synthesize_answer        L 563-667   -> (none)
_handle_infer_with_tools           L 670-1260  -> _handle_infer_plain, _run_two_stage_if_available
_handle_infer_with_image           L 1263-1434 -> _cloud_multimodal_infer
_handle_infer_with_audio           L 1437-1626 -> _cloud_multimodal_infer
_handle_infer_local                L 1629-1715 -> (none)
_cloud_multimodal_infer            L 1724-1809 -> (none)
_handle_vram_free_mb               L 1812-1818 -> (none)
_resolve_loaded_model_name         L 1821-1845 -> (none)
_handle_health                     L 1848-1869 -> _handle_vram_free_mb, _resolve_loaded_model_name
_handle_load_slot                  L 1876-1887 -> (none)
_handle_unload_slot                L 1890-1901 -> (none)
_handle_slot_status                L 1904-1906 -> (none)
_handle_unload                     L 1909-1955 -> (none)
_handle_swap_model                 L 1958-2058 -> _resolve_loaded_model_name
_ensure_drafter_only               L 2061-2078 -> (none)
_handle_infer_draft                L 2081-2167 -> _ensure_drafter_only
```

## Target layout

```
src/core/inference/handlers/__init__.py   → re-exports all 20 handlers + _detect_capabilities
src/core/inference/handlers/chat.py       → _handle_infer, _handle_infer_plain
src/core/inference/handlers/tools.py      → _run_two_stage_if_available, _nemotron_synthesize_answer, _handle_infer_with_tools
src/core/inference/handlers/multimodal.py → _handle_infer_with_image, _handle_infer_with_audio, _handle_infer_local, _cloud_multimodal_infer
src/core/inference/handlers/ops.py        → _handle_vram_free_mb, _resolve_loaded_model_name, _handle_health
src/core/inference/handlers/slots.py      → _handle_load_slot, _handle_unload_slot, _handle_slot_status, _handle_unload, _handle_swap_model
src/core/inference/handlers/draft.py      → _ensure_drafter_only, _handle_infer_draft
```

`git mv src/core/inference/handlers.py src/core/inference/handlers/__init__.py` first (keeps `from .handlers import ...` resolving), then shrink `__init__.py` to re-exports slice by slice — the same proven pattern as the api split.

## Slices

- **S1 — package conversion + chat.py:** git mv to package; move the 2 chat handlers into chat.py; re-export from __init__.py. Gate: handler tests green.
- **S2 — tools.py:** 3 tool-calling handlers (import `_handle_infer_plain` from .chat).
- **S3 — multimodal.py:** 4 multimodal handlers.
- **S4 — ops.py + slots.py:** telemetry + slot ops; slots imports `_resolve_loaded_model_name` from .ops.
- **S5 — draft.py + final:** draft handlers; `__init__.py` = pure re-exports; full scan + full gate; PR.

## Test gate
- Handler offline tests: `tests/test_cloud_fallback.py test_model_server_history.py test_nemotron_native_agentic.py test_tool_arg_parse_sweep.py test_tool_critique_repair.py test_tool_loop_validation.py` (~47 tests) after EVERY slice.
- Full: also `tests/test_api.py -m "not inference"` + telegram gate (89 passed baseline).

## Out of scope
- `api.py` split (done, PR #12), `thought_engine.py` (done, PR #9).