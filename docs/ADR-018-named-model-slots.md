# ADR-018: Named Model Slots

**Date:** 2026-05-26  
**Status:** Accepted  
**Branch:** feat/named-model-slots

---

## Context

kernel-evolving previously used a single-slot architecture for model management:
module-level globals (`_model`, `_processor`, `_stt_model`, etc.) in `model_server.py`
held exactly one primary model and one optional STT model. This design served well
initially but became limiting as the system matured:

1. **Voice/audio use case:** The Gemma 4 E2B-it model is the optimal audio inference
   backend, while Nemotron 3B is preferred for text reasoning. To handle a voice note
   the server had to either run the audio model as the primary (wasting capability for
   text) or load it as a fixed `_stt_model` side-channel with no management API.

2. **Replica routing:** When spawning a voice-capable replica, there was no way to
   declare "use the audio model for this replica" — all replicas shared the same
   primary slot.

3. **VRAM visibility:** No health endpoint exposed which models were actually loaded,
   making debugging VRAM pressure opaque.

4. **On-demand loading:** The STT model loaded lazily but could not be unloaded on
   demand, and new slot types (draft, embed, speculative decoder) had no registration
   mechanism.

---

## Decision

We introduce a **named slot registry** (`SlotRegistry`) as an optional, additive layer
on top of the existing globals. The system operates in two modes:

- **Legacy mode** (default, `model_slots:` absent from `config.yaml`): zero behaviour
  change. All existing handlers use globals as before.
- **Slot mode** (`model_slots:` present): `SlotRegistry` is built at startup, slots
  are registered from config, and models wired after loading. The globals are still
  kept in sync via `_sync_globals_from_slot()`.

### Slot Roles

| Role      | Description                                      | Eviction policy     |
|-----------|--------------------------------------------------|---------------------|
| `primary` | Main reasoning model (Nemotron 3B)               | **Never evict**     |
| `audio`   | Audio/voice model (Gemma 4 E2B-it)               | LRU eligible        |
| `draft`   | Fast speculative decoder (future)                | LRU eligible        |
| `embed`   | Embedding model (future)                         | LRU eligible        |

### LRU Eviction Guard

`SlotRegistry` checks free VRAM (`torch.cuda.mem_get_info()`) before every `load()`.
If free VRAM drops below `vram_threshold_mb` (default 3000 MB), the slot with the
oldest `loaded_at` timestamp among non-primary loaded slots is evicted. Primary
slots are **never evicted**.

### VRAM Reality (RTX 4090 Laptop, 16.4 GB)

| Model                      | VRAM (4-bit quant) |
|----------------------------|--------------------|
| Nemotron 3B (primary)      | ~2 GB              |
| Gemma 4 E2B-it (audio)     | ~2.5 GB            |
| Both simultaneously        | ~4.5 GB            |
| FLUX (Fantasia, when active)| ~8 GB              |

Both models fit simultaneously in normal operation. Eviction is a **safety backstop**
for when Fantasia's FLUX model is loaded (consumes ~8 GB), leaving ~8 GB for the AI
stack. The 3 GB threshold ensures we never attempt to load into a slot that would OOM.

---

## Config Schema

```yaml
# model_slots:
#   primary:
#     model_path: null          # uses existing model.name/path if omitted
#     role: primary
#   audio:
#     model_path: ~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/...
#     dtype: bfloat16
#     quantize: 4bit
#     device: auto
#     role: audio
```

When `model_slots:` is absent, `_slot_registry` remains `None` and every code path
falls back to the pre-existing globals behaviour. This guarantees zero regression for
any deployment that does not opt in.

---

## Replica Integration

`Replica` dataclass gains an optional `slot: str = "primary"` field. When
`slot != "primary"`, `_run_task()` calls `model_client.load_slot(slot)` before
inference to ensure the named slot is loaded. Audio replicas (voice conversation
agents) should set `slot="audio"`.

`spawn_named()` in `replica.py` and `NamedReplicaIn` in `api.py` both accept the
new optional `slot` parameter, defaulting to `"primary"` so all existing callers
remain unchanged.

---

## New RPC Methods

| Method         | Description                                       |
|----------------|---------------------------------------------------|
| `load_slot`    | Load a named slot on demand; params: `{"slot": name}` |
| `unload_slot`  | Unload a named slot, free VRAM                    |
| `slot_status`  | Return JSON list of all registered slots          |

`health` response now always includes a `"slots"` key (`[]` when registry absent).

`model_client` exposes: `load_slot(name)`, `unload_slot(name)`, `slot_status()`.

---

## Backwards Compatibility

| Scenario                         | Behaviour                        |
|----------------------------------|----------------------------------|
| `model_slots:` absent in config  | `_slot_registry = None`, no change |
| Existing globals `_model` etc.   | Always kept in sync via `_sync_globals_from_slot()` |
| All existing RPC handlers        | Unchanged — read globals as before |
| Existing tests                   | All pass; 406 passed, 0 failures |
| Pre-slot model server             | `health()` test skips gracefully |

---

## Implementation Steps (this branch)

1. `src/model_slots.py` — `SlotSpec`, `SlotState`, `SlotRegistry` scaffold
2. `tests/test_model_slots.py` — 22 unit tests (eviction, LRU, status, compat)
3. Wire `SlotRegistry` into `model_server.py` alongside existing globals (non-breaking)
4. `config.yaml` — commented example `model_slots:` section + startup registry build
5. New RPC handlers: `load_slot`, `unload_slot`, `slot_status` + `health` slot field
6. `model_client.py` — `load_slot()`, `unload_slot()`, `slot_status()` client methods
7. `_handle_infer_with_audio` — slot-aware routing (slot param → registry → legacy)
8. `replica.py` + `api.py` — `slot` field on `Replica` and `NamedReplicaIn`
9. Integration test `TestNamedSlots` in `test_e2e_flow.py`; full suite green (406 passed)
10. This ADR

---

## Consequences

### Positive
- **Flexibility:** Multiple models can be managed independently with a clean API.
- **Autonomy:** kernel can load/unload audio model on demand without human intervention.
- **No external dependency:** Audio inference stays on local HF path — no cloud needed.
- **VRAM guard:** Eviction prevents OOM when Fantasia loads FLUX.
- **Observability:** `health()` and `slot_status()` expose exactly which models are loaded.
- **Extensibility:** Future slots (draft, embed, speculative) need only a config entry.

### Neutral
- Slot config is opt-in; legacy deployments are unaffected.
- `_sync_globals_from_slot()` is a thin shim; can be removed after all handlers migrate.

### Negative / Risks
- If both Nemotron and Gemma 4 are loaded simultaneously while Fantasia runs FLUX,
  total VRAM ≈ 12.5 GB — leaving ~4 GB margin on RTX 4090 (16.4 GB). Acceptable.
- `SlotRegistry._release()` deletes Python references and calls `cuda.empty_cache()`;
  if the model is referenced elsewhere (e.g. `_stt_model` global still pointing at it),
  the CUDA memory is not actually freed until all references are released. This is a
  known Python GC limitation; `_sync_globals_from_slot()` mitigates it for the primary
  path but callers should avoid caching `SlotState.model` long-term.
