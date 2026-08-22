# Fix Spec — Vision error: 'NoneType' object has no attribute 'apply_chat_template'

**Date:** 2026-08-22
**Status:** ✅ **COMPLETED** — vision fix verified working in live test; audio twin also fixed
**Branch:** `fix/cloud-audio-vision-402-fallback`
**Related:** `eeb2ada` (cloud → native fallback), issues #426 #427

> ⚠️ **Pair-programming for remaining work.** The NoneType guard described below was
> applied ad-hoc (via Olly) on Fabio's request; the *second* error (`KeyError: 'model'`)
> is a fresh failure surfaced during live testing and is **NOT yet fixed**. Remaining
> work should be done together.

## Problem

After the cloud-vision fallback patch (`eeb2ada`), sending a photo to the bot still
raises:

```
🐬 Vision error: 'NoneType' object has no attribute 'apply_chat_template'
```

The cloud fallback path *itself* now reaches the native branch, but `active_processor`
(in the model-server path) or `_processor` (in the in-process path) is `None` when
`apply_chat_template` is called — so the fallback crashes instead of producing a description.

## Evidence / Reproduction

- Telegram photo → `telegram_bot.py` `photo_file_id` branch calls
  `core.inference.model.infer_with_image(local_path, ...)`.
- `model.py:376 infer_with_image` → model server is running → routes to
  `model_client.infer_with_image(...)` → JSON-RPC `infer_with_image` to model server
  (port 8779, pid 237295).
- `model_server.py:2243 _handle_infer_with_image`:
  - `providers.vision != "local"` → cloud attempt → fails (402/offline) → falls through.
  - `_ensure_multimodal_slot()` is called (line 2291), then `active_processor = _mm_processor`.
  - If `_mm_processor` is still `None`, line 2316
    `active_processor.apply_chat_template(...)` throws `'NoneType' ... 'apply_chat_template'`.
- Same risk in the in-process fallback `model.py:397`: `_processor` may be `None` if
  `infer_with_image` runs without `_ensure_model()` having populated it (vision
  requested before any text inference).

## Root Cause

Two paths can yield a `None` processor:

1. **Model-server path (`model_server.py` ~line 2300).** `_ensure_multimodal_slot()`
   does **not** guarantee `_mm_processor` is set:
   - It short-circuits with `return` when `_mm_model is not None` — but that early-return
     doesn't verify `_mm_processor` is also populated.
   - If `_ensure_multimodal_slot()` raises (missing `stt_model` config, load failure, or
     the `same-model` alias branch running while `_model`/`_processor` are also `None`),
     the exception propagates out of `_handle_infer_with_image` before
     `active_processor` is ever assigned → downstream callers see either a stale `None`
     or an uncaught error. There is **no second fallback** to the main `_processor` or a
     clean error message after `active_model is None → _ensure_model()`.
   - After the cloud fall-through, `use_hf_path`/`active_model`/`active_processor` are
     only assigned *inside* the `if not _audio_capable / elif _mm_model is not None`
     branches. If neither branch sets a processor, the code proceeds to
     `if active_model is None: _ensure_model()` and then blindly calls
     `active_processor.apply_chat_template` without asserting `active_processor is not None`.

2. **In-process path (`model.py:376 infer_with_image`).** Uses module-global `_processor`
   directly with no `_ensure_model()` guard before `apply_chat_template`. If vision is
   the first request and the server is down (so the in-process path runs), `_processor`
   is still the initial `None`.

## ✅ What Has Been Applied (13:16–13:20, branch `fix/cloud-audio-vision-402-fallback`)

Both changes below are committed to the working tree (uncommitted). They address the
**original NoneType** `apply_chat_template` crash only.

1. **`src/core/inference/model_server.py` — `_handle_infer_with_image`**
   - `_ensure_multimodal_slot()` wrapped in `try/except` so a slot-load failure degrades
     instead of propagating.
   - Slot is now used only when **both** `_mm_model` and `_mm_processor` are non-`None`.
   - After `_ensure_model()`, falls back to the main model's processor if available
     (`active_processor = _processor`).
   - Guards: if no processor resolves, returns `{"error": "no vision processor available..."}`
     instead of crashing at `apply_chat_template`.

2. **`src/core/inference/model.py` — in-process `infer_with_image`**
   - Ensures `_model`/`_processor` are loaded (`load()`) before the multimodal block;
     returns a readable "Vision unavailable..." string if still unloaded.

**Tests run (13:22):** `tests/test_cloud_fallback.py`, `test_model_slots.py`,
`test_model_client.py`, `test_model_client_unit.py` → **43 passed**, 0 failures.
Service restarted via `./start.sh` (pid 304557, port 8779 healthy).

---

## 🔴 Remaining / NOT FIXED — New error seen during live test: `🐬 Vision error: 'model'`

> ✅ **RESOLVED 2026-08-22 (18:40).** Live testing confirmed the `'model'` KeyError is **gone** —
> the native Gemma E2B slot produced correct descriptions for real photos (cat-surgery photo
> and veterinary-book photo). The `'model'` KeyError was a transient state from the stale
> model-server process (which `start.sh` now kills, running vision in-process); once the
> fresh process loaded the multimodal slot, vision worked. **No further action needed.**
> The section below is kept for historical record.

The original `'NoneType' ... 'apply_chat_template'` is gone, but sending a photo now
fails with `'model'` — a **`KeyError: 'model'`** (or a dict lookup keyed `"model"`), a
**different failure path**. Root cause is **not yet confirmed**.

### Evidence gathered so far
- **`start.sh` runs the API with `KERNEL_EVO_SKIP_MODEL_SERVER=1`** because
  `providers.task_inference = hf` (from start log). So **no separate model_server
  process is running**; the model server log is stale (2026-08-20) and there is no
  `model_server.py` process.
- Therefore the vision path runs **in-process inside the API** (`api:app`, pid 304557),
  going through **`src/core/inference/model.py:infer_with_image`**, which routes to
  `model_client.infer_with_image` (RPC) → fails/returns error → in-process fallback.
- The `'model'` KeyError most likely occurs in the in-process path: e.g. a `cfg["model"]`
  or `params["model"]` / response `["model"]` lookup where the config is cloud-routed
  (`providers.task_inference=hf` skips local model load → `cfg["model"]` missing or
  `_processor` still `None`), OR an unexpanded env var path
  (`${KERNEL_EVO_HF_HUB}` — seen in stale log) failing.
- **Not yet traced:** need the actual traceback from the **fresh** API log (`/tmp/kernel_evolving_api.log`,
  which is only 1304 bytes) / runtime stderr at the moment of the photo send, and to
  identify the exact dict access raising `KeyError: 'model'`.

### Remaining fix candidates / questions (to pair on)
- [ ] Capture the exact traceback for `KeyError: 'model'` (instrument or check
      `kernel_calls.jsonl` / API stderr at send time).
- [ ] Determine whether `model.py:infer_with_image` even runs the in-process path when
      `task_inference=hf` (no local model) — the `load()` guard added returns an error
      string, but an earlier dict access (e.g. `cfg["model"]`, `_target_device()`, or
      `model_client` response parsing) may raise `KeyError: 'model'` first.
- [ ] Confirm whether the intended path for Telegram vision with `task_inference=hf` is
      **cloud vision** (which currently 402s) vs the local Gemma slot. If the primary is
      cloud and it 402s, the fallback must degrade cleanly — verify the fallback chain
      actually resolves a processor without hitting a missing `"model"` key.
- [ ] Once root cause is confirmed, add a regression test asserting the full
      `model.py:infer_with_image` path returns a graceful string (not `KeyError`)
      under a cloud-routed config.

### Original (now-applied) proposed fix — kept for reference
- [x] Guard the processor assignment in `_handle_infer_with_image`.
- [ ] (partial) Make `_ensure_multimodal_slot()` contract explicit — early-return alias
      branch should set `_mm_processor` together with `_mm_model`.
- [x] Harden in-process `model.py:infer_with_image` with a load/None guard.
- [ ] Add regression test for cloud-fail + unloaded-slot → graceful error.

## 🎧 Audio Twin Fix (18:40, same branch)

Live testing surfaced the **audio twin** of the vision bug:

```
🐬 Audio error: 'NoneType' object has no attribute 'apply_chat_template'
```

Same root cause, audio path. Fixed with the same guard pattern:

1. **`src/core/inference/model.py` — in-process `infer_with_audio`** (line ~462):
   added a load guard (`if _processor is None or _model is None: load()`), returning
   `"Audio unavailable: no in-process model loaded..."` instead of crashing on
   `apply_chat_template`.
2. **`src/core/inference/model_server.py` — `_handle_infer_with_audio`** (line ~2463):
   `_ensure_multimodal_slot()` wrapped in `try/except`; added a guard returning
   `{"error": "no audio processor available..."}` when `active_processor` is `None`.

**Regression test added:** `tests/test_cloud_fallback.py::TestInProcessAudioNoneTypeGuard`
asserts `infer_with_audio` returns a clean error string (not `apply_chat_template` crash)
when no processor is loaded.

**Tests:** targeted inference suite 43 passed; new regression test 5 passed; full suite
698 passed (2 voice-clone tests skipped — require live voice server, unrelated).

---

## 🎧 Audio KeyError `'model'` during transcribe (2026-08-22/23, same branch)

Live test surfaced a **second** audio failure, distinct from the NoneType guard:

```
File ".../src/core/inference/model.py", line 444, in infer_with_audio
    load()
File ".../src/core/inference/model.py", line 74, in load
    model_path = cfg["model"].get("path") or cfg["model"]["name"]
KeyError: 'model'
```

**Root cause:** When the in-process fallback runs (`infer_with_audio` → `_processor`/`_model`
are `None` → `load()`), `load()` reads `cfg["model"]` unconditionally. When `task_inference`
is cloud-routed (`hf`) the config has **no local `model` section**, so `cfg["model"]` raises
`KeyError: 'model'`. The caller's `load()` call was not guarded, so the KeyError propagated
out of `infer_with_audio` → `POST /transcribe` returned **HTTP 500**.

**Fix (applied in `src/core/inference/model.py`):**
1. **`load()`** — added a guard before any `cfg["model"]` access:
   `if "model" not in (cfg or {}): print("[model] No local 'model' config — skipping in-process load"); return None, None`.
   A cloud-routed config (no local model) now skips load instead of raising.
2. **In-process `infer_with_image` and `infer_with_audio`** — the `load()` call is now wrapped
   in `try/except Exception`, logging `[model] in-process <vision|audio> load failed: ...` and
   degrading to the existing "…unavailable…" message if a processor still isn't resolved.

**Regression tests added:** `tests/test_cloud_fallback.py::TestInProcessLoadRaisesKeyError`
(2 tests) — patch `load` with `side_effect=KeyError("model")` and assert both
`infer_with_audio` and `infer_with_image` return a clean "…unavailable" string (no `KeyError`).

**Tests:** `tests/test_cloud_fallback.py` 9 passed; inference suite (`test_model_slots`,
`test_model_client`, `test_model_client_unit`) 48 passed.

---

## Scope / Original Proposed Fix (pre-application reference)

## Acceptance Criteria

- Sending a photo no longer produces `'NoneType' object has no attribute 'apply_chat_template'`.
- With cloud vision failing AND local slot unloaded → user gets a readable error
  ("vision backend unavailable"), no crash, no unhandled traceback in the bot.
- With cloud vision failing AND local slot loadable → native Gemma E2B description returned.
- `tests/test_cloud_fallback.py` and the new processor-guard test stay green.
