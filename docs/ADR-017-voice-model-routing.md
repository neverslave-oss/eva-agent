# ADR-017 — Adaptive Model Switching for Voice Requests

**Date:** 2026-05-26  
**Status:** PROPOSED  
**Author:** Olly (on behalf of Fabio)

---

## Context

kernel-evolving currently runs **Nemotron-Labs-Diffusion-3B** as the primary inference model (task_inference). This is the right default: it is fast, agentic, and handles multi-turn conversation well.

Voice requests require:
1. **STT** — transcribing the incoming voice note (audio → text)
2. **Inference** — agentic reply (text → text), same as normal chat
3. **TTS clone** — synthesising a reply as the user's cloned voice (text → WAV)

The STT step is the problem. Nemotron-Labs-Diffusion-3B is **text-only** — it cannot process audio. The existing workaround (`stt_model` config in `model_server.py`) lazy-loads Gemma 4 as a secondary STT model. This works but has costs:

- **VRAM spike** — Gemma 4 E2B-it at 4-bit needs ~4–5 GB additional VRAM. Loading it alongside Nemotron while handling a voice note may OOM on consumer hardware.
- **Latency** — loading Gemma 4 cold adds 8–15s to first voice response.
- **Idle waste** — Gemma 4 stays resident after STT, wasting VRAM until GC.

---

## Decision

**Option A (recommended): STT via faster-whisper, Nemotron unchanged**

Use `faster-whisper` (already installed in `olly-voice-server`) for STT instead of Gemma 4. The voice pipeline becomes:

```
Voice note → olly-voice-server /stt → transcript text
           → agent.triage(transcript) → Nemotron reply
           → olly-voice-server /tts/clone → WAV reply
```

- No model swap. No VRAM spike.
- faster-whisper `base.en` uses ~200MB RAM only, runs on CPU or CUDA.
- Already wired: `olly-voice-server` has `/stt` and `/tts/clone` endpoints.
- `telegram_bot.py` calls `model.infer_with_audio` for STT. Route that to `olly-voice-server /stt` instead of loading Gemma 4 in `model_server`.

**Implementation (2 files):**

1. `src/telegram_bot.py`: Replace the `from model import infer_with_audio` STT call with a direct HTTP call to `http://127.0.0.1:8766/stt` (or keep the model module call but route it there).
2. `src/model_server.py` or new `src/stt_client.py`: Add `stt_via_voice_server(path) → str` that POSTs to olly-voice-server `/stt`. Fall back to Gemma 4 if voice server unreachable.

**Risk:** olly-voice-server offline → STT fails. Mitigated by the Gemma 4 fallback already in `model_server.py`.

---

**Option B (deferred): Gemma 4 ↔ Nemotron hot-swap on voice request**

Unload Nemotron, load Gemma 4 for the full voice turn (STT + optional multimodal reply), then swap back. 

- Correct approach when you want Gemma 4's native multimodal replies (audio grounding, image understanding).
- **Not recommended now**: swap latency ~20–30s round-trip on consumer VRAM. Acceptable for occasional voice notes; unacceptable for rapid back-and-forth.
- Becomes viable when: Nemotron swap is complete AND we have a pooled VRAM manager.

**Marker:** `# TODO(gemma-swap)` in `telegram_bot.py` voice handler.

---

## Proposed implementation steps (Option A)

| # | File | Change |
|---|------|--------|
| 1 | `src/stt_client.py` (new) | `transcribe(path) → str` — POST to `http://127.0.0.1:8766/stt`, returns text. Falls back to `model.infer_with_audio` if voice server unreachable. |
| 2 | `src/telegram_bot.py` | Replace `from model import infer_with_audio` + call with `from stt_client import transcribe`. |
| 3 | `olly-voice-server` | Verify `/stt` endpoint accepts multipart `audio` field + returns `{"text": "..."}`. Patch if needed. |
| 4 | `tests/test_e2e_flow.py` | Add `TestVoicePipeline::test_stt_routes_to_voice_server` — mock `/stt` endpoint, assert transcript returned without loading Gemma 4. |

**Acceptance criteria:**
- Voice note → transcript → triage → WAV reply without VRAM spike
- If olly-voice-server is down: falls back gracefully to Gemma 4 STT
- All existing voice pipeline tests pass
- No change to Nemotron as default text model

---

## Current state (2026-05-26)

The voice pipeline is fully wired and tested end-to-end today (v1.19.14):
- STT: `model.infer_with_audio` → `model_server._handle_infer_with_audio` → lazy-loads Gemma 4
- Inference: `agent.triage` → Nemotron via `infer_with_tools`
- TTS clone: `_clone_voice_reply` → `olly-voice-server /tts/clone` ✅ confirmed working
- Tests: `test_voice_clone_endpoint_reachable` PASSED, `test_voice_pipeline_wired_in_bot` PASSED

Option A (stt_client.py) is the next step — low risk, no architecture change to inference.  
Option B is a future milestone, not blocking.

---

## Consequences

- **Option A implemented:** voice works reliably, no VRAM pressure, Gemma 4 only loads for tasks it's actually needed for.
- **Option B deferred:** multimodal voice replies (Gemma 4 reasoning on the audio content) not available until VRAM manager is built.
- **No Nemotron changes** in either option.
