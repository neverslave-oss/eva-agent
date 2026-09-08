# ElevenLabs Realtime Voice — Expansion Module (DESIGN)

Status: **ISOLATION BUILD** (workspace only). Port to codebase once verified.

Follows the same pattern proven with:
- `expansions/expertise-field/` → `context_provider.py` (ADR-022 integration seam)
- `voice-clone` skill → self-contained, actively used + integrated
- `olly-voice-server` skill → FastAPI server on :8766

## Goal
Give Kernel-Evo a **bidirectional realtime voice channel** via **ElevenLabs Realtime / Conversational (Agents) API**, alongside the existing native Olly voice-server integration (TTS + STT on :8766).

Two realtime surfaces, both in one module:
1. **ElevenLabs Agents / Conversational AI** — full duplex: client streams PCM mic audio → agent (ElevenLabs-hosted brain) → streams assistant PCM audio back. `agent_id`-driven.
2. **ElevenLabs streaming TTS** (`/v1/text-to-speech/{voice_id}/stream-input`) — low-latency text→audio stream, useful for injecting Kernel-Evo's own replies as voice with minimal latency.

## Interface into Kernel-Evo
Mirror the `expertise-field` seam: a `context_provider.py`-style module that is **purely additive** and exposes functions the kernel can call — it never intercepts or suppresses tool calls.

Exposed entry points (to be wired in during port):
- `start_conversation(agent_id, audio_cb, text_cb)` → opens WS, returns a handle
- `send_audio_chunk(handle, pcm_bytes)`
- `send_text(handle, text)` (for the TTS stream path)
- `on_assistant_audio(cb)` / `on_interruption(cb)` / `on_transcript(cb)`
- `close_conversation(handle)`

## Protocol facts (researched, Aug 24 2026)

### A. Streaming TTS WebSocket (confirmed)
- Endpoint: `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input`
- Query params: `model_id`, `language_code`, `output_format` (e.g. `pcm_44100`), `enable_logging`
- Handshake headers: `xi-api-key`
- Init message (subscribe-ready): `client_start_session` with `voice_settings`, `xi_api_key`
- Stream text in small publish chunks as `{"text": "..."}`
- Assistant audio returns in subscribe events: `{audio: <base64>, alignment: {...}, is_final: bool}`
- End with `{"text": ""}` final + `is_final`

### B. Realtime / Conversational Agents API (to confirm exact wire during build)
- WSS endpoints under `wss://api.elevenlabs.io/v1/convai/conversation/...` / `wss://api.elevenlabs.io/v1/convai/realtime`
- Client sends an init message (`client_start_session` / `conversational_initiation`) carrying `agent_id` (created on ElevenLabs dashboard) + auth
- Microphone audio streamed to server in `audio` events as **16-bit PCM, 44.1kHz mono**
- Assistant audio returned as base64 PCM; interjections/`is_final` flags mark utterance boundaries
- `end_of_conversation`, ping/pong keepalive, interruption events supported
- Requires: an `XI_API_KEY` **and** an `agent_id` provisioned on the ElevenLabs platform

## Environment / config
- `ELEVENLABS_API_KEY` — secret. Never committed. Loaded from `.env` / agent env.
- `ELEVENLABS_AGENT_ID` — provisioned agent for the realtime channel.
- `ELEVENLABS_DEFAULT_VOICE_ID` — default voice for the TTS stream path (fallback clones).

## Security notes
- Key is held server-side (WSL). The WS credential is sent once at handshake, not persisted.
- No reference audio or keys committed. Uses existing `.env` convention.
- Real-time PCM is in-memory only; nothing written to disk by default.

## Isolation → Port plan
1. Build + test in `expansions/elevenlabs-realtime-voice/` (workspace only). ✅ (in progress)
2. Verify the two realtime paths against the real API with the user's key.
3. Promote to the skill ecosystem as `elevenlabs-realtime-voice` skill (mirrors `voice-clone`).
4. Wire entry points into the kernel's audio path (alongside `olly-voice-server`), serving on a dedicated port (e.g. :8767) to avoid clashing with the Olly voice server's :8766.