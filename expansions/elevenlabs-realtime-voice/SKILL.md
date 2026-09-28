---
name: elevenlabs-realtime-voice
description: Bidirectional realtime voice via ElevenLabs (Conversational Agents + streaming TTS). Full-duplex PCM audio channel beside the native Olly voice server on :8766. Serves on :8767.
commands:
  - /eleven-voice
intents:
  - realtime voice
  - eleven labs voice
  - voice conversation
  - talk to me in real time
  - real-time voice api
exec: python3 -m elevenlabs_rt.server {args}
---

# ElevenLabs Realtime Voice — Expansion Skill

## Status
EXPANSION — lives in the kernel-evolving repo under
`expansions/elevenlabs-realtime-voice/`.
Scaffolded in the kernel-evolving workspace, then promoted to a tracked
expansion in the kernel-evolving repository. Adds a bidirectional realtime
voice channel (ADR-022-style context provider seam).
A copy is also installed as a first-party skill in the private ecosystem
(`~/.kernel/ecosystem/private/skills/elevenlabs-realtime-voice/`) for autoload;
the kernel-evolving repo is the canonical source.

## What it does
Two realtime surfaces, mirroring the proven `voice-clone` + `olly-voice-server` pattern:

1. **Realtime / Conversational Agents** (`/rt/*`) — full duplex: stream 16-bit
   PCM mono 44.1kHz mic audio to an ElevenLabs agent, get assistant PCM audio +
   transcripts back. Requires an `agent_id` provisioned on the platform.
2. **Streaming TTS** (`/tts`, `/tts-synth`) — low-latency text→audio on
   `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input`.

## Configuration (env — never commit keys)
| Variable | Purpose |
|---|---|
| `ELEVENLABS_API_KEY` | API key (required) |
| `ELEVENLABS_AGENT_ID` | Provisioned agent for the realtime channel |
| `ELEVENLABS_DEFAULT_VOICE_ID` | Default voice for TTS stream path |
| `ELEVENLABS_RT_PORT` | Server port (default 8767) |

Copy `.env.example` → `.env` and fill in secrets. Alternatively export the vars
in the agent environment.

## Run
```bash
pip install -r requirements.txt
uvicorn elevenlabs_rt.app:app --host 0.0.0.0 --port 8767
```

> `exec` runs `python3 -m elevenlabs_rt.server {args}` — ensure the `src/`
> directory is on `PYTHONPATH` (or `pip install -e .` from the skill root).

## Endpoints
- `GET  /health`
- `POST /tts       {text}`            → PCM audio bytes
- `POST /tts-synth {text}`            → one-shot HTTP fallback
- `POST /rt/start  {agent_id?}`       → `{session_id, conversation_id}`
- `POST /rt/audio  {session_id, pcm_b64}`
- `POST /rt/text   {session_id, text}`
- `POST /rt/end    {session_id}`

## Files
- `src/elevenlabs_rt/realtime.py`       — Agents WebSocket client (full duplex)
- `src/elevenlabs_rt/tts_stream.py`     — streaming TTS client
- `src/elevenlabs_rt/app.py`            — FastAPI server (:8767)
- `src/elevenlabs_rt/context_provider.py` — ADR-022 additive integration seam
- `DESIGN.md`                            — architecture + protocol notes