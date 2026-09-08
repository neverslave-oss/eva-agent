"""
elevenlabs_rt.context_provider — integration seam into the live kernel.

Mirrors `expertise-field/context_provider.py` (ADR-022): a purely ADDITIVE
module that exposes live realtime-voice context to the kernel's system prompt.
It NEVER intercepts or suppresses tool calls; it only FEEDS context.

It reports whether the ElevenLabs realtime channel is configured so the kernel
knows to route audio work to /rt/* and /tts* (alongside the Olly voice server
on :8766/:8767). If the channel is not configured, it adds nothing.
"""
from __future__ import annotations

import os
from typing import Optional

MODULE_NAME = "elevenlabs-realtime-voice"
SERVER_PORT = int(os.environ.get("ELEVENLABS_RT_PORT", "8768"))


def _configured() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def realtime_voice_section() -> str:
    """Build a markdown 'Realtime voice channel' section for the system prompt.

    Returns an empty string when not configured — a non-configured kernel is
    unchanged (same additive invariant as expertise-field).
    """
    if not _configured():
        return ""

    lines = [
        "## Realtime voice channel (ElevenLabs)",
        "A bidirectional realtime voice module is active beside the Olly voice "
        "server. Endpoints (server :{0}):".format(SERVER_PORT),
    ]
    base = "http://127.0.0.1:{0}".format(SERVER_PORT)
    lines += [
        f"- Streaming TTS      POST {base}/tts           {{text}}",
        f"- One-shot TTS       POST {base}/tts-synth     {{text}}",
        f"- RT session start   POST {base}/rt/start",
        f"- RT mic audio       POST {base}/rt/audio      {{session_id, pcm_b64}}",
        f"- RT text            POST {base}/rt/text       {{session_id, text}}",
        f"- RT end             POST {base}/rt/end        {{session_id}}",
        "Audio is 16-bit PCM mono 44.1kHz. Assistant audio + transcripts are "
        "delivered via session callbacks.",
    ]
    return "\n".join(lines)


# Alias matching the expertise-field seam naming convention.
active_section = realtime_voice_section