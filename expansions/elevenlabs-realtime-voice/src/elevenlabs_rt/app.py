"""
elevenlabs_rt/app.py — FastAPI server exposing the ElevenLabs realtime module.

Mirrors `olly-voice-server` (FastAPI on :8766) but on a dedicated port :8767 so
it does NOT clash with the existing Olly voice server. Exposes:
  GET  /health
  POST /tts            {text} -> stream back audio (streaming TTS)
  POST /tts-synth      {text} -> one-shot HTTP fallback
  POST /rt/start       {agent_id?} -> opens a realtime session, returns session_id
  POST /rt/audio       {session_id, pcm_b64} -> forward mic PCM to the agent
  POST /rt/text        {session_id, text} -> forward text to the agent
  POST /rt/end         {session_id} -> close conversation

Lazy module load; no model weights held in memory. Purely additive.
"""
from __future__ import annotations

import base64
import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from . import realtime
from .tts_stream import TTSStreamClient

app = FastAPI(title="elevenlabs-realtime-voice", version="0.1.0")

# Active realtime sessions keyed by an internal session_id -> RealtimeSession.
_SESSIONS: Dict[str, realtime.RealtimeSession] = {}


def _api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY not configured")
    return key


class TTSBody(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model_id: Optional[str] = None


class RTStart(BaseModel):
    agent_id: Optional[str] = None
    voice_id: Optional[str] = None


class RTAudio(BaseModel):
    session_id: str
    pcm_b64: str


class RTText(BaseModel):
    session_id: str
    text: str


class RTEnd(BaseModel):
    session_id: str


@app.get("/health")
async def health():
    return {
        "status": "up",
        "realtime_configured": bool(os.environ.get("ELEVENLABS_API_KEY")),
        "agent_id": bool(os.environ.get("ELEVENLABS_AGENT_ID")),
        "active_sessions": len(_SESSIONS),
    }


@app.post("/tts")
async def tts(body: TTSBody):
    """Streaming TTS: returns the full synthesized audio bytes (PCM)."""
    client = TTSStreamClient(
        api_key=_api_key(),
        voice_id=body.voice_id or os.environ.get("ELEVENLABS_DEFAULT_VOICE_ID"),
        model_id=body.model_id,
    )
    data = await client.stream(body.text)
    if not data:
        raise HTTPException(status_code=502, detail="no audio produced")
    return _pcm_response(data)


@app.post("/tts-synth")
async def tts_synth(body: TTSBody):
    """One-shot HTTP synthesis fallback."""
    client = TTSStreamClient(
        api_key=_api_key(),
        voice_id=body.voice_id or os.environ.get("ELEVENLABS_DEFAULT_VOICE_ID"),
        model_id=body.model_id,
    )
    data = await client.synth_http(body.text)
    return _pcm_response(data)


@app.post("/rt/start")
async def rt_start(body: RTStart):
    """Open a realtime session with an ElevenLabs agent.

    Creates the client WITH on_audio/on_text callbacks so assistant PCM and
    transcripts accumulate into per-session buffers retrievable via
    GET /rt/receive.
    """
    sid = uuid.uuid4().hex

    async def _on_audio(pcm: bytes) -> None:
        session = _SESSIONS.get(sid)
        if session is not None:
            session.audio_buffer += pcm

    async def _on_text(text: str) -> None:
        session = _SESSIONS.get(sid)
        if session is not None:
            session.transcripts.append(text)

    try:
        client = realtime.RealtimeClient(
            api_key=_api_key(),
            agent_id=body.agent_id or os.environ.get("ELEVENLABS_AGENT_ID"),
            voice_id=body.voice_id or os.environ.get("ELEVENLABS_DEFAULT_VOICE_ID"),
            on_audio=_on_audio,
            on_text=_on_text,
        )
        session = await client.connect()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"realtime connect failed: {exc}")

    _SESSIONS[sid] = session
    return {"session_id": sid, "conversation_id": session.conversation_id}


@app.get("/rt/receive")
async def rt_receive(session_id: str):
    """Retrieve accumulated assistant audio + transcripts for a session.

    Returns `{audio_b64, transcripts, done}`. The audio buffer is cleared after
    each call so polling is incremental (each poll returns only NEW audio since
    the previous poll). `done` is true when the session's websocket is closed or
    an end was received.
    """
    session = _get(session_id)
    audio = session.audio_buffer
    session.audio_buffer = b""  # clear so the next poll is incremental
    return {
        "audio_b64": base64.b64encode(audio).decode("ascii") if audio else "",
        "transcripts": list(session.transcripts),
        "done": session.closed,
    }


@app.post("/rt/audio")
async def rt_audio(body: RTAudio):
    session = _get(body.session_id)
    pcm = base64.b64decode(body.pcm_b64)
    await session.send_audio(pcm)
    return {"status": "sent", "bytes": len(pcm)}


@app.post("/rt/text")
async def rt_text(body: RTText):
    session = _get(body.session_id)
    await session.send_text(body.text)
    return {"status": "sent"}


@app.post("/rt/end")
async def rt_end(body: RTEnd):
    session = _SESSIONS.pop(body.session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    await session.end()
    return {"status": "closed"}


def _get(sid: str) -> realtime.RealtimeSession:
    session = _SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def _pcm_response(data: bytes) -> Response:
    return Response(
        content=data,
        media_type="audio/pcm",
        headers={"Content-Type": "audio/pcm", "Content-Length": str(len(data))},
    )