"""
elevenlabs_rt/app.py — FastAPI server exposing the ElevenLabs realtime module.

Mirrors `olly-voice-server` (FastAPI on :8766) but on a dedicated port :8767 so
it does NOT clash with the existing Olly voice server. Exposes:
  GET  /health
  POST /tts            {text} -> stream back audio (streaming TTS)
  POST /tts-synth      {text} -> one-shot HTTP fallback
  POST /stt            (multipart audio file) -> {text} speech-to-text
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

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from . import realtime
from .tts_stream import TTSStreamClient

app = FastAPI(title="elevenlabs-realtime-voice", version="0.1.0")

# The desktop app is served from a different origin (e.g. localhost:8000) and
# calls this server cross-origin from the browser. Allow all origins so the
# browser doesn't block the fetch (mirrors kernel-evolving's API).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
STT_MODEL = "scribe_v1"  # current default ElevenLabs speech-to-text model


async def _elevenlabs_stt(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio via the ElevenLabs Speech-to-Text API.

    Sends a multipart request with the audio file + model_id and returns the
    transcribed text. Raises HTTPException on upstream failure.
    """
    key = _api_key()
    data = aiohttp.FormData()
    data.add_field("file", audio_bytes, filename=filename, content_type="application/octet-stream")
    data.add_field("model_id", STT_MODEL)
    headers = {"xi-api-key": key}
    async with aiohttp.ClientSession() as http:
        async with http.post(STT_URL, data=data, headers=headers) as resp:
            if resp.status != 200:
                detail = await resp.text()
                raise HTTPException(status_code=502, detail=f"ElevenLabs STT failed ({resp.status}): {detail[:300]}")
            payload = await resp.json()
    text = (payload or {}).get("text") or ""
    return text.strip()


@app.post("/stt")
async def stt(request: Request):
    """Speech-to-text: audio -> {text}.

    Accepts either a multipart file upload (`file`) or a JSON body with
    `{audio_b64, format}`. Returns `{"text": "..."}`. Both paths are handled
    from the raw request so a single endpoint serves the browser and simple
    curl/JSON clients.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="multipart request missing 'file' field")
        audio_bytes = await upload.read()
        filename = getattr(upload, "filename", None) or "audio.wav"
    elif "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid JSON body")
        audio_b64 = (payload or {}).get("audio_b64")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="JSON body must include 'audio_b64'")
        audio_bytes = base64.b64decode(audio_b64)
        filename = f"audio.{(payload.get('format') or 'wav')}"
    else:
        raise HTTPException(status_code=400, detail="provide multipart 'file' or JSON {audio_b64}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio payload")

    text = await _elevenlabs_stt(audio_bytes, filename)
    return {"text": text}


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