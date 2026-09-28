"""
elevenlabs_rt/realtime.py — ElevenLabs Realtime / Conversational Agents WebSocket client.

Full-duplex voice channel: streams 16-bit PCM mono (44.1 kHz) microphone audio
up to an ElevenLabs agent, and streams assistant PCM audio + transcripts back.

Designed to mirror the proven `olly-voice-server` + `voice-clone` pattern:
self-contained, additive, no interception of kernel tool calls.

Endpoint (ElevenLabs Agents / Conversational AI):
  wss://api.elevenlabs.io/v1/convai/...
Requires:
  - ELEVENLABS_API_KEY  (server env, never committed)
  - ELEVENLABS_AGENT_ID (provisioned agent on the ElevenLabs platform)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import aiohttp

# Default realtime endpoint template. The exact convai path is confirmed at
# build time against the live docs; we default to the realtime gateway.
REALTIME_WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation/{conversation_id}"

# Client audio format sent up to the agent.
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


# ----------------------------- callbacks ------------------------------
AudioCallback = Callable[[bytes], Awaitable[None]]          # assistant PCM
TextCallback = Callable[[str], Awaitable[None]]             # transcript
EventCallback = Callable[[dict], Awaitable[None]]           # raw event


@dataclass
class RealtimeSession:
    """Handle for an active conversation returned to the caller."""
    conversation_id: str
    agent_id: str = ""
    ws: Optional[aiohttp.ClientWebSocketResponse] = None
    client: Optional["RealtimeClient"] = None
    _closed: bool = False
    # Receive-path buffers, populated by the app's on_audio/on_text callbacks.
    audio_buffer: bytes = b""       # accumulated assistant PCM (16-bit mono)
    transcripts: list = field(default_factory=list)  # assistant/user transcript strings

    @property
    def closed(self) -> bool:
        return self._closed or (self.ws is not None and self.ws.closed)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self.client is None:
            raise RuntimeError("session has no client bound")
        await self.client.send_audio(self, pcm_bytes)

    async def send_text(self, text: str) -> None:
        if self.client is None:
            raise RuntimeError("session has no client bound")
        await self.client.send_text(self, text)

    async def end(self) -> None:
        if self.client is None:
            return
        await self.client.end_conversation(self)


class RealtimeClient:
    """Minimal, additive client. Does nothing until a method is called."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        *,
        on_audio: Optional[AudioCallback] = None,
        on_text: Optional[TextCallback] = None,
        on_event: Optional[EventCallback] = None,
        voice_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.agent_id = agent_id or os.environ.get("ELEVENLABS_AGENT_ID")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_DEFAULT_VOICE_ID")
        self.on_audio = on_audio
        self.on_text = on_text
        self.on_event = on_event
        self._http: Optional[aiohttp.ClientSession] = None
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

    # ------------------------------------------------------------------
    async def connect(self, conversation_id: Optional[str] = None) -> RealtimeSession:
        """Open the realtime websocket and set up a conversation."""
        cid = conversation_id
        url = REALTIME_WS_URL.format(conversation_id=cid or "{conversation_id}")
        if cid is None:
            # fresh conversation: server assigns the id
            url = url.replace("/{conversation_id}", "")

        session = RealtimeSession(conversation_id=cid or "", agent_id=self.agent_id, client=self)
        async with aiohttp.ClientSession() as http:
            self._http = http
            ws = await http.ws_connect(
                url,
                headers={"xi-api-key": self.api_key},
                heartbeat=15.0,
            )
            session.ws = ws

            init = {
                "type": "conversational_initiation",
                "agent_id": self.agent_id,
                "auth": {"api_key": self.api_key},
            }
            await ws.send_str(json.dumps(init))

            asyncio.ensure_future(self._reader(session))
            return session

    async def _reader(self, session: RealtimeSession) -> None:
        ws = session.ws
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_event(session, json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if self.on_audio:
                        await self.on_audio(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            session._closed = True

    async def _handle_event(self, session: RealtimeSession, ev: dict) -> None:
        etype = ev.get("type", "")
        if self.on_event:
            await self.on_event(ev)

        if etype == "conversation_initiation_metadata":
            cid = ev.get("conversation_id")
            if cid:
                session.conversation_id = cid
        elif etype == "ping":
            await session.ws.send_str(json.dumps({"type": "pong"}))
        elif etype == "pong":
            pass
        elif etype == "audio":
            b64 = ev.get("audio", "")
            if b64 and self.on_audio:
                await self.on_audio(base64.b64decode(b64))
        elif etype in ("transcript", "agent_response", "user_transcript", "final_transcript", "partial_transcript"):
            txt = (ev.get("text") or ev.get("transcript") or "").strip()
            if txt and self.on_text:
                await self.on_text(txt)

    # ------------------------------------------------------------------
    async def send_audio(self, session: RealtimeSession, pcm_bytes: bytes) -> None:
        """Send a chunk of 16-bit PCM mono audio up to the agent."""
        if session.closed:
            raise RuntimeError("session closed")
        payload = {
            "type": "audio",
            "audio_data": base64.b64encode(pcm_bytes).decode("ascii"),
            "audio_format": {
                "sample_rate": SAMPLE_RATE,
                "channels": CHANNELS,
                "sample_width": SAMPLE_WIDTH,
            },
        }
        await session.ws.send_str(json.dumps(payload))

    async def send_text(self, session: RealtimeSession, text: str) -> None:
        """Send user-side text to the agent channel."""
        if session.closed:
            raise RuntimeError("session closed")
        await session.ws.send_str(json.dumps({"type": "text", "text": text}))

    async def end_conversation(self, session: RealtimeSession) -> None:
        if session.closed:
            return
        try:
            await session.ws.send_str(json.dumps({"type": "end_of_conversation"}))
        finally:
            await session.ws.close()
            session._closed = True