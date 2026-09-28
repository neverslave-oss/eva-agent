"""
elevenlabs_rt/tts_stream.py — low-latency ElevenLabs streaming TTS.

Text in → streamed audio out, over the confirmed WebSocket endpoint:
  wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input

Also exposes the standalone HTTP path (POST /v1/text-to-speech/{voice_id})
for one-shot synthesis, mirroring how `voice-clone` has both a direct and a
server path. Purely additive — no kernel interception.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Awaitable, Callable, Optional

import aiohttp

TTS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
TTS_HTTP_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

DEFAULT_MODEL = "eleven_multilingual_v2"   # multilingual, good for EN/IT mix
DEFAULT_OUTPUT = "pcm_44100"               # PCM for lowest-latency playback


AudioChunkCallback = Callable[[bytes, bool], Awaitable[None]]  # (pcm_chunk, is_final)


class TTSStreamClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        *,
        model_id: str = DEFAULT_MODEL,
        output_format: str = DEFAULT_OUTPUT,
        on_audio: Optional[AudioChunkCallback] = None,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.environ.get("ELEVENLABS_DEFAULT_VOICE_ID")
        self.model_id = model_id
        self.output_format = output_format
        self.on_audio = on_audio
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        if not self.voice_id:
            raise RuntimeError("ELEVENLABS_DEFAULT_VOICE_ID is not set")

    async def stream(self, text: str, voice_settings=None) -> bytes:
        """Synthesise `text` through the streaming websocket.

        Returns the full accumulated audio if on_audio is not set; otherwise
        chunks are delivered to on_audio and returns b"".
        """
        url = TTS_WS_URL.format(voice_id=self.voice_id)
        params = {
            "model_id": self.model_id,
            "output_format": self.output_format,
            "enable_logging": "false",
        }

        collected: list[bytes] = []
        async with aiohttp.ClientSession() as http:
            ws = await http.ws_connect(url, params=params, headers={"xi-api-key": self.api_key})

            init = {
                "text": " ",
                "voice_settings": voice_settings or {
                    "similarity_boost": 0.8,
                    "speed": 1.0,
                    "stability": 0.5,
                },
                "xi_api_key": self.api_key,
            }
            await ws.send_str(json.dumps(init))

            # Push text in small chunks (bounded buffering), then the final empty chunk.
            # Flush per sentence-ish buffer to keep latency low.
            buffer = ""
            for i in range(0, len(text), 200):
                buffer = text[i:i + 200]
                await ws.send_str(json.dumps({"text": buffer}))
                # allow the reader to run concurrently
                await asyncio.sleep(0)
            await ws.send_str(json.dumps({"text": ""}))

            # Consume audio events until is_final.
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    ev = json.loads(msg.data)
                    if "audio" in ev:
                        chunk = base64.b64decode(ev["audio"])
                        is_final = bool(ev.get("is_final", False))
                        if self.on_audio:
                            await self.on_audio(chunk, is_final)
                        else:
                            collected.append(chunk)
                        if is_final:
                            break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

            await ws.close()

        return b"".join(collected)

    async def synth_http(self, text: str) -> bytes:
        """One-shot HTTP synthesis (fallback / non-streaming path)."""
        url = TTS_HTTP_URL.format(voice_id=self.voice_id)
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        body = {
            "text": text,
            "model_id": self.model_id,
            "output_format": self.output_format,
        }
        async with aiohttp.ClientSession() as http:
            async with http.post(url, headers=headers, json=body) as resp:
                resp.raise_for_status()
                return await resp.read()