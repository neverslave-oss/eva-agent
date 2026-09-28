"""elevenlabs_rt — ElevenLabs realtime voice expansion module."""

__version__ = "0.1.0"

from .context_provider import realtime_voice_section, active_section
from .realtime import RealtimeClient, RealtimeSession
from .tts_stream import TTSStreamClient

__all__ = [
    "realtime_voice_section",
    "active_section",
    "RealtimeClient",
    "RealtimeSession",
    "TTSStreamClient",
]