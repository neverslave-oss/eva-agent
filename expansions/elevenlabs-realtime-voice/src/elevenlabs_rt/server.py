"""
elevenlabs_rt/server.py — small CLI runner for the ElevenLabs realtime voice server.

Usage:
  python -m elevenlabs_rt.server [--host 0.0.0.0] [--port 8768] [--reload]
"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Run the ElevenLabs realtime voice server")
    p.add_argument("--host", default=os.environ.get("ELEVENLABS_RT_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ELEVENLABS_RT_PORT", "8768")))
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    uvicorn.run(
        "elevenlabs_rt.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()