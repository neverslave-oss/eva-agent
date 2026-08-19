"""
capture.py — Grab a single frame from an eye's MJPEG stream.

Used by the `look` tool when a raw image is needed (e.g. the Phase 2 "describe"
path). Reads the MJPEG stream and returns the latest complete JPEG frame.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def grab_frame(stream_url: str, timeout_s: float = 5.0, retries: int = 3) -> Optional[bytes]:
    """Return the latest JPEG frame from an MJPEG stream, or None on failure.

    MJPEG is a sequence of JPEG frames separated by a `--boundary` delimiter.
    We read the stream until we have at least one complete frame, then return
    the last complete JPEG we saw.

    The Pi's V4L2 webcam only allows ONE exclusive open at a time, so a frame
    grab can intermittently fail ("Could not open webcam") when another viewer
    (browser on /video_feed, or the hourly cron) holds the camera. We retry a
    few times with a short backoff so a momentary collision doesn't fail the
    whole describe/look call.
    """
    if not stream_url:
        return None
    import time as _time
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            import requests  # type: ignore

            r = requests.get(stream_url, stream=True, timeout=timeout_s)
            if r.status_code != 200:
                last_exc = RuntimeError(f"stream returned HTTP {r.status_code}")
                _time.sleep(0.5)
                continue

            # Read bytes until we have at least one complete JPEG frame, then
            # return it immediately. The MJPEG stream is infinite — if we kept
            # reading until the buffer exceeded a size cap we'd hang forever.
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=4096):
                buf.extend(chunk)
                # A JPEG frame ends with the SOI/EOI markers; find the last EOI.
                start = buf.find(b"\xff\xd8")  # SOI
                end = buf.find(b"\xff\xd9", start + 2)  # EOI after SOI
                if start != -1 and end != -1:
                    return bytes(buf[start:end + 2])
                # Safety cap: if we got a huge chunk with no complete frame,
                # this attempt failed — retry.
                if len(buf) > 2_000_000:
                    break
            # No complete frame this attempt — retry.
            last_exc = RuntimeError("no complete JPEG frame from stream")
            _time.sleep(0.5)
        except Exception as exc:  # network errors
            last_exc = exc
            _time.sleep(0.5)
    if last_exc is not None:
        logger.warning("[vision] capture failed from %s after %d attempt(s): %s",
                       stream_url, retries, last_exc)
    return None
