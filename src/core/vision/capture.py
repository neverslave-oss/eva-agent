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


def grab_frame(stream_url: str, timeout_s: float = 5.0) -> Optional[bytes]:
    """Return the latest JPEG frame from an MJPEG stream, or None on failure.

    MJPEG is a sequence of JPEG frames separated by a `--boundary` delimiter.
    We read the stream until we have at least one complete frame, then return
    the last complete JPEG we saw.
    """
    if not stream_url:
        return None
    try:
        import requests  # type: ignore

        r = requests.get(stream_url, stream=True, timeout=timeout_s)
        if r.status_code != 200:
            return None

        # Read bytes until we have a complete JPEG (ends with FFD9).
        # Cap the read to avoid hanging on a long stream.
        buf = bytearray()
        last_frame: Optional[bytes] = None
        for chunk in r.iter_content(chunk_size=4096):
            buf.extend(chunk)
            # A JPEG frame ends with the SOI/EOI markers; find the last EOI.
            while True:
                start = buf.find(b"\xff\xd8")  # SOI
                end = buf.find(b"\xff\xd9", start + 2)  # EOI after SOI
                if start == -1 or end == -1:
                    break
                last_frame = bytes(buf[start:end + 2])
                del buf[:end + 2]
            if last_frame is not None and len(buf) > 2_000_000:
                break
        return last_frame
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("[vision] capture failed from %s: %s", stream_url, exc)
        return None
