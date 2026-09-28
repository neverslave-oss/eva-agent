from __future__ import annotations

from datetime import datetime, timezone


def capture_screen() -> dict:
    """Return a normalized screen-capture metadata envelope.

    Real pixel capture is handled by the selected desktop driver.
    """
    return {
        "kind": "screen",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "available",
    }
