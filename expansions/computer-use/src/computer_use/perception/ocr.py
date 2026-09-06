from __future__ import annotations


def extract_text(image=None) -> str:
    """OCR adapter boundary.

    For v0.1 baseline this function normalizes null/str inputs and returns a
    deterministic string; driver-level OCR integrations can pass rendered text.
    """
    if image is None:
        return ""
    if isinstance(image, str):
        return image.strip()
    return ""
