"""Activity-file tracking + log-preview helpers for the model server.

Extracted from the original model_server.py monolith (refactor). The activity
markers maintain a small in-flight counter JSON file (Think-at-Rest); the
preview helper pretty-prints a payload for logs. The canonical activity path is
owned here and the no-arg `mark_activity_start`/`mark_activity_end` wrappers let
loader/inference modules call them without importing back into model_server.py.
"""

import json
import os
import time

from runtime_paths import MODEL_ACTIVITY_FILE as _MODEL_ACTIVITY_PATH_IMPORT

# Activity tracking for Think-at-Rest — mirrors model_client.py counters
# so thought_engine doesn't fire during image/audio inference.
ACTIVITY_PATH = str(_MODEL_ACTIVITY_PATH_IMPORT)


def _mark_start(activity_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(activity_path), exist_ok=True)
        try:
            with open(activity_path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["in_flight"] = int(data.get("in_flight", 0)) + 1
        data["last_start_ts"] = time.time()
        with open(activity_path, "w") as fw:
            json.dump(data, fw)
    except Exception:
        pass


def _mark_end(activity_path: str) -> None:
    try:
        try:
            with open(activity_path) as f:
                data = json.load(f)
        except Exception:
            data = {}
        data["in_flight"] = max(0, int(data.get("in_flight", 0)) - 1)
        data["last_end_ts"] = time.time()
        with open(activity_path, "w") as fw:
            json.dump(data, fw)
    except Exception:
        pass


def mark_activity_start() -> None:
    _mark_start(ACTIVITY_PATH)


def mark_activity_end() -> None:
    _mark_end(ACTIVITY_PATH)


def preview_payload_for_log(payload) -> str:
    """Return full, pretty-formatted JSON for readable server logs."""
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(payload)
    return text
