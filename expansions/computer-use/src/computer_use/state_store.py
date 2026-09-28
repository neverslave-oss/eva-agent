from __future__ import annotations

import json
from pathlib import Path


class StateStore:
    """Per-chat state store with optional per-run checkpoints.

    Shape:
    {
      "<chat_id>": {
        "last_run_id": "run-1",
        "cursor": {"step": 2, "status": "ok"},
        "runs": {
          "run-1": {"step": 2, "status": "ok"}
        }
      }
    }
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._state = raw if isinstance(raw, dict) else {}
            except Exception:
                self._state = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _chat(self, chat_id: str) -> dict:
        return self._state.setdefault(chat_id, {"last_run_id": None, "cursor": {}, "runs": {}})

    def get_chat(self, chat_id: str) -> dict:
        return dict(self._chat(chat_id))

    def update_chat(self, chat_id: str, patch: dict) -> dict:
        node = self._chat(chat_id)
        node.update(patch)
        self.save()
        return dict(node)

    def get_run(self, chat_id: str, run_id: str) -> dict:
        runs = self._chat(chat_id).get("runs", {})
        return dict(runs.get(run_id, {}))

    def update_run(self, chat_id: str, run_id: str, patch: dict) -> dict:
        node = self._chat(chat_id)
        runs = node.setdefault("runs", {})
        cur = dict(runs.get(run_id, {}))
        cur.update(patch)
        runs[run_id] = cur
        node["last_run_id"] = run_id
        node["cursor"] = dict(cur)
        self.save()
        return dict(cur)
