"""
registry.py — Load expertise_field definitions and manage the registry + per-chat
hot-field state.

Mirrors the kernel's src/core/skills.py patterns (load_all / _parse_skill /
dedup logic) so the module behaves consistently when merged into the kernel's
agent pipeline. Works standalone here for isolated testing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml optional for .yaml registry
    yaml = None


# ---------------------------------------------------------------------------
# Field definition parsing (mirrors skills._parse_skill)
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> dict | None:
    if yaml is None:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_field_file(path: Path) -> dict | None:
    """Load a single field definition from .json or .yaml/.yml."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = _read_json(path)
    elif suffix in (".yaml", ".yml"):
        data = _read_yaml(path)
    else:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("_src", str(path))
    return data


def _rglob(base: Path, exts: tuple[str, ...]) -> list[Path]:
    """Follow-symlink rglob for field definition files."""
    hits: list[Path] = []
    if not base.exists():
        return hits
    for root, _dirs, files in os.walk(str(base), followlinks=True):
        for fn in files:
            if fn.lower().endswith(exts):
                hits.append(Path(root) / fn)
    return sorted(hits)


REQUIRED_KEYS = ("id", "name", "version")


class Registry:
    """Holds loaded field definitions + the active-field index.

    Mirrors kernel skills.py load_all(): dedups by field id with a priority
    policy so a repo-local private copy can shadow a workspace copy.
    """

    def __init__(self, fields: list[dict], active: list[str], src_dir: str):
        self.fields = fields
        self.active = list(active)
        self.src_dir = src_dir
        self._by_id = {f["id"]: f for f in fields}

    # -- lookups -----------------------------------------------------------
    def field(self, fid: str) -> dict | None:
        return self._by_id.get(fid)

    def all_fields(self) -> list[dict]:
        return list(self.fields)

    def is_active(self, fid: str) -> bool:
        return fid in self.active

    def active_fields(self) -> list[dict]:
        return [f for f in self.fields if f["id"] in self.active]

    def skill_names_for(self, fid: str) -> list[str]:
        f = self._by_id.get(fid)
        if not f:
            return []
        return [s for s in f.get("skills", []) if isinstance(s, str)]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "active": list(self.active),
            "fields": [
                {"id": f["id"], "name": f["name"], "version": f["version"],
                 "skills": f.get("skills", []),
                 "learn_on_idle": f.get("acquisition_plan", {}).get("learn_on_idle", False)}
                for f in self.fields
            ],
        }


def _priority(path: Path, base_dir: Path) -> int:
    """Lower wins. private/ fields shadow workspace/repo top-level copies.

    Mirrors kernel skills._skill_priority(): any path with a `private/` segment
    wins over everything else. Plain `fields/` copies come second.
    """
    try:
        rel = path.relative_to(base_dir).as_posix()
    except ValueError:
        return 1
    segments = rel.split("/")
    if "private" in segments:
        return 0
    if rel.startswith("fields/"):
        return 1
    return 1


def load(base_dir: str | Path, registry_file: str = "expertise_registry.json") -> Registry:
    """Load fields from <base_dir>/fields/*.{json,yaml,yml} and resolve the
    active index from <base_dir>/<registry_file>.

    Mirrors kernel skills.load_all(): dedup by field id (lowest priority wins).
    """
    base = Path(base_dir).expanduser()
    fields_dir = base / "fields"

    # 1. Parse all field definition files
    raw: list[dict] = []
    for fpath in _rglob(fields_dir, (".json", ".yaml", ".yml")):
        f = _load_field_file(fpath)
        if f and f.get("id"):
            raw.append(f)

    # 2. Dedup by id, priority shadows (keep _src for comparison; strip on output)
    seen: dict[str, dict] = {}
    for f in raw:
        fid = f["id"]
        prio = _priority(Path(f["_src"]), base)
        if fid not in seen or prio < _priority(Path(seen[fid]["_src"]), base):
            seen[fid] = dict(f)  # keep _src for subsequent priority checks
    # drop private/underscore keys on the public copies
    fields = [{k: v for k, v in f.items() if not k.startswith("_")} for f in seen.values()]

    # 3. Resolve active index
    reg_path = base / registry_file
    active: list[str] = []
    if reg_path.exists():
        reg = _read_json(reg_path)
        if reg and isinstance(reg.get("active"), list):
            active = [a for a in reg["active"] if a in seen]

    return Registry(fields=fields, active=active, src_dir=str(base))


def load_registry(base_dir: str | Path) -> Registry:
    """Convenience alias for load()."""
    return load(base_dir)


# ---------------------------------------------------------------------------
# Per-chat hot-field state (decision #2: JSON store)
# ---------------------------------------------------------------------------
class HotFieldState:
    """Persist per-chat hot fields to a JSON file.

    Structure:
        {
          "<chat_id>": {
            "hot": ["plant-science"],        # currently-hot field ids
            "last_used": "2026-08-23T03:02:00Z",
            "usage": {"plant-science": 7}
          }
        }

    JSON chosen (decision #2) over SQLite: cheap read/write, no dependency, and
    the volume is small (1-3 hot fields per chat, low cardinality).
    """

    DEFAULTS = {"hot": [], "last_used": None, "usage": {}}

    def __init__(self, path: str | Path = "hot_fields.json", max_hot: int = 3):
        self.path = Path(path).expanduser()
        self.max_hot = int(max_hot)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        if not isinstance(self._data, dict):
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )

    # -- per-chat accessors -------------------------------------------------
    def _chat(self, chat_id: str) -> dict:
        return self._data.setdefault(chat_id, dict(self.DEFAULTS))

    def hot_fields(self, chat_id: str) -> list[str]:
        return list(self._chat(chat_id).get("hot", []))

    def is_hot(self, chat_id: str, fid: str) -> bool:
        return fid in self.hot_fields(chat_id)

    def set_hot(self, chat_id: str, fid: str, save: bool = True) -> bool:
        """Mark a field hot. Returns True if a field was evicted to fit max_hot."""
        chat = self._chat(chat_id)
        hot = self.hot_fields(chat_id)
        evicted = False
        if fid not in hot:
            hot.append(fid)
        # enforce cap (LRU-ish: evict from the front/least-recently-used)
        while len(hot) > self.max_hot:
            hot.pop(0)
            evicted = True
        chat["hot"] = hot
        chat["usage"][fid] = chat.get("usage", {}).get(fid, 0) + 1
        chat["last_used"] = _iso_now()
        if save:
            self.save()
        return evicted

    def touch(self, chat_id: str, fid: str, save: bool = True) -> None:
        chat = self._chat(chat_id)
        chat["usage"][fid] = chat.get("usage", {}).get(fid, 0) + 1
        chat["last_used"] = _iso_now()
        if save:
            self.save()

    def snapshot(self, chat_id: str | None = None) -> dict:
        """Return a debug-friendly snapshot (used by /debug/fields)."""
        if chat_id is not None:
            return {"chat_id": chat_id, **self._chat(chat_id)}
        return dict(self._data)


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")