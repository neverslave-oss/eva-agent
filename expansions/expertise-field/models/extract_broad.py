#!/usr/bin/env python3
"""
extract_broad.py
================
Broaden seed coverage across Kernel-Evo's per-domain workspaces.

Runs the existing seed_extractor parser over MULTIPLE memory dirs and merges
the rows into one `raw_seed.jsonl`, tagging each row with its `source_workspace`.

Workspaces covered (each is Kernel-Evo's OWN activity, just in a different
domain — correct "self-only, cross-coverage" scope decided in the spec):
  - main        : ~/.openclaw/workspace/memory            (the existing 389)
  - invest      : ~/.openclaw/workspace-invest/memory      (finance/portfolio)
  - marketing   : ~/.openclaw/workspace-marketing/memory   (content/branding)
  - hack        : ~/.openclaw/workspace-hack/memory        (expected to converge
                                                             to the main dev cluster)

Each output row gains `source_workspace`. Files already seen by the main extractor
are not re-scanned there; this simply runs the parser per dir and merges.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from seed_extractor import parse_file  # noqa: E402

# (workspace_name, memory_dir)
SOURCES = [
    ("main",      os.path.expanduser("~/.openclaw/workspace/memory")),
    ("invest",    os.path.expanduser("~/.openclaw/workspace-invest/memory")),
    ("marketing", os.path.expanduser("~/.openclaw/workspace-marketing/memory")),
    ("hack",      os.path.expanduser("~/.openclaw/workspace-hack/memory")),
]

OUT = HERE / "datasets" / "raw_seed.jsonl"


def main() -> int:
    all_rows = []
    seen = set()
    per_ws = {}
    for ws, memdir in SOURCES:
        d = Path(memdir)
        if not d.is_dir():
            print(f"[skip] workspace ...{ws}: dir not found: {d}", file=sys.stderr)
            continue
        rows = []
        for path in sorted(d.glob("*.md")):
            row = parse_file(path)
            if row is None:
                continue
            # Tag with source workspace + de-duplicate by source_file name
            key = (ws, row["source_file"])
            if key in seen:
                continue
            seen.add(key)
            row["source_workspace"] = ws
            rows.append(row)
        per_ws[ws] = len(rows)
        all_rows.extend(rows)
        print(f"[ok] workspace '{ws}': {len(rows)} rows")

    # Write merged raw_seed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nMerged {len(all_rows)} total rows -> {OUT}")
    print(f"per-workspace: {per_ws}")

    # Quick topic distribution across the whole set
    from collections import Counter
    dist = Counter()
    for r in all_rows:
        for t in r.get("skill_domains", []):
            dist[t] += 1
    print("topic distribution:", dict(dist.most_common()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)