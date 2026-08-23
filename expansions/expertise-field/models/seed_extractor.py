#!/usr/bin/env python3
"""
seed_extractor.py
=================
Parse OpenClaw/Kernel-Evo daily memory logs into structured ActivityEvent rows.

This is the REAL implementation reading the actual memory files on disk.
It produces `raw_seed.jsonl` — the unembedded seed dataset for the
expertise-field clustering discovery layer (models/README.md, SPEC Part 7).

Two input formats are handled:
  1. SESSION files  — have a `# Session:` header, `## Conversation Summary`,
                      optionally `## Key Decisions` / `## Next Steps`.
  2. DAILY files    — have a `### <Month> <Day>, <Year>` header + bullet list.

Embedding dimension contract: the local embeddings server (`:8770`) is
`embeddinggemma-300m` → 768-d vectors. Rows are stored UNEMBEDDED here by
design (inspect before paying for embeddings); the embed step is separate.
"""

import argparse
import glob
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Topic lexicon — cosine-sim-independent keyword anchors used to tag a row
# with coarse discipline hints. These are INTERPRETATION labels, not the
# clustering features (the embeddings are). They help us eyeball the seed.
# ---------------------------------------------------------------------------
TOPICS = {
    "plant": ["soil", "moisture", "nutrient", "greenhouse", "hydropon", "yield",
              "crop", "ph level", "ec level", "leaf", "photosynth"],
    "finance": ["portfolio", "stock", "trading", "alpha vantage", "trading212",
                "dividend", "p&l", "position sizing", "capital", "equity", "ticker"],
    "dev": ["debug", "refactor", "branch", "merge", "pull request", "compile",
            "module", "package.json", "requirements.txt", "stack trace", "exception",
            "typeerror", "syntax", "linter"],
    "infra": ["deploy", "tunnel", "nginx", "systemd", "container", "ssh",
              "backup", "volume", "provision", "kubernetes", "k8s", "reverse proxy"],
    "security": ["vuln", "hardening", "permission", "firewall", "crypto", "exploit",
                 "patching", "cve", "auth bypass", "privilege", "sandbox"],
    "media": ["flux", "stable diffusion", "imagegen", "voice clone", "lora",
              "pptx", "narration", "render video", "sdxl", "gfpgan", "upscale"],
    "ops": ["watcher", "alert", "heartbeat", "notification", "cron", "monitor",
            "log rotate", "uptime", "probe", "polling"],
    "data": ["pandas", "timeseries", "csv", "sql", "dataset", "analytics",
             "pipeline", "dataframe", "matplotlib", "scikit-learn", "numpy"],
}

# Native tools we can detect in transcripts (assistant tool invocations).
TOOL_PATTERNS = [
    "exec_shell", "read_file", "write_file", "http_get", "web_search",
    "browser_use", "run_skill", "run_routine", "send_file", "look",
    "sensors", "recall_memory", "search_skills", "list_routines",
]

_TS_RE = re.compile(r"#\s*Session:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9:]{2,8})")
_DAILY_RE = re.compile(r"^###\s+(January|February|March|April|May|June|July|August|"
                       r"September|October|November|December)\s+\d{1,2},\s+(\d{4})")


def parse_ts(text: str, filename: str):
    """Best-effort extraction of an ISO timestamp from a session header or the filename."""
    m = _TS_RE.search(text)
    if m:
        raw = m.group(1).replace(" ", "T")
        try:
            return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    # Fall back to filename date prefix YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", Path(filename).name)
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    return None


def detect_topics(text: str) -> list:
    text_l = text.lower()
    found = []
    for topic, kws in TOPICS.items():
        for kw in kws:
            if kw in text_l:
                found.append(topic)
                break
    return found


def detect_tools(text: str) -> list:
    return [t for t in TOOL_PATTERNS if t in text]


def extract_section(text: str, name: str) -> str:
    """Return the body of a ## <name> section, or '' if absent."""
    m = re.search(rf"^##\s+{re.escape(name)}\s*\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def extract_daily_bullets(text: str) -> str:
    """Daily files: grab the bullet items as a single text block."""
    bullets = [
        line.lstrip("- ").strip()
        for line in text.splitlines()
        if line.strip().startswith("- ") or re.match(r"^\s*[-*]\s", line)
    ]
    return "\n".join(b for b in bullets if b)


def parse_file(path: Path) -> dict | None:
    """Parse a single memory file into an ActivityEvent dict (unembedded)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None

    is_session = _TS_RE.search(text) is not None
    daily = _DAILY_RE.search(text.splitlines()[0] if text.splitlines() else "")

    if is_session:
        summary = extract_section(text, "Conversation Summary")
        decisions = extract_section(text, "Key Decisions")
        next_steps = extract_section(text, "Next Steps")
        body = "\n".join(x for x in [summary, decisions, next_steps] if x)
        if not body:
            body = text  # fall back to whole file
    elif daily:
        body = extract_daily_bullets(text)
        if not body:
            body = text
    else:
        body = text

    ts = parse_ts(text, path.name)
    topics = detect_topics(body)
    tools = detect_tools(body)

    return {
        "event_id": str(uuid.uuid4()),
        "ts": ts,
        "source_file": path.name,
        "source_format": "session" if is_session else ("daily" if daily else "raw"),
        "intent": _summarize_intent(body),
        "intent_text": _shorten(body),
        "skill_domains": topics,
        "tools_used": tools,
        "routing_outcome": "field_match" if topics else "fallback",
        "hot_field": (topics[0] if topics else None),
        "embedding_dim": None,  # not embedded yet
        "embedding_model": None,
    }


def _summarize_intent(text: str) -> str:
    """A terse, single-line intent descriptor derived from the body."""
    m = re.search(r"assistant:\s*([^\n]{10,160})", text)
    if m:
        return m.group(1).strip().replace("\n", " ")
    # fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "user:", "assistant:", "##", "- ")):
            return line[:160]
    return text[:160]


def _shorten(text: str, n: int = 400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n]


def main():
    ap = argparse.ArgumentParser(description="Extract ActivityEvent seed rows from memory logs")
    ap.add_argument("--memory", default=os.path.expanduser("~/.openclaw/workspace/memory"),
                    help="Path to memory dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "datasets", "raw_seed.jsonl"),
                    help="Output JSONL path")
    args = ap.parse_args()

    memory_dir = Path(args.memory)
    if not memory_dir.is_dir():
        print(f"ERROR: memory dir not found: {memory_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = 0
    for path in sorted(memory_dir.glob("*.md")):
        row = parse_file(path)
        if row:
            rows.append(row)
        else:
            skipped += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Summary
    n_session = sum(1 for r in rows if r["source_format"] == "session")
    n_daily = sum(1 for r in rows if r["source_format"] == "daily")
    n_raw = sum(1 for r in rows if r["source_format"] == "raw")
    with_tools = sum(1 for r in rows if r["tools_used"])
    with_topics = sum(1 for r in rows if r["skill_domains"])
    topics_dist = {}
    for r in rows:
        for t in r["skill_domains"]:
            topics_dist[t] = topics_dist.get(t, 0) + 1

    print(f"Extracted {len(rows)} ActivityEvent rows")
    print(f"  files scanned: {len(rows) + skipped}, skipped: {skipped}")
    print(f"  session:{n_session} daily:{n_daily} raw:{n_raw}")
    print(f"  with tools: {with_tools} | with topics: {with_topics}")
    print(f"  topic distribution: {dict(sorted(topics_dist.items(), key=lambda x:-x[1]))}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()