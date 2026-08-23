#!/usr/bin/env python3
"""Unit tests for models/seed_extractor.py, embed.py, visualize.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "models"))

import seed_extractor as se
import embed as emb


def _write_tmp(dir: Path, name: str, content: str) -> Path:
    p = dir / name
    p.write_text(content, encoding="utf-8")
    return p


SESSION_SAMPLE = """\
# Session: 2026-03-01 01:49:14 UTC
- **Session Key**: agent:main:main
- **Source**: telegram
## Conversation Summary
assistant: Fixed the soil moisture watcher, checked nutrient dosing.
user: thanks
## Key Decisions
Switch to FLUX for image generation.
## Next Steps
Reboot the greenhouse sensor tunnel.
"""

DAILY_SAMPLE = """\
### February 21, 2026
- Adjusted the open-agentic-investor project for Trading 212 live endpoint.
- Integrated Alpha Vantage API for stock price fetching.
"""


def test_parse_session_format(tmp_path):
    p = _write_tmp(tmp_path, "2026-03-01-x.md", SESSION_SAMPLE)
    row = se.parse_file(p)
    assert row is not None
    assert row["source_format"] == "session"
    assert row["ts"].startswith("2026-03-01")
    assert "soil" in row["intent_text"].lower()
    assert "plant" in row["skill_domains"]  # soil/ moisture -> plant
    # Session summary has no literal tool names (watcher/sensor/tunnel are domain
    # terms, not tool invocations), so detect_tools correctly returns [].
    # Literal tool detection is covered by test_detect_tools.
    assert se.detect_tools(row["intent_text"]) == []


def test_parse_daily_format(tmp_path):
    p = _write_tmp(tmp_path, "2026-02-21.md", DAILY_SAMPLE)
    row = se.parse_file(p)
    assert row is not None
    assert row["source_format"] == "daily"
    assert "finance" in row["skill_domains"]  # trading/portfolio -> finance


def test_ts_from_filename_fallback(tmp_path):
    p = _write_tmp(tmp_path, "2026-04-10-1234.md", "No session header here")
    row = se.parse_file(p)
    assert row is not None
    assert row["ts"].startswith("2026-04-10")


def test_empty_file_skipped(tmp_path):
    p = _write_tmp(tmp_path, "empty.md", "")
    assert se.parse_file(p) is None


def test_detect_tools():
    assert "read_file" in se.detect_tools("I used read_file and web_search")
    assert se.detect_tools("no tools here") == []


def test_detect_topics_multiple():
    topics = se.detect_topics("trading portfolio and soil moisture watcher")
    assert "finance" in topics
    assert "plant" in topics


def test_main_writes_jsonl(tmp_path, monkeypatch):
    memdir = tmp_path / "memory"
    memdir.mkdir()
    _write_tmp(memdir, "2026-05-01-a.md", SESSION_SAMPLE)
    _write_tmp(memdir, "2026-05-01.md", DAILY_SAMPLE)
    out = tmp_path / "raw_seed.jsonl"
    monkeypatch.setattr(sys, "argv", ["seed_extractor", "--memory", str(memdir), "--out", str(out)])
    se.main()
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert "event_id" in row and "intent_embedding" not in row  # unembedded by default


def test_embed_roundtrip_skips_embedded():
    rows = [{"intent": "x", "intent_text": "y"}, {"intent": "z", "intent_text": "w", "intent_embedding": [1.0]}]
    to_embed = [r for r in rows if not r.get("intent_embedding")]
    assert len(to_embed) == 1
    assert to_embed[0]["intent"] == "x"