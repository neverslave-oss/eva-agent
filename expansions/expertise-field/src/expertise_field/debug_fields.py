"""
debug_fields.py — Standalone debugging/CLI harness for the expertise_field module.

Lets us inspect the registry, route sample queries, and dump per-chat hot state
without wiring into the kernel yet. Mirrors the `/debug/fields` endpoint
(decision #4) that this module will eventually expose.

Usage:
    python src/expertise_field/debug_fields.py [--query "text"] [--repo PATH] [--chat chatA]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # module repo root
sys.path.insert(0, str(REPO / "src"))

from expertise_field.registry import HotFieldState, load_registry
from expertise_field import router

DEFAULT_REPO = str(REPO)


def main() -> int:
    ap = argparse.ArgumentParser(description="expertise_field debug harness")
    ap.add_argument("--query", type=str, default="the plants look dry, check soil moisture")
    ap.add_argument("--repo", type=str, default=DEFAULT_REPO)
    ap.add_argument("--chat", type=str, default="chatA")
    ap.add_argument("--hot", action="store_true", help="print full hot-field state snapshot")
    args = ap.parse_args()

    reg = load_registry(args.repo)
    st = HotFieldState(REPO / "hot_fields.json")

    print("== Registry ==")
    print(json.dumps(reg.to_dict(), indent=2))

    print("\n== Trigger match for:", args.query)
    matched = router.match_by_triggers(args.query, reg)
    print(json.dumps([f["id"] for f in matched]))

    hot, cands = router.choose_fields(args.query, reg, st, chat_id=args.chat)
    print("\n== chosen hot fields ==")
    print(hot)
    print("\n== candidate skills (fed to semantic matcher) ==")
    print(cands)

    if args.hot:
        print("\n== hot-field state snapshot ==")
        print(json.dumps(st.snapshot(args.chat), indent=2))

    # field-tagged skill narrowing example (decision #5)
    tagged = ["soil-analyzer::plant-science", "yield-predictor::plant-science"]
    _h2, c2 = router.choose_fields(args.query, reg, st, chat_id=args.chat, all_skill_names=tagged)
    print("\n== candidate skills incl. field-tagged refs ==")
    print(c2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())