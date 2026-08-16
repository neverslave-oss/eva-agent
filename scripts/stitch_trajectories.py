#!/usr/bin/env python3
"""
stitch_trajectories.py
======================
Stitch together PASS trajectories to create longer multi-turn sequences.
This multiplies training signal by teaching the model to chain tasks.

Usage:
  python3 scripts/stitch_trajectories.py [--input path] [--max-pairs N] [--max-triples N]
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

_DEFAULT_INPUT = Path.home() / ".kernel-evolving/workspace/artifacts/trajectories/sft_ready_final.jsonl"
_DEFAULT_OUTPUT = Path.home() / ".kernel-evolving/workspace/artifacts/trajectories/stitched_trajectories.jsonl"


def load_pass_trajectories(path: Path) -> list[dict]:
    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            if traj.get("verdict") == "PASS" and traj.get("critic_score", 0) >= 0.7:
                trajectories.append(traj)
    return trajectories


def strip_system_prompt(messages: list[dict]) -> list[dict]:
    """Remove the system prompt message(s) from the front of a message list."""
    return [m for m in messages if m.get("role") != "system"]


def make_connector_turn(task_b: str) -> dict:
    """Create a brief user connector turn between two stitched trajectories."""
    return {
        "role": "user",
        "content": f"Good. Now do the next task: {task_b}"
    }


def stitch_pair(traj_a: dict, traj_b: dict, pair_index: int) -> dict:
    """Stitch two trajectories into one multi-turn conversation."""
    sys_msg = next(m for m in traj_a["messages"] if m["role"] == "system")
    msgs_a = strip_system_prompt(traj_a["messages"])
    msgs_b = strip_system_prompt(traj_b["messages"])

    # Remove the leading user message from B (it becomes the connector)
    # The connector replaces B's original user turn
    msgs_b_no_user_first = msgs_b[1:] if msgs_b and msgs_b[0]["role"] == "user" else msgs_b

    connector = make_connector_turn(traj_b["task"])

    combined_messages = [sys_msg] + msgs_a + [connector] + msgs_b_no_user_first

    return {
        "id": f"stitched-pair-{pair_index:04d}",
        "task": f"[stitched-2] {traj_a['task']} + {traj_b['task']}",
        "provider": "stitched",
        "model": "stitched",
        "call_type": "task_inference",
        "messages": combined_messages,
        "artifacts": traj_a.get("artifacts", []) + traj_b.get("artifacts", []),
        "critic_score": min(traj_a["critic_score"], traj_b["critic_score"]),
        "verdict": "PASS",
    }


def stitch_triple(traj_a: dict, traj_b: dict, traj_c: dict, triple_index: int) -> dict:
    """Stitch three trajectories into one multi-turn conversation."""
    sys_msg = next(m for m in traj_a["messages"] if m["role"] == "system")
    msgs_a = strip_system_prompt(traj_a["messages"])
    msgs_b = strip_system_prompt(traj_b["messages"])
    msgs_c = strip_system_prompt(traj_c["messages"])

    msgs_b_no_user_first = msgs_b[1:] if msgs_b and msgs_b[0]["role"] == "user" else msgs_b
    msgs_c_no_user_first = msgs_c[1:] if msgs_c and msgs_c[0]["role"] == "user" else msgs_c

    connector_b = make_connector_turn(traj_b["task"])
    connector_c = make_connector_turn(traj_c["task"])

    combined_messages = (
        [sys_msg]
        + msgs_a
        + [connector_b]
        + msgs_b_no_user_first
        + [connector_c]
        + msgs_c_no_user_first
    )

    return {
        "id": f"stitched-triple-{triple_index:04d}",
        "task": f"[stitched-3] {traj_a['task']} + {traj_b['task']} + {traj_c['task']}",
        "provider": "stitched",
        "model": "stitched",
        "call_type": "task_inference",
        "messages": combined_messages,
        "artifacts": (
            traj_a.get("artifacts", [])
            + traj_b.get("artifacts", [])
            + traj_c.get("artifacts", [])
        ),
        "critic_score": min(
            traj_a["critic_score"], traj_b["critic_score"], traj_c["critic_score"]
        ),
        "verdict": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description="Stitch PASS trajectories into longer sequences")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--max-triples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading PASS trajectories from {args.input}...")
    trajectories = load_pass_trajectories(args.input)
    print(f"  Loaded {len(trajectories)} PASS trajectories")

    # Generate all unique pairs (no self-pairs, no reverse duplicates)
    all_pairs = list(itertools.combinations(range(len(trajectories)), 2))
    random.shuffle(all_pairs)
    selected_pairs = all_pairs[: args.max_pairs]

    # Generate triples (sample from combinations)
    all_triples = list(itertools.combinations(range(len(trajectories)), 3))
    random.shuffle(all_triples)
    selected_triples = all_triples[: args.max_triples]

    print(f"  Generating {len(selected_pairs)} pairs and {len(selected_triples)} triples...")

    stitched = []

    for i, (a_idx, b_idx) in enumerate(selected_pairs):
        pair_traj = stitch_pair(trajectories[a_idx], trajectories[b_idx], i)
        stitched.append(pair_traj)

    for i, (a_idx, b_idx, c_idx) in enumerate(selected_triples):
        triple_traj = stitch_triple(
            trajectories[a_idx], trajectories[b_idx], trajectories[c_idx], i
        )
        stitched.append(triple_traj)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for traj in stitched:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"\n✅ Stitching complete:")
    print(f"   Pairs:   {len(selected_pairs)}")
    print(f"   Triples: {len(selected_triples)}")
    print(f"   Total:   {len(stitched)}")
    print(f"   Output:  {args.output}")


if __name__ == "__main__":
    main()
