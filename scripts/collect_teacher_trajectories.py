#!/usr/bin/env python3
"""
collect_teacher_trajectories.py
================================
Uses the configured teacher provider (OpenAI gpt-5.4 or similar) to run
the synthetic task set and capture real tool-call chains into the trajectory DB.

This bypasses the local VRAM constraint entirely — the teacher model runs in
the cloud, executes tools locally, and we record the full chain.

These become the POSITIVE examples for DPO fine-tuning:
  teacher PASS + tool_calls → positive
  local Gemma FAIL on same tasks → negative (from existing sim runs)

Usage:
  cd repositories/kernel-evolving
  python3 scripts/collect_teacher_trajectories.py [--count N] [--export] [--dry-run]

Requires: OPENAI_API_KEY or TMP_OPEN_AI_API_KEY in .env
"""

import argparse
import json
import os
import re
import sys
import time
import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Import task templates from synthetic generator
sys.path.insert(0, str(_ROOT / "scripts"))
from generate_synthetic_trajectories import TASK_TEMPLATES


def run_task_with_teacher(task: str, provider, tools: list, workspace: str) -> tuple[list, str]:
    """Run infer_with_tools through teacher provider. Returns (steps, reply)."""
    steps = []

    def _step_cb(n, tool_name, args, result):
        steps.append({"tool": tool_name, "args": args, "result": str(result)[:500]})

    from model import infer_with_tools as local_iwt

    # Route through teacher provider (OpenAI)
    reply = provider.infer_with_tools(
        messages=[
            {"role": "system", "content": (
                "You are a precise AI agent. Complete tasks step by step using the tools available. "
                "Always write files when asked. Always run commands when asked. "
                "Never skip tool calls. Return a brief confirmation when done."
            )},
            {"role": "user", "content": task},
        ],
        tools=tools,
        workspace=workspace,
        max_steps=10,
        step_callback=_step_cb,
        call_type="trajectory_teacher",
    )
    return steps, reply


def critic_score(task: str, reply: str, steps: list, provider) -> float:
    """Score the result. Uses teacher provider for reliable scoring."""
    try:
        step_summary = ", ".join(s["tool"] for s in steps) if steps else "none"
        prompt = (
            f"Rate this task execution 0.0–1.0.\n"
            f"Task: {task[:300]}\n"
            f"Tools used: {step_summary}\n"
            f"Reply: {reply[:400]}\n\n"
            f"Score: 1.0 = perfect, 0.7 = acceptable, <0.5 = failed.\n"
            f"Respond with score only (e.g. 0.85):"
        )
        raw = provider.infer(
            [{"role": "user", "content": prompt}],
            max_new_tokens=16,
            call_type="critic",
        )
        m = re.search(r"\b(0\.\d+|1\.0|1)\b", raw)
        return float(m.group(1)) if m else 0.5
    except Exception as e:
        print(f"  [critic] error: {e}")
        return 0.5


def main():
    parser = argparse.ArgumentParser(description="Collect teacher trajectories via OpenAI")
    parser.add_argument("--count", type=int, default=len(TASK_TEMPLATES))
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-filter", type=str, default="")
    args = parser.parse_args()

    import yaml
    config_path = _ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Verify teacher provider is configured
    teacher_provider_name = config.get("providers", {}).get("trajectory_teacher", "openai")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")
    if teacher_provider_name == "openai" and not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or set trajectory_teacher to another provider.")
        sys.exit(1)

    workspace = os.path.expanduser("~/.kernel-evolving/workspace")
    os.makedirs(workspace, exist_ok=True)

    from trajectory_collector import get_collector, TrajectoryCollector
    from tools import TOOLS
    from provider import get_provider

    col = get_collector(config)
    provider = get_provider()

    tasks = TASK_TEMPLATES[:args.count]
    if args.task_filter:
        tasks = [(t, e, d) for t, e, d in tasks if args.task_filter.lower() in t.lower()]

    if args.dry_run:
        print(f"Would run {len(tasks)} tasks via teacher provider: {teacher_provider_name}")
        for i, (task, expected, diff) in enumerate(tasks, 1):
            print(f"  {i:2d}. [{diff:6s}] {task[:80]}")
        return

    print(f"\n{'='*60}")
    print(f"Teacher trajectory collector")
    print(f"Provider: {teacher_provider_name}")
    print(f"Tasks: {len(tasks)}  Min score: {args.min_score}")
    print(f"{'='*60}\n")

    # Count before
    try:
        import sqlite3
        _DB_PATH = os.path.expanduser("~/.kernel-evolving/workspace/data/evolution.db")
        conn = sqlite3.connect(_DB_PATH)
        before_count = conn.execute("SELECT COUNT(*) FROM task_trajectories WHERE provider LIKE 'teacher/%'").fetchone()[0]
        conn.close()
        print(f"Teacher trajectories in DB before: {before_count}")
    except Exception:
        before_count = 0

    saved = 0
    skipped = 0
    failed = 0

    for i, (task, expected_tools, difficulty) in enumerate(tasks, 1):
        print(f"\n[{i:2d}/{len(tasks)}] [{difficulty:6s}] {task[:70]}")
        t0 = time.time()
        try:
            steps, reply = run_task_with_teacher(task, provider, TOOLS, workspace)
            elapsed = time.time() - t0

            actual_tools = [s["tool"] for s in steps]
            score = critic_score(task, reply, steps, provider)

            print(f"         Tools:  {actual_tools}")
            print(f"         Score:  {score:.2f}  ({elapsed:.1f}s)")
            print(f"         Reply:  {reply[:100]}")

            # Detect artifacts
            now = time.time()
            artifacts = []
            for s in steps:
                if s.get("tool") == "write_file":
                    p = os.path.expanduser(s.get("args", {}).get("path", ""))
                    if p and os.path.isfile(p):
                        artifacts.append(p)
            # Also check evo_* files
            for p in Path.home().glob("evo_*"):
                if p.is_file() and (now - p.stat().st_mtime) < 90:
                    artifacts.append(str(p))
            artifacts = list(set(artifacts))

            verdict = "PASS" if score >= args.min_score else "FAIL"
            print(f"         Verdict: {verdict}  tools={len(steps)}  artifacts={len(artifacts)}")

            if score >= args.min_score and steps:
                col.record(
                    task=task,
                    provider=f"teacher/{teacher_provider_name}/{difficulty}",
                    model_name=config.get("providers", {}).get("models", {}).get(teacher_provider_name, "unknown"),
                    call_type="trajectory_teacher",
                    tool_calls=steps,
                    final_reply=reply,
                    artifacts=artifacts,
                    critic_score=score,
                    critic_verdict=verdict,
                )
                saved += 1
                print(f"         ✅ Saved")
            else:
                skipped += 1
                reason = "no steps" if not steps else f"score {score:.2f} < {args.min_score}"
                print(f"         ⏭  Skipped ({reason})")

        except Exception as e:
            failed += 1
            print(f"         ❌ Error: {e}")

        time.sleep(1)  # rate limit

    print(f"\n{'='*60}")
    print(f"Done. Saved: {saved}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*60}\n")

    # Count after
    try:
        conn = sqlite3.connect(_DB_PATH)
        after_count = conn.execute("SELECT COUNT(*) FROM task_trajectories WHERE provider LIKE 'teacher/%'").fetchone()[0]
        total_with_tools = conn.execute("SELECT COUNT(*) FROM task_trajectories WHERE tool_calls != '[]'").fetchone()[0]
        conn.close()
        print(f"Teacher trajectories in DB after: {after_count} (+{after_count - before_count})")
        print(f"Total rows with tool_calls: {total_with_tools}")
    except Exception:
        pass

    if args.export and saved > 0:
        print("Exporting JSONL...")
        path, count = col.export_jsonl(min_score=args.min_score)
        print(f"Exported {count} records → {path}")
        print(f"\nTo fine-tune the drafter:")
        print(f"  uv run scripts/finetune_from_trajectories.py --dataset {path}")


if __name__ == "__main__":
    main()
