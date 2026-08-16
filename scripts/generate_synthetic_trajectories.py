#!/usr/bin/env python3
"""
generate_synthetic_trajectories.py
===================================
Generates synthetic (task, tool_calls, reply) trajectories and writes them
directly into the trajectory DB + exports JSONL for drafter fine-tuning.

Usage:
  cd /path/to/kernel-evolving
  python3 scripts/generate_synthetic_trajectories.py [--count N] [--export]

What it does:
  1. Defines a curated set of task templates covering: file ops, shell exec,
     fetch+summarise, multi-step planning, skill dispatch, data conversion.
  2. For each task, generates a realistic tool-call chain using the configured
     provider (default: local Gemma, override with --provider openai).
  3. Runs a critic pass to score the result.
  4. Saves PASS trajectories (score >= 0.7) to the DB.
  5. Optionally exports JSONL for HuggingFace TRL SFTTrainer.

Synthetic vs real:
  Real trajectories come from live user interactions — they capture real gaps.
  Synthetic trajectories cover known-good patterns the drafter should learn:
  reliable write_file + exec_shell chains, multi-tool sequences, etc.
  Together they form a DPO dataset: synthetic PASS = positive, Gemma FAIL = negative.
"""

import argparse
import json
import os
import re
import sys
import time
import datetime
from pathlib import Path

# Add src/ to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Task templates ────────────────────────────────────────────────────────────
# Each entry: (task_description, expected_tools, difficulty)
# difficulty: easy / medium / hard

TASK_TEMPLATES = [
    # File ops
    ("Write a Python script that prints 'Hello from Evo' and save it to ~/evo_hello.py",
     ["write_file"], "easy"),
    ("Read the file ~/evo_hello.py and summarise what it does",
     ["read_file"], "easy"),
    ("Create a file ~/evo_notes/daily.md with today's date as a heading and one placeholder note",
     ["exec_shell", "write_file"], "easy"),
    ("List all .py files in the current directory and save the list to ~/evo_filelist.txt",
     ["exec_shell", "write_file"], "easy"),

    # Shell exec
    ("Check the current disk usage and report the top 3 largest directories under /home",
     ["exec_shell"], "easy"),
    ("Run 'python3 --version' and report the result",
     ["exec_shell"], "easy"),
    ("Create a directory ~/evo_workspace/data and confirm it exists",
     ["exec_shell"], "easy"),

    # Multi-step
    ("Write a Python function that reverses a string, save it to ~/evo_utils.py, then run it with the input 'kernel'",
     ["write_file", "exec_shell"], "medium"),
    ("Fetch the content of https://example.com, extract the page title, and save it to ~/evo_fetch.txt",
     ["http_get", "write_file"], "medium"),
    ("Write a bash script that counts lines in all .py files under /tmp and saves the result to ~/evo_linecount.txt",
     ["write_file", "exec_shell"], "medium"),
    ("Create ~/evo_config.json with keys: name='evo', version='1.0', active=true",
     ["write_file"], "easy"),
    ("Read ~/evo_config.json, update the version to '1.1', and write it back",
     ["read_file", "write_file"], "medium"),

    # Data conversion
    ("Convert this CSV to a JSON array and save to ~/evo_data.json:\nname,age\nFabio,35\nOlly,1",
     ["write_file"], "medium"),
    ("Write a Python script that reads a CSV from stdin and prints it as a markdown table, save to ~/evo_csv2md.py",
     ["write_file"], "medium"),

    # Planning / multi-step hard
    ("Create a small project: write ~/evo_project/main.py (hello world), ~/evo_project/README.md, and run main.py",
     ["exec_shell", "write_file", "exec_shell"], "hard"),
    ("Write a Python script that generates 5 random numbers, saves them to ~/evo_random.txt, then reads and sums them",
     ["write_file", "exec_shell", "read_file"], "hard"),

    # Skill-like tasks
    ("Summarise the following text in 3 bullet points and save to ~/evo_summary.md:\n"
     "Kernel-evolving is a self-evolving AI agent that runs locally, learns from gaps, "
     "and improves its own skills over time through a think-at-rest cycle.",
     ["write_file"], "easy"),
    ("Extract all URLs from this text and save them as a JSON list to ~/evo_urls.json:\n"
     "Visit https://kernel.ai and https://fabiopacifici.com for more info. Also check https://openclaw.ai",
     ["write_file"], "easy"),

    # ── Multi-turn follow-through (exec → write, read → reason → write) ───────────
    ("Run 'uname -a && python3 --version && free -h' and save the complete output to ~/evo_sysinfo.txt",
     ["exec_shell", "write_file"], "easy"),
    ("Check which Python packages are installed with 'pip list' and save the first 20 lines to ~/evo_packages.txt",
     ["exec_shell", "write_file"], "easy"),
    ("Run 'df -h' to check disk usage and save the output to ~/evo_disk.txt, then read it back and tell me the root partition usage",
     ["exec_shell", "write_file", "read_file"], "medium"),
    ("Write a file ~/evo_greeting.txt with the content 'Hello Fabio', then read it back and confirm its contents",
     ["write_file", "read_file"], "easy"),
    ("Create ~/evo_counter.py that counts from 1 to 5, run it, and save the output to ~/evo_count_output.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),
    ("Run 'git --version && node --version && python3 --version' and write a short tech stack summary to ~/evo_stack.md",
     ["exec_shell", "write_file"], "easy"),
    ("Fetch https://httpbin.org/get, extract the 'origin' IP field, and save it to ~/evo_ip.txt",
     ["http_get", "write_file"], "medium"),
    ("Write ~/evo_fibonacci.py that prints the first 10 Fibonacci numbers, run it, and save the output to ~/evo_fib_output.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),

    # ── Read → reason → write patterns ───────────────────────────────────────
    ("Read the file ~/kernel-evo-notes/g4bf16_eval.txt (if it exists) and summarise what it contains in one sentence saved to ~/evo_read_test.txt",
     ["read_file", "write_file"], "easy"),
    ("Check if ~/evo_config.json exists. If yes, read it and write a summary to ~/evo_config_summary.txt. If no, create it with default values.",
     ["read_file", "write_file"], "medium"),
    ("Read the file /etc/os-release, extract the OS name and version, and write them to ~/evo_os_info.txt",
     ["read_file", "write_file"], "easy"),

    # ── Error recovery — agent should handle gracefully ────────────────────────
    ("Try to read ~/nonexistent_file_xyz.txt. When it fails, create it with the content 'created by Evo' and confirm.",
     ["read_file", "write_file"], "medium"),
    ("Run 'ls ~/evo_workspace/data 2>/dev/null || mkdir -p ~/evo_workspace/data && echo created' and report the result",
     ["exec_shell"], "easy"),

    # ── Structured data tasks ─────────────────────────────────────────────
    ("Create ~/evo_tasks.json with a list of 3 fake todo items (id, title, done), then run 'python3 -m json.tool ~/evo_tasks.json' to validate it",
     ["write_file", "exec_shell"], "medium"),
    ("Write a YAML file ~/evo_config.yaml with keys: agent, version, enabled. Then run python3 to verify it parses correctly and save the result to ~/evo_yaml_check.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),
    ("Fetch https://httpbin.org/json, save the raw JSON to ~/evo_httpbin.json, then extract the 'slideshow.title' field and report it",
     ["http_get", "write_file"], "hard"),

    # ── Agent self-knowledge tasks ──────────────────────────────────────────
    ("Run 'ls ~/.kernel-evolving/ecosystem/private/skills/ | head -10' and save the skill list to ~/evo_skills_list.txt",
     ["exec_shell", "write_file"], "easy"),
    ("Check if the kernel-evolving API is running: 'curl -s http://localhost:8779/health'. Save the response to ~/evo_health.txt",
     ["exec_shell", "write_file"], "easy"),
    ("Run 'nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total --format=csv,noheader' and save GPU stats to ~/evo_gpu_stats.txt",
     ["exec_shell", "write_file"], "easy"),

    # ── Code generation + execution ────────────────────────────────────────────
    ("Write ~/evo_sort.py that sorts a list [5,2,8,1,9,3] and prints sorted and reversed versions. Run it and save output to ~/evo_sort_output.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),
    ("Write ~/evo_wordcount.py that counts words in a given string. Run it with 'kernel evolving is a self-evolving AI agent'. Save output to ~/evo_wordcount_output.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),
    ("Create ~/evo_env_report.py that prints all environment variables starting with 'KERNEL'. Run it and save to ~/evo_env_report.txt",
     ["write_file", "exec_shell", "write_file"], "medium"),

    # ── Long-form writing ───────────────────────────────────────────────────
    ("Write a README.md for the kernel-evolving project (5 sections: Overview, Features, Setup, Usage, Architecture) and save to ~/evo_readme.md",
     ["write_file"], "medium"),
    ("Write a technical blog post draft about self-evolving AI agents (intro, 3 sections, conclusion) and save to ~/evo_blog_draft.md",
     ["write_file"], "hard"),
    ("Create a weekly work log template at ~/evo_weekly_log.md with sections: Completed, In Progress, Blockers, Next Week",
     ["write_file"], "easy"),
]


# ── Generator ─────────────────────────────────────────────────────────────────

def run_task(task: str, provider, tools: list, workspace: str) -> tuple[list, str]:
    """Run infer_with_tools via the configured provider and collect steps."""
    steps = []

    def _step_cb(n, tool_name, args, result):
        steps.append({"tool": tool_name, "args": args, "result": str(result)[:500]})

    reply = provider.infer_with_tools(
        messages=[
            {"role": "system", "content": (
                "You are Evo, a precise AI agent. Complete tasks step by step using the tools available. "
                "Always write files when asked. Always run commands when asked. "
                "Never skip tool calls. Return a brief confirmation when done."
            )},
            {"role": "user", "content": task},
        ],
        tools=tools,
        workspace=workspace,
        max_steps=10,
        step_callback=_step_cb,
        call_type="task_inference",
    )
    return steps, reply


def critic_score(task: str, reply: str, steps: list, provider) -> float:
    """Quick critic pass. Returns 0.0–1.0."""
    try:
        step_summary = ", ".join(s["tool"] for s in steps) if steps else "none"
        prompt = (
            f"Rate this task execution 0.0–1.0.\n"
            f"Task: {task[:300]}\n"
            f"Tools used: {step_summary}\n"
            f"Reply: {reply[:400]}\n\n"
            f"Score criteria:\n"
            f"- 0.9–1.0: task fully completed, correct tools used, artifact on disk\n"
            f"- 0.7–0.9: mostly correct, minor issues\n"
            f"- 0.5–0.7: partial completion\n"
            f"- <0.5: failed or wrong approach\n\n"
            f"Respond with the score only (e.g. 0.85):"
        )
        from provider import get_provider as _get_prov
        raw = _get_prov().infer([{"role": "user", "content": prompt}], max_new_tokens=16, call_type="critic")
        m = re.search(r"\b(0\.\d+|1\.0|1)\b", raw)
        return float(m.group(1)) if m else 0.5
    except Exception as e:
        print(f"  [critic] error: {e}")
        return 0.5


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trajectories for drafter fine-tuning")
    parser.add_argument("--count", type=int, default=len(TASK_TEMPLATES),
                        help="Number of tasks to run (default: all)")
    parser.add_argument("--export", action="store_true", help="Export JSONL after generation")
    parser.add_argument("--min-score", type=float, default=0.7, help="Min critic score to save (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running")
    parser.add_argument("--task-filter", type=str, default="", help="Only run tasks containing this string")
    parser.add_argument("--provider", type=str, default="",
                        help="Force a specific provider (e.g. openai, local). Overrides config.")
    args = parser.parse_args()

    # Bootstrap
    import yaml
    config_path = _ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override provider if specified
    if args.provider:
        config.setdefault("providers", {})["task_inference"] = args.provider
        config["providers"]["critic"] = args.provider
        print(f"Provider override: task_inference={args.provider}")

    workspace = os.path.expanduser("~/.kernel-evolving/workspace")
    os.makedirs(workspace, exist_ok=True)

    from trajectory_collector import get_collector
    from tools import TOOLS
    from provider import get_provider

    col = get_collector(config)
    provider = get_provider()

    tasks = TASK_TEMPLATES[:args.count]
    if args.task_filter:
        tasks = [(t, e, d) for t, e, d in tasks if args.task_filter.lower() in t.lower()]

    if args.dry_run:
        print(f"Would run {len(tasks)} tasks:")
        for i, (task, expected, diff) in enumerate(tasks, 1):
            print(f"  {i:2d}. [{diff:6s}] {task[:80]}")
        return

    print(f"\n{'='*60}")
    print(f"Synthetic trajectory generator — {len(tasks)} tasks")
    print(f"Min critic score: {args.min_score}")
    print(f"{'='*60}\n")

    saved = 0
    skipped = 0
    failed = 0

    for i, (task, expected_tools, difficulty) in enumerate(tasks, 1):
        print(f"\n[{i:2d}/{len(tasks)}] [{difficulty:6s}] {task[:70]}")
        print(f"         Expected tools: {expected_tools}")
        t0 = time.time()
        try:
            steps, reply = run_task(task, provider, TOOLS, workspace)
            elapsed = time.time() - t0

            actual_tools = [s["tool"] for s in steps]
            score = critic_score(task, reply, steps, provider)

            print(f"         Tools used:     {actual_tools}")
            print(f"         Critic score:   {score:.2f}  ({elapsed:.1f}s)")
            print(f"         Reply:          {reply[:100]}")

            # Detect artifacts (files written in last 90s)
            now = time.time()
            artifacts = [
                f for f in Path(workspace).rglob("*")
                if f.is_file() and (now - f.stat().st_mtime) < 90
            ]
            # Also check ~/evo_* files
            home_evo = list(Path.home().glob("evo_*")) + list(Path.home().glob("evo_*/**/*"))
            artifacts += [str(p) for p in home_evo if p.is_file() and (now - p.stat().st_mtime) < 90]
            artifacts = list(set(str(a) for a in artifacts))

            verdict = "PASS" if score >= args.min_score else "FAIL"
            print(f"         Verdict:        {verdict}  artifacts={len(artifacts)}")

            # Reject error/OOM responses before saving — these corrupt the training data
            _error_markers = (
                "[model_server error]", "[model_client error]", "CUDA error",
                "out of memory", "cudaErrorMemoryAllocation", "(max steps reached)",
                "model_server timeout",
            )
            _is_error_reply = any(m in (reply or "") for m in _error_markers)
            if _is_error_reply:
                skipped += 1
                print(f"         ⏭  Skipped (error reply — not training data)")
                continue

            if score >= args.min_score and steps:
                provider_label = args.provider or config.get('providers', {}).get('task_inference', 'local')
                col.record(
                    task=task,
                    provider=f"synthetic/{difficulty}/{provider_label}",
                    model_name=config.get('providers', {}).get('models', {}).get(provider_label, provider_label),
                    call_type="task_inference",
                    tool_calls=steps,
                    final_reply=reply,
                    artifacts=artifacts,
                    critic_score=score,
                    critic_verdict=verdict,
                )
                saved += 1
                print(f"         ✅ Saved to DB")
            else:
                skipped += 1
                print(f"         ⏭  Skipped (score < {args.min_score})")

        except Exception as e:
            failed += 1
            print(f"         ❌ Error: {e}")

        # Brief pause between tasks to avoid model overload
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Done. Saved: {saved}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*60}\n")

    if args.export and saved > 0:
        print("Exporting JSONL...")
        path = col.export_jsonl(min_score=args.min_score)
        print(f"Exported → {path}")
        print(f"Ready for: uv run scripts/finetune_from_trajectories.py --dataset {path}")


if __name__ == "__main__":
    main()
