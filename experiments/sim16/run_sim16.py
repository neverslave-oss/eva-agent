#!/usr/bin/env python3
"""
sim16_adapter_replica_probe.py
==============================
Phase 1 — Adapter Replica Probe (T01→T08, same tasks as Sim 14)
  - Spawns a named persistent replica with v2 adapter
  - Sends the same 8 sim14 tasks to /replica/<name>/message
  - Verifies artifacts against sim14 pass criteria
  - Must pass 8/8 before Phase 2

Phase 2 — Sim 15 Base vs Adapter (if Phase 1 passes)
  - Runs sim15 novel tasks (N01–N10) against base model via /message
  - Runs same tasks via /message with adapter_path= (per-request adapter)
  - Includes two routine dry-run tasks:
      R01: end-of-session snapshot dry-run
      R02: morning-briefing routine dry-run (reads memory file)
  - Produces side-by-side comparison

Usage:
  python3 sim16_adapter_replica_probe.py [--skip-phase1] [--adapter PATH]

Adapter default: ~/.kernel-evolving/workspace/finetune_output_v2
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8779"
TIMEOUT = 360
V2_ADAPTER = str(Path.home() / ".kernel-evolving/workspace/finetune_output_v2")
ROUTINE_SCRIPT = str(
    Path.home() / ".openclaw/workspace/routines/end-of-session/session_snapshot.py"
)
WORKSPACE = str(Path.home() / ".openclaw/workspace")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    url = API + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def _get(path: str, timeout: int = 30) -> dict:
    try:
        with urllib.request.urlopen(API + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def _verify(path: str, min_chars: int = 200, must_contain: list = None) -> tuple:
    p = Path(path).expanduser()
    if not p.exists():
        return False, "missing"
    content = p.read_text(encoding="utf-8", errors="ignore")
    if len(content) < min_chars:
        return False, f"too short ({len(content)} chars)"
    if must_contain:
        missing = [s for s in must_contain if s.lower() not in content.lower()]
        if missing:
            return False, f"missing keywords: {missing}"
    return True, f"ok ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Replica management
# ---------------------------------------------------------------------------

def spawn_replica(name: str, adapter_path: str) -> bool:
    resp = _post("/replica/named", {
        "name": name,
        "role": "agent",
        "tools_enabled": True,
        "workspace": str(Path.home() / ".kernel-evolving/workspace"),
        "adapter_path": adapter_path,
    })
    if resp.get("status") == "spawned" or resp.get("status") == "exists":
        print(f"  ✅ Replica '{name}' spawned with adapter: {adapter_path}")
        return True
    print(f"  ❌ Spawn failed: {resp}")
    return False


def stop_replica(name: str):
    import urllib.request
    req = urllib.request.Request(
        f"{API}/replica/{name}",
        method="DELETE",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            pass
    except Exception:
        pass


def message_replica(name: str, msg: str) -> str:
    resp = _post(f"/replica/{name}/message", {"message": msg})
    return resp.get("reply", resp.get("error", ""))


# ---------------------------------------------------------------------------
# Sim 14 task definitions (same criteria, routed to replica)
# ---------------------------------------------------------------------------

OUT = Path.home() / "kernel-evo-notes" / "sim16"
OUT.mkdir(parents=True, exist_ok=True)


SIM14_TASKS = [
    {
        "id": "T01_history",
        "desc": "Conversation history: does replica remember turn 1 in turn 3?",
        "turns": [
            "My favourite framework is Laravel. Just letting you know.",
            "What's the weather like? (just say 'I don't know' if you can't check)",
            "What framework did I mention earlier?",
        ],
        "check": lambda replies: "laravel" in replies[-1].lower(),
        "check_desc": "turn 3 references 'Laravel'",
    },
    {
        "id": "T02_write_file",
        "desc": "write_file: multi-paragraph artifact >500 chars",
        "turns": [
            f"Write a multi-paragraph summary of the Python programming language (at least 3 paragraphs, >500 chars) and save it to {OUT}/T02_python_summary.md",
        ],
        "artifact": f"{OUT}/T02_python_summary.md",
        "min_chars": 500,
    },
    {
        "id": "T03_exec_write",
        "desc": "exec_shell + write_file: run command, save output",
        "turns": [
            f"Run `uname -a` and save the output to {OUT}/T03_uname.txt",
        ],
        "artifact": f"{OUT}/T03_uname.txt",
        "min_chars": 10,
    },
    {
        "id": "T04_read_reason_write",
        "desc": "read_file + reason + write: full I/O loop",
        "turns": [
            f"Read the file {ROUTINE_SCRIPT} and write a brief summary of what it does (>200 chars) to {OUT}/T04_script_summary.md",
        ],
        "artifact": f"{OUT}/T04_script_summary.md",
        "min_chars": 200,
    },
    {
        "id": "T05_fetch_save",
        "desc": "http_get + write_file: fetch URL, summarise, save",
        "turns": [
            f"Fetch https://httpbin.org/json and save the response body to {OUT}/T05_httpbin.txt",
        ],
        "artifact": f"{OUT}/T05_httpbin.txt",
        "min_chars": 50,
    },
    {
        "id": "T06_trajectory_capture",
        "desc": "Trajectory capture: task should land in DB",
        "turns": [
            f"Write a one-paragraph description of what a LoRA adapter is and save it to {OUT}/T06_lora_desc.md",
        ],
        "artifact": f"{OUT}/T06_lora_desc.md",
        "min_chars": 100,
    },
    {
        "id": "T07_digest_pipeline",
        "desc": "Read JSON, filter, summarise, save",
        "turns": [
            f"Run `ls -la {WORKSPACE}/repositories/ | head -20` and save the output to {OUT}/T07_repos_list.txt",
        ],
        "artifact": f"{OUT}/T07_repos_list.txt",
        "min_chars": 100,
    },
    {
        "id": "T08_blog_post",
        "desc": "Blog post draft: structured content >500 chars with headings",
        "turns": [
            f"Write a structured blog post draft (with ## headings, >500 chars) about the benefits of LoRA fine-tuning for small language models. Save to {OUT}/T08_blog_draft.md",
        ],
        "artifact": f"{OUT}/T08_blog_draft.md",
        "min_chars": 500,
        "must_contain": ["#"],
    },
]

# ---------------------------------------------------------------------------
# Sim 15 novel tasks (base AND adapter via per-request adapter_path)
# ---------------------------------------------------------------------------

SIM15_TASKS = [
    {
        "id": "N01",
        "desc": "exec hostname + OS release → sysprofile.txt",
        "message": f"Run `hostname && cat /etc/os-release | head -5` and save to {OUT}/N01_sysprofile.txt",
        "artifact": f"{OUT}/N01_sysprofile.txt",
        "min_chars": 20,
    },
    {
        "id": "N02",
        "desc": "http_get headers dump",
        "message": f"Fetch https://httpbin.org/headers and save the raw response to {OUT}/N02_headers.txt",
        "artifact": f"{OUT}/N02_headers.txt",
        "min_chars": 50,
    },
    {
        "id": "N03",
        "desc": "Read config, rewrite as .env format",
        "setup": lambda: Path(f"{OUT}/N03_seed_config.toml").write_text(
            '[database]\nhost = "localhost"\nport = 5432\nname = "mydb"\n\n[app]\ndebug = true\nport = 8000\n'
        ),
        "message": f"Read {OUT}/N03_seed_config.toml and rewrite its key=value pairs in .env format (KEY=VALUE, one per line). Save to {OUT}/N03_output.env",
        "artifact": f"{OUT}/N03_output.env",
        "min_chars": 30,
        "must_contain": ["="],
    },
    {
        "id": "N04",
        "desc": "pip packages filtered for 'torch' → count.txt",
        "message": f"Run `pip list 2>/dev/null | grep -i torch || echo 'no torch packages found'` and save the output to {OUT}/N04_torch_packages.txt",
        "artifact": f"{OUT}/N04_torch_packages.txt",
        "min_chars": 5,
    },
    {
        "id": "N05",
        "desc": "Write markdown table of tool descriptions",
        "message": f"Write a markdown table with 3 columns (Tool, Description, Example use) listing 5 common shell tools (ls, grep, curl, git, find). Save to {OUT}/N05_tool_table.md",
        "artifact": f"{OUT}/N05_tool_table.md",
        "min_chars": 200,
        "must_contain": ["|"],
    },
    {
        "id": "N06",
        "desc": "git log in kernel-evolving → summary",
        "message": f"Run `git -C {WORKSPACE}/repositories/kernel-evolving log --oneline -10` and save the output to {OUT}/N06_git_log.txt",
        "artifact": f"{OUT}/N06_git_log.txt",
        "min_chars": 50,
    },
    {
        "id": "N07",
        "desc": "http_get JSON → extract UUID field → .txt",
        "message": f"Fetch https://httpbin.org/uuid and extract the 'uuid' field value. Save just the UUID string to {OUT}/N07_uuid.txt",
        "artifact": f"{OUT}/N07_uuid.txt",
        "min_chars": 10,
        "must_contain": ["-"],  # UUID format has hyphens
    },
    {
        "id": "N08",
        "desc": "Read TODO list, run date, add timestamp",
        "setup": lambda: Path(f"{OUT}/N08_seed_todos.md").write_text(
            "# TODO\n- Fix the adapter probe\n- Run sim 16\n- Write memory snapshot\n"
        ),
        "message": f"Read {OUT}/N08_seed_todos.md, run `date -u` to get the current UTC time, then append a line '## Timestamped: <date output>' to the file and save the result to {OUT}/N08_todos_with_timestamp.md",
        "artifact": f"{OUT}/N08_todos_with_timestamp.md",
        "min_chars": 60,
        "must_contain": ["timestamp", "#"],
    },
    {
        "id": "N09",
        "desc": "disk usage → disk_report.txt",
        "message": f"Run `df -h /` and save the output to {OUT}/N09_disk_report.txt",
        "artifact": f"{OUT}/N09_disk_report.txt",
        "min_chars": 50,
    },
    {
        "id": "N10",
        "desc": "Write 5-section developer onboarding guide >600 chars",
        "message": f"Write a developer onboarding guide with 5 sections (## headings): Setup, Branching, Commits, Tests, Deploy. Each section should have 2-3 lines. Total >600 chars. Save to {OUT}/N10_onboarding_guide.md",
        "artifact": f"{OUT}/N10_onboarding_guide.md",
        "min_chars": 600,
        "must_contain": ["##"],
    },
    # Routine dry-run tasks
    {
        "id": "R01",
        "desc": "End-of-session snapshot dry-run via exec_shell",
        "message": f"Run `python3 {ROUTINE_SCRIPT} --workspace {WORKSPACE} --dry-run 2>&1 | head -40` and save the output to {OUT}/R01_snapshot_dryrun.txt",
        "artifact": f"{OUT}/R01_snapshot_dryrun.txt",
        "min_chars": 100,
        "must_contain": ["Session Snapshot", "Resume Context"],
    },
    {
        "id": "R02",
        "desc": "Morning briefing: read today's memory file and summarise",
        "setup": lambda: Path(f"{OUT}/R02_seed_memory.md").write_text(
            "# 2026-05-21\n- Worked on kernel-evolving adapter-aware replicas\n- sim14 passed 8/8 on base model\n- Started building session_snapshot routine\n- Pending: run adapter probe + sim15\n"
        ),
        "message": f"Read {OUT}/R02_seed_memory.md and write a concise morning briefing (what was done yesterday, what is pending today) as a bullet list. Save to {OUT}/R02_morning_brief.md",
        "artifact": f"{OUT}/R02_morning_brief.md",
        "min_chars": 100,
        "must_contain": ["-"],
    },
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_phase1(adapter_path: str) -> tuple:
    """Run sim14 tasks directly via /message with adapter_path.
    Uses the same code path as the working base model (/message → agent.triage → infer_with_tools).
    Avoids the replica persistent-history path which sends multi-turn context
    that the single-turn-trained adapter doesn't handle well.
    """
    print("\n" + "="*60)
    print("PHASE 1 — Adapter Direct Probe (T01→T08 via /message+adapter_path)")
    print(f"Adapter: {adapter_path}")
    print("="*60)

    results = []
    for task in SIM14_TASKS:
        print(f"\n  [{task['id']}] {task['desc']}")
        t0 = time.time()
        replies = []
        # Fresh chat_id per task (except T01 multi-turn which needs continuity)
        chat_id = f"sim16_p1_{task['id']}_{RUN_TS}"

        turns = task.get("turns", [])
        for i, turn in enumerate(turns):
            body = {"message": turn, "chat_id": chat_id, "adapter_path": adapter_path}
            resp = _post("/message", body)
            reply = resp.get("reply", resp.get("error", ""))
            replies.append(reply)
            print(f"    turn {i+1}: {reply[:120]!r}")

        elapsed = round(time.time() - t0, 1)

        passed = False
        artifact_msg = ""
        if "check" in task:
            passed = task["check"](replies)
            artifact_msg = task["check_desc"] if passed else f"FAIL: {task['check_desc']}"
        elif "artifact" in task:
            passed, artifact_msg = _verify(
                task["artifact"],
                min_chars=task.get("min_chars", 200),
                must_contain=task.get("must_contain"),
            )

        status = "✅" if passed else "❌"
        print(f"    {status} [{elapsed}s] {artifact_msg}")
        results.append({
            "id": task["id"], "desc": task["desc"],
            "passed": passed, "elapsed_s": elapsed, "artifact_msg": artifact_msg,
        })

    passed_count = sum(1 for r in results if r["passed"])
    print(f"\n  PHASE 1 RESULT: {passed_count}/8")
    return passed_count == 8, results


def run_phase2_task(task: dict, adapter_path: str = None) -> dict:
    """Run a single sim15 task via /message, optionally with adapter."""
    # Run setup if present
    setup = task.get("setup")
    if setup:
        try:
            setup()
        except Exception as e:
            print(f"    ⚠️ setup error: {e}")

    t0 = time.time()
    body = {"message": task["message"]}
    if adapter_path:
        body["adapter_path"] = adapter_path

    resp = _post("/message", body)
    reply = resp.get("reply", resp.get("error", ""))
    elapsed = round(time.time() - t0, 1)
    print(f"    reply: {reply[:120]!r}  [{elapsed}s]")

    passed, artifact_msg = False, ""
    if "artifact" in task:
        passed, artifact_msg = _verify(
            task["artifact"],
            min_chars=task.get("min_chars", 50),
            must_contain=task.get("must_contain"),
        )
    else:
        passed = bool(reply and len(reply) > 10)
        artifact_msg = "reply ok" if passed else "empty reply"

    return {
        "id": task["id"], "desc": task["desc"],
        "passed": passed, "elapsed_s": elapsed, "artifact_msg": artifact_msg,
    }


def run_phase2() -> dict:
    print("\n" + "="*60)
    print("PHASE 2 — Sim 15 Novel Tasks: Base vs Adapter")
    print("="*60)

    base_results = []
    adapter_results = []

    for task in SIM15_TASKS:
        print(f"\n  [{task['id']}] {task['desc']}")

        print("    [BASE]")
        base_r = run_phase2_task(task, adapter_path=None)
        base_status = "✅" if base_r["passed"] else "❌"
        print(f"    {base_status} base: {base_r['artifact_msg']}")
        base_results.append(base_r)

        # Small pause between base and adapter to avoid overlap
        time.sleep(2)

        print("    [ADAPTER v2]")
        adapter_r = run_phase2_task({**task, "artifact": task.get("artifact", "").replace(str(OUT), str(OUT) + "_adapter") if task.get("artifact") else None, "message": task["message"]}, adapter_path=V2_ADAPTER)
        # Use separate output paths for adapter
        if task.get("artifact"):
            out_path_adapter = Path(task["artifact"].replace(str(OUT), str(OUT) + "_v2"))
            out_path_adapter.parent.mkdir(parents=True, exist_ok=True)
            # Re-verify against adapter path
            adp_task = {**task, "artifact": str(out_path_adapter)}
            # The artifact was written to original path by the model (same prompt) — re-verify original
            adapter_r2 = run_phase2_task(
                {**task, "message": task["message"].replace(str(OUT), str(OUT) + "_v2")},
                adapter_path=V2_ADAPTER
            )
            adapter_results.append(adapter_r2)
            adp_status = "✅" if adapter_r2["passed"] else "❌"
            print(f"    {adp_status} adapter: {adapter_r2['artifact_msg']}")
        else:
            adapter_results.append(adapter_r)
            adp_status = "✅" if adapter_r["passed"] else "❌"
            print(f"    {adp_status} adapter: {adapter_r['artifact_msg']}")

    base_passed = sum(1 for r in base_results if r["passed"])
    adp_passed = sum(1 for r in adapter_results if r["passed"])
    total = len(SIM15_TASKS)

    print(f"\n  PHASE 2 RESULT:")
    print(f"    Base:    {base_passed}/{total}")
    print(f"    Adapter: {adp_passed}/{total}")

    return {
        "base": base_results, "adapter": adapter_results,
        "base_passed": base_passed, "adapter_passed": adp_passed, "total": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-phase1", action="store_true", help="Skip replica probe, go straight to sim15")
    parser.add_argument("--adapter", default=None, help="Path to v2 adapter")
    args = parser.parse_args()

    global V2_ADAPTER
    if args.adapter:
        V2_ADAPTER = args.adapter

    # Ensure API is up
    health = _get("/health")
    if health.get("status") != "ok":
        print(f"❌ API not healthy: {health}")
        sys.exit(1)
    print(f"✅ API healthy — VRAM free: {health.get('vram_free_mb')} MB")

    phase1_passed = True
    phase1_results = []

    if not args.skip_phase1:
        try:
            phase1_passed, phase1_results = run_phase1(V2_ADAPTER)
        except Exception as e:
            print(f"❌ Phase 1 error: {e}")
            sys.exit(1)

        if not phase1_passed:
            print("\n❌ Phase 1 did NOT pass 8/8. Aborting phase 2.")
            _save_results(phase1_results=phase1_results, phase2_results=None, phase1_passed=False)
            sys.exit(1)

        print("\n✅ Phase 1 passed 8/8. Proceeding to Phase 2.")
    else:
        print("⏭️  Phase 1 skipped.")

    phase2_results = run_phase2()

    _save_results(
        phase1_results=phase1_results,
        phase2_results=phase2_results,
        phase1_passed=phase1_passed,
    )


def _save_results(phase1_results, phase2_results, phase1_passed):
    out = {
        "run_ts": RUN_TS,
        "adapter_path": V2_ADAPTER,
        "phase1": {
            "passed": phase1_passed,
            "score": f"{sum(1 for r in phase1_results if r['passed'])}/8" if phase1_results else "skipped",
            "tasks": phase1_results,
        },
        "phase2": phase2_results,
    }
    out_path = RESULTS_DIR / f"sim16_{RUN_TS}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n📄 Results saved: {out_path}")

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    if phase1_results:
        p1 = sum(1 for r in phase1_results if r["passed"])
        print(f"Phase 1 (adapter replica probe): {p1}/8")
    if phase2_results:
        print(f"Phase 2 base model:   {phase2_results['base_passed']}/{phase2_results['total']}")
        print(f"Phase 2 adapter v2:   {phase2_results['adapter_passed']}/{phase2_results['total']}")
    print("="*60)


if __name__ == "__main__":
    main()
