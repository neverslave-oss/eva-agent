#!/usr/bin/env python3
"""
Sim 8 — Real Task Completion Validation.

Previous sims tested: did Evo ACQUIRE a skill? (Sims 1-5)
                      did the pipeline RESPOND? (Sim 7)
Sim 8 tests:          did Evo produce a REAL ARTIFACT on disk?

Each task has a concrete verifiable output (file on disk, command output, etc.).
Tasks are designed to exercise the full chain:
  - skill exec dispatch (kernel-doc-retrieval /markdown)
  - infer_with_tools + write_file
  - micro-planner multi-step decomposition
  - evolution gap → Tier 2 synthesis → real use of synthesised skill
  - routine execution producing output

Sim 8 submits via /message (same path as a real user), NOT /evolution/trigger.
This tests the complete triage → (plan) → execute → verify loop.

Results written to sim8/results/ for paper_v9.
"""
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API       = "http://localhost:8779"
WORKSPACE = Path.home() / "kernel-evo-notes" / "sim8"
WORKSPACE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEOUT_PER_TASK = 360   # 6 min per task — covers PDF conversion (~220s) + inference


def _post(path, body, timeout=TIMEOUT_PER_TASK):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{API}{path}", data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read())


def _verify_file(path: str) -> tuple[bool, str]:
    """Check a file exists on disk and is non-empty."""
    p = Path(os.path.expanduser(path))
    if not p.exists():
        return False, f"file not found: {path}"
    size = p.stat().st_size
    if size == 0:
        return False, f"file is empty: {path}"
    return True, f"exists, {size} bytes"


def _verify_contains(path: str, keyword: str) -> tuple[bool, str]:
    """Check file exists and contains a keyword."""
    ok, msg = _verify_file(path)
    if not ok:
        return ok, msg
    content = Path(os.path.expanduser(path)).read_text(errors="replace")
    if keyword.lower() in content.lower():
        return True, f"contains '{keyword}' ({len(content)} chars)"
    return False, f"file exists but missing '{keyword}'"


TASKS = [
    # ── T01: Skill exec dispatch — kernel-doc-retrieval /markdown ─────────────
    {
        "id": "T01_doc_to_markdown",
        "description": "kernel-doc-retrieval exec dispatch — convert PDF to Markdown",
        "message": f"/markdown ~/.openclaw/workspace/publications/kernel-paper/kernel_paper_v2.pdf",
        "verify": lambda reply: _verify_contains(
            "~/.kernel/workspace/docs/kernel_paper_v2.md", "Kernel"
        ),
        "artifact": "~/.kernel/workspace/docs/kernel_paper_v2.md",
        "tier": "skill-exec",
    },

    # ── T02: write_file tool — produce a Python script artifact ───────────────
    {
        "id": "T02_write_python_script",
        "description": "write_file tool — produce ~/kernel-evo-notes/sim8/hello_evo.py",
        "message": (
            "Use write_file to create ~/kernel-evo-notes/sim8/hello_evo.py with this content:\n"
            "#!/usr/bin/env python3\n"
            "# Written by Kernel Evo via write_file tool\n"
            "print('Hello from Kernel Evo!')\n"
            "print('Sim 8 — task completion validation')\n"
        ),
        "verify": lambda reply: _verify_contains(
            "~/kernel-evo-notes/sim8/hello_evo.py", "Kernel Evo"
        ),
        "artifact": "~/kernel-evo-notes/sim8/hello_evo.py",
        "tier": "infer_with_tools",
    },

    # ── T03: exec_shell — run a command and capture output ────────────────────
    {
        "id": "T03_shell_output",
        "description": "exec_shell — run skill_inspector and save output to file",
        "message": (
            "Run the command: python3 ~/kernel-evo-notes/skill_inspector.py "
            "and save the full output to ~/kernel-evo-notes/sim8/skills_table.txt using write_file."
        ),
        "verify": lambda reply: _verify_file("~/kernel-evo-notes/sim8/skills_table.txt"),
        "artifact": "~/kernel-evo-notes/sim8/skills_table.txt",
        "tier": "infer_with_tools",
    },

    # ── T04: multi-step (micro-planner) — fetch + summarise + save ────────────
    {
        "id": "T04_fetch_summarise_save",
        "description": "micro-planner — fetch arXiv page, extract title+abstract, save summary",
        "message": "http_get https://fabiopacifici.com/how-to-write-an-agent-brief-that-actually-works then write_file ~/kernel-evo-notes/sim8/web-summary.md with 3 bullet point summary",
        "verify": lambda reply: _verify_contains(
            "~/kernel-evo-notes/sim8/web-summary.md", "-"
        ),
        "artifact": "~/kernel-evo-notes/sim8/web-summary.md",
        "tier": "micro-planner",
    },

    # ── T05: read + analyse + write — self-knowledge task ─────────────────────
    {
        "id": "T05_self_assessment",
        "description": "read evolution DB + write self-assessment report",
        "message": (
            "Open ~/.kernel-evolving/workspace/thoughts/2026-05-11.md and count "
            "how many entries are gap_reflection, self_improvement, curiosity, retrospective. "
            "Save a report with those counts to ~/kernel-evo-notes/sim8/thought-analysis.md"
        ),
        "verify": lambda reply: _verify_contains(
            "~/kernel-evo-notes/sim8/thought-analysis.md", "gap"
        ),
        "artifact": "~/kernel-evo-notes/sim8/thought-analysis.md",
        "tier": "infer_with_tools",
    },
]


def run_task(task: dict) -> dict:
    ts_start = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n{'='*64}")
    print(f"[{task['id']}] {task['description']}")
    print(f"  Tier: {task['tier']}")

    reply = ""
    error = None
    try:
        result = _post("/message", {"message": task["message"], "chat_id": f"sim8-{task['id']}"})
        reply  = result.get("reply", "")
    except Exception as e:
        error = str(e)
        print(f"  ❌ ERROR: {error}")

    elapsed = round(time.time() - t0, 1)

    # Verify artifact on disk
    artifact_ok, artifact_msg = task["verify"](reply) if not error else (False, "task errored")

    status = "✅ PASS" if artifact_ok else "❌ FAIL"
    print(f"  {status} | {elapsed}s | artifact: {artifact_msg}")
    if reply:
        print(f"  Reply preview: {reply[:120].replace(chr(10),' ')}")

    return {
        "id":          task["id"],
        "description": task["description"],
        "tier":        task["tier"],
        "ts":          ts_start,
        "elapsed_s":   elapsed,
        "artifact":    task.get("artifact"),
        "artifact_ok": artifact_ok,
        "artifact_msg":artifact_msg,
        "error":       error,
        "reply_len":   len(reply),
        "reply_head":  reply[:300],
    }


def main():
    print(f"\n{'#'*64}")
    print("SIM 8 — Real Task Completion Validation")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Output dir: {WORKSPACE}")
    print(f"{'#'*64}")

    try:
        health = _get("/health")
        version = _get("/version")
        print(f"\n[health] skills={health['skills']} routines={health['routines']} replicas={health['active_replicas']}")
        print(f"[version] {version['version']}")
        # Enable SIM_MODE: disables exec_shell approval gate for clean trajectory collection
        _post("/sim/mode", {"enabled": True}, timeout=5)
        print("[sim] SIM_MODE enabled — exec_shell approval gate bypassed")
    except Exception as e:
        print(f"[health] ERROR: {e} — aborting")
        return

    all_results = []
    for task in TASKS:
        r = run_task(task)
        all_results.append(r)

    # Summary
    passed  = sum(1 for r in all_results if r["artifact_ok"])
    failed  = sum(1 for r in all_results if not r["artifact_ok"] and not r["error"])
    errored = sum(1 for r in all_results if r["error"])
    total_t = sum(r["elapsed_s"] for r in all_results)

    print(f"\n{'#'*64}")
    print(f"SUMMARY: {passed}/{len(TASKS)} artifacts produced | {errored} errors | {round(total_t/60,1)} min total")
    print(f"{'#'*64}")
    for r in all_results:
        icon = "✅" if r["artifact_ok"] else ("💥" if r["error"] else "❌")
        print(f"  {icon} {r['id']:<30} tier={r['tier']:<20} {r['elapsed_s']}s")
        if not r["artifact_ok"]:
            print(f"       → {r['artifact_msg']}")

    ts_str  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sim8_{ts_str}.json"
    out_path.write_text(json.dumps({
        "sim": "sim8",
        "ts":  datetime.now(timezone.utc).isoformat(),
        "focus": "real_task_completion",
        "adrs_active": ["ADR-004","ADR-005","ADR-006","ADR-007","ADR-008","ADR-009","ADR-010","ADR-011","ADR-012"],
        "summary": {
            "total":    len(TASKS),
            "passed":   passed,
            "failed":   failed,
            "errored":  errored,
            "pass_rate": f"{passed}/{len(TASKS)}",
            "total_elapsed_min": round(total_t / 60, 1),
        },
        "results": all_results,
    }, indent=2))
    print(f"\nResults → {out_path}")
    try:
        _post("/sim/mode", {"enabled": False}, timeout=5)
        print("[sim] SIM_MODE disabled")
    except Exception:
        pass


if __name__ == "__main__":
    main()
