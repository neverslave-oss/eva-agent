#!/usr/bin/env python3
"""
Sim 7 — ADR-008 Critic Replica Pipeline Validation.

Previous sims tested: did Evo ACQUIRE a skill?
Sim 7 tests:         does the ADR-008 critic replica pipeline produce real results?

Tests the multi-stage replica pipeline (writer→critic) introduced in ADR-008.
Focus: result quality and pipeline chaining — not evolution acquisition speed.

Tasks:
  T01: Basic writer→critic pipeline (BDI explanation)
  T02: Code analysis + review pipeline
  T03: Three-stage planner→writer→critic
  T04: Gap analysis → skill synthesis pipeline
  T05: Evolution history reflection + strategy pipeline

Results written to sim7/results/ for paper_v8 analysis.
"""
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8779"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEOUT = 300  # longer timeout — CPU inference is slow (~60-90s per 2-stage call) — model may need to load


def _post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}", data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def _get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read())


TASKS = [
    {
        "id": "T01_bdi_pipeline",
        "description": "Basic writer→critic pipeline (BDI agent explanation)",
        "stages": [
            {
                "name": "writer",
                "role": "custom",
                "brief": "You are a concise AI systems technical writer. Write clearly and accurately.",
                "task": "Write a 3-sentence explanation of the BDI (Belief-Desire-Intention) agent model.",
                "tools_enabled": False,
            },
            {
                "name": "critic",
                "role": "custom",
                "brief": "You are a rigorous quality critic for AI systems documentation. Check for technical accuracy.",
                "task": "Review the draft above. Flag any inaccuracies or missing concepts. Reply: PASS or FAIL with one annotation.",
                "tools_enabled": False,
                "input_from": "writer",
            },
        ],
        "verify": lambda r: "writer" in r.get("results", {}) and "critic" in r.get("results", {}),
    },
    {
        "id": "T02_code_review_pipeline",
        "description": "Code analysis + review pipeline",
        "stages": [
            {
                "name": "analyst",
                "role": "custom",
                "brief": "You are a Python code analyst.",
                "task": "Describe what this Python function does and identify any bugs:\ndef divide(a, b):\n    return a / b",
                "tools_enabled": False,
            },
            {
                "name": "reviewer",
                "role": "custom",
                "brief": "You are a senior engineer doing code review.",
                "task": "Given the analysis above, write a 2-point code review with specific improvement suggestions.",
                "tools_enabled": False,
                "input_from": "analyst",
            },
        ],
        "verify": lambda r: "analyst" in r.get("results", {}) and "reviewer" in r.get("results", {}),
    },
    {
        "id": "T03_three_stage",
        "description": "Three-stage planner→writer→critic pipeline",
        "stages": [
            {
                "name": "planner",
                "role": "custom",
                "brief": "You are a task planner. Output a 3-step numbered plan only.",
                "task": "Plan how to evaluate whether a local AI agent is self-improving over time.",
                "tools_enabled": False,
            },
            {
                "name": "writer",
                "role": "custom",
                "brief": "You are a technical writer. Expand a plan into prose.",
                "task": "Turn the plan above into a 2-paragraph methodology description.",
                "tools_enabled": False,
                "input_from": "planner",
            },
            {
                "name": "critic",
                "role": "custom",
                "brief": "You are a research methodology critic.",
                "task": "Review the methodology above. Is it sound? What is missing? One paragraph.",
                "tools_enabled": False,
                "input_from": "writer",
            },
        ],
        "verify": lambda r: all(
            k in r.get("results", {}) for k in ["planner", "writer", "critic"]
        ),
    },
    {
        "id": "T04_gap_synthesis_pipeline",
        "description": "Gap analysis → skill synthesis pipeline",
        "stages": [
            {
                "name": "gap_analyst",
                "role": "custom",
                "brief": "You are an AI agent capability analyst.",
                "task": (
                    "Given that Kernel-Evo has skills for: web-scraping, PDF conversion, "
                    "code review, and translation — identify the top 3 capability gaps "
                    "for a developer using it daily."
                ),
                "tools_enabled": False,
            },
            {
                "name": "synthesiser",
                "role": "custom",
                "brief": "You are a skill synthesiser. Given identified gaps, propose skill names.",
                "task": (
                    "For each gap identified above, propose a skill name (kebab-case) "
                    "and a one-line description of what it does."
                ),
                "tools_enabled": False,
                "input_from": "gap_analyst",
            },
        ],
        "verify": lambda r: (
            "gap_analyst" in r.get("results", {}) and
            "synthesiser" in r.get("results", {})
        ),
    },
    {
        "id": "T05_evolution_reflection",
        "description": "Evolution history reflection + strategy pipeline",
        "stages": [
            {
                "name": "historian",
                "role": "custom",
                "brief": (
                    "You are a historian of this AI system's evolution. "
                    "Sims 1-5 tested keyword→semantic→verification→enriched-ecosystem strategies."
                ),
                "task": (
                    "Summarise the key insight from each simulation condition "
                    "(Sim 1 through Sim 5) in one sentence each."
                ),
                "tools_enabled": False,
            },
            {
                "name": "strategist",
                "role": "custom",
                "brief": "You are a research strategist identifying next steps.",
                "task": (
                    "Based on the simulation history above, what should Sim 8 focus on "
                    "to advance the research? Give 3 specific recommendations."
                ),
                "tools_enabled": False,
                "input_from": "historian",
            },
        ],
        "verify": lambda r: (
            "historian" in r.get("results", {}) and
            "strategist" in r.get("results", {})
        ),
    },
]


def run_task(task: dict) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"[{task['id']}] {task['description']}")
    t_start = time.time()
    try:
        result = _post("/replica/pipeline", {"stages": task["stages"]})
    except Exception as e:
        elapsed = round(time.time() - t_start, 1)
        print(f"  ERROR ({elapsed}s): {e}")
        return {"id": task["id"], "description": task["description"], "error": str(e), "ts": ts, "elapsed_s": elapsed}

    elapsed = round(time.time() - t_start, 1)
    verified = task["verify"](result)
    status_str = result.get("status", "?")
    stages_done = list(result.get("results", {}).keys())

    print(f"  status={status_str} | stages={stages_done} | verified={verified} | {elapsed}s")
    for stage_name, stage_result in result.get("results", {}).items():
        preview = str(stage_result)[:150].replace("\n", " ")
        print(f"    [{stage_name}]: {preview}...")

    return {
        "id": task["id"],
        "description": task["description"],
        "ts": ts,
        "elapsed_s": elapsed,
        "status": status_str,
        "stages_completed": stages_done,
        "verified": verified,
        "results": result.get("results", {}),
    }


def main():
    print(f"\n{'#'*60}")
    print("SIM 7 — ADR-008 Critic Replica Pipeline Validation")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Tasks: {len(TASKS)}")
    print(f"{'#'*60}")

    try:
        health = _get("/health")
        print(f"\n[health] {health}")
        version = _get("/version")
        print(f"[version] {version}")
    except Exception as e:
        print(f"[health] ERROR: {e} — aborting")
        return

    all_results = []
    for task in TASKS:
        r = run_task(task)
        all_results.append(r)

    # Summary
    verified_count = sum(1 for r in all_results if r.get("verified"))
    error_count = sum(1 for r in all_results if "error" in r)
    print(f"\n{'#'*60}")
    print(f"SUMMARY: {verified_count}/{len(TASKS)} verified | {error_count} errors")
    print(f"{'#'*60}")
    for r in all_results:
        icon = "✅" if r.get("verified") else ("💥" if "error" in r else "❌")
        stages = r.get("stages_completed", [])
        elapsed = r.get("elapsed_s", "?")
        print(f"  {icon} {r['id']:35s} stages={stages} ({elapsed}s)")

    # Save results
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sim7_{ts_str}.json"
    out_path.write_text(json.dumps({
        "sim": "sim7",
        "ts": datetime.now(timezone.utc).isoformat(),
        "adr": "ADR-008",
        "test_focus": "critic_replica_pipeline",
        "summary": {
            "total": len(TASKS),
            "verified": verified_count,
            "errors": error_count,
            "pass_rate": f"{verified_count}/{len(TASKS)}",
        },
        "results": all_results,
    }, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
