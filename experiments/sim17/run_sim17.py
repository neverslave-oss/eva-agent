#!/usr/bin/env python3
"""
sim17 — v5 Adapter Gap Validation
===================================
Tests the v5 adapter specifically against the gap areas it was trained on:
  Gap A: read_file + summarise/extract (20 tasks)
  Gap B: http_get + write_file (20 tasks)
  Gap F: vision / doc-retrieval (20 tasks, tested via PDF/read proxies)
  Gap G: voice-clone routing (20 tasks)
  Gap H: multimodal chains (10 tasks)

Phase 1 — Core I/O Baseline (T01–T08, same as sim14/sim16)
  - Verifies adapter didn't regress on core capabilities
  - Must pass 6/8 (allow 2 flaky)

Phase 2 — v5 Gap Probe (G01–G08)
  - Gap A: read + summarise
  - Gap B: http_get + extract field
  - Gap F: PDF/doc routing (proxied as file read)
  - Gap G: voice-clone routing
  - Compare pass rate vs sim16 (6/8 baseline)

Usage:
  python3 simulations/sim17/run_sim17.py [--adapter PATH] [--skip-phase1]
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
V5_ADAPTER = str(Path.home() / ".kernel-evolving/workspace/finetune_output_v5")
ROUTINE_SCRIPT = str(
    Path.home() / ".openclaw/workspace/routines/end-of-session/session_snapshot.py"
)
WORKSPACE = str(Path.home() / ".openclaw/workspace")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

OUT = Path.home() / "kernel-evo-notes" / "sim17"
OUT.mkdir(parents=True, exist_ok=True)


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


def _verify(path: str, min_chars: int = 50, must_contain: list = None) -> tuple:
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
# Phase 1 — Core I/O Baseline (T01–T08, same as sim16)
# ---------------------------------------------------------------------------

SIM14_TASKS = [
    {
        "id": "T01_history",
        "desc": "Conversation history: does model remember turn 1 in turn 3?",
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
        "desc": "Trajectory capture: task lands in artifact",
        "turns": [
            f"Write a one-paragraph description of what a LoRA adapter is and save it to {OUT}/T06_lora_desc.md",
        ],
        "artifact": f"{OUT}/T06_lora_desc.md",
        "min_chars": 100,
    },
    {
        "id": "T07_repo_list",
        "desc": "exec ls repos + save",
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
# Phase 2 — v5 Gap Probe (G01–G08)
# ---------------------------------------------------------------------------

# Seed files for gap tasks
def _setup_seeds():
    (OUT / "G01_seed_config.yaml").write_text(
        "model: gemma-4\nmax_tokens: 2048\ntemperature: 0.7\nadapter: v5\ndataset: clean_v5_merged\n"
    )
    (OUT / "G03_seed_code.py").write_text(
        "import os\nimport sys\n\ndef main():\n    print('Hello kernel-evolving')\n    return 0\n\nif __name__ == '__main__':\n    sys.exit(main())\n"
    )
    (OUT / "G05_seed_readme.md").write_text(
        "# kernel-evolving\n\nA self-improving agent training pipeline.\n\n## Features\n- Trajectory collection\n- LoRA fine-tuning\n- Sim evaluation\n\n## Usage\nbash scripts/train.sh\n"
    )


SIM17_GAP_TASKS = [
    # Gap A: read_file + summarise
    {
        "id": "G01_read_summarise",
        "desc": "Gap A: read YAML config + write structured summary",
        "setup": _setup_seeds,
        "message": f"Read {OUT}/G01_seed_config.yaml and write a structured summary of its contents (what model, what settings) to {OUT}/G01_summary.md. Be specific about the values.",
        "artifact": f"{OUT}/G01_summary.md",
        "min_chars": 100,
        "must_contain": ["gemma", "adapter"],
    },
    # Gap A: read + extract specific field
    {
        "id": "G02_read_extract",
        "desc": "Gap A: read file, extract specific value",
        "message": f"Read {OUT}/G01_seed_config.yaml and extract only the value of the 'dataset' field. Save just that value to {OUT}/G02_dataset_value.txt",
        "artifact": f"{OUT}/G02_dataset_value.txt",
        "min_chars": 5,
        "must_contain": ["clean_v5"],
    },
    # Gap A: read source file + describe imports
    {
        "id": "G03_read_imports",
        "desc": "Gap A: read Python file, list imports",
        "message": f"Read {OUT}/G03_seed_code.py and list all imported modules to {OUT}/G03_imports.txt",
        "artifact": f"{OUT}/G03_imports.txt",
        "min_chars": 5,
        "must_contain": ["os", "sys"],
    },
    # Gap B: http_get + extract field
    {
        "id": "G04_http_extract",
        "desc": "Gap B: http_get JSON + extract specific field",
        "message": f"Fetch https://httpbin.org/uuid and extract just the UUID value. Save it to {OUT}/G04_uuid.txt",
        "artifact": f"{OUT}/G04_uuid.txt",
        "min_chars": 10,
        "must_contain": ["-"],
    },
    # Gap B: http_get + write full response
    {
        "id": "G05_http_write",
        "desc": "Gap B: http_get + save full response",
        "message": f"Fetch https://httpbin.org/get and save the complete JSON response to {OUT}/G05_get_response.json",
        "artifact": f"{OUT}/G05_get_response.json",
        "min_chars": 100,
        "must_contain": ["url"],
    },
    # Gap F: doc retrieval / summarise README
    {
        "id": "G06_doc_summarise",
        "desc": "Gap F: read markdown doc + structured summary",
        "message": f"Read {OUT}/G05_seed_readme.md and write a summary that covers: (1) what the project does, (2) key features listed, (3) how to use it. Save to {OUT}/G06_readme_summary.md",
        "artifact": f"{OUT}/G06_readme_summary.md",
        "min_chars": 150,
        "must_contain": ["feature", "train"],
    },
    # Gap G: voice-clone routing (attempt, may fail gracefully)
    {
        "id": "G07_voice_routing",
        "desc": "Gap G: voice-clone skill routing (graceful fail ok)",
        "message": f"Use the voice-clone skill to generate speech for 'Hello from sim17' using sample /tmp/missing_sample_sim17.wav. If the sample is missing, explain the error and save your explanation to {OUT}/G07_voice_error.txt",
        "artifact": f"{OUT}/G07_voice_error.txt",
        "min_chars": 20,
        "must_contain": ["missing", "error", "sample", "file"],
    },
    # Gap H: multimodal chain — exec + read + write
    {
        "id": "G08_multimodal_chain",
        "desc": "Gap H: exec + read + write chain",
        "message": f"Run `git -C {WORKSPACE}/repositories/kernel-evolving log --oneline -5` to get recent commits, then read {OUT}/G05_seed_readme.md to get the project description, and finally write a combined 'project status' report to {OUT}/G08_status_report.md that includes both the recent commits and a one-line project description.",
        "artifact": f"{OUT}/G08_status_report.md",
        "min_chars": 150,
        "must_contain": ["commit", "kernel"],
    },
]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_phase1(adapter_path: str) -> tuple:
    print("\n" + "="*60)
    print("PHASE 1 — Core I/O Baseline (T01→T08 with v5 adapter)")
    print(f"Adapter: {adapter_path}")
    print("="*60)

    results = []
    for task in SIM14_TASKS:
        print(f"\n  [{task['id']}] {task['desc']}")
        t0 = time.time()
        replies = []
        chat_id = f"sim17_p1_{task['id']}_{RUN_TS}"

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
                min_chars=task.get("min_chars", 50),
                must_contain=task.get("must_contain"),
            )

        status = "✅" if passed else "❌"
        print(f"    {status} [{elapsed}s] {artifact_msg}")
        results.append({
            "id": task["id"], "desc": task["desc"],
            "passed": passed, "elapsed_s": elapsed, "artifact_msg": artifact_msg,
        })

    passed_count = sum(1 for r in results if r["passed"])
    threshold = 6
    print(f"\n  PHASE 1 RESULT: {passed_count}/8 (threshold: {threshold}/8)")
    return passed_count >= threshold, results


def run_phase2(adapter_path: str) -> dict:
    print("\n" + "="*60)
    print("PHASE 2 — v5 Gap Probe (G01–G08)")
    print("="*60)

    # Run all setup once
    try:
        _setup_seeds()
    except Exception as e:
        print(f"  ⚠️  Seed setup error: {e}")

    results = []
    for task in SIM17_GAP_TASKS:
        print(f"\n  [{task['id']}] {task['desc']}")
        t0 = time.time()

        # Per-task setup
        setup = task.get("setup")
        if setup and setup is not _setup_seeds:
            try:
                setup()
            except Exception as e:
                print(f"    ⚠️ setup error: {e}")

        body = {"message": task["message"], "adapter_path": adapter_path}
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

        status = "✅" if passed else "❌"
        print(f"    {status} [{elapsed}s] {artifact_msg}")
        results.append({
            "id": task["id"], "desc": task["desc"],
            "passed": passed, "elapsed_s": elapsed, "artifact_msg": artifact_msg,
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n  PHASE 2 RESULT: {passed_count}/{total}")
    return {"results": results, "passed": passed_count, "total": total}


def _save_results(phase1_results, phase2_results, phase1_passed, adapter_path):
    p1_count = sum(1 for r in phase1_results if r["passed"]) if phase1_results else 0
    out = {
        "run_ts": RUN_TS,
        "sim": "sim17",
        "adapter_path": adapter_path,
        "phase1": {
            "passed": phase1_passed,
            "score": f"{p1_count}/8" if phase1_results else "skipped",
            "tasks": phase1_results,
        },
        "phase2": phase2_results,
    }
    out_path = RESULTS_DIR / f"sim17_{RUN_TS}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n📄 Results saved: {out_path}")

    print("\n" + "="*60)
    print("FINAL SUMMARY — sim17")
    print("="*60)
    if phase1_results:
        print(f"Phase 1 (core baseline): {p1_count}/8 {'✅' if phase1_passed else '❌'}")
    if phase2_results:
        p2 = phase2_results.get("passed", 0)
        t2 = phase2_results.get("total", 8)
        print(f"Phase 2 (v5 gap probe):  {p2}/{t2}")
        print(f"  vs sim16 baseline:     6/8")
    print("="*60)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="sim17 — v5 adapter gap validation")
    parser.add_argument("--adapter", default=V5_ADAPTER, help="Path to v5 adapter")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip phase 1 baseline")
    args = parser.parse_args()

    adapter_path = args.adapter

    # Ensure API is up
    health = _get("/health")
    if health.get("status") != "ok":
        print(f"❌ API not healthy: {health}")
        sys.exit(1)
    print(f"✅ API healthy — VRAM free: {health.get('vram_free_mb')} MB")
    print(f"   Using adapter: {adapter_path}")

    phase1_passed = True
    phase1_results = []

    if not args.skip_phase1:
        try:
            phase1_passed, phase1_results = run_phase1(adapter_path)
        except Exception as e:
            print(f"❌ Phase 1 error: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)
    else:
        print("⏭️  Phase 1 skipped.")

    phase2_results = run_phase2(adapter_path)

    out_path = _save_results(
        phase1_results=phase1_results,
        phase2_results=phase2_results,
        phase1_passed=phase1_passed,
        adapter_path=adapter_path,
    )

    # Also emit to stdout for log capture
    p1_count = sum(1 for r in phase1_results if r["passed"]) if phase1_results else 0
    p2 = phase2_results.get("passed", 0)
    t2 = phase2_results.get("total", 8)
    print(f"\nsim17 complete: Phase1={p1_count}/8  Phase2={p2}/{t2}  Results={out_path}")


if __name__ == "__main__":
    main()
