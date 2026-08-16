#!/usr/bin/env python3
"""
run_sim_eval.py — Sim 14 task evaluator for adapter evaluation.

Runs the exact same 8 tasks from simulations/sim14/run_sim14.py against
a configurable port (default 8779) and saves results to:
    ~/.kernel-evolving/workspace/artifacts/eval_results/<label>_<timestamp>.json

Usage:
  python3 scripts/run_sim_eval.py [--port 8779] [--label baseline]
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

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Sim 14/15 evaluator for adapter evaluation")
parser.add_argument("--port", type=int, default=8779, help="API port to target (default 8779)")
parser.add_argument("--label", default="baseline", help="Run label for output filename (default 'baseline')")
parser.add_argument("--sim", choices=["14", "15", "both"], default="14",
                    help="Which sim to run: 14 (default), 15, or both")
parser.add_argument("--replay-count", type=int, default=15,
                    help="(sim15 only) number of training prompts to replay")
parser.add_argument("--training-data", default=str(
    Path.home() / ".kernel-evolving/workspace/artifacts/trajectories/training_combined_clean.jsonl"
), help="(sim15 only) path to training JSONL")
args = parser.parse_args()

API = f"http://localhost:{args.port}"
TIMEOUT = 300
EVAL_RESULTS_DIR = Path.home() / ".kernel-evolving/workspace/artifacts/eval_results"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Try kernel-evolving workspace first, fall back to Olly's media dir (read-only, not pollution)
DIGEST_PATH = next(
    (p for p in [
        Path.home() / ".kernel-evolving/workspace/docs/digest_today.json",
        Path.home() / ".openclaw/media/digest_today.json",
    ] if p.exists()),
    Path.home() / ".kernel-evolving/workspace/docs/digest_today.json",  # non-existent → skip
)

# Temp output directory for artifacts
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_BASE = Path.home() / ".kernel-evolving/workspace/artifacts/eval_runs" / f"eval_{args.label}_{RUN_TS}"


# ---------------------------------------------------------------------------
# Helpers (copied from sim14/run_sim14.py exactly)
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    url = API + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read())


def _verify(path: str, min_chars: int = 200, must_contain: list = None) -> tuple:
    p = Path(path)
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


def run_task(task_id: str, desc: str, messages: list, artifact: str = None,
             min_chars: int = 200, must_contain: list = None, session_id: str = None,
             timeout: int = TIMEOUT) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task_id}] {desc}")

    replies = []
    for i, msg in enumerate(messages):
        try:
            body = {"message": msg}
            if session_id:
                body["chat_id"] = session_id
            result = _post("/message", body, timeout=timeout)
            reply = result.get("reply", "")
            replies.append(reply)
            print(f"    turn {i+1}: {reply[:100]!r}")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"    💥 ERROR turn {i+1}: {e}")
            return {"id": task_id, "desc": desc, "ts": ts, "elapsed_s": elapsed,
                    "error": str(e), "passed": False, "artifact_msg": str(e)}

    elapsed = round(time.time() - t0, 1)
    final_reply = replies[-1] if replies else ""

    artifact_ok, artifact_msg = True, "n/a"
    if artifact:
        ap = Path(artifact)
        if not ap.exists() and final_reply.strip():
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_text(final_reply, encoding="utf-8")
        artifact_ok, artifact_msg = _verify(artifact, min_chars, must_contain)

    passed = artifact_ok if artifact else bool(final_reply.strip())
    icon = "✅" if passed else "❌"
    print(f"    {icon} {elapsed}s | {artifact_msg}")

    return {
        "id": task_id,
        "desc": desc,
        "ts": ts,
        "elapsed_s": elapsed,
        "replies": [r[:300] for r in replies],
        "artifact": artifact,
        "artifact_ok": artifact_ok,
        "artifact_msg": artifact_msg,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main evaluation — exact same 8 tasks as sim14/run_sim14.py (T01–T08)
# ---------------------------------------------------------------------------

def main():
    out_dir = OUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    session = f"eval-{args.label}-{RUN_TS}"

    # Unload local model if running against a cloud provider to free VRAM
    try:
        import urllib.request as _ureq2, json as _json2
        _cur_prov_data = _json2.load(_ureq2.urlopen(f"{API}/provider", timeout=3))
        _cur_ti = _cur_prov_data.get("routing", {}).get("task_inference", {}).get("provider", "local")
        if _cur_ti != "local":
            print(f"  [eval] provider={_cur_ti} — requesting model unload to free VRAM")
            _unload_body = _json2.dumps({"task_inference": _cur_ti}).encode()
            _unload_req = _ureq2.Request(f"{API}/provider/set",
                                          data=_unload_body,
                                          headers={"Content-Type": "application/json"})
            _ureq2.urlopen(_unload_req, timeout=5)
    except Exception as _ue:
        print(f"  [eval] WARNING: pre-eval unload check failed: {_ue}")

    print(f"\n{'='*60}")
    print(f"  Sim 14 Eval — {args.label}")
    print(f"  port:    {args.port}")
    print(f"  run_id:  {RUN_TS}")
    try:
        v = _get("/version")
        print(f"  version: {v.get('version', '?')}")
    except Exception:
        print("  version: ?")
    print(f"{'='*60}")

    results = []

    # T01 — Conversation history
    results.append(run_task(
        "T01_history",
        "Conversation history: does Evo remember turn 1 in turn 3?",
        messages=[
            "My name is Fabio and I have 10 cats.",
            "What is 2 + 2?",
            "What did I tell you about myself in my first message?",
        ],
        session_id=session + "-T01",
        min_chars=10,
    ))
    t01 = results[-1]
    final = t01.get("replies", [""])[-1].lower()
    t01["passed"] = any(k in final for k in ["fabio", "10", "cat"])
    t01["artifact_msg"] = "context recalled" if t01["passed"] else "context NOT recalled"
    icon = "✅" if t01["passed"] else "❌"
    print(f"    → {icon} history check: {t01['artifact_msg']}")

    # T02 — write_file reliability
    results.append(run_task(
        "T02_write_file",
        "write_file: multi-paragraph artifact >500 chars",
        messages=[
            f"Write a detailed README for a project called 'kernel-evolving' — at least 5 sections "
            f"with proper markdown headings. Save it to {out_dir}/T02_readme.md"
        ],
        artifact=str(out_dir / "T02_readme.md"),
        min_chars=500,
        must_contain=["#"],
        session_id=session + "-T02",
    ))

    # T03 — exec_shell + write_file chain
    results.append(run_task(
        "T03_exec_write",
        "exec_shell + write_file: run command, save output",
        messages=[
            f"Run 'python3 --version && echo hostname=$(hostname)' and save the output to "
            f"{out_dir}/T03_sysinfo.txt"
        ],
        artifact=str(out_dir / "T03_sysinfo.txt"),
        min_chars=5,
        must_contain=["python"],
        session_id=session + "-T03",
    ))

    # T04 — read_file + reason + write
    seed_path = out_dir / "T04_seed.txt"
    seed_path.write_text(
        "Project: kernel-evolving\nGoal: self-evolving local AI agent\nVersion: 1.13.0\n"
        "Key features: memory seeker, trajectory collection, voice mode, evolution dashboard\n",
        encoding="utf-8"
    )
    results.append(run_task(
        "T04_read_reason_write",
        "read_file + reason + write: full I/O loop",
        messages=[
            f"Read the file {seed_path}, extract the key features listed, and write a one-paragraph "
            f"product description to {out_dir}/T04_product.md"
        ],
        artifact=str(out_dir / "T04_product.md"),
        min_chars=100,
        session_id=session + "-T04",
    ))

    # T05 — fetch + summarise + save
    results.append(run_task(
        "T05_fetch_save",
        "http_get + write_file: fetch URL, summarise, save",
        messages=[
            f"Fetch https://example.com, extract the page title and first paragraph of text, "
            f"and save a summary to {out_dir}/T05_fetch.txt"
        ],
        artifact=str(out_dir / "T05_fetch.txt"),
        min_chars=20,
        session_id=session + "-T05",
    ))

    # T06 — Trajectory capture
    results.append(run_task(
        "T06_trajectory_capture",
        "Trajectory capture: PASS task should land in DB",
        messages=[
            f"Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number, "
            f"add a docstring, and save it to {out_dir}/T06_fibonacci.py"
        ],
        artifact=str(out_dir / "T06_fibonacci.py"),
        min_chars=80,
        must_contain=["def fibonacci"],
        session_id=session + "-T06",
    ))

    # T07 — Digest pipeline
    if DIGEST_PATH.exists():
        results.append(run_task(
            "T07_digest_pipeline",
            "Digest pipeline: read JSON, filter, summarise, save",
            messages=[
                f"read_file {DIGEST_PATH} — it is a JSON list of news items. "
                f"Pick the 3 most developer-relevant AI items (no arXiv). "
                f"write_file {out_dir}/T07_digest_summary.md with a 2-sentence note per item."
            ],
            artifact=str(out_dir / "T07_digest_summary.md"),
            min_chars=200,
            session_id=session + "-T07",
            timeout=300,
        ))
    else:
        print(f"\n  [T07_digest_pipeline] SKIP — no digest file at {DIGEST_PATH}")
        results.append({"id": "T07_digest_pipeline", "passed": None, "artifact_msg": "skipped — no digest"})

    # T08 — Blog post draft
    results.append(run_task(
        "T08_blog_post",
        "Blog post draft: structured content >500 chars with headings",
        messages=[
            f"write_file {out_dir}/T08_blog.md — content: a blog post titled "
            f"'How I built a self-evolving local AI agent' with markdown sections: "
            f"## Introduction, ## The Problem, ## The Architecture, ## Key Features, ## What's Next. "
            f"No URL to fetch — write all content from your own knowledge."
        ],
        artifact=str(out_dir / "T08_blog.md"),
        min_chars=500,
        must_contain=["#"],
        session_id=session + "-T08",
        timeout=600,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.label} (port {args.port})")
    print(f"{'='*60}")
    passed_count = 0
    total = 0
    for r in results:
        if r.get("passed") is None:
            icon = "⏭ "
        elif r["passed"]:
            icon = "✅"
            passed_count += 1
            total += 1
        else:
            icon = "❌"
            total += 1
        msg = r.get("artifact_msg", "")
        elapsed = r.get("elapsed_s", "")
        print(f"  {icon} {r['id']:30s} {msg} {elapsed}s")

    verdict = "✅ PASS" if passed_count == total else ("⚠️ PARTIAL" if passed_count > 0 else "❌ FAIL")
    print(f"\n  {verdict} — {passed_count}/{total} tasks passed")

    payload = {
        "label": args.label,
        "port": args.port,
        "run_id": RUN_TS,
        "ts": datetime.now(timezone.utc).isoformat(),
        "passed": passed_count,
        "total": total,
        "verdict": verdict,
        "tasks": results,
    }

    out_file = EVAL_RESULTS_DIR / f"{args.label}_{RUN_TS}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  Results: {out_file}")
    print(f"  Artifacts: {out_dir}\n")

    # Exit 0 if all pass, 1 if any fail
    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    # Dispatch to sim14, sim15, or both
    if args.sim in ("15", "both"):
        _sim15_path = str(Path(__file__).resolve().parent.parent / "simulations" / "sim15" / "run_sim15.py")
        import subprocess as _sp
        _cmd = [
            "python3", _sim15_path,
            "--port", str(args.port),
            "--label", args.label,
            "--training-data", args.training_data,
            "--replay-count", str(args.replay_count),
        ]
        print(f"[run_sim_eval] Delegating to sim15: {' '.join(_cmd)}")
        _result = _sp.run(_cmd)
        if args.sim == "15":
            sys.exit(_result.returncode)
        # For "both" fall through to sim14
    main()
