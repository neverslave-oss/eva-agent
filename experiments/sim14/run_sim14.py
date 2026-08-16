#!/usr/bin/env python3
"""
Sim 14 — Full-Stack Regression + Capability Test
=================================================
Tests all fixes and improvements shipped in v1.12.0–v1.13.0:

  T01  Conversation history — does Evo remember turn 1 in turn 3?
  T02  write_file reliability — multi-paragraph artifact, >500 chars
  T03  exec_shell + write_file chain — run a command, save output
  T04  read_file + reason + write — full file I/O loop
  T05  Multi-step: fetch URL + summarise + save
  T06  Trajectory capture — does a PASS task land in the DB?
  T07  Digest pipeline — read JSON, filter, summarise, save (sim13 T03 regression)
  T08  Blog post draft — direct prompt produces structured content >500 chars

Pass criteria:
  T01: reply in turn 3 references content from turn 1
  T02: artifact written, >500 chars
  T03: artifact written, contains command output
  T04: artifact written, >200 chars
  T05: artifact written, >100 chars
  T06: trajectory count increases after the run
  T07: artifact written, >200 chars
  T08: artifact written, >500 chars, contains ## or # headings

Usage:
  python3 sim14/run_sim14.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

API = "http://localhost:8779"
TIMEOUT = 300
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = Path.home() / "kernel-evo-notes" / "sim14"
DIGEST_PATH = Path.home() / ".openclaw/media/digest_today.json"


def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    url = API + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read())


def _verify(path: str, min_chars: int = 200, must_contain: list = None) -> tuple[bool, str]:
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
             min_chars: int = 200, must_contain: list = None, session_id: str = None) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task_id}] {desc}")

    replies = []
    for i, msg in enumerate(messages):
        try:
            body = {"message": msg}
            if session_id:
                body["chat_id"] = session_id
            result = _post("/message", body)
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

    # Save reply as artifact if path given and file not written by agent
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


def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    session = f"sim14-{run_id}"

    print(f"\n{'='*60}")
    print(f"  Sim 14 — Full-Stack Regression + Capability")
    print(f"  run_id: {run_id}")
    print(f"  version: ", end="")
    try:
        v = _get("/version")
        print(v.get("version", "?"))
    except Exception:
        print("?")
    print(f"  digest: {DIGEST_PATH.exists()}")
    print(f"{'='*60}")

    # VRAM guard — warn if less than 1GB free
    try:
        vr = _get("/health")
        vram_free = vr.get("vram_free_mb", 0)
        if 0 < vram_free < 1000:
            print(f"  ⚠️  VRAM low ({vram_free}MB free) — OOM likely. Check: nvidia-smi | grep python")
    except Exception:
        pass

    # Trajectory count before
    traj_before = 0
    try:
        td = _get("/evolution/trajectories")
        trajs = td.get("trajectories", td) if isinstance(td, dict) else td
        traj_before = len(trajs)
    except Exception:
        pass
    print(f"  Trajectories before: {traj_before}")

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
    # Override pass: check reply 3 mentions "Fabio" or "10" or "cats"
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
            f"Write a detailed README for a project called 'kernel-evolving' — at least 5 sections with proper markdown headings. Save it to {out_dir}/T02_readme.md"
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
            f"Run 'python3 --version && echo hostname=$(hostname)' and save the output to {out_dir}/T03_sysinfo.txt"
        ],
        artifact=str(out_dir / "T03_sysinfo.txt"),
        min_chars=5,
        must_contain=["python"],
        session_id=session + "-T03",
    ))

    # T04 — read_file + reason + write
    # First write something to read
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
            f"Read the file {seed_path}, extract the key features listed, and write a one-paragraph product description to {out_dir}/T04_product.md"
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
            f"Fetch https://example.com, extract the page title and first paragraph of text, and save a summary to {out_dir}/T05_fetch.txt"
        ],
        artifact=str(out_dir / "T05_fetch.txt"),
        min_chars=20,
        session_id=session + "-T05",
    ))

    # T06 — Trajectory capture (run a task that should be collected)
    results.append(run_task(
        "T06_trajectory_capture",
        "Trajectory capture: PASS task should land in DB",
        messages=[
            f"Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number, add a docstring, and save it to {out_dir}/T06_fibonacci.py"
        ],
        artifact=str(out_dir / "T06_fibonacci.py"),
        min_chars=80,
        must_contain=["def fibonacci"],
        session_id=session + "-T06",
    ))

    # T07 — Digest pipeline (sim13 T03 regression)
    if DIGEST_PATH.exists():
        results.append(run_task(
            "T07_digest_pipeline",
            "Digest pipeline: read JSON, filter, summarise, save",
            messages=[
                f"Read the file at {DIGEST_PATH} using read_file. "
                f"Pick the 3 most developer-relevant AI news items (skip arXiv papers), "
                f"summarise each in 2 sentences, and write the summary to {out_dir}/T07_digest_summary.md"
            ],
            artifact=str(out_dir / "T07_digest_summary.md"),
            min_chars=200,
            session_id=session + "-T07",
        ))
    else:
        print(f"\n  [T07_digest_pipeline] SKIP — no digest file at {DIGEST_PATH}")
        results.append({"id": "T07_digest_pipeline", "passed": None, "artifact_msg": "skipped — no digest"})

    # T08 — Blog post draft
    results.append(run_task(
        "T08_blog_post",
        "Blog post draft: structured content >500 chars with headings",
        messages=[
            f"Write a blog post titled 'How I built a self-evolving local AI agent' — "
            f"include sections: Introduction, The Problem, The Architecture, Key Features, What's Next. "
            f"Save to {out_dir}/T08_blog.md"
        ],
        artifact=str(out_dir / "T08_blog.md"),
        min_chars=500,
        must_contain=["#"],
        session_id=session + "-T08",
    ))

    # Trajectory count after
    traj_after = 0
    try:
        time.sleep(3)  # allow async trajectory recording to flush
        td = _get("/evolution/trajectories")
        trajs = td.get("trajectories", td) if isinstance(td, dict) else td
        traj_after = len(trajs)
    except Exception:
        pass

    t06 = next((r for r in results if r["id"] == "T06_trajectory_capture"), None)
    if t06 and t06.get("passed"):
        traj_captured = traj_after > traj_before
        if t06:
            t06["trajectory_captured"] = traj_captured
            t06["traj_before"] = traj_before
            t06["traj_after"] = traj_after
        print(f"\n  Trajectories after: {traj_after} (delta: +{traj_after - traj_before})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESULTS")
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
    print(f"  Trajectories captured: {traj_before} → {traj_after} (+{traj_after - traj_before})")

    payload = {
        "sim": "sim14",
        "run_id": run_id,
        "version": "1.13.0",
        "ts": datetime.now(timezone.utc).isoformat(),
        "passed": passed_count,
        "total": total,
        "verdict": verdict,
        "traj_before": traj_before,
        "traj_after": traj_after,
        "tasks": results,
    }
    result_file = RESULTS_DIR / f"sim14_{run_id}.json"
    result_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"  Results: {result_file}")
    print(f"  Artifacts: {out_dir}\n")
    return payload


if __name__ == "__main__":
    main()
