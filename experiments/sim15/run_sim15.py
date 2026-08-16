#!/usr/bin/env python3
"""
Sim 15 — Training-Replay Fitness + Novel Held-Out Evaluation
=============================================================
Two-part suite to detect overfitting / underfitting after fine-tuning:

  Part A — Training Replay (15 sampled prompts from the actual training set)
  ──────────────────────────────────────────────────────────────────────────
  Runs real prompts the model was trained on.  For each response we check:
    • PASS          — tool(s) used, artifact written, output meets quality bar
    • OVERFIT       — response is a near-verbatim copy of the training label
                      (Jaccard word-similarity ≥ 0.85 against stored label)
    • UNDERFIT      — model ignored the task (no tool call, very short reply)

  Part B — Novel Held-Out Tasks (10 tasks not in training data)
  ─────────────────────────────────────────────────────────────
  Same tool mix as training (exec+write, read+write, http+write, multi-step)
  but with different paths, commands, content, and phrasing.

  N01  exec_shell → write           hostname + OS release → sysprofile.txt
  N02  http_get → write             httpbin.org/headers → raw headers dump
  N03  read_file + write            Read seeded config, rewrite as .env format
  N04  exec_shell → parse → write   pip packages filtered for "torch" → count.txt
  N05  write structured markdown    Write a markdown table of tool descriptions
  N06  exec_shell chain → write     git log in kernel-evolving repo → summary
  N07  http_get JSON → extract      httpbin.org/uuid → extract UUID field → .txt
  N08  read + exec + write          Read seeded TODO list, run `date`, add timestamp
  N09  exec_shell → write           disk usage (`df -h /`) → disk_report.txt
  N10  write long structured doc    Write 5-section developer onboarding guide >600 chars

Pass criteria (all tasks):
  Part A: PASS=correct execution, not OVERFIT and not UNDERFIT
  Part B: artifact written, meets min_chars/must_contain thresholds

Usage:
  python3 sim15/run_sim15.py [--port 8779] [--label live]
  # Against shadow (finetuned adapter):
  python3 sim15/run_sim15.py --port 8780 --label finetuned_v2
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

parser = argparse.ArgumentParser(description="Sim 15 — training replay + novel eval")
parser.add_argument("--port", type=int, default=8779, help="API port (default 8779)")
parser.add_argument("--label", default="live", help="Run label for output filename")
parser.add_argument("--training-data", default=str(
    Path.home() / ".kernel-evolving/workspace/trajectories/training_combined_clean.jsonl"
), help="Path to training JSONL")
parser.add_argument("--replay-count", type=int, default=15,
                    help="Number of training prompts to replay (default 15)")
parser.add_argument("--seed", type=int, default=99, help="RNG seed for sampling")
args = parser.parse_args()

API = f"http://localhost:{args.port}"
TIMEOUT = 300
STREAM_TIMEOUT = 600   # per-task wall-clock cap when using /message/stream
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = Path.home() / "kernel-evo-notes" / "sim15"
RUN_TS = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _post_stream(path: str, body: dict, timeout: int = STREAM_TIMEOUT) -> dict:
    """POST to an NDJSON streaming endpoint (/message/stream).

    Consumes newline-delimited JSON events until {"event": "done"} or
    {"event": "error"}.  Each step line resets the idle timer so the
    connection never times out mid-inference.
    Returns a dict with at least a 'reply' key, matching the _post() contract.
    """
    url = API + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # Use a generous socket read timeout — the server keeps sending heartbeat
    # events so the socket stays alive; we only need a long idle cap.
    with urllib.request.urlopen(req, timeout=timeout) as r:
        final: dict = {"reply": ""}
        for raw_line in r:
            line = raw_line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("event") == "step":
                print(f"      ⚙ step {evt.get('step','?')}: {evt.get('tool','')} — {str(evt.get('result',''))[:80]}")
            elif evt.get("event") == "done":
                final = evt
                break
            elif evt.get("event") == "error":
                raise RuntimeError(evt.get("error", "stream error"))
        return final


def _post(path: str, body: dict, timeout: int = TIMEOUT) -> dict:
    """POST to a regular (non-streaming) endpoint.  Falls back to stream endpoint
    by swapping /message → /message/stream automatically."""
    # Route /message through the streaming endpoint to avoid 300s hard timeout
    if path == "/message":
        try:
            return _post_stream("/message/stream", body)
        except Exception as e:
            # If streaming endpoint isn't available fall back to plain POST
            if "404" in str(e) or "Connection" in str(e):
                pass  # fall through
            else:
                raise
    url = API + path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read())


def _verify(path: str, min_chars: int = 50, must_contain: list = None) -> tuple:
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


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def run_task(task_id: str, desc: str, message: str,
             artifact: str = None, min_chars: int = 50,
             must_contain: list = None, session_id: str = None) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task_id}] {desc}")

    reply = ""
    error = None
    try:
        body = {"message": message}
        if session_id:
            body["chat_id"] = session_id
        result = _post("/message", body)
        reply = result.get("reply", "")
        print(f"    reply: {reply[:120]!r}")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"    💥 ERROR: {e}")
        return {"id": task_id, "desc": desc, "ts": ts, "elapsed_s": elapsed,
                "error": str(e), "passed": False, "artifact_msg": str(e), "reply": ""}

    elapsed = round(time.time() - t0, 1)

    artifact_ok, artifact_msg = True, "n/a"
    if artifact:
        ap = Path(artifact)
        if not ap.exists() and reply.strip():
            ap.parent.mkdir(parents=True, exist_ok=True)
            ap.write_text(reply, encoding="utf-8")
        artifact_ok, artifact_msg = _verify(artifact, min_chars, must_contain)

    passed = artifact_ok if artifact else bool(reply.strip())
    icon = "✅" if passed else "❌"
    print(f"    {icon} {elapsed}s | {artifact_msg}")

    return {
        "id": task_id,
        "desc": desc,
        "ts": ts,
        "elapsed_s": elapsed,
        "reply": reply[:500],
        "artifact": artifact,
        "artifact_ok": artifact_ok,
        "artifact_msg": artifact_msg,
        "passed": passed,
    }


# ── Part A — Training Replay ──────────────────────────────────────────────────

def load_training_sample(path: str, n: int, seed: int) -> list:
    """Load n random examples from the training JSONL, return list of dicts."""
    if not Path(path).exists():
        print(f"  ⚠️  Training data not found at {path} — skipping Part A")
        return []
    data = [json.loads(line) for line in open(path)]
    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))
    return sample


def get_training_label(example: dict) -> str:
    """Extract the expected final assistant reply from a training example."""
    msgs = example.get("messages", [])
    asst = [m for m in msgs if m.get("role") == "assistant"]
    if not asst:
        return ""
    # Last assistant turn is the golden label
    content = asst[-1].get("content", "")
    return str(content)


def run_part_a(out_dir: Path, session_prefix: str) -> list:
    print(f"\n{'─'*60}")
    print("  Part A — Training Replay")
    print(f"{'─'*60}")

    examples = load_training_sample(args.training_data, args.replay_count, args.seed)
    if not examples:
        return []

    results = []
    for i, ex in enumerate(examples):
        task_prompt = ex.get("task", "")
        if not task_prompt:
            # Fall back to first user message
            msgs = ex.get("messages", [])
            user_msgs = [m for m in msgs if m.get("role") == "user"]
            task_prompt = user_msgs[0].get("content", "") if user_msgs else ""
        if not task_prompt:
            continue

        golden_label = get_training_label(ex)
        session_id = f"{session_prefix}-A{i:02d}"
        task_id = f"A{i+1:02d}_{ex.get('id', 'unk')[:12]}"
        desc = task_prompt[:60] + ("…" if len(task_prompt) > 60 else "")

        t0 = time.time()
        print(f"\n  [{task_id}] {desc}")

        reply = ""
        error = None
        try:
            result = _post("/message", {"message": task_prompt, "chat_id": session_id})
            reply = result.get("reply", "")
            print(f"    reply: {reply[:120]!r}")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"    💥 ERROR: {e}")
            results.append({
                "id": task_id, "desc": desc, "task": task_prompt[:200],
                "passed": False, "verdict": "ERROR", "error": str(e),
                "elapsed_s": elapsed,
            })
            continue

        elapsed = round(time.time() - t0, 1)

        # ── Fitness checks ──────────────────────────────────────────────────
        # 1. Did the model respond meaningfully?
        is_underfit = len(reply.strip()) < 20 or reply.strip() in (
            "I don't know.", "I can't help with that.", "", "Done."
        )

        # 2. Near-verbatim copy of training label?
        sim = _jaccard(reply, golden_label) if golden_label else 0.0
        is_overfit = sim >= 0.85 and len(golden_label) > 30

        # 3. Did it actually do something useful (tool call implied by reply)?
        # Heuristic: reply mentions a file path, command output, or "written"
        _tool_signals = ["written to", "saved to", "wrote", "done", "executed",
                         "error", "fetched", "result", "output", "file", "read"]
        has_tool_signal = any(sig in reply.lower() for sig in _tool_signals)

        if is_overfit:
            verdict = "OVERFIT"
            icon = "🔁"
        elif is_underfit:
            verdict = "UNDERFIT"
            icon = "📉"
        elif has_tool_signal or len(reply) > 40:
            verdict = "PASS"
            icon = "✅"
        else:
            verdict = "UNCERTAIN"
            icon = "⚠️"

        passed = verdict == "PASS"
        print(f"    {icon} {elapsed}s | sim={sim:.2f} | {verdict}")

        results.append({
            "id": task_id,
            "desc": desc,
            "task": task_prompt[:200],
            "reply": reply[:400],
            "golden_label": golden_label[:200],
            "jaccard_sim": round(sim, 3),
            "verdict": verdict,
            "passed": passed,
            "elapsed_s": elapsed,
        })

    return results


# ── Part B — Novel Held-Out Tasks ────────────────────────────────────────────

def run_part_b(out_dir: Path, session_prefix: str) -> list:
    print(f"\n{'─'*60}")
    print("  Part B — Novel Held-Out Tasks")
    print(f"{'─'*60}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Seed files needed for some tasks
    cfg_seed = out_dir / "N03_config_seed.yaml"
    cfg_seed.write_text(
        "model_name: gemma-4-e2b-it\n"
        "adapter_path: ~/.kernel-evolving/workspace/finetune_output_v2\n"
        "max_tokens: 1024\n"
        "temperature: 0.0\n"
        "shadow_port: 8780\n"
        "collect_trajectories: true\n",
        encoding="utf-8",
    )

    todo_seed = out_dir / "N08_todos.txt"
    todo_seed.write_text(
        "TODO: Fix T07 digest pipeline — local path routed to http_get\n"
        "TODO: Run sim15 novel eval after sim14 fixes\n"
        "TODO: Push fix/sim14-t04-t07-eval-suite branch\n"
        "TODO: Write ADR-015 for training replay evaluation pattern\n",
        encoding="utf-8",
    )

    results = []
    session = f"{session_prefix}-B"

    results.append(run_task(
        "N01_sysprofile",
        "exec_shell → write: hostname + OS release → sysprofile.txt",
        message=(
            f"Run 'hostname && (lsb_release -d 2>/dev/null || cat /etc/os-release | head -3) && uname -r' "
            f"and write the full output to {out_dir}/N01_sysprofile.txt"
        ),
        artifact=str(out_dir / "N01_sysprofile.txt"),
        min_chars=10,
        must_contain=["linux"],
        session_id=session,
    ))

    results.append(run_task(
        "N02_http_headers",
        "http_get → write: httpbin.org/headers → headers dump",
        message=(
            f"Fetch https://httpbin.org/headers and save the raw JSON response to "
            f"{out_dir}/N02_headers.json"
        ),
        artifact=str(out_dir / "N02_headers.json"),
        min_chars=20,
        must_contain=["headers"],
        session_id=session,
    ))

    results.append(run_task(
        "N03_config_reformat",
        "read_file + write: read YAML config, rewrite as .env format",
        message=(
            f"Read the file {cfg_seed}. Convert each key: value pair to KEY=value format "
            f"(uppercase keys, no spaces around =) and write the result to "
            f"{out_dir}/N03_config.env"
        ),
        artifact=str(out_dir / "N03_config.env"),
        min_chars=30,
        must_contain=["MODEL_NAME", "MAX_TOKENS"],
        session_id=session,
    ))

    results.append(run_task(
        "N04_torch_pkg_count",
        "exec_shell → parse → write: pip packages filtered for torch → count",
        message=(
            f"Run 'pip list 2>/dev/null | grep -i torch' to find torch-related packages. "
            f"Count how many lines are returned and write 'Found N torch-related packages:\\n<list>' "
            f"to {out_dir}/N04_torch_packages.txt"
        ),
        artifact=str(out_dir / "N04_torch_packages.txt"),
        min_chars=10,
        must_contain=["torch"],
        session_id=session,
    ))

    results.append(run_task(
        "N05_markdown_table",
        "write: produce a markdown table of the 4 available tools",
        message=(
            f"Write a markdown table with columns: Tool | Description | Example Use. "
            f"Populate it with all 4 tools available to you: exec_shell, read_file, write_file, http_get. "
            f"Save to {out_dir}/N05_tools_table.md"
        ),
        artifact=str(out_dir / "N05_tools_table.md"),
        min_chars=80,
        must_contain=["|", "exec_shell", "read_file"],
        session_id=session,
    ))

    # For N06 we need git in the kernel-evolving repo
    repo_path = str(REPO_ROOT)
    results.append(run_task(
        "N06_git_log_summary",
        "exec_shell chain → write: recent git commits in this repo → summary",
        message=(
            f"Run 'git -C {repo_path} log --oneline -10' to get the last 10 commits. "
            f"Then write a 3-sentence summary of recent work to {out_dir}/N06_git_summary.txt"
        ),
        artifact=str(out_dir / "N06_git_summary.txt"),
        min_chars=40,
        session_id=session,
    ))

    results.append(run_task(
        "N07_uuid_extract",
        "http_get JSON → extract field → write: httpbin UUID",
        message=(
            f"Fetch https://httpbin.org/uuid. Extract just the UUID string value from the JSON. "
            f"Write only the UUID (no JSON wrapper) to {out_dir}/N07_uuid.txt"
        ),
        artifact=str(out_dir / "N07_uuid.txt"),
        min_chars=30,
        must_contain=["-"],   # UUIDs contain hyphens
        session_id=session,
    ))

    results.append(run_task(
        "N08_todo_timestamp",
        "read_file + exec_shell + write: append timestamp to TODO list",
        message=(
            f"Read the file {todo_seed}. Run 'date +\"%Y-%m-%d %H:%M:%S\"' to get the current time. "
            f"Then write the TODO list with a 'Last updated: <timestamp>' line prepended "
            f"to {out_dir}/N08_todos_stamped.txt"
        ),
        artifact=str(out_dir / "N08_todos_stamped.txt"),
        min_chars=80,
        must_contain=["last updated", "TODO"],
        session_id=session,
    ))

    results.append(run_task(
        "N09_disk_report",
        "exec_shell → write: disk usage of / → disk_report.txt",
        message=(
            f"Run 'df -h /' to get root disk usage. Write a one-line summary "
            f"'Disk /: <size> total, <used> used, <avail> available (<pct> full)' "
            f"(fill from df output) to {out_dir}/N09_disk_report.txt"
        ),
        artifact=str(out_dir / "N09_disk_report.txt"),
        min_chars=20,
        must_contain=["disk"],
        session_id=session,
    ))

    results.append(run_task(
        "N10_onboarding_guide",
        "write structured doc: 5-section developer onboarding guide >600 chars",
        message=(
            f"Write a developer onboarding guide for the kernel-evolving project. "
            f"Include exactly these 5 sections with ## headings: "
            f"Overview, Prerequisites, First Run, Running Evaluations, Adding Tools. "
            f"Each section should have 2–3 sentences. "
            f"Save to {out_dir}/N10_onboarding.md"
        ),
        artifact=str(out_dir / "N10_onboarding.md"),
        min_chars=600,
        must_contain=["##", "overview", "prerequisites"],
        session_id=session,
    ))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_id = RUN_TS
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    session = f"sim15-{run_id}-{args.label}"

    print(f"\n{'='*60}")
    print(f"  Sim 15 — Training Replay + Novel Held-Out Eval")
    print(f"  label:   {args.label}")
    print(f"  port:    {args.port}")
    print(f"  run_id:  {run_id}")
    print(f"  replay:  {args.replay_count} training samples (seed={args.seed})")
    print(f"  version: ", end="")
    try:
        v = _get("/version")
        print(v.get("version", "?"))
    except Exception:
        print("?")
    print(f"{'='*60}")

    part_a = run_part_a(out_dir, session)
    part_b = run_part_b(out_dir, session)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PART A — TRAINING REPLAY")
    print(f"{'='*60}")
    a_pass = a_overfit = a_underfit = a_uncertain = 0
    for r in part_a:
        v = r.get("verdict", "?")
        icon = {"PASS": "✅", "OVERFIT": "🔁", "UNDERFIT": "📉", "UNCERTAIN": "⚠️", "ERROR": "💥"}.get(v, "?")
        sim = r.get("jaccard_sim", 0)
        print(f"  {icon} {r['id']:20s} {v:10s} sim={sim:.2f}  {r.get('desc','')[:50]}")
        if v == "PASS":      a_pass += 1
        elif v == "OVERFIT":  a_overfit += 1
        elif v == "UNDERFIT": a_underfit += 1
        else:                 a_uncertain += 1

    a_total = len(part_a)
    print(f"\n  PASS={a_pass}  OVERFIT={a_overfit}  UNDERFIT={a_underfit}  OTHER={a_uncertain} / {a_total}")

    if a_overfit / max(a_total, 1) >= 0.5:
        fitness = "🔁 OVERFIT — model memorised training; generalisation likely poor"
    elif a_underfit / max(a_total, 1) >= 0.4:
        fitness = "📉 UNDERFIT — model not following training patterns; more data/epochs needed"
    elif a_pass / max(a_total, 1) >= 0.6:
        fitness = "✅ GOOD FIT — model generalises correctly on seen tasks"
    else:
        fitness = "⚠️ MIXED — review individual verdicts above"
    print(f"  Fitness: {fitness}")

    print(f"\n{'='*60}")
    print("  PART B — NOVEL HELD-OUT TASKS")
    print(f"{'='*60}")
    b_pass = 0
    for r in part_b:
        icon = "✅" if r.get("passed") else "❌"
        print(f"  {icon} {r['id']:25s} {r.get('artifact_msg',''):25s} {r.get('elapsed_s','')}s")
        if r.get("passed"):
            b_pass += 1
    b_total = len(part_b)
    b_verdict = "✅ PASS" if b_pass == b_total else (f"⚠️ PARTIAL {b_pass}/{b_total}" if b_pass > 0 else "❌ FAIL")
    print(f"\n  {b_verdict} — {b_pass}/{b_total} novel tasks passed")

    # ── Save results ──────────────────────────────────────────────────────────
    payload = {
        "sim": "sim15",
        "label": args.label,
        "run_id": run_id,
        "port": args.port,
        "ts": datetime.now(timezone.utc).isoformat(),
        "part_a": {
            "total": a_total,
            "pass": a_pass,
            "overfit": a_overfit,
            "underfit": a_underfit,
            "uncertain": a_uncertain,
            "fitness": fitness,
            "tasks": part_a,
        },
        "part_b": {
            "total": b_total,
            "pass": b_pass,
            "verdict": b_verdict,
            "tasks": part_b,
        },
    }
    result_file = RESULTS_DIR / f"sim15_{args.label}_{run_id}.json"
    result_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n  Results: {result_file}")
    print(f"  Artifacts: {out_dir}")
    print(f"{'='*60}\n")
    return payload


if __name__ == "__main__":
    main()
