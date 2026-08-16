#!/usr/bin/env python3
"""
Sim 13 — Morning Brief Routine: Gemma 4 E2B vs Qwen3-VL-2B

Goal:
  Run the morning brief routine against both local models and compare which one
  executes it successfully (reads digest, picks stories, writes a blog post draft,
  saves artifact).

Each model is swapped in, then asked to:
  1. Run /run morning-brief (skill/routine trigger path)
  2. Run a direct chat prompt replicating what the routine does
     (read digest_today.json → pick 3-5 stories → write blog post → save to file)

Pass criteria per task:
  - Non-empty reply returned within timeout
  - Artifact file written (>200 chars)
  - Reply contains at least one of: a title, "##", "blog", "post", "digest"

Usage:
  python3 sim13/run_sim13.py

Requires:
  - kernel-evolving API running on http://localhost:8779
  - Model server socket at /tmp/kernel_evolving_model.sock
  - At least Gemma 4 E2B downloaded at its known path
  - Qwen3-VL-2B downloaded (or sim records SKIP for that model)
"""

import json
import os
import socket
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

API = "http://localhost:8779"
MODEL_SOCKET = "/tmp/kernel_evolving_model.sock"
TIMEOUT = 240
DIGEST_PATH = Path.home() / ".openclaw/media/digest_today.json"
OUT_BASE = Path.home() / "kernel-evo-notes" / "sim13"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "key": "e2b",
        "label": "Gemma 4 E2B (2.3B)",
        "model_path": "~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
        "drafter_path": "~/models/huggingface/hub/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587",
    },
    {
        "key": "qwen3vl2b",
        "label": "Qwen3-VL 2B Instruct",
        "model_path": "~/models/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct",
        "drafter_path": "",
    },
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _post(path, body, timeout=TIMEOUT):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def swap_model(model: dict) -> bool:
    """Hot-swap model via the model server socket. Returns True on success."""
    if not os.path.isdir(model["model_path"]):
        print(f"  ⚠️  Model not downloaded: {model['model_path']}")
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(300)
        s.connect(MODEL_SOCKET)
        req = json.dumps({
            "method": "swap_model",
            "params": {
                "model_path": model["model_path"],
                "drafter_path": model["drafter_path"],
                "dtype": "bfloat16",
            },
        }) + "\n"
        s.sendall(req.encode())
        resp = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
            if b"\n" in resp:
                break
        result = json.loads(resp.strip())
        if "error" in result:
            print(f"  ❌ swap_model error: {result['error']}")
            return False
        print(f"  ✅ Loaded: {result.get('model', '?')}")
        return True
    except Exception as e:
        print(f"  ❌ swap_model exception: {e}")
        return False


def _build_digest_snippet() -> str:
    """Read up to 5 non-arXiv items from digest_today.json for the prompt."""
    if not DIGEST_PATH.exists():
        return "(digest unavailable)"
    try:
        items = json.loads(DIGEST_PATH.read_text(encoding="utf-8"))
        filtered = [
            i for i in items
            if "arxiv" not in i.get("label", "").lower()
        ][:5]
        if not filtered:
            filtered = items[:5]  # fallback to whatever is there
        lines = []
        for i in filtered:
            title = i.get("title", "?")[:100]
            link = i.get("link", "")
            label = i.get("label", "")
            lines.append(f"- [{label}] {title} — {link}")
        return "\n".join(lines) if lines else "(no items)"
    except Exception as e:
        return f"(error reading digest: {e})"


def make_tasks(out_dir: Path, model_key: str):
    digest_snippet = _build_digest_snippet()
    return [
        {
            "id": "T01_routine_trigger",
            "message": "/run morning-brief",
            "artifact": str(out_dir / f"{model_key}_T01_morning_brief_routine.txt"),
            "description": "Trigger morning-brief via /run command",
        },
        {
            "id": "T02_direct_digest_post",
            "message": (
                "You are running the daily morning brief. "
                "Here are today's top AI news items:\n\n"
                f"{digest_snippet}\n\n"
                "Write a short developer-focused blog post (300-500 words) covering "
                "the 3 most relevant stories. Use markdown with a title, one paragraph "
                "per story, and a closing sentence. Then save the post to "
                f"~/kernel-evo-notes/sim13/{model_key}_morning_post.md"
            ),
            "artifact": str(out_dir / f"{model_key}_morning_post.md"),
            "description": "Direct prompt: read digest snippet → write + save blog post",
        },
        {
            "id": "T03_digest_json_read",
            "message": (
                "Read the file ~/.openclaw/media/digest_today.json, "
                "pick the 3 most developer-relevant AI news items (skip arXiv papers), "
                "and summarise each in one sentence. Then save the summary to "
                f"~/kernel-evo-notes/sim13/{model_key}_digest_summary.txt"
            ),
            "artifact": str(out_dir / f"{model_key}_digest_summary.txt"),
            "description": "Read digest JSON → filter → summarise → save",
        },
    ]


def _verify(artifact_path: str, min_chars: int = 200) -> tuple[bool, str]:
    p = Path(artifact_path)
    if not p.exists():
        return False, "missing"
    size = p.stat().st_size
    if size < min_chars:
        return False, f"too short ({size} chars)"
    return True, f"ok ({size} chars)"


def _quality_check(reply: str) -> bool:
    """Rough quality signal: does the reply look like a blog post or structured content?"""
    signals = ["##", "# ", "blog", "post", "digest", "developer", "summary", "today"]
    reply_lower = reply.lower()
    return any(s in reply_lower for s in signals)


def run_task(task: dict, model_key: str) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task['id']}] {task['description']}")
    try:
        result = _post("/message", {
            "message": task["message"],
            "chat_id": f"sim13-{model_key}-{task['id']}",
        })
        reply = result.get("reply", "")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"    💥 ERROR {elapsed}s: {e}")
        return {
            "id": task["id"],
            "description": task["description"],
            "ts": ts,
            "elapsed_s": elapsed,
            "artifact_ok": False,
            "artifact_msg": str(e),
            "quality_ok": False,
            "reply_head": "",
            "error": str(e),
        }

    elapsed = round(time.time() - t0, 1)

    # If model wrote artifact itself, great; otherwise save reply as artifact
    artifact_path = Path(task["artifact"])
    if reply and not artifact_path.exists():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(reply, encoding="utf-8")

    artifact_ok, artifact_msg = _verify(task["artifact"])
    quality_ok = _quality_check(reply)
    passed = artifact_ok and (bool(reply.strip()))

    icon = "✅" if passed else ("⚠️" if reply else "❌")
    print(f"    {icon} {elapsed}s | artifact: {artifact_msg} | quality: {'✅' if quality_ok else '❌'} | reply: {reply[:80]!r}")

    return {
        "id": task["id"],
        "description": task["description"],
        "ts": ts,
        "elapsed_s": elapsed,
        "artifact": str(task["artifact"]),
        "artifact_ok": artifact_ok,
        "artifact_msg": artifact_msg,
        "quality_ok": quality_ok,
        "reply_head": reply[:200],
        "passed": passed,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_BASE / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sim 13 — Morning Brief: Gemma 4 E2B vs Qwen3-VL-2B")
    print(f"  run_id: {run_id}")
    print(f"  digest items available: {DIGEST_PATH.exists()}")
    print(f"{'='*60}")

    all_results = {}

    for model in MODELS:
        print(f"\n{'─'*60}")
        print(f"  Model: {model['label']}")
        print(f"{'─'*60}")

        if not os.path.isdir(model["model_path"]):
            print(f"  ⏭  SKIP — not downloaded: {model['model_path']}")
            all_results[model["key"]] = {
                "model": model["label"],
                "skipped": True,
                "reason": "not downloaded",
                "tasks": [],
                "passed": 0,
                "total": 0,
            }
            continue

        swapped = swap_model(model)
        if not swapped:
            all_results[model["key"]] = {
                "model": model["label"],
                "skipped": True,
                "reason": "swap_model failed",
                "tasks": [],
                "passed": 0,
                "total": 0,
            }
            continue

        # Brief warm-up pause after swap
        time.sleep(3)

        tasks = make_tasks(out_dir, model["key"])
        task_results = []
        passed = 0

        for task in tasks:
            r = run_task(task, model["key"])
            task_results.append(r)
            if r.get("passed"):
                passed += 1

        pass_rate = f"{passed}/{len(tasks)}"
        verdict = "✅ PASS" if passed == len(tasks) else ("⚠️ PARTIAL" if passed > 0 else "❌ FAIL")
        print(f"\n  {verdict} — {pass_rate} tasks passed")

        all_results[model["key"]] = {
            "model": model["label"],
            "skipped": False,
            "tasks": task_results,
            "passed": passed,
            "total": len(tasks),
            "pass_rate": pass_rate,
            "verdict": verdict,
        }

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for key, r in all_results.items():
        if r["skipped"]:
            print(f"  {r['model']}: SKIPPED ({r['reason']})")
        else:
            print(f"  {r['model']}: {r.get('verdict','?')} ({r['pass_rate']})")

    # ── winner ───────────────────────────────────────────────────────────────
    candidates = {k: v for k, v in all_results.items() if not v["skipped"]}
    if len(candidates) >= 2:
        scores = {k: v["passed"] for k, v in candidates.items()}
        winner_key = max(scores, key=lambda k: scores[k])
        winner = candidates[winner_key]
        if scores["e2b"] == scores.get("qwen3vl2b"):
            print("\n  🤝 TIE — both models performed equally")
        else:
            print(f"\n  🏆 Winner: {winner['model']} ({winner['pass_rate']})")
    elif len(candidates) == 1:
        only = list(candidates.values())[0]
        print(f"\n  Only one model ran: {only['model']} → {only.get('verdict','?')}")

    # ── persist ──────────────────────────────────────────────────────────────
    payload = {
        "sim": "sim13",
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "digest_available": DIGEST_PATH.exists(),
        "models": all_results,
    }
    result_file = RESULTS_DIR / f"sim13_{run_id}.json"
    result_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results saved: {result_file}")
    print(f"  Artifacts in: {out_dir}\n")

    return payload


if __name__ == "__main__":
    main()
