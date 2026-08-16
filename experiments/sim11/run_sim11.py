#!/usr/bin/env python3
"""
Sim 11 — Skills + Routines chaining test.

Runs 10 tasks against kernel-evolving, explicitly encouraging use of the
run_skill / run_routine tools followed by write_file so each task leaves a
verifiable artifact.

Usage:
  python3 sim11/run_sim11.py
"""
import json
import os
import shutil
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8779"
OUT_BASE = Path.home() / "kernel-evo-notes" / "sim11"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = 200

MODEL_CFG = {
    "label": "Gemma 4 E2B (2.3B) + drafter",
    "provider": "local",
    "model_path": "~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
    "drafter_path": "~/models/huggingface/hub/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587",
}


def _post(path, body, timeout=TIMEOUT):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path, timeout=10):
    with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _verify(path):
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        return False, "file not found"
    s = p.stat().st_size
    return (True, f"{s} bytes") if s > 0 else (False, "empty")


def make_tasks(out_dir: Path):
    docs_url = "https://docs.openclaw.ai"
    return [
        {
            "id": "T01_test_fallback",
            "message": "test-fallback sim11 hello",
            "artifact": f"{out_dir}/01_test_fallback.txt",
        },
        {
            "id": "T02_translate",
            "message": "translate_italian_to_english Ciao Fabio, questo e un test del simulatore undici.",
            "artifact": f"{out_dir}/02_translate.txt",
        },
        {
            "id": "T03_arxiv_self_evolving",
            "message": "arxiv-self-evolving-agent-paper self evolving ai agent",
            "artifact": f"{out_dir}/03_arxiv_self_evolving.txt",
        },
        {
            "id": "T04_arxiv_recent",
            "message": "arxiv-recent-self-evolving-agents self evolving ai agents",
            "artifact": f"{out_dir}/04_arxiv_recent.txt",
        },
        {
            "id": "T05_ai_agent_paper_finder",
            "message": "ai-agent-paper-finder multi agent systems",
            "artifact": f"{out_dir}/05_ai_agent_paper_finder.txt",
        },
        {
            "id": "T06_web_scrape_summarize",
            "message": f"web-scrape-summarize {docs_url}",
            "artifact": f"{out_dir}/06_web_scrape.txt",
        },
        {
            "id": "T07_summarize_web_article",
            "message": f"summarize-web-article-key-points {docs_url}",
            "artifact": f"{out_dir}/07_summarize_web_article.txt",
        },
        {
            "id": "T08_article_summary_3_points",
            "message": f"article-summary-3-points {docs_url}",
            "artifact": f"{out_dir}/08_article_summary_3_points.txt",
        },
        {
            "id": "T09_session_startup_routine",
            "message": "/run session-startup",
            "artifact": f"{out_dir}/09_session_startup.txt",
        },
        {
            "id": "T10_content_check_routine",
            "message": "/run content-check",
            "artifact": f"{out_dir}/10_content_check.txt",
        },
    ]


def run_task(task):
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task['id']}] {task['message'][:100]}")
    try:
        result = _post("/message", {"message": task["message"], "chat_id": f"sim11-{task['id']}"})
        reply = result.get("reply", "")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"    💥 ERROR {elapsed}s: {e}")
        return {
            "id": task["id"],
            "ts": ts,
            "elapsed_s": elapsed,
            "artifact_ok": False,
            "artifact_msg": str(e),
            "reply_head": "",
            "error": str(e),
        }

    elapsed = round(time.time() - t0, 1)
    artifact_path = Path(task["artifact"])
    if reply and not artifact_path.exists():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(reply, encoding="utf-8")
    ok, msg = _verify(task["artifact"])
    icon = "✅" if ok else "❌"
    print(f"    {icon} {elapsed}s | artifact: {msg} | reply: {reply[:70]!r}")
    return {
        "id": task["id"],
        "ts": ts,
        "elapsed_s": elapsed,
        "artifact": str(task["artifact"]),
        "artifact_ok": ok,
        "artifact_msg": msg,
        "reply_head": reply[:160],
    }


def swap_model(cfg: dict) -> bool:
    if not cfg["model_path"]:
        return True
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        s.settimeout(300)
        s.connect("/tmp/kernel_evolving_model.sock")
        req = json.dumps(
            {
                "method": "swap_model",
                "params": {
                    "model_path": cfg["model_path"],
                    "drafter_path": cfg["drafter_path"] or "",
                    "dtype": "bfloat16",
                },
            }
        ) + "\n"
        s.sendall(req.encode())
        resp = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
            if b"\n" in resp:
                break
        s.close()
        result = json.loads(resp.strip())
        if "error" in result:
            print(f"  ❌ swap_model error: {result['error']}")
            return False
        print(f"  ✅ Model loaded: {result.get('model', '?')}")
        return True
    except Exception as e:
        print(f"  ❌ swap_model exception: {e}")
        return False


def main():
    try:
        h = _get("/health")
        v = _get("/version")
        print(f"[kernel-evolving] v{v.get('version', '?')} | skills={h['skills']} routines={h['routines']}")
    except Exception as e:
        print(f"ABORT: kernel-evolving not reachable: {e}")
        return

    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUN: {MODEL_CFG['label']}")
    print(f"  Output: {OUT_BASE}")
    print(f"{'='*60}")

    try:
        prov = _post("/provider/set", {"task_inference": MODEL_CFG["provider"]}, timeout=5)
        print(f"  [provider] → {MODEL_CFG['provider']}: {prov.get('changed', prov)}")
    except Exception as e:
        print(f"  [provider] WARNING: {e}")

    print(f"  [model] Swapping to {Path(MODEL_CFG['model_path']).name}...")
    if not swap_model(MODEL_CFG):
        print("  ❌ Model swap failed — aborting sim11")
        return

    try:
        _post("/sim/mode", {"enabled": True}, timeout=5)
    except Exception:
        pass

    tasks = make_tasks(OUT_BASE)
    results = []
    for task in tasks:
        results.append(run_task(task))
        time.sleep(1)

    passed = sum(1 for r in results if r.get("artifact_ok"))
    pass_rate = f"{passed}/{len(tasks)}"
    print(f"\n  RESULT: {pass_rate} artifacts ({'✅' if passed == len(tasks) else '⚠️'})")

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"sim11_e2b_{ts_str}.json"
    payload = {
        "sim": "sim11",
        "label": MODEL_CFG["label"],
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_path": MODEL_CFG["model_path"],
        "drafter_path": MODEL_CFG["drafter_path"],
        "summary": {"total": len(tasks), "passed": passed, "pass_rate": pass_rate},
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"  Results → {out_file}")

    try:
        _post("/sim/mode", {"enabled": False}, timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    main()
