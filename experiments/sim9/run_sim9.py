#!/usr/bin/env python3
"""
Sim 9 — Local Model Usability Test.

Three provider runs:
  - Run A: E2B local  (gemma-4-E2B-it + drafter)
  - Run B: E4B local  (gemma-4-E4B-it + drafter, if available)
  - Run C: GPT-5.4    (baseline reference)

5 tasks per run. Each produces a verifiable file artifact.
Results saved separately per run for paper comparison.

Usage:
  python3 run_sim9.py              # all runs
  python3 run_sim9.py --run e2b    # single run
  python3 run_sim9.py --run e4b
  python3 run_sim9.py --run gpt
"""
import argparse, json, os, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "http://localhost:8779"
OUT_BASE = Path.home() / "kernel-evo-notes" / "sim9"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = 90  # increased from 60 — local model needs more time

MODEL_CONFIGS = {
    "e2b": {
        "label": "Gemma 4 E2B (2.3B) + drafter",
        "provider": "local",
        "model_path": "~/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3",
        "drafter_path": "~/models/huggingface/hub/hub/models--google--gemma-4-E2B-it-assistant/snapshots/be0358c16076890848a1344a34209aa7c1df7587",
    },
    "e4b": {
        "label": "Gemma 4 E4B (4.5B) + drafter",
        "provider": "local",
        "model_path": "~/models/huggingface/hub/models--google--gemma-4-E4B-it",
        "drafter_path": "~/models/huggingface/hub/models--google--gemma-4-E4B-it-assistant",
    },
    "gpt": {
        "label": "GPT-5.4 (baseline reference)",
        "provider": "openai",
        "model_path": None,
        "drafter_path": None,
    },
}

def _post(path, body, timeout=TIMEOUT):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{API}{path}", data=data,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _get(path):
    with urllib.request.urlopen(f"{API}{path}", timeout=10) as r:
        return json.loads(r.read())

def _verify(path):
    p = Path(os.path.expanduser(str(path)))
    if not p.exists(): return False, "file not found"
    s = p.stat().st_size
    return (True, f"{s} bytes") if s > 0 else (False, "empty")

def make_tasks(out_dir: Path):
    return [
        {
            "id": "T01_write_json",
            "message": f'Write this JSON to {out_dir}/config.json: {{"agent":"kernel-evo","version":"1.10.2","sim":"sim9"}}',
            "artifact": f"{out_dir}/config.json",
        },
        {
            "id": "T02_write_script",
            "message": f'Write a Python script to {out_dir}/hello.py with content: #!/usr/bin/env python3\nprint("kernel-evo sim9 local")',
            "artifact": f"{out_dir}/hello.py",
        },
        {
            "id": "T03_exec_and_save",
            "message": f'Run: echo "sim9 passed $(date)" and save the output to {out_dir}/timestamp.txt',
            "artifact": f"{out_dir}/timestamp.txt",
        },
        {
            "id": "T04_read_and_summarise",
            "message": f'Read the file {out_dir}/config.json and write a one-line summary to {out_dir}/summary.txt',
            "artifact": f"{out_dir}/summary.txt",
        },
        {
            "id": "T05_chain",
            "message": f'Run: python3 {out_dir}/hello.py and save its output to {out_dir}/result.txt',
            "artifact": f"{out_dir}/result.txt",
        },
    ]

def run_task(task):
    ts = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    print(f"\n  [{task['id']}] {task['message'][:80]}")
    try:
        result = _post("/message", {"message": task["message"], "chat_id": f"sim9-{task['id']}"})
        reply = result.get("reply", "")
    except Exception as e:
        elapsed = round(time.time()-t0, 1)
        print(f"    💥 ERROR {elapsed}s: {e}")
        return {"id": task["id"], "ts": ts, "elapsed_s": elapsed,
                "artifact_ok": False, "artifact_msg": str(e), "reply_head": "", "error": str(e)}
    elapsed = round(time.time()-t0, 1)
    ok, msg = _verify(task["artifact"])
    icon = "✅" if ok else "❌"
    print(f"    {icon} {elapsed}s | artifact: {msg} | reply: {reply[:60]!r}")
    return {"id": task["id"], "ts": ts, "elapsed_s": elapsed,
            "artifact": str(task["artifact"]), "artifact_ok": ok,
            "artifact_msg": msg, "reply_head": reply[:120]}

def swap_model(cfg: dict) -> bool:
    """Hot-swap to the model for this run. Returns True on success."""
    if cfg["model_path"] is None:
        return True  # GPT — no swap needed
    import socket as _sock
    try:
        sock_path = "/tmp/kernel_evolving_model.sock"
        s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        s.settimeout(300)
        s.connect(sock_path)
        req = json.dumps({"method": "swap_model", "params": {
            "model_path": cfg["model_path"],
            "drafter_path": cfg["drafter_path"] or "",
            "dtype": "bfloat16",
        }}) + "\n"
        s.sendall(req.encode())
        resp = b""
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            resp += chunk
            if b"\n" in resp: break
        s.close()
        result = json.loads(resp.strip())
        if "error" in result:
            print(f"  ❌ swap_model error: {result['error']}")
            return False
        print(f"  ✅ Model loaded: {result.get('model','?')}")
        return True
    except Exception as e:
        print(f"  ❌ swap_model exception: {e}")
        return False

def run_suite(run_key: str) -> dict:
    cfg = MODEL_CONFIGS[run_key]
    out_dir = OUT_BASE / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(out_dir)

    print(f"\n{'='*60}")
    print(f"  RUN: {cfg['label']}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Set provider
    try:
        prov = _post("/provider/set", {"task_inference": cfg["provider"]}, timeout=5)
        print(f"  [provider] → {cfg['provider']}: {prov.get('changed', prov)}")
    except Exception as e:
        print(f"  [provider] WARNING: {e}")

    # Swap model if local
    if cfg["model_path"]:
        print(f"  [model] Swapping to {Path(cfg['model_path']).name}...")
        if not swap_model(cfg):
            print(f"  ❌ Model swap failed — skipping run {run_key}")
            return {"run": run_key, "label": cfg["label"], "skipped": True, "summary": {"total": 5, "passed": 0}}

    # Enable sim mode
    try:
        _post("/sim/mode", {"enabled": True}, timeout=5)
    except Exception:
        pass

    results = []
    for task in tasks:
        results.append(run_task(task))
        time.sleep(1)

    passed = sum(1 for r in results if r.get("artifact_ok"))
    pass_rate = f"{passed}/{len(tasks)}"
    print(f"\n  RESULT: {pass_rate} artifacts ({'✅' if passed == len(tasks) else '⚠️'})")

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"sim9_{run_key}_{ts_str}.json"
    payload = {
        "sim": "sim9", "run": run_key, "label": cfg["label"],
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_path": cfg.get("model_path"),
        "drafter_path": cfg.get("drafter_path"),
        "summary": {"total": len(tasks), "passed": passed, "pass_rate": pass_rate},
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"  Results → {out_file}")

    try:
        _post("/sim/mode", {"enabled": False}, timeout=5)
    except Exception:
        pass

    return payload

def main():
    parser = argparse.ArgumentParser(description="Sim 9 — Local model usability test")
    parser.add_argument("--run", choices=["e2b", "e4b", "gpt", "all"], default="all")
    args = parser.parse_args()

    try:
        h = _get("/health")
        v = _get("/version")
        print(f"[kernel-evolving] v{v.get('version','?')} | skills={h['skills']} routines={h['routines']}")
    except Exception as e:
        print(f"ABORT: kernel-evolving not reachable: {e}"); return

    runs = ["e2b", "e4b", "gpt"] if args.run == "all" else [args.run]

    # Skip e4b if drafter not downloaded yet
    if "e4b" in runs:
        drafter_path = Path(MODEL_CONFIGS["e4b"]["drafter_path"])
        if not drafter_path.exists():
            print(f"  ⚠️  E4B drafter not yet downloaded ({drafter_path}) — will run without drafter")
            MODEL_CONFIGS["e4b"]["drafter_path"] = ""

    all_results = []
    for run_key in runs:
        result = run_suite(run_key)
        all_results.append(result)
        if run_key != runs[-1]:
            print("\n  [pause 3s between runs...]")
            time.sleep(3)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  SIM 9 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Run':<8} {'Model':<35} {'Pass Rate'}")
    print(f"  {'-'*55}")
    for r in all_results:
        skipped = r.get("skipped", False)
        label = r["label"][:34]
        rate = "SKIPPED" if skipped else r["summary"]["pass_rate"]
        print(f"  {r['run']:<8} {label:<35} {rate}")

if __name__ == "__main__":
    main()
