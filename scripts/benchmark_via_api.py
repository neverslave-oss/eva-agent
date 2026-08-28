#!/usr/bin/env python3
"""
benchmark_via_api.py — benchmark kernel-evolving behavior through its HTTP API.

Sends a battery of natural-language messages to POST /message and measures
latency + captures the agent's replies. This exercises the REAL agent path
(triage → inference → tools → reply), so it reflects how the system actually
behaves under a given config, not just raw model throughput.

Usage (run once per config; switch configs with start.sh --config=<file>):
    python3 scripts/benchmark_via_api.py --label configA [--out configs/configA/results.json]

The script assumes the server is already up with the config you want to test.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

API = "http://localhost:8779"

# A battery of representative messages. Each has a label + the message text.
# chat_id is fixed per label so conversation continuity is exercised too.
TESTS = [
    {"label": "text", "message": "Explain in 3 short sentences what a transformer neural network is."},
    {"label": "tool_calc", "message": "What is 15% of 240? Use the calculator tool if available."},
    {"label": "tool_followup", "message": "And what is 33% of that result? Use the calculator tool."},
    {"label": "memory_context", "message": "Summarize what we've been doing in this conversation so far, briefly."},
]


def send_message(text: str, chat_id: str) -> dict:
    """POST /message and return {latency_s, reply, error}."""
    body = json.dumps({"message": text, "chat_id": chat_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/message",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        dt = time.perf_counter() - t0
        return {"latency_s": round(dt, 3), "reply": payload.get("reply", ""), "error": None}
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"latency_s": round(dt, 3), "reply": "", "error": str(e)}


def health() -> bool:
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark kernel-evolving via its HTTP API")
    parser.add_argument("--label", required=True, help="Config label, e.g. configA")
    parser.add_argument("--out", default=None, help="Path to write results JSON (default configs/<label>/results.json)")
    parser.add_argument("--models", default=None, help="Comma list of model labels to record (informational)")
    args = parser.parse_args()

    if not health():
        print("ERROR: kernel-evolving API not reachable. Start it with ./start.sh --config=<file> first.")
        sys.exit(1)

    chat_id = f"bench-{args.label}"
    results = []
    print(f"=== API benchmark: {args.label} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"Server: {API} | chat_id: {chat_id}\n")

    for t in TESTS:
        print(f"--- {t['label']} ---")
        r = send_message(t["message"], chat_id)
        r["label"] = t["label"]
        results.append(r)
        status = "OK" if not r["error"] else f"ERROR: {r['error'][:120]}"
        print(f"  {status} | {r['latency_s']}s | {r['reply'][:160]}")
        print()

    # Summary
    ok = [r for r in results if not r["error"]]
    avg = round(sum(r["latency_s"] for r in ok) / len(ok), 3) if ok else None
    print("=== Summary ===")
    print(f"Config: {args.label} | tests: {len(results)} | ok: {len(ok)} | avg_latency: {avg}s")
    if args.models:
        print(f"Models: {args.models}")

    # Persist
    out = args.out or os.path.join("configs", args.label, "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": args.models,
        "tests_run": len(results),
        "tests_ok": len(ok),
        "avg_latency_s": avg,
        "results": results,
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
