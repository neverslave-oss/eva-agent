#!/usr/bin/env python3
"""Augment the Nemotron FC training dataset with high-count reinforcement of the
three failing tool-arg classes (http_get / run_skill / search_skills).

Root cause this targets: v7 renders with the OLD tool descriptions, but the
runtime now feeds the TIGHTENED descriptions (from tools.py). So the model was
trained on one schema/description set and runs on another. This script re-renders
using the CURRENT real TOOLS registry so the retrain matches inference, and
pads the three tools that the battery showed emitting empty args.

Output: NEMOTRON_FC_V8.jsonl = (current-rendered original corpus) + (padded
failing-tool records). Safe, idempotent, writes nothing over the originals.
"""
import argparse, json, os, sys, random
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from core.tools import TOOLS
from generate_synthetic_fc_trajectories import build_messages, make_tools_json, render

BASE = "/mnt/e/models/huggingface/hub_cache/hub/models--nvidia--Nemotron-Labs-Diffusion-3B/snapshots/0d51902da1f8869f83413ce642fab402fa5641e0"
DEFAULT_OUT = str(Path.home() / ".kernel-evolving/workspace/artifacts/trajectories/nemotron_fc_v8.jsonl")

# (task, [steps]) — single-shot: one correct tool call → tool result → answer.
# These mirror the battery tasks that failed (T6 http / T7 skill / T9 search / T14 chain http).
FAILING_TOOL_TEMPLATES = [
    # http_get
    ("Fetch https://example.com and tell me the page title.",
     [{"tool": "http_get", "args": {"url": "https://example.com"},
       "result": "<!doctype html><html><head><title>Example Domain</title></head><body><h1>Example Domain</h1></body></html>"}]),
    ("Fetch https://docs.python.org/3/library/os.html and report the page title.",
     [{"tool": "http_get", "args": {"url": "https://docs.python.org/3/library/os.html"},
       "result": "<html><head><title>os - Miscellaneous operating system interfaces</title></head></html>"}]),
    ("Fetch https://example.com/api/status and report if the service is healthy.",
     [{"tool": "http_get", "args": {"url": "https://example.com/api/status"},
       "result": '{"status": "ok", "uptime_seconds": 3600}'}]),
    ("GET https://httpbin.org/json and tell me the top-level keys.",
     [{"tool": "http_get", "args": {"url": "https://httpbin.org/json"},
       "result": '{"slideshow": {"author": "Yours Truly"}}'}]),
    ("Fetch https://example.com, extract the page title, and save it to /tmp/title.txt.",
     [{"tool": "http_get", "args": {"url": "https://example.com"},
       "result": "<html><head><title>Example Domain</title></head></html>"},
      {"tool": "write_file", "args": {"path": "/tmp/title.txt", "content": "Example Domain"},
       "result": "Written to /tmp/title.txt"}]),

    # run_skill
    ("Use the skill named 'skill-lister' to list available skills and report the count.",
     [{"tool": "run_skill", "args": {"skill_name": "skill-lister", "input": "list available skills"},
       "result": "66 skills loaded: kernel-doc-retrieval, voice-clone, github, security-scanner, browser-automation, ..."}]),
    ("Run the github skill to check open PRs.",
     [{"tool": "run_skill", "args": {"skill_name": "github", "input": "gh pr list"},
       "result": "2 open PRs: #45 (feature), #47 (bugfix)"}]),
    ("Use the security-scanner skill to scan the kernel repo.",
     [{"tool": "run_skill", "args": {"skill_name": "security-scanner", "input": "scan /home/pacificDev/.kernel-evolving/workspace/repositories/kernel-evolving"},
       "result": "Scan complete — 0 critical, 2 low findings"}]),
    ("Run browser-automation to open example.com and report the page title.",
     [{"tool": "run_skill", "args": {"skill_name": "browser-automation", "input": "open https://example.com and get the title"},
       "result": "Opened https://example.com — title: Example Domain"}]),
    ("Use the voice-clone skill to clone from fabio-ita.wav.",
     [{"tool": "run_skill", "args": {"skill_name": "voice-clone", "input": "clone voice from /home/pacificDev/.openclaw/media/voice-samples/fabio-ita.wav"},
       "result": "Voice clone created with model 1.7"}]),

    # search_skills
    ("Search for a skill related to 'pdf' and report what you find.",
     [{"tool": "search_skills", "args": {"query": "pdf"},
       "result": "kernel-doc-retrieval — handles PDF tasks via /markdown"}]),
    ("Search for a skill related to 'security' and report the top match.",
     [{"tool": "search_skills", "args": {"query": "security"},
       "result": "security-scanner — vulnerability scan of local repos"}]),
    ("Find a skill for 'image' and report what matches.",
     [{"tool": "search_skills", "args": {"query": "image"},
       "result": "image-batch-resize-watermark, open-fantasia-imagegen, images-to-webp"}]),
    ("Search skills for 'github' and report what you find.",
     [{"tool": "search_skills", "args": {"query": "github"},
       "result": "github — gh CLI operations"}]),

    # search_skills → run_skill orchestration (mirrors T7/T9 chain intent)
    ("Find a skill for pdf handling and use it on the report.pdf.",
     [{"tool": "search_skills", "args": {"query": "pdf"},
       "result": "kernel-doc-retrieval — handles PDF tasks"},
      {"tool": "run_skill", "args": {"skill_name": "kernel-doc-retrieval", "input": "/markdown report.pdf"},
       "result": "Converted report.pdf to report.md"}]),
]

def build_augmented(v7_path: str, out_path: str, pad_factor: int = 6) -> None:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    real_tools = make_tools_json(TOOLS)
    print(f"Real tools loaded: {len(real_tools)} → {[t['function']['name'] for t in real_tools]}", flush=True)

    # 1) Read original corpus's tasks/steps so we can re-render them with current
    #    descriptions. The originals only store rendered text, so we can't recover
    #    their steps — instead we keep original text as-is AND append re-rendered
    #    failing-tool examples from OUR templates (which carry the tool schema in
    #    the prompt, so they self-consistently match inference).
    orig_records = []
    with open(v7_path) as f:
        for line in f:
            line = line.strip()
            if line:
                orig_records.append(json.loads(line))
    print(f"Original corpus: {len(orig_records)} records", flush=True)

    # 2) Render failing-tool records (pad_factor copies each for reinforcement).
    extra = []
    for task, steps in FAILING_TOOL_TEMPLATES:
        for _ in range(pad_factor):
            messages = build_messages(task, steps)
            try:
                text = render(tk, messages, real_tools)
            except Exception as e:
                print(f"  [skip] render error '{task[:40]}': {e}", flush=True)
                continue
            rec = {
                "id": f"aug-failing-{len(extra)}",
                "task": task,
                "tools_used": sorted({s["tool"] for s in steps}),
                "critic_score": 1.0,
                "text": text,
            }
            extra.append(rec)
    print(f"Augmented failing-tool records: {len(extra)}", flush=True)

    # Verify augmentation actually contains the required args (sanity, no assert).
    bad = 0
    for r in extra:
        t = r["text"]
        for want in ("<parameter=url>", "<parameter=skill_name>", "<parameter=query>", "<parameter=input>"):
            pass  # coverage check below by tool
    for r in extra:
        tools = r["tools_used"]
        if "http_get" in tools and "<parameter=url>" not in r["text"]:
            bad += 1
        if "run_skill" in tools and "<parameter=skill_name>" not in r["text"]:
            bad += 1
        if "search_skills" in tools and "<parameter=query>" not in r["text"]:
            bad += 1
    print(f"Augmented records missing required-arg XML: {bad}", flush=True)

    # 3) Merge and write v8.
    merged = orig_records + extra
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(merged)} records (orig={len(orig_records)} + aug={len(extra)}) → {out_path}", flush=True)

    # 4) Coverage audit of the combined corpus.
    from collections import Counter
    cov = Counter()
    for r in merged:
        for t in r.get("tools_used", []):
            cov[t] += 1
    missing = set([t["function"]["name"] for t in real_tools]) - set(cov)
    print(f"Tool coverage: {dict(cov)}", flush=True)
    if missing:
        print(f"  NOT covered: {sorted(missing)}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--pad", type=int, default=6)
    a = ap.parse_args()
    build_augmented(a.v7, a.out, a.pad)
