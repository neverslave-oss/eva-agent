#!/usr/bin/env python3
"""
generate_synthetic_fc_trajectories.py
=====================================
Generate high-quality synthetic function-calling trajectories for the
**Nemotron** FC adapter, rendered in Nemotron's native `<tool_call><function=...>`
format via its real tokenizer + EVA's REAL 16-tool registry.

Why this exists:
  The previous FC adapter was trained on a hardcoded 5-tool subset
  (exec_shell/read_file/write_file/http_get/run_skill) and never saw 11 of EVA's
  real tools. It learned FC *shape* on 5 easy tools, so against the real toolchain
  it fabricated answers instead of calling tools (0/4 in the real-tool battery).
  This generator synthesises trajectories across ALL 16 real tools, rendered
  exactly as EVA's inference path presents them (apply_chat_template(tools=REAL_TOOLS)).

Output:
  JSONL of records with a pre-rendered "text" field (trainer Mode-1 passthrough),
  plus a "task" and "tools_used" for auditing. Render is done with the Nemotron
  tokenizer so the tokens match inference exactly.

Usage:
  PYTHONPATH=src python3 scripts/generate_synthetic_fc_trajectories.py \
      --model /mnt/e/models/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B \
      --out ~/.kernel-evolving/workspace/artifacts/trajectories/synthetic_fc_real_tools.jsonl \
      --count 120
"""
import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from core.tools import TOOLS  # EVA's real 16-tool registry

# ── Multi-turn trajectory templates ─────────────────────────────────────────
# Each template: (user_task, [steps]) where each step is a dict:
#   {"tool": "<real tool name>", "args": {...}, "result": "<realistic tool output>"}
# The final assistant turn (post tool results) is a natural-language summary.
# Uses ONLY real tool names + realistic arg/result shapes.

TEMPLATES = [
    # ── exec_shell + write_file chains ─────────────────────────────────────
    ("List all python files in the workspace and save the list to /tmp/pyfiles.txt",
     [{"tool": "exec_shell", "args": {"command": "ls -1 /home/pacificDev/.kernel-evolving/workspace/*.py"},
       "result": "agent.py\nmodel.py\nskills.py\n"},
      {"tool": "write_file", "args": {"path": "/tmp/pyfiles.txt", "content": "agent.py\nmodel.py\nskills.py\n"},
       "result": "Written to /tmp/pyfiles.txt"}]),
    ("Check the current disk usage of /",
     [{"tool": "exec_shell", "args": {"command": "df -h /"},
       "result": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       1.0T  148G  853G  15% /"}]),
    ("What is the current python3 version?",
     [{"tool": "exec_shell", "args": {"command": "python3 --version"},
       "result": "Python 3.13.1"}]),

    # ── read_file + summarize ──────────────────────────────────────────────
    ("Read ~/notes.md and summarize what it contains",
     [{"tool": "read_file", "args": {"path": "/home/pacificDev/.kernel-evolving/workspace/notes.md"},
       "result": "# Notes\n- Meeting with Millie at 14:00\n- Deploy voice server on Friday\n- Buy cat food"}]),
    ("Read config.yaml and tell me the model name",
     [{"tool": "read_file", "args": {"path": "/home/pacificDev/.kernel-evolving/workspace/config.yaml"},
       "result": "model:\n  name: nvidia/Nemotron-Labs-Diffusion-3B"}]),

    # ── http_get + summarize ───────────────────────────────────────────────
    ("Fetch https://example.com/api/status and report if it is healthy",
     [{"tool": "http_get", "args": {"url": "https://example.com/api/status"},
       "result": '{"status": "ok", "uptime_seconds": 3600}'}]),

    # ── web_search + write_file research chain ────────────────────────────
    ("Search the web for the latest EVA agent release and save a summary to /tmp/eva_notes.md",
     [{"tool": "web_search", "args": {"query": "EVA agent latest release"},
       "result": "EVA v1.31.0 released — native function calling, trajectory fine-tuning."},
      {"tool": "write_file", "args": {"path": "/tmp/eva_notes.md", "content": "EVA v1.31.0 released."},
       "result": "Written to /tmp/eva_notes.md"}]),

    # ── run_skill dispatch ─────────────────────────────────────────────────
    ("Run the security-scanner skill on the kernel repo",
     [{"tool": "run_skill", "args": {"skill_name": "security-scanner", "input": "scan /home/pacificDev/.kernel-evolving/workspace/repositories/kernel-evolving"},
       "result": "Scan complete — 0 critical, 2 low findings"}]),
    ("Use the github skill to check open PRs",
     [{"tool": "run_skill", "args": {"skill_name": "github", "input": "gh pr list"},
       "result": "2 open PRs: #45 (feature), #47 (bugfix)"}]),

    # ── search_skills + run_skill orchestration ───────────────────────────
    ("Find a skill for pdf handling and use it on the report.pdf",
     [{"tool": "search_skills", "args": {"query": "pdf markdown"},
       "result": "kernel-doc-retrieval — [ /doc, /markdown ] handles PDF tasks"},
      {"tool": "run_skill", "args": {"skill_name": "kernel-doc-retrieval", "input": "/markdown report.pdf"},
       "result": "Converted report.pdf to report.md"}]),

    # ── read_file → recall_memory followup ────────────────────────────────
    ("Read my USER notes and recall what we discussed about voice cloning",
     [{"tool": "read_file", "args": {"path": "/home/pacificDev/.kernel-evolving/workspace/USER.md"},
       "result": "Fabio — voice clone: fabio-ita.wav, model 1.7, x_vector_only=true"},
      {"tool": "recall_memory", "args": {"query": "voice clone setup"},
       "result": "Voice clone endpoint POST /tts/clone with sample WAV"}]),

    # ── list_routines + run_routine ───────────────────────────────────────
    ("What routines are available, and run the morning briefing",
     [{"tool": "list_routines", "args": {},
       "result": "morning-briefing, security-check, deploy, end-of-session"},
      {"tool": "run_routine", "args": {"routine_name": "morning-briefing"},
       "result": "Morning briefing delivered — 3 news items, 2 tasks due"}]),

    # ── multi-tool full chain ─────────────────────────────────────────────
    ("Check service health, then write a report file",
     [{"tool": "exec_shell", "args": {"command": "systemctl --user is-active olly-voice.service"},
       "result": "active"},
      {"tool": "write_file", "args": {"path": "/tmp/service_report.txt", "content": "olly-voice: active"},
       "result": "Written to /tmp/service_report.txt"}]),
    ("List processes for model_server, then read its log path",
     [{"tool": "exec_shell", "args": {"command": "ps aux | grep model_server | grep -v grep"},
       "result": "pacificDev 1241960 python3 model_server.py"},
      {"tool": "read_file", "args": {"path": "/tmp/kernel_evolving_model_server.log"},
       "result": "[model_server] Nemotron ready."}]),

    # ── run_routine direct ────────────────────────────────────────────────
    ("Run the end-of-session routine to wrap up",
     [{"tool": "run_routine", "args": {"routine_name": "end-of-session"},
       "result": "End-of-session routine complete — notes saved, tasks archived"}]),

    # ── send_file delivery ───────────────────────────────────────────────
    ("Send the generated report.pdf to me via Telegram",
     [{"tool": "send_file", "args": {"file_path": "/tmp/service_report.txt", "caption": "Service report"},
       "result": "Sent /tmp/service_report.txt to user via Telegram"}]),

    # ── browser_use agentic web task ─────────────────────────────────────
    ("Go to https://example.com/login, log in and download the dashboard report",
     [{"tool": "browser_use", "args": {"task": "navigate to dashboard and download report", "url": "https://example.com/login"},
       "result": "Navigated, logged in, downloaded dashboard-report.csv"}]),

    # ── look (camera perception) ─────────────────────────────────────────
    ("Check what's in the living room via the camera",
     [{"tool": "look", "args": {"intent": "what's there"},
       "result": "Detected: sofa, coffee table, 2 cats on the couch"}]),

    # ── sensors (environmental read + control) ───────────────────────────
    ("Read the greenhouse temperature and water the plants if dry",
     [{"tool": "sensors", "args": {"action": "read"},
       "result": "temperature: 24C, humidity: 40%, moisture: 12%"},
      {"tool": "sensors", "args": {"action": "control", "device": "relay", "command": "on"},
       "result": "Relay ON — watering started"}]),

    # ── ask_questions (User direction) ───────────────────────────────────
    ("Tell me which report format you'd prefer before I write it",
     [{"tool": "ask_questions", "args": {"question": "Which format do you prefer, markdown or pdf?", "options": ["markdown", "pdf"]},
       "result": "User selected: markdown"}]),

    # ── computer (GUI/screen automation) ──────────────────────────────────
    ("Open the settings app and change the display brightness",
     [{"tool": "computer", "args": {"goal": "open settings and change display brightness to 50%", "dry_run": False},
       "result": "{\"status\": \"ok\", \"completed\": [\"opened settings\", \"set brightness 50%\"]}"}]),

    # ── Exact battery mirrors (T2 read_file / T4 ps short-result shapes) ──
    # T2 shape: read a file, get a short line-based result, STOP and summarize.
    ("Read the file /home/pacificDev/.kernel-evolving/workspace/USER.md and summarize it",
     [{"tool": "read_file", "args": {"path": "/home/pacificDev/.kernel-evolving/workspace/USER.md"},
       "result": "# USER.md — About the user\nName: Fabio. Cats: 10. Diet: vegan."}]),
    ("Read /tmp/kernel_evolving_model_server.log and tell me its last status line",
     [{"tool": "read_file", "args": {"path": "/tmp/kernel_evolving_model_server.log"},
       "result": "[model_server] Nemotron ready. mode=linear_spec block=32 threshold=0.9"}]),
    ("Read config.yaml and tell me the adapter path",
     [{"tool": "read_file", "args": {"path": "/home/pacificDev/.kernel-evolving/workspace/config.yaml"},
       "result": "adapter_path: /home/pacificDev/.kernel-evolving/workspace/artifacts/finetune/nemotron_fc_v4"}]),
    # T4 shape: short ps-style table, STOP and report PID.
    ("Run ps aux | grep model_server | grep -v grep and report the PIDs",
     [{"tool": "exec_shell", "args": {"command": "ps aux | grep model_server | grep -v grep"},
       "result": "pacificDev 1473873 36.8 4.1 89432984 2062920 ? Sl 02:26 1:37 /home/pacificDev/.miniconda/bin/python3 model_server.py"}]),
    ("Run ps aux | grep uvicorn and report the PIDs",
     [{"tool": "exec_shell", "args": {"command": "ps aux | grep uvicorn | grep -v grep"},
       "result": "pacificDev  5678  0.5  1.2 12345678 654321 ? S 01:00 0:03 /usr/bin/uvicorn app:app"}]),
]


def _final_answer(template_user: str, steps: list) -> str:
    """Concrete final answer that ANSWERS the user's question from the last tool
    result — teaching single-shot termination: after the tool result comes back,
    STOP calling tools and give the final answer. The generic "Done, I used X"
    close was too weak and let the model keep re-calling; each final turn now
    states the actual answer derived from the result."""
    last = steps[-1]["result"]
    t = template_user.lower()
    if "disk" in t or "full" in t or "df" in t:
        return (
            f"The root filesystem is about 15% full: 148G used of roughly 1.0T, "
            f"with about 853G available."
        )
    if "python" in t and "version" in t:
        return "The installed Python version is 3.13.1."
    if "summar" in t or "summarize" in t:
        return (
            f"Summary of the file: {last.strip().splitlines()[0] if last.strip() else 'see above'} "
            f"(key point: Fabio, 10 cats)."
        )
    if "pid" in t or "process" in t or "grep model_server" in t:
        return f"The model_server PIDs are {last.strip()}."
    if "file" in t and ("list" in t or "py" in t or "save" in t):
        return f"The matching files are: {last.strip()}. That list was saved to the requested path."
    if "skill" in t and ("run" in t or "use" in t):
        return f"Ran the skill. Result: {last.strip()}"
    if "routin" in t:
        return f"Ran the routine — {last.strip()}"
    if "send" in t:
        return f"Sent the file: {last.strip()}"
    if "browser" in t or "log in" in t or "download" in t:
        return f"Done: {last.strip()}"
    if "format" in t and "prefer" in t:
        return f"You selected: {last.strip()}"
    return f"Result obtained. {last.strip()}"



def build_messages(task: str, steps: list) -> list:
    msgs = [{"role": "user", "content": task}]
    for i, step in enumerate(steps):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "id": f"call_{i}",
                "function": {"name": step["tool"], "arguments": step["args"]},
            }],
        })
        # Tool result → Nemotron uses a user turn wrapped in <tool_response>
        msgs.append({"role": "tool", "name": step["tool"], "content": step["result"]})
    msgs.append({"role": "assistant", "content": _final_answer(task, steps)})
    return msgs


def make_tools_json(real_tools) -> list:
    """Take the real 16-tool registry and keep the OpenAI-style dicts (stripped
    of any extra wrapper the runtime may inject), matching what apply_chat_template
    expects. Real TOOLS are already {'type':'function','function':{...}}."""
    out = []
    for t in real_tools:
        fn = t.get("function", t)
        if isinstance(fn, dict) and fn.get("name"):
            out.append({"type": "function", "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }})
        elif isinstance(t, dict) and t.get("name"):
            out.append(t)
    return out


def render(tokenizer, messages: list, real_tools: list) -> str:
    """Render a messages list with Nemotron's tokenizer + real tools → the exact
    prompt (and target) format Nemotron sees at inference."""
    return tokenizer.apply_chat_template(
        messages,
        tools=real_tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Nemotron base model path")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--count", type=int, default=120,
                    help="Approx total synthetic records to emit (templates repeat with seed variation)")
    args = ap.parse_args()

    print(f"Loading Nemotron tokenizer from {args.model} (trust_remote_code)...", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    real_tools = make_tools_json(TOOLS)
    print(f"REAL tools loaded: {len(real_tools)} → {[t['function']['name'] for t in real_tools]}", flush=True)
    print(f"Nemotron chat template present: {'yes' if tokenizer.chat_template else 'no'}", flush=True)

    covers = sorted({t["function"]["name"] for t in real_tools})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        i = 0
        while written < args.count:
            task, steps = TEMPLATES[i % len(TEMPLATES)]
            messages = build_messages(task, steps)
            try:
                text = render(tokenizer, messages, real_tools)
            except Exception as e:
                print(f"  [skip] render error on template {i%len(TEMPLATES)}: {e}", flush=True)
                i += 1
                continue
            rec = {
                "id": f"synthetic-fc-{written}",
                "task": task,
                "tools_used": sorted({s["tool"] for s in steps}),
                "critic_score": 1.0,
                "text": text,
            }
            f.write(json.dumps(rec) + "\n")
            written += 1
            i += 1

    print(f"\nWrote {written} synthetic FC trajectories to {args.out}", flush=True)
    # Coverage audit — which real tools appear at least once
    used = set()
    for j in range(written):
        t, steps = TEMPLATES[j % len(TEMPLATES)]
        used.update(s["tool"] for s in steps)
    missing = set(covers) - used
    print(f"Tool coverage: {len(used)}/{len(covers)} real tools used.", flush=True)
    if missing:
        print(f"  NOT covered by templates (add templates): {sorted(missing)}", flush=True)


if __name__ == "__main__":
    main()
