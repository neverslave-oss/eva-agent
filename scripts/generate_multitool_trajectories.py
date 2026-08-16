#!/usr/bin/env python3
"""
generate_multitool_trajectories.py
====================================
Dedicated batch for deep multi-tool chains (3-6 steps).
Each task is engineered to REQUIRE a specific tool sequence —
the agent cannot complete it without calling all the tools in order.

Target sequences:
  exec → read → write                  (3-step)
  http → read → write                  (3-step)
  web  → exec → write                  (3-step)
  skill → read → write                 (3-step)
  exec → exec → read → write           (4-step)
  exec → http → write                  (3-step)
  read → exec → exec → write           (4-step)
  exec → read → exec → write           (4-step)
  web  → read → write                  (3-step)
  http → exec → read → write           (4-step)
  exec → exec → exec → write           (4-step - pipeline)
  skill → exec → write                 (3-step)
  exec → write → exec → write          (4-step - create + verify)
  web  → exec → read → write           (4-step)
  exec → http → exec → write           (4-step)
  read → read → write                  (3-step - combine two files)
  exec → exec → exec → exec → write    (5-step - full env audit)
  web  → web  → write                  (3-step - multi-search synthesise)
  skill → skill → write                (3-step - multi-skill synthesise)
  http → http → write                  (3-step - multi-api combine)
"""
import argparse, json, os, sys, time, re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Task set — each task FORCES a specific tool sequence ─────────────────────

MULTI_TOOL_TASKS = [

    # ── 3-step: exec → read → write ───────────────────────────────────────────
    ("Run 'cat ~/.kernel-evolving/config.yaml | head -20' to get the config header, then read ~/evo_identity.txt, and write both combined into ~/evo_config_snapshot.md",
     "mt3:exec-read-write"),
    ("Run 'pip list | grep -E \"torch|transformers|trl\"' to check ML packages, read ~/evo_status.txt if it exists, and write an ML environment summary to ~/evo_ml_env.md",
     "mt3:exec-read-write-ml"),
    ("Run 'ps aux | grep python | grep -v grep' to list Python processes, read ~/evo_identity.txt, and write a combined runtime snapshot to ~/evo_runtime.md",
     "mt3:exec-read-write-runtime"),

    # ── 3-step: http → read → write ───────────────────────────────────────────
    ("Fetch https://httpbin.org/uuid to get a unique ID, read ~/evo_status.txt if it exists, and write a versioned snapshot combining both to ~/evo_snapshot.txt",
     "mt3:http-read-write"),
    ("GET https://httpbin.org/get to get connection info, read ~/evo_config.json if it exists, and write a combined connectivity report to ~/evo_conn_report.md",
     "mt3:http-read-write-conn"),

    # ── 3-step: web → exec → write ────────────────────────────────────────────
    ("Search for 'Python asyncio best practices 2026', then run 'python3 -c \"import asyncio; print(asyncio.__version__)\"' to check local version, and write a combined note to ~/evo_asyncio.md",
     "mt3:web-exec-write"),
    ("Search for 'CUDA toolkit version compatibility PyTorch 2026', then run 'nvidia-smi --query-gpu=driver_version --format=csv,noheader' to get local driver, and write a compatibility note to ~/evo_cuda_compat.md",
     "mt3:web-exec-gpu"),
    ("Search for 'FastAPI health check endpoint pattern 2026', then run 'curl -s http://localhost:8779/health' to check our local API, and write a comparison to ~/evo_health_pattern.md",
     "mt3:web-exec-healthcheck"),

    # ── 3-step: skill → read → write ──────────────────────────────────────────
    ("Use the collective-memory skill to search for 'kernel-evolving architecture', then read ~/evo_identity.txt if it exists, and write a combined context note to ~/evo_arch_context.md",
     "mt3:skill-read-write"),
    ("Run the open-workspace-tracker skill to get current todos, then read ~/evo_status.txt if it exists, and write a combined session brief to ~/evo_session_brief.md",
     "mt3:skill-read-write-brief"),

    # ── 3-step: read → exec → write ───────────────────────────────────────────
    ("Read ~/evo_identity.txt, then run 'date && uptime' to get current time and load, and append the runtime info to ~/evo_identity.txt",
     "mt3:read-exec-append"),
    ("Read ~/evo_config.json if it exists (create it with default values if not), then run 'python3 --version' and add the Python version to the config, and write it back",
     "mt3:read-exec-write-config"),

    # ── 4-step: exec → exec → read → write ────────────────────────────────────
    ("Run 'df -h' to check disk, then run 'free -h' to check RAM, then read ~/evo_status.txt if it exists, and write a full system health report to ~/evo_health_full.md",
     "mt4:exec-exec-read-write"),
    ("Run 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader' for VRAM, run 'python3 -c \"import torch; print(torch.cuda.is_available())\"' for PyTorch, read ~/evo_ml_env.md if it exists, and write an updated ML status to ~/evo_ml_status.md",
     "mt4:exec-exec-read-write-gpu"),

    # ── 4-step: read → exec → exec → write ────────────────────────────────────
    ("Read ~/evo_config.json, run 'curl -s http://localhost:8779/health' to check the API, run 'pgrep -c python' to count Python processes, and write a combined status report to ~/evo_full_status.md",
     "mt4:read-exec-exec-write"),

    # ── 4-step: exec → read → exec → write ────────────────────────────────────
    ("Run 'git -C ~/.kernel-evolving log --oneline -5' to see recent commits, read the latest changelog or README if it exists at ~/.kernel-evolving/CHANGELOG.md, run 'curl -s http://localhost:8779/health | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115)))\"' for status, and write a release summary to ~/evo_release.md",
     "mt4:exec-read-exec-write"),
    ("Run 'ls ~/evo_*.md 2>/dev/null | wc -l' to count our trajectory output files, read the newest one, run 'du -sh ~/ 2>/dev/null | head -1' for disk usage, and write a data collection summary to ~/evo_data_summary.md",
     "mt4:exec-read-exec-write-summary"),

    # ── 4-step: http → exec → read → write ────────────────────────────────────
    ("Fetch https://api.github.com/repos/fabiopacifici-bot/kernel-evolving/releases/latest to get the latest version, run 'curl -s http://localhost:8779/health' to get running version, read ~/evo_status.txt if it exists, and write a version comparison to ~/evo_version_diff.md",
     "mt4:http-exec-read-write"),

    # ── 4-step: web → read → exec → write ─────────────────────────────────────
    ("Search for 'Gemma 4 context window token limit 2026', read ~/evo_identity.txt if it exists, run 'python3 -c \"import transformers; print(transformers.__version__)\"' to check local version, and write a compatibility note to ~/evo_gemma4_compat.md",
     "mt4:web-read-exec-write"),

    # ── 4-step: exec → write → exec → write (create + verify) ────────────────
    ("Write a Python script to ~/evo_test_script.py that prints 'kernel-evolving health check OK', then run it with python3, then write the execution result to ~/evo_test_output.txt",
     "mt4:write-exec-write-verify"),
    ("Create a shell script at ~/evo_check.sh that curls http://localhost:8779/health, then make it executable with chmod +x, then run it, and save the output to ~/evo_check_result.txt",
     "mt4:write-exec-exec-write"),

    # ── 5-step: full environment audit ────────────────────────────────────────
    ("Run 'python3 --version', 'pip list | grep -c .', 'df -h | tail -1', 'free -h | grep Mem', and 'curl -s http://localhost:8779/health' — collect all outputs and write a complete environment audit to ~/evo_env_audit.md",
     "mt5:5exec-write"),
    ("Run 'hostname', 'uname -r', 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', 'python3 -c \"import torch; print(torch.version.cuda)\"', then read ~/evo_identity.txt, and write a full hardware + software profile to ~/evo_hw_profile.md",
     "mt5:exec-exec-exec-exec-read-write"),

    # ── 3-step: multi-search → synthesise ─────────────────────────────────────
    ("Search for 'HuggingFace TRL GRPO training 2026', then search for 'DPO reward model training 2026', and write a comparison of both methods to ~/evo_rl_methods.md",
     "mt3:web-web-write"),
    ("Search for 'Gemma 4 multimodal capabilities 2026', then search for 'Gemma 4 E2B-it tool calling performance', and synthesise findings to ~/evo_gemma4_full.md",
     "mt3:web-web-write-gemma"),

    # ── 3-step: multi-http → write ────────────────────────────────────────────
    ("Fetch https://httpbin.org/uuid for a unique session ID, then fetch https://httpbin.org/get for connection details, and write both to a combined session record at ~/evo_session.json",
     "mt3:http-http-write"),

    # ── 4-step: skill → web → exec → write ────────────────────────────────────
    ("Use collective-memory to search for 'kernel-evolving dataset', then search the web for 'SFT dataset size recommendations 2026', then run 'python3 -c \"import sqlite3,os; db=sqlite3.connect(os.path.expanduser(chr(126)+chr(47)+chr(46)+chr(107)+chr(101)+chr(114)+chr(110)+chr(101)+chr(108)+chr(45)+chr(101)+chr(118)+chr(111)+chr(108)+chr(118)+chr(105)+chr(110)+chr(103)+chr(47)+chr(119)+chr(111)+chr(114)+chr(107)+chr(115)+chr(112)+chr(97)+chr(99)+chr(101)+chr(47)+chr(101)+chr(118)+chr(111)+chr(108)+chr(117)+chr(116)+chr(105)+chr(111)+chr(110)+chr(46)+chr(100)+chr(98))); print(db.execute(chr(83)+chr(69)+chr(76)+chr(69)+chr(67)+chr(84)+chr(32)+chr(67)+chr(79)+chr(85)+chr(78)+chr(84)+chr(40)+chr(42)+chr(41)+chr(32)+chr(70)+chr(82)+chr(79)+chr(77)+chr(32)+chr(116)+chr(97)+chr(115)+chr(107)+chr(95)+chr(116)+chr(114)+chr(97)+chr(106)+chr(101)+chr(99)+chr(116)+chr(111)+chr(114)+chr(105)+chr(101)+chr(115)).fetchone()[0])\"' to count our trajectories, and write a readiness assessment to ~/evo_dataset_readiness.md",
     "mt4:skill-web-exec-write"),

    # ── Multi-turn 4-step chains ───────────────────────────────────────────────
    (["The model I want to fine-tune is at ~/models/huggingface/hub/models--google--gemma-4-E2B-it",
      "Check how big it is on disk, also check free VRAM with nvidia-smi, and write a fine-tuning feasibility note to ~/evo_finetune_feasibility.md"],
     "mt4:multiturn-exec-exec-write"),
    (["I need to track progress of our trajectory generation",
      "Count total records in ~/.kernel-evolving/workspace/data/evolution.db trajectories table, check how many are olly-teacher provider, then read ~/evo_status.txt if exists, and write a progress report to ~/evo_progress_report.md"],
     "mt4:multiturn-exec-read-write"),
]

# ── System prompt with Olly context ──────────────────────────────────────────

_OLLY_WORKSPACE = Path.home() / ".openclaw" / "workspace"

def _load_file(path: Path, max_chars: int = 1500) -> str:
    try:
        t = path.read_text().strip()
        return t[:max_chars] + "\n...[truncated]" if len(t) > max_chars else t
    except Exception:
        return ""

def _extract_relevant(text: str, keywords: list, max_chars: int = 1200) -> str:
    lines = text.split("\n"); relevant, window = [], []
    for line in lines:
        window.append(line)
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant.extend(window); window = []
        if len(window) > 5: window = window[-5:]
    result = "\n".join(relevant)
    return (result[:max_chars] + "\n...[truncated]") if len(result) > max_chars else result.strip()

def build_system_prompt() -> str:
    soul = _load_file(_OLLY_WORKSPACE / "SOUL.md", 1200)
    user = _load_file(_OLLY_WORKSPACE / "USER.md", 500)
    memory_text = _load_file(_OLLY_WORKSPACE / "MEMORY.md", 99999)
    memory_rel = _extract_relevant(memory_text,
        ["kernel-evolving","drafter","trajectory","fine-tun","Gemma","dataset","multi-tool","tool_call"], 1200)
    from datetime import datetime
    today = _load_file(_OLLY_WORKSPACE / "memory" / f"{datetime.now().strftime('%Y-%m-%d')}.md", 1000)

    return f"""You are Olly — generating training data for kernel-evolving's Gemma 4 MTP drafter.

You are demonstrating MULTI-TOOL CHAINS — tasks that require 3-6 tool calls in sequence.
The drafter must learn to plan and execute ordered chains without shortcuts.

## Your identity
{soul}

## Your user
{user}

## kernel-evolving context
{memory_rel}

## Today
{today}

## Tools
  • exec_shell(command)           — run shell commands
  • read_file(path)               — read a file
  • write_file(path, content)     — write a file
  • http_get(url)                 — HTTP GET
  • web_search(query)             — DuckDuckGo search
  • run_skill(skill_name, input)  — execute a skill
  • run_routine(routine_name)     — execute a routine

## Critical rules for multi-tool training data
1. CHAIN ALL TOOLS — every tool in the task description MUST be called.
2. Each tool result feeds the next step — use actual outputs, never fabricate.
3. write_file must contain REAL content from prior tool results.
4. Correct order matters — plan before executing. Follow the task sequence.
5. Use ~ for paths. Never use /home/pacificDev or any absolute home path.
6. Final reply: one line confirming what was written and where.
7. Never skip a tool call and claim completion without it."""

# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {"type":"function","function":{"name":"exec_shell","description":"Run a shell command. Returns stdout+stderr.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file's contents.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write content to a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"http_get","description":"HTTP GET request.","parameters":{"type":"object","properties":{"url":{"type":"string"},"timeout":{"type":"integer"}},"required":["url"]}}},
    {"type":"function","function":{"name":"web_search","description":"DuckDuckGo search. Returns real results.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"run_skill","description":"Execute an installed skill.","parameters":{"type":"object","properties":{"skill_name":{"type":"string"},"input":{"type":"string"}},"required":["skill_name","input"]}}},
    {"type":"function","function":{"name":"run_routine","description":"Execute a routine.","parameters":{"type":"object","properties":{"routine_name":{"type":"string"}},"required":["routine_name"]}}},
]

# ── Token loader ──────────────────────────────────────────────────────────────

def _load_token() -> str:
    import json as _j, time as _t
    cred = Path.home() / ".openclaw/credentials/github-copilot.token.json"
    if cred.exists():
        try:
            d = _j.loads(cred.read_text())
            remaining = d.get("expiresAt", 0)/1000 - _t.time()
            if remaining > 30:
                token = d["token"]; os.environ["GITHUB_TOKEN"] = token; return token
            else:
                print(f"[mt-teacher] Token expires in {remaining:.0f}s — waiting..."); _t.sleep(35)
                d = _j.loads(cred.read_text()); token = d["token"]
                os.environ["GITHUB_TOKEN"] = token; return token
        except Exception as e:
            print(f"[mt-teacher] Token error: {e}")
    return os.environ.get("GITHUB_TOKEN", "")

# ── Save ──────────────────────────────────────────────────────────────────────

def save_trajectory(col, task_label, category, steps, reply, prov, min_score=0.7) -> bool:
    if len(steps) < 2: return False  # must have at least 2 tool calls
    error_markers = ("[model_server error]","CUDA error","out of memory","(error:")
    if any(m in (reply or "") for m in error_markers): return False

    write_steps = [s for s in steps if s.get("tool") == "write_file"]
    if not write_steps: return False  # must write something

    try:
        step_summary = " → ".join(s["tool"] for s in steps)
        tool_results = "\n".join(f"  {s['tool']}: {str(s.get('result',''))[:120]}" for s in steps[:5])
        crit_prompt = (
            f"Rate this multi-tool AI agent response 0.0-1.0.\n\n"
            f"1.0 = all required tools called in order, real outputs used, file written with actual content\n"
            f"0.8 = most tools called, file written, minor gap\n"
            f"0.6 = some tools called but chain incomplete or content vague\n"
            f"0.4 = skipped key tools or file content is fabricated\n"
            f"0.0 = no chain, no file, or completely wrong\n\n"
            f"Task: {str(task_label)[:200]}\nTool chain: {step_summary}\n"
            f"Tool results:\n{tool_results}\nFinal reply: {(reply or '')[:200]}\n\nRate:"
        )
        crit = prov.infer([{"role":"user","content":crit_prompt}], max_new_tokens=10, call_type="critic")
        m = re.search(r"\b(0\.\d+|1\.0|1)\b", crit)
        score = float(m.group(1)) if m else 0.7
    except Exception:
        score = 0.8 if len(steps) >= 3 and write_steps else 0.6

    n = len(steps)
    effective_min = max(0.4, min_score - 0.3) if n >= 3 else min_score
    if score < effective_min: return False

    verdict = "PASS" if score >= min_score else "PARTIAL"
    artifacts = json.dumps([s["args"].get("path","") for s in write_steps])
    col.record(
        task=f"[mt-teacher] {str(task_label)[:200]}",
        provider=f"mt-teacher/{category}",
        model_name="gpt-5.4",
        call_type="task_inference",
        tool_calls=steps,
        final_reply=reply or "",
        artifacts=artifacts,
        critic_score=score,
        critic_verdict=verdict,
    )
    return True

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = MULTI_TOOL_TASKS[:args.count] if args.count else MULTI_TOOL_TASKS

    if args.dry_run:
        print(f"Multi-tool tasks ({len(tasks)}):")
        for spec in tasks:
            label = " → ".join(spec[0]) if isinstance(spec[0], list) else spec[0]
            print(f"  [{spec[1]}] {str(label)[:80]}")
        return

    import yaml
    with open(_ROOT / "config.yaml") as f: config = yaml.safe_load(f)

    from provider import InferenceProvider
    from trajectory_collector import get_collector

    config.setdefault("providers", {})
    config["providers"]["task_inference"] = "copilot"
    config["providers"].setdefault("models", {})["copilot"] = "gpt-5.4"

    prov = InferenceProvider(config)
    col  = get_collector(config)
    workspace = os.path.expanduser(config.get("kernel_workspace","~/.kernel-evolving/workspace"))
    system_prompt = build_system_prompt()

    saved_total = 0
    print(f"\n{'='*60}")
    print(f"Multi-Tool Chain Generator (GPT-5.4 + Olly context)")
    print(f"Tasks: {len(tasks)} | Min depth: 2 tools | Min score: {args.min_score}")
    print(f"{'='*60}")

    for i, spec in enumerate(tasks, 1):
        is_multiturn = isinstance(spec[0], list)
        label = spec[0] if not is_multiturn else spec[0]
        category = spec[1]
        print(f"\n[{i}/{len(tasks)}] [{category}] {str(label[0] if is_multiturn else label)[:80]}")
        _load_token()

        steps = []
        def _cb(n, tool, targs, result, _s=steps):
            _s.append({"tool": tool, "args": targs, "result": str(result)[:600]})

        try:
            if is_multiturn:
                turns = spec[0]; history = []
                for turn in turns:
                    msgs = [{"role":"system","content":system_prompt}] + history + [{"role":"user","content":turn}]
                    reply = prov.infer_with_tools(msgs, tools=TOOLS, workspace=workspace,
                                                   max_steps=8, step_callback=_cb, call_type="task_inference")
                    history += [{"role":"user","content":turn}, {"role":"assistant","content":reply or ""}]
            else:
                reply = prov.infer_with_tools(
                    messages=[{"role":"system","content":system_prompt}, {"role":"user","content":label}],
                    tools=TOOLS, workspace=workspace, max_steps=8,
                    step_callback=_cb, call_type="task_inference",
                )

            saved = save_trajectory(col, label, category, steps, reply, prov, args.min_score)
            tool_names = [s["tool"] for s in steps]
            depth_str = f"depth={len(steps)}"
            print(f"  {depth_str} Tools: {' → '.join(tool_names)} | Saved: {'✅' if saved else '⏭'}")
            if saved: saved_total += 1
        except Exception as e:
            print(f"  ❌ {e}")
        time.sleep(1.5)

    print(f"\n{'='*60}")
    print(f"Done. Saved {saved_total}/{len(tasks)}")
    print(f"{'='*60}")

    if args.export:
        path, count = get_collector(config).export_jsonl(min_score=args.min_score)
        print(f"\nExported {count} records → {path}")

if __name__ == "__main__":
    main()
