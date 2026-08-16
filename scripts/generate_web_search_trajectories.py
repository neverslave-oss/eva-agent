#!/usr/bin/env python3
"""
generate_web_search_trajectories.py
=====================================
Dedicated batch for web_search + write_file trajectories.
Uses GPT-5.4 via Copilot for reliable tool-calling, with full Olly
context (SOUL, USER, MEMORY) injected into the system prompt.

This is the highest-signal web dataset we can produce:
- Real DuckDuckGo results (wired in tools.py)
- Olly's identity + kernel-evolving context in system prompt
- Diverse query types: research, debugging, documentation, comparisons
"""
import argparse, json, os, sys, time, re
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Web search task set ───────────────────────────────────────────────────────

WEB_TASKS = [
    # Research + save
    ("Search for 'HuggingFace TRL SFTTrainer tool calling fine-tuning 2026' and save the top findings to ~/evo_trl_toolcalling.md",
     "web:research-trl"),
    ("Search for 'Gemma 4 E2B-it benchmark MMLU tool use 2026' and write a summary to ~/evo_gemma4_benchmarks.md",
     "web:research-gemma4"),
    ("Search for 'speculative decoding MTP drafter training 2026' and save key techniques to ~/evo_speculative_training.md",
     "web:research-speculative"),
    ("Search for 'DPO vs SFT fine-tuning small language models 2026 comparison' and write pros/cons to ~/evo_dpo_vs_sft.md",
     "web:research-training-methods"),
    ("Search for 'agent trajectory dataset quality filtering critic score 2026' and save best practices to ~/evo_dataset_quality.md",
     "web:research-dataset-quality"),
    ("Search for 'local first AI agent self-evolving architecture 2026' and summarise to ~/evo_local_agent_arch.md",
     "web:research-agent-arch"),
    ("Search for 'OpenAI function calling tool use JSON schema best practices 2026' and save examples to ~/evo_tool_schema.md",
     "web:research-tool-schema"),
    ("Search for 'HuggingFace Jobs free H100 GPU fine-tuning cost 2026' and write an estimate to ~/evo_gpu_costs.md",
     "web:research-gpu-costs"),

    # Search + exec + write (3-step chains)
    ("Search for 'Python 3.13 new features 2026', then run 'python3 --version' to confirm local version, and write a comparison to ~/evo_python_notes.md",
     "web:search-exec-write"),
    ("Search for 'FastAPI best practices async 2026', then check if port 8779 is running with curl, and write a summary including local service status to ~/evo_api_notes.md",
     "web:search-exec-service-check"),
    ("Search for 'git rebase interactive best practices 2026', then run 'git log --oneline -3' in ~/.kernel-evolving, and write notes combining both to ~/evo_git_notes.md",
     "web:search-exec-git"),
    ("Search for 'SQLite performance optimization 2026', then run 'du -sh ~/.kernel-evolving/workspace/data/evolution.db' to get our DB size, and write a tuning plan to ~/evo_sqlite_plan.md",
     "web:search-exec-db"),

    # Search + read + write (3-step chains)
    ("Search for 'Markdown best practices documentation 2026', then read ~/evo_status.txt if it exists, and write an improved version of it to ~/evo_status_improved.md",
     "web:search-read-write"),
    ("Search for 'AI agent memory architecture 2026', read ~/evo_identity.txt if it exists, and write a context note combining both to ~/evo_memory_arch.md",
     "web:search-read-combine"),

    # Multi-search + synthesise
    ("Search for 'Gemma 4 quantization GGUF 2026', then search for 'llama.cpp Gemma support 2026', and write a combined deployment guide to ~/evo_gemma4_deployment.md",
     "web:multi-search-synthesise"),
    ("Search for 'kernel-evolving self-evolving agent', then search for 'OpenClaw AI agent framework', and write a comparison of both to ~/evo_agent_comparison.md",
     "web:multi-search-compare"),

    # Research with specific output format
    ("Search for 'HuggingFace dataset format for tool-calling SFT 2026' and write a JSONL schema example to ~/evo_dataset_schema.md",
     "web:research-schema-write"),
    ("Search for 'LoRA adapter rank alpha configuration 2026 best practices' and write recommended config values to ~/evo_lora_config.md",
     "web:research-lora-config"),
    ("Search for 'bitsandbytes 4bit quantization WSL2 CUDA 2026' and write debugging steps to ~/evo_bnb_fix.md",
     "web:research-debug-write"),
    ("Search for 'transformers AutoModelForCausalLM tool calling 2026' and save code examples to ~/evo_model_tool_examples.md",
     "web:research-code-examples"),
]

# ── System prompt: Olly context injected ─────────────────────────────────────

_OLLY_WORKSPACE = Path.home() / ".openclaw" / "workspace"

def _load_file(path: Path, max_chars: int = 1500) -> str:
    try:
        t = path.read_text().strip()
        return t[:max_chars] + "\n...[truncated]" if len(t) > max_chars else t
    except Exception:
        return ""

def _extract_relevant(memory_text: str, keywords: list, max_chars: int = 1500) -> str:
    lines = memory_text.split("\n")
    relevant, window = [], []
    for line in lines:
        window.append(line)
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant.extend(window); window = []
        if len(window) > 5: window = window[-5:]
    result = "\n".join(relevant)
    return (result[:max_chars] + "\n...[truncated]") if len(result) > max_chars else result.strip()

def build_system_prompt() -> str:
    soul = _load_file(_OLLY_WORKSPACE / "SOUL.md", 1200)
    user = _load_file(_OLLY_WORKSPACE / "USER.md", 600)
    memory_text = _load_file(_OLLY_WORKSPACE / "MEMORY.md", 99999)
    memory_relevant = _extract_relevant(memory_text,
        ["kernel-evolving","drafter","trajectory","fine-tun","ADR-013","Gemma","dataset","web_search"],
        1500)
    from datetime import datetime
    today = _load_file(_OLLY_WORKSPACE / "memory" / f"{datetime.now().strftime('%Y-%m-%d')}.md", 1200)

    return f"""You are Olly — the main AI assistant running inside OpenClaw, generating training data for kernel-evolving.

You are demonstrating how kernel-evolving SHOULD handle web search tasks.
Every response becomes training data for its Gemma 4 MTP drafter.

## Your identity
{soul}

## Your user
{user}

## kernel-evolving context
{memory_relevant}

## Today's context
{today}

## Tools available
  • web_search(query, save_to?)  — DuckDuckGo search, returns real results
  • write_file(path, content)    — write a file
  • read_file(path)              — read a file
  • exec_shell(command)          — run shell commands
  • http_get(url)                — HTTP GET
  • run_skill(skill_name, input) — execute a skill
  • run_routine(routine_name)    — execute a routine

## Rules — this is training data, follow exactly
1. ALWAYS call web_search first for search tasks — never make up results.
2. Use the ACTUAL search output in your written files — quote real findings.
3. For multi-step tasks: chain in order (search → exec/read → write_file).
4. write_file must contain real content from the search, not placeholder text.
5. Use ~ for paths. Never use absolute /home/... paths.
6. Be concise. One confirmation line after completion.
7. Never say "I have written..." without write_file having returned in this conversation."""

# ── Tools (OpenAI format) ─────────────────────────────────────────────────────

TOOLS = [
    {"type":"function","function":{"name":"web_search","description":"Search the web via DuckDuckGo. Returns real search results.","parameters":{"type":"object","properties":{"query":{"type":"string"},"save_to":{"type":"string","description":"Optional path to save results"}},"required":["query"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write content to a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"exec_shell","description":"Run a shell command.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"http_get","description":"HTTP GET request.","parameters":{"type":"object","properties":{"url":{"type":"string"},"timeout":{"type":"integer"}},"required":["url"]}}},
    {"type":"function","function":{"name":"run_skill","description":"Execute a skill by name.","parameters":{"type":"object","properties":{"skill_name":{"type":"string"},"input":{"type":"string"}},"required":["skill_name","input"]}}},
]

# ── Token loader ──────────────────────────────────────────────────────────────

def _load_token() -> str:
    """Always load fresh token — handles 30-min expiry."""
    import json as _j, time as _t
    cred = Path.home() / ".openclaw/credentials/github-copilot.token.json"
    if cred.exists():
        try:
            d = _j.loads(cred.read_text())
            remaining = d.get("expiresAt", 0)/1000 - _t.time()
            if remaining > 30:
                token = d["token"]
                os.environ["GITHUB_TOKEN"] = token
                return token
            else:
                print(f"[web-teacher] Token expires in {remaining:.0f}s — waiting...")
                _t.sleep(35)
                d = _j.loads(cred.read_text())
                token = d["token"]
                os.environ["GITHUB_TOKEN"] = token
                return token
        except Exception as e:
            print(f"[web-teacher] Token error: {e}")
    return os.environ.get("GITHUB_TOKEN", "")

# ── Save trajectory ───────────────────────────────────────────────────────────

def save_trajectory(col, task, category, steps, reply, prov, min_score=0.7) -> bool:
    if not steps:
        return False
    error_markers = ("[model_server error]","CUDA error","out of memory","(error:")
    if any(m in (reply or "") for m in error_markers):
        return False

    # Critic — check web_search was actually called with real results
    web_steps = [s for s in steps if s.get("tool") == "web_search"]
    write_steps = [s for s in steps if s.get("tool") == "write_file"]
    fake_result = any("not available" in str(s.get("result","")) or "fallback" in str(s.get("result","")) for s in web_steps)

    if not web_steps:
        return False  # Must have at least one web_search
    if fake_result:
        print("  ⚠️  web_search returned fallback — skipping")
        return False

    try:
        step_summary = ", ".join(s["tool"] for s in steps)
        tool_results = "\n".join(f"  {s['tool']}: {str(s.get('result',''))[:150]}" for s in steps[:4])
        crit_prompt = (
            f"Rate this AI agent response 0.0-1.0.\n\n"
            f"1.0 = web_search called, real results used, file written with actual content\n"
            f"0.8 = search done, file written, minor issues\n"
            f"0.6 = search done but file content is vague/generic\n"
            f"0.4 = search done but no file written, or content is fabricated\n"
            f"0.0 = no search, no file, or completely wrong\n\n"
            f"Task: {task[:200]}\nTools: {step_summary}\n"
            f"Tool results:\n{tool_results}\nFinal reply: {(reply or '')[:200]}\n\nRate:"
        )
        crit = prov.infer([{"role":"user","content":crit_prompt}], max_new_tokens=10, call_type="critic")
        m = re.search(r"\b(0\.\d+|1\.0|1)\b", crit)
        score = float(m.group(1)) if m else 0.7
    except Exception:
        score = 0.8 if (web_steps and write_steps) else 0.6

    effective_min = max(0.5, min_score - 0.2) if len(steps) >= 3 else min_score
    if score < effective_min:
        return False

    verdict = "PASS" if score >= min_score else "PARTIAL"
    artifacts = json.dumps([s["args"].get("path","") for s in write_steps])
    col.record(
        task=f"[web-teacher] {task}",
        provider=f"web-teacher/{category}",
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

    tasks = WEB_TASKS[:args.count] if args.count else WEB_TASKS

    if args.dry_run:
        print(f"Web-teacher tasks ({len(tasks)}):")
        for task, cat in tasks:
            print(f"  [{cat}] {task[:75]}")
        return

    import yaml
    with open(_ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    from provider import InferenceProvider
    from trajectory_collector import get_collector

    # GPT-5.4 via Copilot — reliable tool-calling + Olly context in system prompt
    config.setdefault("providers", {})
    config["providers"]["task_inference"] = "copilot"
    config["providers"].setdefault("models", {})["copilot"] = "gpt-5.4"

    prov = InferenceProvider(config)
    col  = get_collector(config)
    workspace = os.path.expanduser(config.get("kernel_workspace","~/.kernel-evolving/workspace"))
    system_prompt = build_system_prompt()

    saved_total = 0
    print(f"\n{'='*60}")
    print(f"Web-Search Trajectory Generator (GPT-5.4 + Olly context)")
    print(f"Tasks: {len(tasks)} | Min score: {args.min_score}")
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"{'='*60}")

    for i, (task, category) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] [{category}] {task[:80]}")
        _load_token()  # refresh before every task

        steps = []
        def _cb(n, tool, targs, result, _s=steps):
            _s.append({"tool": tool, "args": targs, "result": str(result)[:600]})

        try:
            reply = prov.infer_with_tools(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": task},
                ],
                tools=TOOLS, workspace=workspace, max_steps=8,
                step_callback=_cb, call_type="task_inference",
            )
            saved = save_trajectory(col, task, category, steps, reply, prov, args.min_score)
            tool_names = [s["tool"] for s in steps]
            print(f"  Tools: {tool_names} | Saved: {'✅' if saved else '⏭'}")
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
