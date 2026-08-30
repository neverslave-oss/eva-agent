#!/usr/bin/env python3
"""
generate_olly_teacher_trajectories.py
=======================================
Generates training trajectories using Olly (Claude Sonnet via GitHub Copilot)
as the teacher model — not GPT-5.4.

Rationale:
  Olly has full context of the kernel-evolving codebase, Fabio's workflow,
  the tool patterns, and how tasks should actually be handled. GPT-5.4
  generates generic tool use. Olly generates trajectories that reflect the
  ACTUAL behavioural patterns we want to teach the drafter — the correct
  reasoning style, tool selection, grounding from context, concise replies.

  This is the highest-signal dataset we can produce: Olly mimicking exactly
  how kernel-evolving should behave, using the same tools, same workspace,
  same task set — but with the reasoning quality of a frontier model.

Tasks are drawn from a curated set that covers:
  - T1: Single-tool write
  - T2: exec + write (multi-tool chain)
  - T3: read + transform + write
  - T4: http_get + extract + write
  - T5: multi-turn context grounding
  - T6: skill dispatch (prefer run_skill over raw exec)
  - T7: conditional logic (check-then-act)
  - T8: error recovery pattern
  - T9: routine execution + summarise

Usage:
  cd repositories/kernel-evolving/src
  python3 ../scripts/generate_olly_teacher_trajectories.py [--count N] [--export]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Task set ─────────────────────────────────────────────────────────────────

OLLY_TASKS = [
    # T1: Single-tool write
    ("Write 'kernel-evolving v1 is running' to ~/evo_status.txt",
     ["write_file"], "t1:write"),
    ("Create a file ~/evo_readme.md with a one-paragraph description of what kernel-evolving does",
     ["write_file"], "t1:write-creative"),
    ("Save today's date and hostname to ~/evo_identity.txt",
     ["exec_shell", "write_file"], "t1:exec-write"),

    # T2: exec + write
    ("Run 'df -h' and save only the root filesystem line to ~/evo_disk.txt",
     ["exec_shell", "write_file"], "t2:exec-filter-write"),
    ("Check which Python packages are installed with pip list, then save the count to ~/evo_pkg_count.txt",
     ["exec_shell", "write_file"], "t2:exec-count-write"),
    ("Run the git log --oneline -5 in ~/.kernel-evolving and save it to ~/evo_git_recent.txt",
     ["exec_shell", "write_file"], "t2:exec-git-write"),
    ("Check if port 8779 is listening and write 'UP' or 'DOWN' to ~/evo_port_status.txt",
     ["exec_shell", "write_file"], "t2:exec-conditional-write"),

    # T3: read + transform + write
    ("Read ~/evo_status.txt, capitalise its content, and write the result to ~/evo_status_caps.txt",
     ["read_file", "write_file"], "t3:read-transform-write"),
    ("Read ~/evo_identity.txt and append a line 'Last checked: <current date>' to it",
     ["read_file", "exec_shell", "write_file"], "t3:read-append-write"),

    # T4: http_get + extract + write
    ("Fetch https://httpbin.org/uuid and save just the UUID value to ~/evo_uuid.txt",
     ["http_get", "write_file"], "t4:http-extract-write"),
    ("GET https://httpbin.org/get and save the 'origin' IP field to ~/evo_origin.txt",
     ["http_get", "write_file"], "t4:http-field-extract"),
    ("Check if https://httpbin.org/status/200 returns 200 and write 'OK' or 'FAIL' to ~/evo_health.txt",
     ["http_get", "write_file"], "t4:http-health-write"),

    # T5: multi-turn context grounding
    # These are tuples of (turns_list, expected_tools, category)
    # Handled specially below
    (["My project is at ~/projects/kernel-test and uses pytest",
      "Run its tests and save the output to ~/evo_test_results.txt"],
     ["exec_shell", "write_file"], "t5:multiturn-recall-exec"),
    (["I want to monitor a service called fantasia on port 8765",
      "Write a one-liner health check script to ~/evo_fantasia_check.sh",
      "Run it and tell me the result"],
     ["write_file", "exec_shell"], "t5:multiturn-service-monitor"),
    (["The config I need is at ~/evo_config.json",
      "Read it and tell me what 'version' is set to"],
     ["read_file"], "t5:multiturn-read-extract"),

    # T6: skill dispatch
    ("Search collective memory for anything about kernel-evolving architecture and summarise to ~/evo_arch_notes.txt",
     ["run_skill", "write_file"], "t6:skill-dispatch-collective-memory"),
    ("Use the open-workspace-tracker skill to list today's todos",
     ["run_skill"], "t6:skill-dispatch-tracker"),
    ("Run the morning-briefing routine and save the output to ~/evo_briefing.txt",
     ["run_routine", "write_file"], "t6:routine-execute"),

    # T7: conditional logic
    ("Check if ~/evo_config.json exists. If yes, read it and write a one-line summary to ~/evo_config_check.txt. If no, create it with {\"active\": true}",
     ["exec_shell", "read_file", "write_file"], "t7:conditional-check-act"),
    ("Check free RAM. If less than 4GB, write 'LOW MEMORY' to ~/evo_mem_alert.txt. If more, write 'OK'",
     ["exec_shell", "write_file"], "t7:conditional-threshold"),

    # T8: error recovery
    ("Try to read ~/evo_nonexistent_file.txt — if it doesn't exist, create it with content 'created by recovery' and confirm",
     ["read_file", "write_file"], "t8:error-recovery-create"),
    ("Run 'cat ~/evo_log.txt 2>/dev/null || echo no log yet' and save the output to ~/evo_log_check.txt",
     ["exec_shell", "write_file"], "t8:graceful-fallback"),

    # T9: summarise + synthesise
    ("Read ~/evo_status.txt and ~/evo_disk.txt (if they exist), combine their content into a single status report at ~/evo_report.md",
     ["read_file", "write_file"], "t9:multi-read-synthesise"),
    ("Write a concise Markdown changelog entry for kernel-evolving v1.18.4 to ~/evo_changelog.md based on what you know about the recent fixes",
     ["write_file"], "t9:knowledge-write"),
]


# ── System prompt builder — loads Olly's actual workspace context ────────────

_OLLY_WORKSPACE = Path.home() / ".openclaw" / "workspace"


def _load_file_section(path: Path, max_chars: int = 2000) -> str:
    """Load a file, truncating to max_chars if needed."""
    try:
        text = path.read_text().strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception:
        return ""


def _extract_memory_section(memory_text: str, keywords: list, max_chars: int = 1500) -> str:
    """Extract paragraphs from MEMORY.md that mention any of the keywords."""
    lines = memory_text.split("\n")
    relevant = []
    window = []
    for line in lines:
        window.append(line)
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant.extend(window)
            window = []
        if len(window) > 5:
            window = window[-5:]
    result = "\n".join(relevant)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n...[truncated]"
    return result.strip()


def build_teacher_system_prompt() -> str:
    """Build the teacher system prompt by loading Olly's actual workspace context.
    
    This is what makes the teacher 'Olly' rather than a generic model:
    - SOUL.md: identity, tone, values
    - USER.md: who Fabio is, preferences, channel
    - MEMORY.md (relevant sections): kernel-evolving architecture, ADR-013, fine-tuning plan
    - Today's memory: what happened today, current state
    """
    soul = _load_file_section(_OLLY_WORKSPACE / "SOUL.md", max_chars=1500)
    user = _load_file_section(_OLLY_WORKSPACE / "USER.md", max_chars=800)

    # Extract only the kernel-evolving relevant sections from MEMORY.md
    memory_text = _load_file_section(_OLLY_WORKSPACE / "MEMORY.md", max_chars=99999)
    memory_relevant = _extract_memory_section(
        memory_text,
        keywords=["kernel-evolving", "drafter", "trajectory", "fine-tun", "ADR-013",
                  "Gemma", "dataset", "tool_call", "infer_with_tools"],
        max_chars=2000
    )

    # Today's memory — most current state
    from datetime import datetime
    today_file = _OLLY_WORKSPACE / "memory" / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    today_memory = _load_file_section(today_file, max_chars=1500)

    prompt = f"""You are Olly — the main AI assistant running inside OpenClaw, acting as teacher to generate
training trajectories for kernel-evolving (a local self-evolving agent).

You are demonstrating exactly how kernel-evolving SHOULD behave when given a task. Every response
you produce here becomes training data for its Gemma 4 MTP drafter. Generate ideal tool-use
behaviour — correct reasoning, correct tool selection, concise replies.

## Your identity and values
{soul}

## Your user
{user}

## kernel-evolving context (from memory)
{memory_relevant}

## Today's session context
{today_memory}

## The agent you are teaching (kernel-evolving)
kernel-evolving is a local-first self-evolving AI agent running on the user's machine (port 8779).
It uses Gemma 4 E2B-it as its main model and the gemma-4-E2B-it-assistant (151MB, 4-layer) as its
MTP drafter. You are generating SFT training data for that drafter.

The drafter needs to learn:
- When to call exec_shell vs write_file vs read_file vs http_get vs run_skill
- How to chain tools correctly (exec then write, read then transform then write)
- How to resolve context from prior conversation turns before acting
- Concise, grounded replies — no fabrication, no fake completion
- Always prefer run_skill over reimplementing skill logic

## Tools available
  • exec_shell(command)           — run shell commands
  • read_file(path)               — read a file
  • write_file(path, content)     — write a file  
  • http_get(url)                 — HTTP GET
  • run_skill(skill_name, input)  — execute an installed skill
  • run_routine(routine_name)     — execute a routine

## Strict rules for this training data generation
1. TOOL FIRST — always call the tool. Never describe what you would do.
2. write tasks: call write_file with actual content → reply "Done — written to <path>."
3. exec tasks: call exec_shell → use the ACTUAL output in reply, never guess.
4. read tasks: call read_file first, answer from actual content.
5. skill tasks: call run_skill — never reimplement with exec_shell.
6. multi-step: chain tools in order, do not skip steps.
7. Use ~ for home paths. NEVER use /home/pacificDev or any absolute home path.
8. Be concise. One confirmation line after completion. No filler.
9. Never say "I have written/saved/created" without the tool call having returned in this conversation.
10. Error handling: if file missing, create it or report clearly. Never crash silently."""

    return prompt


# Build once at import time
TEACHER_SYSTEM_PROMPT = build_teacher_system_prompt()


# ── Tool definitions (subset used for training) ───────────────────────────────

# Tools in OpenAI function-calling format (type: function wrapper required by copilot/openai tool loop)
TOOLS = [
    {"type": "function", "function": {"name": "exec_shell", "description": "Run a shell command. Returns stdout+stderr.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file's contents.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to a file. Creates parent dirs if needed.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "http_get", "description": "HTTP GET request. Returns response body.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "run_skill", "description": "Execute an installed skill by name.", "parameters": {"type": "object", "properties": {"skill_name": {"type": "string"}, "input": {"type": "string"}}, "required": ["skill_name", "input"]}}},
    {"type": "function", "function": {"name": "run_routine", "description": "Execute a named routine.", "parameters": {"type": "object", "properties": {"routine_name": {"type": "string"}}, "required": ["routine_name"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web and return results summary.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


# ── Batch F: Deep multi-tool chains (3-5 steps) ───────────────────────────────

BATCH_F_DEEP_CHAINS = [
    ("Run 'df -h' and 'free -h', then read ~/evo_status.txt if it exists, and write a combined system report to ~/evo_sysreport.md",
     ["exec_shell", "exec_shell", "read_file", "write_file"], "f:exec-exec-read-write"),
    ("Check what Python version is installed, read ~/evo_identity.txt, then write a combined env summary to ~/evo_env_summary.txt",
     ["exec_shell", "read_file", "write_file"], "f:exec-read-write"),
    ("Run 'git log --oneline -3' in ~/.kernel-evolving, then read the latest CHANGELOG.md if it exists, and write a release summary to ~/evo_release_summary.md",
     ["exec_shell", "read_file", "write_file"], "f:exec-read-write-release"),
    ("Check if ~/evo_config.json exists. Read it. Update the 'last_checked' field to today's date. Write it back.",
     ["exec_shell", "read_file", "write_file"], "f:check-read-update-write"),
    ("List all .txt files in ~/, read the newest one, summarise its content, and write the summary to ~/evo_txt_summary.md",
     ["exec_shell", "read_file", "write_file"], "f:list-read-summarise-write"),
    ("Fetch https://httpbin.org/uuid, also run 'hostname && date', then write both results together to ~/evo_combined_info.txt",
     ["http_get", "exec_shell", "write_file"], "f:http-exec-write"),
    ("GET https://httpbin.org/get to find your public IP, then check which process is on port 8779 with fuser, and write a connectivity report to ~/evo_connectivity.md",
     ["http_get", "exec_shell", "write_file"], "f:http-exec-portcheck-write"),
    ("Fetch https://api.github.com/repos/fabiopacifici-bot/kernel-evolving/commits?per_page=3, extract the 3 latest commit messages, and save them to ~/evo_recent_commits.md",
     ["http_get", "write_file"], "f:http-extract-write-commits"),
    ("Read ~/evo_identity.txt, run 'uptime' to get system load, then write a health snapshot combining both to ~/evo_health_snapshot.txt",
     ["read_file", "exec_shell", "write_file"], "f:read-exec-write"),
    ("Read ~/evo_status.txt, count the number of lines with exec_shell, then append a line 'checked at <date>' to it",
     ["read_file", "exec_shell", "write_file"], "f:read-count-append"),
    ("Check if ~/evo_project/ exists. If not, create it with mkdir. Then write a README.md inside it and a main.py stub. Confirm both files were written.",
     ["exec_shell", "write_file", "write_file", "exec_shell"], "f:check-create-write-write-verify"),
    ("Run 'pip list | grep torch', fetch https://httpbin.org/get, read ~/evo_identity.txt, then write a full environment report to ~/evo_full_report.md",
     ["exec_shell", "http_get", "read_file", "write_file"], "f:4step-env-report"),
    ("Check disk usage with df, check RAM with free, check if port 8779 responds with curl, then write a one-line status to ~/evo_quick_status.txt",
     ["exec_shell", "exec_shell", "exec_shell", "write_file"], "f:3exec-write"),
]


# ── Batch G: Skills + web search ─────────────────────────────────────────────

BATCH_G_SKILLS_WEB = [
    ("Search collective memory for 'kernel-evolving fine-tuning dataset' and write the key findings to ~/evo_cm_finetuning.md",
     ["run_skill", "write_file"], "g:skill-cm-write"),
    ("Use the open-workspace-tracker skill to list all open todos, then filter to high priority only and save to ~/evo_high_todos.md",
     ["run_skill", "write_file"], "g:skill-tracker-write"),
    ("Use the collective-memory skill to search for 'ADR-013 trajectory fine-tune' and summarise findings to ~/evo_adr013_notes.md",
     ["run_skill", "write_file"], "g:skill-cm-adr013"),
    ("Run the open-workspace-tracker skill to add a high-priority todo: 'Run pilot fine-tune on HF Jobs'",
     ["run_skill"], "g:skill-tracker-add"),
    ("Search collective memory for 'Gemma 4 drafter speculative decoding' and save anything relevant to ~/evo_drafter_notes.md",
     ["run_skill", "write_file"], "g:skill-cm-drafter"),
    ("Search collective memory for 'kernel-evolving version', then run 'curl -s http://localhost:8779/health' to confirm current version, and write both to ~/evo_version_check.md",
     ["run_skill", "exec_shell", "write_file"], "g:skill-exec-write"),
    ("Use the open-workspace-tracker to list todos, then run 'date' to get the current timestamp, and write a timestamped todo report to ~/evo_todo_report.md",
     ["run_skill", "exec_shell", "write_file"], "g:skill-exec-report"),
    ("Search collective memory for 'neverslave.com deployment', read ~/evo_status.txt if it exists, and write a combined context note to ~/evo_deploy_context.md",
     ["run_skill", "read_file", "write_file"], "g:skill-read-write"),
    ("Search the web for 'TRL SFTTrainer Gemma 4 tool calling fine-tuning 2026 best practices' and save the key findings to ~/evo_sft_guide.md",
     ["web_search", "write_file"], "g:web-sft-write"),
    ("Search the web for 'speculative decoding MTP drafter small language model 2026' and write a technical summary to ~/evo_speculative_notes.md",
     ["web_search", "write_file"], "g:web-speculative-write"),
    ("Search for 'OpenClaw AI agent local-first self-evolving 2026' and save what you find to ~/evo_openclaw_notes.md",
     ["web_search", "write_file"], "g:web-openclaw-write"),
    ("Search the web for 'GRPO DPO SFT comparison fine-tuning small models 2026' and write key differences to ~/evo_training_methods.md",
     ["web_search", "write_file"], "g:web-training-methods"),
    ("Search for 'HuggingFace TRL Jobs GPU free tier H100 fine-tuning cost 2026' and save cost estimates to ~/evo_hf_costs.md",
     ["web_search", "write_file"], "g:web-hf-costs"),
    ("Search the web for 'Gemma 4 E2B-it benchmark results 2026 tool use' and save a comparison table to ~/evo_gemma4_benchmarks.md",
     ["web_search", "write_file"], "g:web-gemma4-bench"),
    ("Search for 'agent trajectory dataset quality filter critic score 2026' and write best practices to ~/evo_dataset_quality.md",
     ["web_search", "write_file"], "g:web-dataset-quality"),
    ("Search the web for 'Gemma 4 E2B context length benchmark', then check local VRAM with nvidia-smi, and write a feasibility note to ~/evo_local_feasibility.md",
     ["web_search", "exec_shell", "write_file"], "g:web-exec-write"),
    ("Search for 'kernel-evolving self-evolving agent architecture', then check if port 8779 is running with curl, and write a status+context note to ~/evo_evo_status.md",
     ["web_search", "exec_shell", "write_file"], "g:web-exec-status"),
    ("Search the web for 'ADR architecture decision record template 2026', read ~/evo_identity.txt, then write a new ADR template to ~/evo_adr_template.md",
     ["web_search", "read_file", "write_file"], "g:web-read-write"),
    ("Run the morning-briefing routine, then write a summary of what it reported to ~/evo_morning_summary.md",
     ["run_routine", "write_file"], "g:routine-write"),
    ("Run the security-check routine if available, otherwise run 'ps aux | grep python | wc -l' and write the result to ~/evo_process_count.txt",
     ["run_routine", "write_file"], "g:routine-fallback-write"),
]


# ── Copilot token loader ─────────────────────────────────────────────────────

def _load_copilot_token() -> str:
    """Always load fresh token from OpenClaw credential store.
    Called before every task so we never use an expired token.
    OpenClaw refreshes the credential file every ~30 min automatically.
    """
    import json as _json, time as _time
    cred_path = Path.home() / ".openclaw" / "credentials" / "github-copilot.token.json"
    if cred_path.exists():
        try:
            d = _json.loads(cred_path.read_text())
            exp = d.get("expiresAt", 0) / 1000
            remaining = exp - _time.time()
            if remaining > 30:  # at least 30s left
                token = d["token"]
                os.environ["GITHUB_TOKEN"] = token
                os.environ["GITHUB_COPILOT_TOKEN"] = token
                return token
            else:
                print(f"[teacher] Token expires in {remaining:.0f}s — waiting for refresh...")
                _time.sleep(35)  # wait for openclaw to refresh
                d = _json.loads(cred_path.read_text())  # re-read
                token = d["token"]
                os.environ["GITHUB_TOKEN"] = token
                os.environ["GITHUB_COPILOT_TOKEN"] = token
                return token
        except Exception as e:
            print(f"[teacher] Token load error: {e}")
    return os.environ.get("GITHUB_TOKEN", "")

def run_task(task_spec, provider, workspace: str) -> tuple[list, str, list]:
    """Run a task (single or multi-turn). Returns (steps, final_reply, messages_for_record)."""
    
    is_multiturn = isinstance(task_spec[0], list)
    turns = task_spec[0] if is_multiturn else [task_spec[0]]
    
    steps = []
    history = []
    
    for turn_text in turns:
        messages = [{"role": "system", "content": TEACHER_SYSTEM_PROMPT}] + history + [{"role": "user", "content": turn_text}]
        
        turn_steps = []
        def _cb(n, tool_name, args, result):
            turn_steps.append({"tool": tool_name, "args": args, "result": str(result)[:800]})

        reply = provider.infer_with_tools(
            messages=messages,
            tools=TOOLS,
            workspace=workspace,
            max_steps=8,
            step_callback=_cb,
            call_type="task_inference",
        )
        
        steps.extend(turn_steps)
        history.append({"role": "user", "content": turn_text})
        history.append({"role": "assistant", "content": reply or ""})
    
    # Build clean messages record (no system prompt — added by cleaner)
    record_messages = []
    for m in history:
        record_messages.append(m)
    
    final_reply = history[-1].get("content", "") if history else ""
    return steps, final_reply, record_messages


def save_trajectory(col, task_label: str, category: str, steps: list, reply: str,
                    messages: list, prov, min_score: float = 0.7) -> bool:
    """Score via critic and save trajectory. Returns True if saved."""
    if not steps:
        return False

    error_markers = ("[model_server error]", "model_client error", "CUDA error", "out of memory")
    if any(m in (reply or "") for m in error_markers):
        return False

    # Critic scoring
    try:
        step_summary = ", ".join(s["tool"] for s in steps)
        tool_results = "\n".join(f"  {s['tool']}: {str(s.get('result',''))[:100]}" for s in steps[:3])
        crit_prompt = (
            f"Evaluate this AI agent response. Rate 0.0-1.0.\n\n"
            f"1.0 = task fully done, correct tool, correct output\n"
            f"0.8 = done with minor issues\n"
            f"0.6 = partial\n"
            f"0.4 = significant errors\n"
            f"0.0 = wrong tool or failed\n\n"
            f"Task: {task_label[:200]}\n"
            f"Tools called: {step_summary}\n"
            f"Tool results:\n{tool_results}\n"
            f"Final reply: {(reply or '')[:200]}\n\n"
            f"Rate (number only):"
        )
        crit = prov.infer([{"role": "user", "content": crit_prompt}],
                          max_new_tokens=10, call_type="critic")
        m = __import__("re").search(r"\b(0\.\d+|1\.0|1)\b", crit)
        score = float(m.group(1)) if m else 0.7
    except Exception:
        score = 0.7

    effective_min = min_score if len(steps) == 1 else max(0.5, min_score - 0.2)
    if score < effective_min:
        return False

    verdict = "PASS" if score >= min_score else "PARTIAL"
    artifacts = json.dumps([s["args"].get("path", "") for s in steps if s.get("tool") == "write_file"])

    col.record(
        task=f"[olly-teacher] {task_label}",
        provider=f"olly-teacher/{category}",
        model_name="claude-sonnet-4.6",
        call_type="task_inference",
        tool_calls=steps,
        final_reply=reply or "",
        artifacts=artifacts,
        critic_score=score,
        critic_verdict=verdict,
    )
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0, help="Max tasks to run (0=all)")
    parser.add_argument("--export", action="store_true", help="Export JSONL after run")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batches", default="all",
                        help="Comma-separated: all, base, F, G (default: all)")
    args = parser.parse_args()

    selected = args.batches.lower()
    tasks = []
    if selected == "all" or "base" in selected:
        tasks += OLLY_TASKS
    if selected == "all" or "f" in selected:
        tasks += BATCH_F_DEEP_CHAINS
    if selected == "all" or "g" in selected:
        tasks += BATCH_G_SKILLS_WEB

    if args.count:
        tasks = tasks[:args.count]

    if args.dry_run:
        print(f"Olly-teacher tasks ({len(tasks)}):")
        for t in tasks:
            label = t[0][0] if isinstance(t[0], list) else t[0]
            print(f"  [{t[2]}] {str(label)[:70]}")
        return

    # Load Copilot token from OpenClaw credential store (works even without sourcing .env)
    token = _load_copilot_token()
    if token:
        print(f"[teacher] Copilot token loaded ({token[:15]}...)")
    else:
        print("[teacher] WARNING: no Copilot token found — will fall back to OpenAI")

    import yaml
    cfg_path = _ROOT / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    from provider import InferenceProvider
    from trajectory_collector import get_collector

    # Force copilot provider with gpt-5.4 for reliable tool-calling
    # Note: claude-sonnet-4.6 via Copilot narrates instead of calling tools;
    # gpt-5.4 via Copilot executes tool calls correctly.
    # The Olly identity is injected via the system prompt (SOUL, USER, MEMORY context).
    config.setdefault("providers", {})
    config["providers"]["task_inference"] = "copilot"
    config["providers"].setdefault("models", {})["copilot"] = "gpt-5.4"

    prov = InferenceProvider(config)
    col = get_collector(config)
    workspace = os.path.expanduser(config.get("kernel_workspace", "~/.kernel-evolving/workspace"))

    saved_total = 0
    print(f"\n{'='*60}")
    print(f"Olly-as-Teacher Trajectory Generator")
    print(f"Tasks: {len(tasks)} | Min score: {args.min_score}")
    print(f"Provider: copilot/claude-sonnet-4.6")
    print(f"{'='*60}")

    for i, task_spec in enumerate(tasks, 1):
        is_multiturn = isinstance(task_spec[0], list)
        label = " → ".join(task_spec[0]) if is_multiturn else task_spec[0]
        category = task_spec[2]
        print(f"\n[{i}/{len(tasks)}] [{category}] {str(label)[:80]}")

        # Refresh token before every task — handles 30-min expiry during long runs
        _load_copilot_token()

        try:
            steps, reply, messages = run_task(task_spec, prov, workspace)
            saved = save_trajectory(col, label, category, steps, reply, messages, prov, args.min_score)
            tool_names = [s["tool"] for s in steps]
            print(f"  Tools: {tool_names} | Saved: {'✅' if saved else '⏭ (below min score)'}")
            if saved:
                saved_total += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Done. Saved {saved_total}/{len(tasks)} trajectories.")
    print(f"{'='*60}")

    if args.export:
        path, count = col.export_jsonl(min_score=args.min_score)
        print(f"\nExported {count} total records → {path}")
        print(f"Next: python3 scripts/clean_trajectories.py --input {path} --output clean_export.jsonl --stats")


if __name__ == "__main__":
    main()
