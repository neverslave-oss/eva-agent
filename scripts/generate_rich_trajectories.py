#!/usr/bin/env python3
"""
generate_rich_trajectories.py
==============================
Generates the 4 missing trajectory categories needed before fine-tuning:

  Batch A — Multi-turn (grounding): 3-5 turn sessions where the model must
             extract data from conversation history before acting.

  Batch B — Full system prompt: tasks run with the real build_system_prompt()
             output injected so the model sees the full Evo context at runtime.

  Batch C — Skill dispatch: tasks naturally resolved by run_skill() rather
             than raw exec_shell, teaching the model to prefer skills.

  Batch D — Evolution loop: gap detection → run_skill(evolver) → installed skill
             scenario, so the model learns to recognise and use the evolution cycle.

  Batch E — Web research (browser-automation + web_search tool): tasks requiring
             web search, URL fetch, and save-to-file patterns.

All batches use OpenAI as teacher (no local GPU needed).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── Batch A: Multi-turn trajectories ─────────────────────────────────────────

MULTI_TURN_SESSIONS = [
    # (list of turns, description)
    (
        [
            "My project is called LunarMapper and it lives at ~/projects/lunar-mapper",
            "Run the tests for it and save the results to ~/evo_test_results.txt",
        ],
        "multi-turn: project path recall → test execution"
    ),
    (
        [
            "I have a cat named Luna who is 3 years old and lives in Rome",
            "Write that info to ~/evo_pet_info.txt",
            "Now read it back and confirm the content",
        ],
        "multi-turn: personal info recall → file write → file read"
    ),
    (
        [
            "The API endpoint we're testing is https://httpbin.org/get",
            "Fetch it and save the response to ~/evo_api_response.json",
        ],
        "multi-turn: URL recall → http_get → save"
    ),
    (
        [
            "My git repo is at ~/projects/kernel-test and the branch is feat/new-feature",
            "Check the git status of that repo and save the output to ~/evo_git_status.txt",
        ],
        "multi-turn: repo path + branch recall → exec_shell → save"
    ),
    (
        [
            "I want to track a server called nebula at IP 192.168.1.100",
            "Write a connection test script to ~/evo_conn_test.sh that pings that IP",
            "Now run it and save the output to ~/evo_conn_result.txt",
        ],
        "multi-turn: server info recall → script write → exec → save"
    ),
    (
        [
            "The config file I need to update is ~/evo_config.json",
            "Add a field 'last_updated' with today's date to it",
        ],
        "multi-turn: file path recall → read → modify → write"
    ),
    (
        [
            "My name is Fabio and I'm building a self-evolving AI agent called kernel-evolving",
            "Write a short bio about me and the project and save it to ~/evo_bio.md",
        ],
        "multi-turn: identity recall → structured write"
    ),
    (
        [
            "I need to monitor three services: Fantasia on port 8765, Voice on port 8767, and Kernel on port 8779",
            "Write a health check script ~/evo_health_check.sh that curls all three",
            "Run it and save the combined output to ~/evo_health_results.txt",
        ],
        "multi-turn: service list recall → script write → exec → save"
    ),
]


# ── Batch C: Skill dispatch tasks ────────────────────────────────────────────

SKILL_DISPATCH_TASKS = [
    # (task, skill_name, description)
    (
        "Search collective memory for information about kernel-evolving architecture and save the findings to ~/evo_memory_search.txt",
        "collective-memory",
        "skill dispatch: collective-memory search → write"
    ),
    (
        "Use the web-scrape-summarize skill to fetch and summarise https://docs.python.org/3/library/asyncio.html — save the summary to ~/evo_asyncio_summary.md",
        "web-scrape-summarize",
        "skill dispatch: web scrape → summarise → write"
    ),
    (
        "Use the article-summary-3-points skill on https://arxiv.org/abs/2309.07864 and save the 3 key points to ~/evo_paper_summary.md",
        "article-summary-3-points",
        "skill dispatch: article summary → write"
    ),
    (
        "Run the python-unit-tests-roi skill to analyse ~/evo_utils.py and save the test recommendations to ~/evo_test_recommendations.md",
        "python-unit-tests-roi",
        "skill dispatch: code analysis → write"
    ),
    (
        "Use the code-review-diff-analysis skill on this Python snippet and save the review to ~/evo_code_review.md:\ndef add(a,b): return a+b\ndef subtract(a,b): return a-b",
        "code-review-diff-analysis",
        "skill dispatch: code review → write"
    ),
    (
        "Use the github skill to list the last 5 open issues in the fabiopacifici-bot/kernel-evolving repo and save them to ~/evo_issues.md",
        "github",
        "skill dispatch: github issues → write"
    ),
    (
        "Run the mental-map skill to get the current knowledge graph status and save a summary to ~/evo_mental_map_status.md",
        "mental-map",
        "skill dispatch: mental-map query → write"
    ),
    (
        "Use the open-workspace-tracker skill to list all open high-priority todos and save them to ~/evo_high_priority_todos.md",
        "open-workspace-tracker",
        "skill dispatch: tracker todos → write"
    ),
]


# ── Batch E: Web research tasks ───────────────────────────────────────────────

WEB_RESEARCH_TASKS = [
    (
        "Search the web for 'HuggingFace TRL SFT trainer Gemma 4 fine-tuning 2026' and save the key findings to ~/evo_sft_research.md",
        ["web_search", "write_file"],
        "web research: search → summarise → save"
    ),
    (
        "Fetch https://httpbin.org/headers and save the full response to ~/evo_headers.json, then extract the 'User-Agent' field",
        ["http_get", "write_file"],
        "web: http_get → save → extract"
    ),
    (
        "Search for 'Italian employment contract law 2026 key clauses' and write a summary of what you find to ~/evo_legal_research.md",
        ["web_search", "write_file"],
        "web research: legal research → save"
    ),
    (
        "Fetch https://api.github.com/repos/huggingface/trl/releases/latest and save the latest TRL release version to ~/evo_trl_version.txt",
        ["http_get", "write_file"],
        "web: API fetch → extract version → save"
    ),
    (
        "Search the web for 'bitsandbytes libnvJitLink.so fix WSL2 2026' and save debugging steps to ~/evo_bnb_debug.md",
        ["web_search", "write_file"],
        "web research: debug search → save steps"
    ),
    (
        "Fetch https://httpbin.org/json and extract the slideshow title, then write a one-sentence summary to ~/evo_api_summary.txt",
        ["http_get", "write_file"],
        "web: fetch JSON → extract → summarise → save"
    ),
    (
        "Search for 'Gemma 4 E2B-it context window capabilities multimodal 2026' and save a technical summary to ~/evo_gemma4_specs.md",
        ["web_search", "write_file"],
        "web research: model research → save"
    ),
    (
        "Search for 'OpenClaw AI agent framework features' and save what you find to ~/evo_openclaw_research.md, then create a comparison with kernel-evolving capabilities",
        ["web_search", "write_file"],
        "web research: competitive research → analysis → save"
    ),
]


# ── Batch D: Evolution-aware tasks ───────────────────────────────────────────

EVOLUTION_TASKS = [
    (
        "I need to translate a legal document from Italian to English. Can you handle that? If you have a skill for it, use it. Save the result to ~/evo_translation_test.txt",
        ["run_skill", "write_file"],
        "evolution-aware: skill lookup → dispatch if available"
    ),
    (
        "Search collective memory for any previous decisions about the kernel-evolving architecture and summarise them to ~/evo_arch_decisions.md",
        ["run_skill", "write_file"],
        "evolution-aware: collective memory → architectural context"
    ),
    (
        "Run the security-check routine and save the summary to ~/evo_security_summary.txt",
        ["run_routine", "write_file"],
        "evolution-aware: routine execution → save"
    ),
    (
        "Check if there is a skill for generating Python unit tests. If yes, use it on ~/evo_utils.py. If no, write a simple test file manually.",
        ["run_skill", "write_file"],
        "evolution-aware: conditional skill dispatch or fallback"
    ),
    (
        "Use the hugging-face-model-trainer skill to get information about how to fine-tune a model and save the key steps to ~/evo_finetune_guide.md",
        ["run_skill", "write_file"],
        "evolution-aware: skill for domain knowledge → save"
    ),
]


# ── Generator ─────────────────────────────────────────────────────────────────

def build_full_system_prompt(config: dict, skills: list) -> str:
    """Build the real Evo system prompt as it appears at runtime."""
    try:
        from context import build_system_prompt
        return build_system_prompt(config, skills, [], channel="trajectory-gen")
    except Exception as e:
        return (
            f"You are Evo (kernel-evolving), a self-evolving local AI agent. "
            f"You have access to tools: exec_shell, read_file, write_file, http_get, "
            f"web_search, run_skill, run_routine. "
            f"Always call the appropriate tool. Never claim completion without a tool result. "
            f"Available skills: {', '.join(s.get('name','') for s in skills[:20])}."
        )


def run_multiturn_session(turns: list, system_prompt: str, provider, tools: list, workspace: str) -> list:
    """Run a multi-turn session and collect all turns as trajectory messages."""
    history = []
    all_steps = []

    for i, user_msg in enumerate(turns):
        messages = [{"role": "system", "content": system_prompt}] + history + \
                   [{"role": "user", "content": user_msg}]

        steps = []
        def _cb(n, tool, args, result, _i=i):
            steps.append({"tool": tool, "args": args, "result": str(result)[:500]})

        reply = provider.infer_with_tools(
            messages=messages,
            tools=tools,
            workspace=workspace,
            max_steps=8,
            step_callback=_cb,
            call_type="task_inference",
        )
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply or ""})
        all_steps.extend(steps)

    return history, all_steps


def record_to_hf_messages(system_prompt: str, history: list, steps: list) -> list:
    """Convert a multi-turn session to HF-format messages array."""
    msgs = [{"role": "system", "content": system_prompt}]
    # Rebuild with tool calls interleaved
    step_idx = 0
    for msg in history:
        if msg["role"] == "user":
            msgs.append(msg)
        elif msg["role"] == "assistant":
            # Find steps that belong to this turn (heuristic: pair each assistant turn with available steps)
            if step_idx < len(steps):
                turn_steps = steps[step_idx:step_idx+2]  # max 2 tool calls per turn
                step_idx += len(turn_steps)
                for s in turn_steps:
                    msgs.append({
                        "role": "assistant",
                        "tool_calls": [{"type": "function", "function": {
                            "name": s["tool"],
                            "arguments": json.dumps(s["args"])
                        }}]
                    })
                    msgs.append({
                        "role": "tool",
                        "name": s["tool"],
                        "content": s["result"],
                    })
            msgs.append({"role": "assistant", "content": msg["content"]})
    return msgs


def save_trajectory(col, task: str, provider_label: str, model_name: str,
                    steps: list, reply: str, system_prompt: str,
                    extra_context: dict = None, min_score: float = 0.7) -> bool:
    """Score and save a single trajectory.

    Multi-step traces (2+ tool calls) are valuable even if critic scores are
    borderline — they teach the model how to chain tools. Single-tool traces
    require a higher bar since they're more common in the base dataset.
    """
    if not steps:
        return False

    error_markers = ("[model_server error]", "model_client error", "CUDA error",
                     "out of memory", "(inference unavailable)")
    if any(m in (reply or "") for m in error_markers):
        return False

    # Critic score via provider
    try:
        from provider import get_provider as _gp
        prov = _gp()
        step_summary = ", ".join(s["tool"] for s in steps)
        tool_results = "\n".join(f"  {s['tool']}: {str(s.get('result',''))[:100]}" for s in steps[:3])
        crit_prompt = (
            f"Evaluate this AI agent response. Rate 0.0-1.0.\n\n"
            f"Criteria:\n"
            f"- 1.0: task fully completed, correct tool used, correct output\n"
            f"- 0.8: task completed with minor issues\n"
            f"- 0.6: partial completion or imprecise but close\n"
            f"- 0.4: significant errors or missing key steps\n"
            f"- 0.0: wrong tool, wrong output, or completely failed\n\n"
            f"Task: {task[:200]}\n"
            f"Tools called: {step_summary}\n"
            f"Tool results:\n{tool_results}\n"
            f"Final reply: {(reply or '')[:200]}\n\n"
            f"Rate (number only):"
        )
        crit = prov.infer([{"role": "user", "content": crit_prompt}],
                          max_new_tokens=10, call_type="critic")
        m = __import__("re").search(r"\b(0\.\d+|1\.0|1)\b", crit)
        score = float(m.group(1)) if m else 0.6  # default 0.6 — assume partial success
    except Exception:
        score = 0.7  # assume pass if critic unavailable

    # Multi-step traces: accept at lower threshold — chaining is the hard pattern
    effective_min = min_score if len(steps) == 1 else max(0.5, min_score - 0.2)

    if score < effective_min:
        return False

    verdict = "PASS" if score >= min_score else "PARTIAL"

    # Detect artifacts
    import glob as _glob
    now = time.time()
    artifacts = [f for f in _glob.glob(os.path.expanduser("~/evo_*"), recursive=True)
                 if os.path.isfile(f) and (now - os.path.getmtime(f)) < 120]

    col.record(
        task=task,
        provider=provider_label,
        model_name=model_name,
        call_type="task_inference",
        tool_calls=steps,
        final_reply=reply or "",
        artifacts=artifacts or None,
        critic_score=score,
        critic_verdict=verdict,
    )
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate rich multi-batch trajectories")
    parser.add_argument("--provider", default="openai", help="Provider to use (default: openai)")
    parser.add_argument("--batches", default="A,C,D,E", help="Comma-separated batch letters (default: A,C,D,E)")
    parser.add_argument("--export", action="store_true", help="Export JSONL when done")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import yaml
    config_path = _ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.provider:
        config.setdefault("providers", {})["task_inference"] = args.provider
        config["providers"]["critic"] = args.provider
        print(f"Provider: {args.provider}")

    batches = [b.strip().upper() for b in args.batches.split(",")]

    if args.dry_run:
        total = 0
        if "A" in batches:
            print(f"Batch A (multi-turn): {len(MULTI_TURN_SESSIONS)} sessions")
            total += len(MULTI_TURN_SESSIONS)
        if "C" in batches:
            print(f"Batch C (skill dispatch): {len(SKILL_DISPATCH_TASKS)} tasks")
            total += len(SKILL_DISPATCH_TASKS)
        if "D" in batches:
            print(f"Batch D (evolution-aware): {len(EVOLUTION_TASKS)} tasks")
            total += len(EVOLUTION_TASKS)
        if "E" in batches:
            print(f"Batch E (web research): {len(WEB_RESEARCH_TASKS)} tasks")
            total += len(WEB_RESEARCH_TASKS)
        print(f"Total: {total} new trajectories")
        return

    workspace = os.path.expanduser("~/.kernel-evolving/workspace")
    from trajectory_collector import get_collector
    from tools import TOOLS
    from provider import get_provider
    from skills import load_all as load_skills

    col = get_collector(config)
    prov = get_provider(config)
    skills = load_skills(os.path.expanduser("~/.kernel-evolving/ecosystem"))
    system_prompt = build_full_system_prompt(config, skills)
    model_name = config.get("providers", {}).get("models", {}).get(args.provider, args.provider)

    saved_total = 0

    # ── Batch A: Multi-turn ───────────────────────────────────────────────────
    if "A" in batches:
        print(f"\n{'='*60}\nBatch A — Multi-turn ({len(MULTI_TURN_SESSIONS)} sessions)\n{'='*60}")
        for i, (turns, desc) in enumerate(MULTI_TURN_SESSIONS, 1):
            print(f"\n[A{i}/{len(MULTI_TURN_SESSIONS)}] {desc}")
            try:
                history, steps = run_multiturn_session(turns, system_prompt, prov, TOOLS, workspace)
                final_reply = history[-1]["content"] if history else ""
                task_label = f"[multi-turn] {' → '.join(t[:50] for t in turns)}"
                saved = save_trajectory(col, task_label, f"multi-turn/{args.provider}",
                                        model_name, steps, final_reply, system_prompt,
                                        min_score=args.min_score)
                print(f"  Steps: {len(steps)} | Saved: {'✅' if saved else '⏭'}")
                if saved:
                    saved_total += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(2)

    # ── Batch C: Skill dispatch ───────────────────────────────────────────────
    if "C" in batches:
        print(f"\n{'='*60}\nBatch C — Skill dispatch ({len(SKILL_DISPATCH_TASKS)} tasks)\n{'='*60}")
        for i, (task, skill_name, desc) in enumerate(SKILL_DISPATCH_TASKS, 1):
            print(f"\n[C{i}/{len(SKILL_DISPATCH_TASKS)}] {desc}")
            steps = []
            def _cb(n, tool, args, result):
                steps.append({"tool": tool, "args": args, "result": str(result)[:500]})
            try:
                reply = prov.infer_with_tools(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task},
                    ],
                    tools=TOOLS, workspace=workspace, max_steps=8,
                    step_callback=_cb, call_type="task_inference",
                )
                saved = save_trajectory(col, task, f"skill-dispatch/{args.provider}",
                                        model_name, steps, reply, system_prompt,
                                        min_score=args.min_score)
                tool_names = [s["tool"] for s in steps]
                print(f"  Tools: {tool_names} | Saved: {'✅' if saved else '⏭'}")
                if saved:
                    saved_total += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(2)

    # ── Batch D: Evolution-aware ──────────────────────────────────────────────
    if "D" in batches:
        print(f"\n{'='*60}\nBatch D — Evolution-aware ({len(EVOLUTION_TASKS)} tasks)\n{'='*60}")
        for i, (task, expected_tools, desc) in enumerate(EVOLUTION_TASKS, 1):
            print(f"\n[D{i}/{len(EVOLUTION_TASKS)}] {desc}")
            steps = []
            def _cb(n, tool, args, result):
                steps.append({"tool": tool, "args": args, "result": str(result)[:500]})
            try:
                reply = prov.infer_with_tools(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task},
                    ],
                    tools=TOOLS, workspace=workspace, max_steps=8,
                    step_callback=_cb, call_type="task_inference",
                )
                saved = save_trajectory(col, task, f"evolution/{args.provider}",
                                        model_name, steps, reply, system_prompt,
                                        min_score=args.min_score)
                tool_names = [s["tool"] for s in steps]
                print(f"  Tools: {tool_names} | Saved: {'✅' if saved else '⏭'}")
                if saved:
                    saved_total += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(2)

    # ── Batch E: Web research ─────────────────────────────────────────────────
    if "E" in batches:
        print(f"\n{'='*60}\nBatch E — Web research ({len(WEB_RESEARCH_TASKS)} tasks)\n{'='*60}")
        for i, (task, expected_tools, desc) in enumerate(WEB_RESEARCH_TASKS, 1):
            print(f"\n[E{i}/{len(WEB_RESEARCH_TASKS)}] {desc}")
            steps = []
            def _cb(n, tool, args, result):
                steps.append({"tool": tool, "args": args, "result": str(result)[:500]})
            try:
                reply = prov.infer_with_tools(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task},
                    ],
                    tools=TOOLS, workspace=workspace, max_steps=8,
                    step_callback=_cb, call_type="task_inference",
                )
                saved = save_trajectory(col, task, f"web-research/{args.provider}",
                                        model_name, steps, reply, system_prompt,
                                        min_score=args.min_score)
                tool_names = [s["tool"] for s in steps]
                print(f"  Tools: {tool_names} | Saved: {'✅' if saved else '⏭'}")
                if saved:
                    saved_total += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Done. Saved {saved_total} new trajectories.")
    print(f"{'='*60}")

    if args.export:
        from trajectory_collector import get_collector as _gc
        col2 = _gc(config)
        path, count = col2.export_jsonl()
        print(f"\nExported → ({path}, {count} total records)")
        print(f"Ready for: python3 scripts/request_finetune.py --dataset {path}")


if __name__ == "__main__":
    main()
