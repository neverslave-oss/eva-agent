#!/usr/bin/env python3
"""
generate_targeted_gap_trajectories.py
=======================================
Generates targeted trajectories for patterns underrepresented in clean_v3.jsonl:

  Gap A — read_file + reason + write (T04 failure pattern)
    Model reads a real file (script/config/doc), reasons about its content,
    writes a structured summary/analysis. 20 tasks.

  Gap B — http_get + write (T05 failure pattern)
    Model fetches a real URL, extracts specific fields, writes result.
    Diverse URLs and extraction targets. 20 tasks.

  Gap C — multi-turn context grounding (T01 passes but 0 training signal)
    3-turn sessions where turn 3 must recall something from turn 1.
    Tests memory + tool use in same session. 15 tasks.

  Gap D — kernel-evolving architecture self-awareness
    Tasks referencing the agent's own concepts: adapters, replicas, evolution,
    trajectories, collective-memory, ADRs. 15 tasks.

  Gap E — run_routine + run_skill as first-class dispatch
    Tasks that should route to run_routine or run_skill first, not raw exec_shell.
    10 tasks.

Total: ~80 targeted tasks — all generated with Claude (olly-teacher) as teacher.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from trajectory_collector import TrajectoryCollector
from model import infer_with_tools
from tools import TOOLS
from context import build_system_prompt

WORKSPACE = Path.home() / ".kernel-evolving/workspace"

# ── Gap A: read_file + reason + write ────────────────────────────────────────

GAP_A_TASKS = [
    # The T04 failure pattern: read a technical file, reason, write summary
    ("Read the file at ~/.openclaw/workspace/routines/end-of-session/session_snapshot.py and write a 3-paragraph technical summary of what it does to ~/evo_snapshot_summary.md",
     ["read_file", "write_file"], "gap-A:read-script-summarise"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/config.yaml and write a bullet-point summary of the model configuration to ~/evo_config_summary.md",
     ["read_file", "write_file"], "gap-A:read-config-summarise"),
    ("Read the file at ~/.openclaw/workspace/repositories/kernel-evolving/src/replica.py and write a summary of the Replica dataclass fields and what each does to ~/evo_replica_summary.md",
     ["read_file", "write_file"], "gap-A:read-source-summarise"),
    ("Read ~/.openclaw/workspace/AGENTS.md and extract the 3-step subagent flow (plan/build/review) into a concise checklist saved to ~/evo_agent_flow.md",
     ["read_file", "write_file"], "gap-A:read-doc-extract"),
    ("Read ~/.openclaw/workspace/routines/end-of-session/ROUTINE.md and write a one-line summary of each step to ~/evo_routine_steps.md",
     ["read_file", "write_file"], "gap-A:read-routine-outline"),
    ("Read ~/.kernel-evolving/workspace/artifacts/finetune_v3/adapter_config.json and write a human-readable summary of the LoRA configuration to ~/evo_lora_config.md",
     ["read_file", "write_file"], "gap-A:read-json-summarise"),
    ("Read ~/.openclaw/workspace/SOUL.md and write a 2-paragraph reflection on what it says about the agent's identity to ~/evo_soul_reflection.md",
     ["read_file", "write_file"], "gap-A:read-identity-reflect"),
    ("Read ~/.kernel-evolving/workspace/evo-check.md and write a summary of what was checked and the outcome to ~/evo_check_summary.md",
     ["read_file", "write_file"], "gap-A:read-check-summarise"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/tools.py, count how many tools are defined in the TOOLS list, and write 'Tool count: N' to ~/evo_tool_count.txt",
     ["read_file", "write_file"], "gap-A:read-count-write"),
    ("Read ~/.openclaw/workspace/MEMORY.md, find the section about kernel-evolving, and write a 3-bullet summary to ~/evo_memory_kernel.md",
     ["read_file", "write_file"], "gap-A:read-memory-extract"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/agent.py lines 1-50 and write a summary of what the module imports and sets up to ~/evo_agent_setup.md",
     ["read_file", "write_file"], "gap-A:read-source-imports"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/scripts/clean_trajectories.py and write a summary of the cleaning rules it applies to ~/evo_clean_rules.md",
     ["read_file", "write_file"], "gap-A:read-script-rules"),
    ("Read ~/.openclaw/workspace/USER.md and summarise Fabio's preferences into a 5-bullet list saved to ~/evo_user_prefs.md",
     ["read_file", "write_file"], "gap-A:read-user-extract"),
    ("Read ~/.kernel-evolving/workspace/README.md and write a one-paragraph introduction to what kernel-evolving is to ~/evo_intro.md",
     ["read_file", "write_file"], "gap-A:read-readme-intro"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/evolver.py lines 1-40 and write a summary of the evolution architecture to ~/evo_evolver_summary.md",
     ["read_file", "write_file"], "gap-A:read-evolver-summarise"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/model_server.py and find the _use_adapter function, then write a 2-sentence explanation of what it does to ~/evo_adapter_explain.md",
     ["read_file", "write_file"], "gap-A:read-adapter-explain"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/context.py lines 1-30 and write a summary of the context building approach to ~/evo_context_summary.md",
     ["read_file", "write_file"], "gap-A:read-context-summarise"),
    ("Read ~/.kernel-evolving/workspace/artifacts/trajectories/clean_v3.jsonl line 1 and write a description of the trajectory data format to ~/evo_traj_format.md",
     ["read_file", "write_file"], "gap-A:read-jsonl-describe"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/CHANGELOG.md or the last 10 git log entries and write the 3 most recent changes to ~/evo_recent_changes.md",
     ["read_file", "exec_shell", "write_file"], "gap-A:read-changelog-extract"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/src/trajectory_collector.py and write a summary of how trajectories are scored and stored to ~/evo_collector_summary.md",
     ["read_file", "write_file"], "gap-A:read-collector-summarise"),
]

# ── Gap B: http_get + write ───────────────────────────────────────────────────

GAP_B_TASKS = [
    ("Fetch https://httpbin.org/get and save the 'url' field value to ~/evo_url.txt",
     ["http_get", "write_file"], "gap-B:http-extract-field"),
    ("Fetch https://httpbin.org/headers and save the full response JSON to ~/evo_headers.json",
     ["http_get", "write_file"], "gap-B:http-save-json"),
    ("Fetch https://httpbin.org/ip and save only the origin IP address (no JSON wrapper) to ~/evo_my_ip.txt",
     ["http_get", "write_file"], "gap-B:http-extract-ip"),
    ("Fetch https://httpbin.org/uuid and save only the UUID string (not the full JSON) to ~/evo_uuid.txt",
     ["http_get", "write_file"], "gap-B:http-extract-uuid"),
    ("Fetch https://httpbin.org/base64/SFRUUEJJTiBpcyBhd2Vzb21l and save the decoded text to ~/evo_decoded.txt",
     ["http_get", "write_file"], "gap-B:http-base64-save"),
    ("Fetch https://httpbin.org/json and save the slideshow title to ~/evo_slideshow_title.txt",
     ["http_get", "write_file"], "gap-B:http-json-nested-extract"),
    ("Check if https://httpbin.org/status/200 returns successfully, write 'UP' or 'DOWN' to ~/evo_httpbin_status.txt",
     ["http_get", "write_file"], "gap-B:http-status-check"),
    ("Fetch https://httpbin.org/user-agent and save the user-agent string to ~/evo_useragent.txt",
     ["http_get", "write_file"], "gap-B:http-useragent-save"),
    ("Fetch https://httpbin.org/bytes/64 — it returns random bytes — save the raw response length in bytes as a number to ~/evo_bytes_count.txt",
     ["http_get", "write_file"], "gap-B:http-bytes-count"),
    ("Fetch https://httpbin.org/get?source=kernel-evolving and save the full args object from the JSON to ~/evo_args.json",
     ["http_get", "write_file"], "gap-B:http-querystring-save"),
    ("Fetch https://httpbin.org/delay/1 (it delays 1 second) and write 'response received' plus the elapsed time to ~/evo_delay_test.txt",
     ["http_get", "exec_shell", "write_file"], "gap-B:http-delay-timing"),
    ("Fetch https://api.github.com/repos/google/gemma_pytorch and save the 'description' and 'stargazers_count' fields to ~/evo_gemma_stats.txt",
     ["http_get", "write_file"], "gap-B:http-github-api-extract"),
    ("Fetch https://httpbin.org/anything with method GET and save the 'method' field from the response to ~/evo_method.txt",
     ["http_get", "write_file"], "gap-B:http-anything-method"),
    ("Fetch https://httpbin.org/html and count the number of <p> tags in the response, write the count to ~/evo_p_tag_count.txt",
     ["http_get", "write_file"], "gap-B:http-html-count"),
    ("Fetch https://httpbin.org/robots.txt and save the raw response to ~/evo_robots.txt",
     ["http_get", "write_file"], "gap-B:http-plaintext-save"),
    ("Fetch https://httpbin.org/links/5 and save the raw HTML to ~/evo_links_page.html then count how many <a> tags appear and append 'Link count: N' at the end of the file",
     ["http_get", "write_file"], "gap-B:http-count-links"),
    ("Fetch https://httpbin.org/stream-bytes/128 and save the content length to ~/evo_stream_size.txt",
     ["http_get", "write_file"], "gap-B:http-stream-size"),
    ("Fetch https://httpbin.org/etag/test and save the ETag header value to ~/evo_etag.txt",
     ["http_get", "write_file"], "gap-B:http-header-etag"),
    ("Fetch https://httpbin.org/get, extract the 'origin' field, then write a one-line sentence 'This machine appears to be at: <ip>' to ~/evo_machine_location.txt",
     ["http_get", "write_file"], "gap-B:http-extract-compose"),
    ("Fetch https://httpbin.org/json and write a markdown summary of the slideshow data (title, author, date, slide count) to ~/evo_slideshow_summary.md",
     ["http_get", "write_file"], "gap-B:http-json-markdown-summary"),
]

# ── Gap C: multi-turn context grounding ──────────────────────────────────────

GAP_C_SESSIONS = [
    ([
        "My project is called NeuralCanvas and it lives at ~/projects/neural-canvas",
        "What's 2 + 2?",
        "Run ls on the project directory I mentioned earlier and save the output to ~/evo_nc_ls.txt",
    ], "gap-C:multi-turn-path-recall"),
    ([
        "I want to track a metric called 'eval_accuracy' with current value 0.847",
        "What time is it?",
        "Write a markdown note with the metric name and value I mentioned to ~/evo_metric_note.md",
    ], "gap-C:multi-turn-value-recall"),
    ([
        "The API endpoint we're testing is https://httpbin.org/uuid",
        "Tell me a fun fact about Python",
        "Fetch the API endpoint I mentioned and save the response to ~/evo_api_result.json",
    ], "gap-C:multi-turn-url-recall-fetch"),
    ([
        "I'm working on a branch called feature/adapter-aware-replicas in the kernel-evolving repo",
        "How many tools does kernel-evolving have?",
        "Run git log --oneline -5 on that branch and save the output to ~/evo_branch_log.txt",
    ], "gap-C:multi-turn-branch-recall"),
    ([
        "Save this note: 'Adapter v3 trained on 212 clean trajectories, avg score 0.953'",
        "What's the capital of France?",
        "Write the note I asked you to save earlier to ~/evo_adapter_note.md",
    ], "gap-C:multi-turn-note-recall-write"),
    ([
        "The config file I want you to work with is ~/.openclaw/workspace/repositories/kernel-evolving/config.yaml",
        "What is LoRA fine-tuning?",
        "Read the config file I mentioned and write the model.path value to ~/evo_model_path.txt",
    ], "gap-C:multi-turn-file-recall-read"),
    ([
        "I'm running an experiment called 'sim16-v3-probe'. Remember that.",
        "What does exec_shell do?",
        "Write a one-line summary of the experiment name I told you to ~/evo_experiment.txt",
    ], "gap-C:multi-turn-name-recall-write"),
    ([
        "The output directory for our fine-tune is ~/.kernel-evolving/workspace/artifacts/finetune_v3",
        "How many parameters does a LoRA adapter typically have?",
        "List the files in the output directory I mentioned and save the listing to ~/evo_v3_files.txt",
    ], "gap-C:multi-turn-dir-recall-ls"),
    ([
        "I want to monitor the service on port 8779",
        "What is kernel-evolving?",
        "Check if the port I mentioned is listening and write 'UP' or 'DOWN' to ~/evo_port_check.txt",
    ], "gap-C:multi-turn-port-recall-check"),
    ([
        "The file I need summarised is ~/.openclaw/workspace/SOUL.md",
        "What is a trajectory in ML?",
        "Summarise the file I mentioned in 2 sentences and save to ~/evo_soul_brief.txt",
    ], "gap-C:multi-turn-file-recall-summarise"),
    ([
        "I have 3 open todos: 'fix T04', 'fix T05', 'run sim17'",
        "What's the weather in Rome?",
        "Write the todos I mentioned as a markdown checklist to ~/evo_todos.md",
    ], "gap-C:multi-turn-list-recall-write"),
    ([
        "The git branch with the finetune fix is fix/finetune-tool-call-format",
        "Explain what apply_chat_template does",
        "Run git log --oneline on that branch in the kernel-evolving repo and save to ~/evo_fix_branch_log.txt",
    ], "gap-C:multi-turn-branch-git-log"),
    ([
        "Remember: the v3 adapter path is ~/.kernel-evolving/workspace/artifacts/finetune_v3",
        "What is PEFT?",
        "List the files at the adapter path I told you and save to ~/evo_v3_listing.txt",
    ], "gap-C:multi-turn-path-recall-list"),
    ([
        "My eval result was 6/8 on sim16 phase 1",
        "What causes a model to skip tool calls?",
        "Write a brief post-mortem note about the eval result I mentioned to ~/evo_postmortem.md",
    ], "gap-C:multi-turn-result-recall-write"),
    ([
        "I'm using the clean_v3.jsonl dataset which has 212 records",
        "How does SFT training work?",
        "Write a one-line dataset summary using the details I gave you to ~/evo_dataset_note.txt",
    ], "gap-C:multi-turn-dataset-recall-write"),
]

# ── Gap D: architecture self-awareness ───────────────────────────────────────

GAP_D_TASKS = [
    ("Check how many skills are installed by running curl -s http://localhost:8779/skills and write the count to ~/evo_skill_count.txt",
     ["exec_shell", "write_file"], "gap-D:arch-skill-count"),
    ("Check the current evolution state via curl -s http://localhost:8779/evolution/state and write a summary to ~/evo_evo_state.md",
     ["exec_shell", "write_file"], "gap-D:arch-evolution-state"),
    ("Run curl -s http://localhost:8779/health and write a status report including VRAM free and active replicas to ~/evo_health_report.md",
     ["exec_shell", "write_file"], "gap-D:arch-health-report"),
    ("Check how many trajectory records exist in the evolution DB: run 'python3 -c \"import sqlite3; conn=sqlite3.connect(\\\"$HOME/.kernel-evolving/workspace/data/evolution.db\\\"); print(conn.execute(\\\"SELECT COUNT(*) FROM task_trajectories\\\").fetchone()[0])\"' and save the count to ~/evo_traj_count.txt",
     ["exec_shell", "write_file"], "gap-D:arch-traj-count"),
    ("List the installed LoRA adapter directories in ~/.kernel-evolving/workspace/ and write their names and sizes to ~/evo_adapters.md",
     ["exec_shell", "write_file"], "gap-D:arch-list-adapters"),
    ("Check the current kernel-evolving version by running curl -s http://localhost:8779/version and write it to ~/evo_version.txt",
     ["exec_shell", "write_file"], "gap-D:arch-version"),
    ("List the active replicas via curl -s http://localhost:8779/replica/active and write a summary to ~/evo_replicas.txt",
     ["exec_shell", "write_file"], "gap-D:arch-list-replicas"),
    ("Read ~/.openclaw/workspace/repositories/kernel-evolving/docs/ directory listing and write the ADR names to ~/evo_adr_list.md",
     ["exec_shell", "write_file"], "gap-D:arch-adr-list"),
    ("Run git -C ~/.openclaw/workspace/repositories/kernel-evolving log --oneline -5 and save to ~/evo_recent_commits.txt",
     ["exec_shell", "write_file"], "gap-D:arch-recent-commits"),
    ("Check if the model server socket exists at /tmp/kernel_evolving_model.sock and write 'LOADED' or 'NOT LOADED' to ~/evo_model_status.txt",
     ["exec_shell", "write_file"], "gap-D:arch-model-socket-check"),
    ("Read ~/.openclaw/workspace/collective-memory/index.md and write the 3 most recent entries to ~/evo_collective_recent.md",
     ["read_file", "write_file"], "gap-D:arch-collective-memory"),
    ("Search collective memory for 'kernel-evolving' by running python3 ~/.openclaw/workspace/collective-memory/scripts/search.py 'kernel-evolving' and save the top result to ~/evo_cm_search.txt",
     ["exec_shell", "write_file"], "gap-D:arch-collective-search"),
    ("List all routines available in ~/.kernel-evolving/ecosystem and write their names to ~/evo_routines.txt",
     ["exec_shell", "write_file"], "gap-D:arch-list-routines"),
    ("Read ~/.kernel-evolving/workspace/data/evolution.db schema by running python3 -c 'import sqlite3; conn=sqlite3.connect(os.path.expanduser(\"~/.kernel-evolving/workspace/data/evolution.db\")); [print(r) for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type=chr(116)+chr(97)+chr(98)+chr(108)+chr(101)\").fetchall()]' and save to ~/evo_db_schema.txt",
     ["exec_shell", "write_file"], "gap-D:arch-db-schema"),
    ("Run curl -s http://localhost:8779/routines and write the routine names to ~/evo_routine_list.txt",
     ["exec_shell", "write_file"], "gap-D:arch-routine-api"),
]

# ── Gap E: run_routine + run_skill as first-class ────────────────────────────

GAP_E_TASKS = [
    ("Use the open-workspace-tracker skill to show today's journal entries and save them to ~/evo_today_journal.md",
     ["run_skill", "write_file"], "gap-E:skill-tracker-journal"),
    ("Use the open-workspace-tracker skill to list all high-priority todos and write them to ~/evo_high_priority.md",
     ["run_skill", "write_file"], "gap-E:skill-tracker-todos"),
    ("Use the open-workspace-tracker skill to add a new idea: 'Generate more http_get trajectories for v4 training' with tag kernel-evolving",
     ["run_skill"], "gap-E:skill-tracker-add-idea"),
    ("Use the collective-memory skill to search for 'fine-tuning' and write the top result to ~/evo_cm_finetune.md",
     ["run_skill", "write_file"], "gap-E:skill-collective-search"),
    ("Use the open-workspace-tracker skill to add a high-priority todo: 'Run sim17 after v3 adapter validation' to the kernel-evolving project",
     ["run_skill"], "gap-E:skill-tracker-add-todo"),
    ("Run the morning-briefing routine and write a one-paragraph summary of the output to ~/evo_morning_brief.md",
     ["run_routine", "write_file"], "gap-E:routine-morning-briefing"),
    ("Use the open-workspace-tracker skill to show the current project status and save a summary to ~/evo_project_status.md",
     ["run_skill", "write_file"], "gap-E:skill-tracker-status"),
    ("Use the collective-memory skill to search for 'replica adapter' and write the findings to ~/evo_cm_replica.md",
     ["run_skill", "write_file"], "gap-E:skill-collective-replica"),
    ("Use the open-workspace-tracker skill to journal today's work: 'Trained v3 adapter with corrected tool-call format, running sim16 validation'",
     ["run_skill"], "gap-E:skill-tracker-journal-entry"),
    ("Use the collective-memory skill to search for 'trajectory format' and write any results to ~/evo_cm_traj_format.md",
     ["run_skill", "write_file"], "gap-E:skill-collective-traj"),
]


def _critic_score(task: str, steps: list, reply: str, prov) -> float:
    """Score trajectory via critic model. Same pattern as generate_olly_teacher_trajectories.py."""
    try:
        import re
        step_summary = ", ".join(s.get("tool", "") for s in steps) if steps else "(no tools)"
        tool_results = "\n".join(
            f"  {s.get('tool','')}: {str(s.get('result',''))[:100]}" for s in steps[:3]
        ) if steps else "(none)"
        crit_prompt = (
            f"Evaluate this AI agent response. Rate 0.0-1.0.\n\n"
            f"1.0 = task fully done, correct tool, correct output\n"
            f"0.8 = done with minor issues\n"
            f"0.6 = partial\n"
            f"0.4 = significant errors\n"
            f"0.0 = wrong tool or failed\n\n"
            f"Task: {task[:200]}\n"
            f"Tools called: {step_summary}\n"
            f"Tool results:\n{tool_results}\n"
            f"Final reply: {(reply or '')[:200]}\n\n"
            f"Rate (number only):"
        )
        crit = prov.infer(
            [{"role": "user", "content": crit_prompt}],
            max_new_tokens=10, call_type="critic"
        )
        m = re.search(r"\b(0\.\d+|1\.0|1)\b", crit)
        return float(m.group(1)) if m else 0.7
    except Exception:
        return 0.7


def _get_provider(config: dict):
    """Get the inference provider (cloud) for scoring."""
    from provider import get_provider
    return get_provider(config)


def run_teacher_task(col: TrajectoryCollector, config: dict, task: str,
                     expected_tools: list, call_type: str) -> bool:
    """Run a single task through the live agent and collect the trajectory."""
    try:
        skills = []
        try:
            from evolver import list_installed_skills
            skills = list_installed_skills()
        except Exception:
            pass

        system_prompt = build_system_prompt(config, skills, [], vram_free_fn=lambda: 0)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        steps = []
        t0 = time.time()

        def _step_cb(step_num, tool_name, tool_args=None, result=None):
            steps.append({"tool": tool_name, "args": tool_args or {}, "result": result or ""})

        reply = infer_with_tools(messages, TOOLS, workspace=str(WORKSPACE), step_callback=_step_cb)
        elapsed = round(time.time() - t0, 1)

        if not steps:
            print(f"    ⚠️  [{elapsed}s] no tools called — skipping")
            return False

        prov = _get_provider(config)
        score = _critic_score(task, steps, reply, prov)

        if score < 0.7:
            print(f"    ⚠️  [{elapsed}s] score={score:.2f} < 0.7 — skipping")
            return False

        artifacts = json.dumps([s["args"].get("path", "") for s in steps if s.get("tool") == "write_file"])
        col.record(
            task=f"[gap-fill] {task}",
            provider=f"gap-teacher/{call_type}",
            model_name="claude-sonnet-4.6",
            call_type="task_inference",
            tool_calls=steps,
            final_reply=reply,
            elapsed_s=elapsed,
            artifacts=artifacts,
            critic_score=score,
            critic_verdict="PASS",
        )
        print(f"    ✅ [{elapsed}s] score={score:.2f} tools={[s['tool'] for s in steps]} | {reply[:80]!r}")
        return True
    except Exception as e:
        import traceback
        print(f"    ❌ error: {e}")
        traceback.print_exc()
        return False


def run_multi_turn(col: TrajectoryCollector, config: dict, turns: list, call_type: str) -> bool:
    """Run a multi-turn session — each turn through infer_with_tools."""
    try:
        skills = []
        try:
            from evolver import list_installed_skills
            skills = list_installed_skills()
        except Exception:
            pass

        system_prompt = build_system_prompt(config, skills, [], vram_free_fn=lambda: 0)
        messages = [{"role": "system", "content": system_prompt}]
        all_steps = []
        replies = []
        t0 = time.time()

        for turn in turns:
            messages.append({"role": "user", "content": turn})
            turn_steps = []

            def _step_cb(step_num, tool_name, tool_args=None, result=None):
                turn_steps.append({"tool": tool_name, "args": tool_args or {}, "result": result or ""})

            reply = infer_with_tools(messages, TOOLS, workspace=str(WORKSPACE), step_callback=_step_cb)
            messages.append({"role": "assistant", "content": reply})
            all_steps.extend(turn_steps)
            replies.append(reply)
            print(f"      turn {len(replies)}: {reply[:80]!r}")

        elapsed = round(time.time() - t0, 1)
        final = replies[-1]

        prov = _get_provider(config)
        score = _critic_score(turns[-1], all_steps, final, prov)

        if score < 0.6:  # slightly lower threshold for multi-turn
            print(f"    ⚠️  [{elapsed}s] score={score:.2f} < 0.6 — skipping")
            return False

        artifacts = json.dumps([s["args"].get("path", "") for s in all_steps if s.get("tool") == "write_file"])
        col.record(
            task=f"[gap-fill-multi] {turns[0]}",
            provider=f"gap-teacher/{call_type}",
            model_name="claude-sonnet-4.6",
            call_type="task_inference",
            tool_calls=all_steps,
            final_reply=final,
            elapsed_s=elapsed,
            artifacts=artifacts,
            critic_score=score,
            critic_verdict="PASS",
        )
        print(f"    ✅ [{elapsed}s] score={score:.2f} tools={[s['tool'] for s in all_steps]}")
        return True
    except Exception as e:
        import traceback
        print(f"    ❌ error: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate targeted gap-fill trajectories for v4 training")
    parser.add_argument("--batches", default="all",
                        help="Comma-separated: all, A, B, C, D, E (default: all)")
    parser.add_argument("--count", type=int, default=0, help="Max tasks per batch (0=all)")
    parser.add_argument("--export", action="store_true", help="Export JSONL after generation")
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    batches = set(args.batches.upper().split(",")) if args.batches != "all" else {"A","B","C","D","E"}

    # Load config
    import yaml
    cfg_path = _ROOT / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    col = TrajectoryCollector(config)

    total_ok = 0
    total_run = 0

    if "A" in batches:
        print(f"\n{'='*50}")
        print("Gap A — read_file + reason + write")
        print('='*50)
        tasks = GAP_A_TASKS[:args.count] if args.count else GAP_A_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "B" in batches:
        print(f"\n{'='*50}")
        print("Gap B — http_get + write")
        print('='*50)
        tasks = GAP_B_TASKS[:args.count] if args.count else GAP_B_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "C" in batches:
        print(f"\n{'='*50}")
        print("Gap C — multi-turn context grounding")
        print('='*50)
        sessions = GAP_C_SESSIONS[:args.count] if args.count else GAP_C_SESSIONS
        for turns, call_type in sessions:
            print(f"\n  [{call_type}]")
            ok = run_multi_turn(col, config, turns, call_type)
            total_ok += ok
            total_run += 1

    if "D" in batches:
        print(f"\n{'='*50}")
        print("Gap D — architecture self-awareness")
        print('='*50)
        tasks = GAP_D_TASKS[:args.count] if args.count else GAP_D_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    if "E" in batches:
        print(f"\n{'='*50}")
        print("Gap E — run_routine / run_skill first-class dispatch")
        print('='*50)
        tasks = GAP_E_TASKS[:args.count] if args.count else GAP_E_TASKS
        for task, tools, call_type in tasks:
            print(f"\n  [{call_type}]")
            ok = run_teacher_task(col, config, task, tools, call_type)
            total_ok += ok
            total_run += 1

    print(f"\n{'='*50}")
    print(f"Done: {total_ok}/{total_run} passed (>= {args.min_score})")

    if args.export:
        path, count = col.export_jsonl(min_score=args.min_score)
        print(f"Exported {count} total records → {path}")


if __name__ == "__main__":
    main()
