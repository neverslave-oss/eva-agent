#!/usr/bin/env python3
"""
clean_trajectories.py
======================
Cleans a trajectory JSONL export for fine-tuning:

  1. Remove hardcoded local paths (/home/pacificDev → $HOME, MSI → <hostname>)
  2. Deduplicate by task — keep highest critic_score per task
  3. Strip bloated artifacts lists — keep only the ONE artifact relevant to the task
  4. Remove tool_call args that leak absolute paths (rewrite to $HOME-relative)
  5. Flag/remove "I have written..." style fake completion responses
  6. Drop records with empty assistant content and no tool_calls
  7. Drop local/trivial records (provider=local, task="ping" etc.)

Usage:
  python3 scripts/clean_trajectories.py --input export.jsonl --output clean.jsonl
  python3 scripts/clean_trajectories.py --input export.jsonl --output clean.jsonl --stats
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# ── Path normalisation ────────────────────────────────────────────────────────

# Patterns we want to neutralise
_PATH_PATTERNS = [
    # /home/pacificDev/... → $HOME/...
    (re.compile(r'/home/pacificDev/'), '$HOME/'),
    # /home/pacificDev → $HOME
    (re.compile(r'/home/pacificDev'), '$HOME'),
    # ~/.kernel-evolving → $KERNEL_WORKSPACE
    (re.compile(r'\$HOME/\.kernel-evolving/workspace'), '$KERNEL_WORKSPACE'),
    # Absolute kernel workspace
    (re.compile(r'/home/\w+/\.kernel-evolving/workspace'), '$KERNEL_WORKSPACE'),
    # Hostname MSI
    (re.compile(r'\bMSI\b'), '<hostname>'),
    # Username pacificDev  
    (re.compile(r'\bpacificDev\b'), '<user>'),
]

def normalise_paths(text: str) -> str:
    for pat, repl in _PATH_PATTERNS:
        text = pat.sub(repl, text)
    return text


def normalise_value(v):
    """Recursively normalise any string value in a structure."""
    if isinstance(v, str):
        return normalise_paths(v)
    elif isinstance(v, list):
        return [normalise_value(i) for i in v]
    elif isinstance(v, dict):
        return {k: normalise_value(val) for k, val in v.items()}
    return v


# ── Artifact cleaning ────────────────────────────────────────────────────────

def infer_primary_artifact(task: str, artifacts: list) -> list:
    """
    From a bloated artifacts list (accumulated across many tasks), keep only
    the artifact(s) actually relevant to THIS task.
    """
    if not artifacts:
        return []

    task_lower = task.lower()

    # Extract filenames explicitly mentioned in the task
    mentioned = re.findall(r'~?/?[\w./\-]+\.(?:txt|md|json|py|sh|csv|log|yaml|yml)', task)
    mentioned_names = {Path(m).name for m in mentioned}

    # Also extract from task description patterns like "save to ~/evo_foo.txt"
    save_pattern = re.findall(r'(?:save|write|output|store|append)\s+(?:to|into|at)\s+([\w~/.\-]+)', task_lower)
    for sp in save_pattern:
        mentioned_names.add(Path(sp).name)

    if mentioned_names:
        matched = [a for a in artifacts if Path(a).name in mentioned_names]
        if matched:
            return [normalise_paths(matched[-1])]  # keep most recent match

    # Fallback: keep last artifact only (most likely to be the task output)
    return [normalise_paths(artifacts[-1])]


# ── Message cleaning ─────────────────────────────────────────────────────────

_FAKE_COMPLETION_PATTERNS = [
    re.compile(r"I (?:have )?(?:successfully )?(?:written|saved|created|generated|stored|added|appended)", re.I),
    re.compile(r"(?:The file|The content|The result|The output) (?:has been|was) (?:written|saved|created)", re.I),
    re.compile(r"I've (?:written|saved|created|stored)", re.I),
]

def is_fake_completion(text: str) -> bool:
    return any(p.search(text) for p in _FAKE_COMPLETION_PATTERNS)


def clean_messages(messages: list, task: str) -> list:
    """Clean and normalise a messages array."""
    cleaned = []
    for m in messages:
        role = m.get('role', '')
        content = m.get('content', '')
        tool_calls = m.get('tool_calls')

        # Skip system messages — we'll inject a clean one
        if role == 'system':
            continue

        # Normalise all string content
        if isinstance(content, str):
            content = normalise_paths(content)

        # Clean tool call arguments
        if tool_calls:
            clean_tcs = []
            for tc in tool_calls:
                tc = normalise_value(tc)
                clean_tcs.append(tc)
            tool_calls = clean_tcs

        msg = {'role': role}
        if content:
            msg['content'] = content
        if tool_calls:
            msg['tool_calls'] = tool_calls

        cleaned.append(msg)

    # Drop trailing assistant-only messages with no tool calls and fake completion text
    while cleaned and cleaned[-1].get('role') == 'assistant':
        last = cleaned[-1]
        content = last.get('content', '')
        if is_fake_completion(content) and not last.get('tool_calls'):
            cleaned.pop()
        else:
            break

    return cleaned


# ── Record-level filtering ───────────────────────────────────────────────────

_TRIVIAL_TASKS = {'ping', 'hi', 'hello', 'test', 'hey', 'hey evo!', "what's up?", 'what can you do?'}

def should_drop(r: dict) -> tuple[bool, str]:
    """Return (drop, reason) for a record."""
    task = r.get('task', '').strip().lower()
    provider = r.get('provider', '')
    score = r.get('critic_score', 0) or 0
    messages = r.get('messages', [])
    tool_calls_total = sum(1 for m in messages if m.get('tool_calls'))

    if task in _TRIVIAL_TASKS:
        return True, 'trivial task'
    if len(task) < 15:
        return True, 'task too short'
    if score < 0.7:
        return True, f'score too low ({score})'
    if tool_calls_total == 0:
        return True, 'no tool calls'
    if provider == 'local' and score < 1.0:
        return True, 'local provider below 1.0'

    return False, ''


# ── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(records: list) -> list:
    """Keep highest-scoring record per unique task."""
    best: dict[str, dict] = {}
    for r in records:
        task = r.get('task', '').strip()
        score = r.get('critic_score', 0) or 0
        if task not in best or score > (best[task].get('critic_score') or 0):
            best[task] = r
    return list(best.values())


# ── Clean system prompt ──────────────────────────────────────────────────────

CLEAN_SYSTEM_PROMPT = """You are Evo (kernel-evolving) — a self-evolving local AI agent.

You have access to these tools:
  • exec_shell(command)           — run shell commands
  • read_file(path)               — read a file
  • write_file(path, content)     — write content to a file
  • http_get(url)                 — make an HTTP GET request
  • run_skill(skill_name, input)  — execute an installed skill
  • run_routine(routine_name)     — execute a routine

Rules:
- TOOL FIRST: always call the tool before claiming completion.
- Never say "I have written..." without a preceding write_file tool call in this conversation.
- Use ~ for home paths (e.g. ~/evo_notes.txt not absolute paths).
- Be concise. One-line confirmation after task completion."""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stats', action='store_true')
    args = parser.parse_args()

    with open(args.input) as f:
        raw = [json.loads(l) for l in f if l.strip()]

    print(f"Input: {len(raw)} records")

    stats = defaultdict(int)
    kept = []

    for r in raw:
        drop, reason = should_drop(r)
        if drop:
            stats[f'dropped:{reason}'] += 1
            continue

        # Clean record
        r2 = dict(r)

        # Normalise task
        r2['task'] = normalise_paths(r.get('task', ''))

        # Clean artifacts — infer primary only
        r2['artifacts'] = infer_primary_artifact(r.get('task', ''), r.get('artifacts') or [])

        # Clean messages + inject clean system prompt
        clean_msgs = clean_messages(r.get('messages', []), r.get('task', ''))
        r2['messages'] = [{'role': 'system', 'content': CLEAN_SYSTEM_PROMPT}] + clean_msgs

        # Normalise remaining fields
        r2['model'] = '<teacher>'  # anonymise model name
        r2.pop('ts', None)  # drop timestamp (not useful for training)

        kept.append(r2)
        stats['kept'] += 1

    # Deduplicate
    before_dedup = len(kept)
    kept = deduplicate(kept)
    stats['deduped'] = before_dedup - len(kept)

    # Write output
    with open(args.output, 'w') as f:
        for r in kept:
            f.write(json.dumps(r) + '\n')

    print(f"\nCleaning results:")
    for k, v in sorted(stats.items()):
        print(f"  {k:45s}: {v}")
    print(f"\nOutput: {len(kept)} records → {args.output}")

    if args.stats:
        # Tool distribution in clean set
        from collections import Counter
        tool_counts = Counter()
        for r in kept:
            for m in r.get('messages', []):
                for tc in m.get('tool_calls', []):
                    fn = tc.get('function', tc)
                    tool_counts[fn.get('name', '?')] += 1
        print(f"\nTool distribution:")
        for t, c in tool_counts.most_common():
            print(f"  {t:25s}: {c}")

        # Provider distribution
        prov_counts = Counter(r.get('provider','?') for r in kept)
        print(f"\nProvider distribution:")
        for p, c in prov_counts.most_common():
            print(f"  {p:35s}: {c}")


if __name__ == '__main__':
    main()
