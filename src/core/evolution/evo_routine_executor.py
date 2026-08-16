"""
evo_routine_executor.py — Evo-only routine executor.

Purpose:
- Keep production ROUTINE.md specs untouched
- Avoid changing the generic routine runner used elsewhere
- Execute routines step-by-step in kernel-evolving by:
  1) parsing the routine into discrete steps
  2) letting the model execute ONE step at a time with tools
  3) synthesizing a final output from completed step results

This avoids the common failure mode where Gemma returns meta-text like
"I have initiated the routine" instead of actually doing the work.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Callable

from core.inference.model import infer, infer_with_tools
from core.tools import TOOLS, WORKSPACE as DEFAULT_WORKSPACE


def _extract_section(body: str, heading: str) -> str:
    m = re.search(rf"##\s+{re.escape(heading)}\s*(.*?)(?=\n##\s+|\Z)", body, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def _extract_steps(body: str) -> list[dict]:
    steps_block = _extract_section(body, "Steps")
    if not steps_block:
        return [{"title": "Routine", "content": body.strip()}]

    matches = list(re.finditer(r"^###\s+(.+?)\n(.*?)(?=^###\s+|\Z)", steps_block, re.DOTALL | re.MULTILINE))
    if not matches:
        return [{"title": "Routine", "content": steps_block.strip()}]

    steps = []
    for m in matches:
        title = m.group(1).strip()
        content = m.group(2).strip()
        if content:
            steps.append({"title": title, "content": content})
    return steps or [{"title": "Routine", "content": steps_block.strip()}]


def _extract_output_format(body: str) -> str:
    out = _extract_section(body, "Output Format")
    return out.strip()


def _looks_incomplete(text: str) -> bool:
    t = text.lower()
    bad_markers = (
        "awaiting the results",
        "please wait",
        "has been initiated",
        "i will now execute",
        "i am now awaiting",
        "i will proceed",
        "i will perform",
        "the routine execution has been initiated",
    )
    raw_markers = ("<html", "<tool_call", "<tool_response", "<!doctype html", "```html")
    return any(m in t for m in bad_markers) or any(m in t for m in raw_markers)


def _extract_backtick_items(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"`([^`]+)`", text) if m.group(1).strip()]


def _is_shell_command(value: str) -> bool:
    prefixes = (
        "python ", "python3 ", "curl ", "pgrep ", "gh ", "cd ", "grep ",
        "cat ", "date ", "ls ", "find ", "echo ", "test ",
    )
    return value.startswith(prefixes) or any(x in value for x in (" && ", " || ", " | ", "; "))


def _collect_direct_artifacts(step_title: str, step_content: str, workspace: str) -> list[dict]:
    artifacts = []
    items = _extract_backtick_items(step_content)
    for item in items:
        try:
            if _is_shell_command(item):
                cmd = _rewrite_command(item, workspace)
                import subprocess
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=workspace)
                out = (res.stdout.strip() or res.stderr.strip() or "(no output)")[:4000]
                artifacts.append({"kind": "command", "source": cmd, "result": out})
            else:
                path = _rewrite_path(item)
                abs_path = os.path.expanduser(path)
                if not abs_path.startswith("/"):
                    abs_path = os.path.join(workspace, abs_path)
                if os.path.exists(abs_path):
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        out = f.read(4000)
                    artifacts.append({"kind": "file", "source": abs_path, "result": out})
                    if "updated today" in step_content.lower():
                        mtime = datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime("%Y-%m-%d")
                        artifacts.append({"kind": "mtime", "source": abs_path, "result": mtime})
        except Exception as e:
            artifacts.append({"kind": "error", "source": item, "result": str(e)})
    return artifacts


def _summarize_step_result(step_title: str, step_content: str, raw_result: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are summarizing the completed result of ONE routine step. "
                "Return a concise result for that step only. Never dump raw HTML/JSON/XML/tool markup."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Step: {step_title}\n\n"
                f"Step instructions:\n{step_content}\n\n"
                f"Raw result:\n{raw_result}\n\n"
                "Now summarize the completed outcome of this step in 2-5 bullet points or a short paragraph."
            ),
        },
    ]
    return infer(messages, max_new_tokens=8192).strip()


def _rewrite_date_tokens(text: str) -> str:
    from core.tool_arg_utils import rewrite_date_tokens
    return rewrite_date_tokens(text)


def _rewrite_command(command: str, workspace: str) -> str:
    # Evo must stay inside its own workspace unless the routine explicitly uses
    # absolute paths or explicit cross-workspace targets.
    return _rewrite_date_tokens(command)


def _rewrite_path(path: str) -> str:
    return _rewrite_date_tokens(path)


def execute_routine(
    routine: dict,
    workspace: str | None = None,
    step_callback: Callable[[int, str, dict, str], None] | None = None,
) -> str:
    body = routine.get("body", routine.get("instructions", ""))
    workspace = os.path.expanduser(workspace or DEFAULT_WORKSPACE)
    name = routine.get("name", "routine")
    description = routine.get("description", "")
    steps = _extract_steps(body)
    output_format = _extract_output_format(body)

    # Evo-only toolset: allow tools needed to execute steps, but prevent recursive routine calls.
    routine_tools = []
    for t in TOOLS:
        if t.get("function", {}).get("name") == "run_routine":
            continue
        tcopy = dict(t)
        routine_tools.append(tcopy)

    completed: list[dict] = []
    global_tool_step = 0

    for i, step in enumerate(steps, start=1):
        step_title = step["title"]
        step_content = step["content"]
        prior = "\n\n".join(
            f"Step {idx+1} — {s['title']}:\n{s['result']}" for idx, s in enumerate(completed[-3:])
        ) or "(none yet)"

        def _nested_cb(n: int, tool_name: str, args: dict, result: str):
            nonlocal global_tool_step
            global_tool_step += 1
            fixed_args = dict(args or {})
            if tool_name == "exec_shell" and fixed_args.get("command"):
                fixed_args["command"] = _rewrite_command(str(fixed_args["command"]), workspace)
            elif tool_name == "read_file" and fixed_args.get("path"):
                fixed_args["path"] = _rewrite_path(str(fixed_args["path"]))
            if step_callback:
                step_callback(
                    global_tool_step,
                    f"{step_title} · {tool_name}",
                    fixed_args,
                    result,
                )

        direct_artifacts = _collect_direct_artifacts(step_title, step_content, workspace)
        artifacts_blob = "\n\n".join(
            f"[{a['kind']}] {a['source']}\n{a['result']}" for a in direct_artifacts
        ) or "(none)"

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are Kernel executing ONE step of the routine '{name}'. "
                    "Complete this step now using the available tools. "
                    "Do not describe future work. Do not say you are starting or waiting. "
                    "Return only the completed result for this step. "
                    "Never dump raw HTML/JSON/XML/tool markup; summarize it instead. "
                    "Prefer the directly collected artifacts when they already answer the step."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Routine: {name}\n"
                    f"Description: {description}\n\n"
                    f"Previous completed step results:\n{prior}\n\n"
                    f"Current step {i}/{len(steps)}: {step_title}\n\n"
                    f"Instructions:\n{step_content}\n\n"
                    f"Execution workspace for this routine: {workspace}\n"
                    "Stay inside this workspace unless the routine explicitly gives an absolute path or explicit external target.\n"
                    "If a file path contains $(date +%Y-%m-%d), resolve it to today's date first.\n\n"
                    f"Directly collected artifacts for this step:\n{artifacts_blob}"
                ),
            },
        ]

        result = infer_with_tools(
            messages,
            tools=routine_tools,
            workspace=workspace,
            step_callback=_nested_cb,
        ).strip()

        if _looks_incomplete(result):
            result = _summarize_step_result(step_title, step_content, result)

        completed.append({
            "title": step_title,
            "content": step_content,
            "result": result,
        })

    results_blob = "\n\n".join(
        f"### {s['title']}\n{s['result']}" for s in completed
    )

    final_messages = [
        {
            "role": "system",
            "content": (
                f"You are Kernel composing the final output for the routine '{name}'. "
                "Use ONLY the completed step results provided. "
                "Return the finished final answer now. No preamble. No meta-commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Routine: {name}\n"
                f"Description: {description}\n\n"
                f"Completed step results:\n{results_blob}\n\n"
                f"Output format (if provided):\n{output_format or '(none specified)'}\n\n"
                "Now produce the final routine output."
            ),
        },
    ]

    try:
        final = infer(final_messages, max_new_tokens=8192).strip()
    except Exception:
        final = ""
    if not final or _looks_incomplete(final) or "model_server error" in final.lower():
        # Last-resort fallback: return completed step results instead of useless meta text.
        return results_blob
    return final
