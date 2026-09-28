"""Structured re-prompt / repair-hint builders for the model server.

Extracted from the original model_server.py monolith (refactor). These build
human-readable guidance strings shown to the model when a tool call fails
(Plan 007 re-prompt) or when schema validation flags bad arguments (T1/T2
repair). Stateless pure functions — a leaf module.
"""


def build_failure_reprompt(tool_name: str, status: str, reason: str, failed_count: int, total_count: int, available_tools: list) -> str:
    """Build a structured re-prompt when a tool call fails.

    Instead of feeding raw error text to the model and hoping it recovers,
    this provides actionable guidance: what failed, why, and what to try next.
    """
    tool_list = ", ".join(sorted(available_tools)) if available_tools else "no alternative tools available"
    status_hints = {
        "empty": "The tool returned no output — it may need different parameters or the target may not exist.",
        "timeout": "The tool timed out — the operation took too long. Try breaking it into smaller steps or using a different approach.",
        "error": f"The tool encountered an error: {reason[:200]}",
        "skill_not_found": f"Skill '{tool_name}' not found. Use search_skills(query='{tool_name}') to find the correct name, then retry.",
        "routine_not_found": f"Routine '{tool_name}' not found. Use list_routines() to see available routines.",
        "no_results": "The tool found no results — try a different query or broader search terms.",
        "skill_execution_failed": f"The skill failed during execution. Try a different approach or use search_skills to find alternatives.",
        "routine_execution_failed": "The routine failed. Check list_routines() for alternatives or try a manual approach.",
        "permission_denied": "Permission denied — you may need to request user approval or use a different path/command.",
    }
    hint = status_hints.get(status, f"Tool call failed (status: {status}). Reason: {reason[:200]}")

    if failed_count == total_count and total_count == 1:
        return (
            f"⚠️ Tool `{tool_name}` failed: {hint}\n"
            f"Available tools: {tool_list}\n"
            f"Please try one of these approaches:\n"
            f"  1. Call the same tool with corrected arguments\n"
            f"  2. Use a different tool that could achieve the same result\n"
            f"  3. If you cannot complete the task with available tools, explain why and suggest escalation"
        )
    elif failed_count == total_count:
        return (
            f"⚠️ All {total_count} tool calls failed. Here's what went wrong with `{tool_name}`: {hint}\n"
            f"Available tools: {tool_list}\n"
            f"Re-evaluate the task and try a different approach."
        )
    else:
        return f"⚠️ This tool call failed ({status}): {reason[:200]}. Some other tools may have succeeded — continue with partial results."


def build_schema_repair_hint(tool_name: str, args: dict, problems: list, query: str = "") -> str:
    """Build a schema-aware repair hint for T2.

    Given the schema validation problems (T1), tell the model WHICH keys are
    missing/typo'd and what the schema expects (with the property description),
    plus the original request to derive the value from. This gives the model
    just enough to complete the one step instead of a generic "retry".
    """
    from core.tool_arg_utils import get_tool_schema
    schema = get_tool_schema(tool_name)
    props = (schema or {}).get("properties", {}) or {}
    lines = [f"Tool `{tool_name}` needs corrected arguments before it can run."]
    for p in problems:
        if p.startswith("missing required"):
            key = p.split("'")[1]
            desc = props.get(key, {}).get("description", "")
            base = f"- {p}"
            if desc:
                base += f" (schema: {desc})"
            lines.append(base)
        else:
            lines.append(f"- {p}")
    if query:
        lines.append(f"Original request (use it to fill the values): {query[:200]}")
    lines.append("Retry the tool with corrected arguments.")
    return " ".join(lines)
