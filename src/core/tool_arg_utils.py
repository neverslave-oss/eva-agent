"""
tool_arg_utils.py — Shared tool-argument sanitization/normalization.

Consolidates logic that was previously duplicated across model.py,
model_server.py, tools.py, and evo_routine_executor.py (R11). Local models
frequently emit malformed tool arguments (leaked XML tags, shell-redirect
injection, alias keys instead of canonical ones) — this module is the single
place that cleans them up.
"""
import re
from datetime import datetime

_XML_CLOSE_RE = re.compile(r'^[a-zA-Z_][\w-]*</\w+>\s*', re.DOTALL)


def sanitize_text(val: str) -> str:
    """Strip newlines and shell redirect chars that local models may inject into arguments.
    E.g. '>\\nvoice-clone' -> 'voice-clone', '>\\nkernel-doc-retrieval' -> 'kernel-doc-retrieval'.
    """
    if not isinstance(val, str):
        return val
    val = val.strip()
    if "\n" in val:
        val = [p.strip() for p in val.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
    val = val.lstrip(">").lstrip("\n").rstrip("<")
    return val


def normalize_tool_args(tool_name: str, args: dict) -> dict:
    """Map common alias keys emitted by local models to canonical tool arguments."""
    if not isinstance(args, dict):
        return {}
    out = dict(args)
    tname = sanitize_text((tool_name or "").strip())

    def _move(src: str, dst: str):
        if src in out and dst not in out:
            out[dst] = out.pop(src)

    # Strip leaked XML closing tags from any value (e.g. "command</name>\n`ls`" → "`ls`")
    # Nemotron sometimes generates <name>key</name>\nvalue and the key leaks into the value.
    for k in list(out.keys()):
        if isinstance(out[k], str):
            cleaned = _XML_CLOSE_RE.sub('', out[k]).strip().strip('`').strip()
            cleaned = sanitize_text(cleaned)
            if cleaned:
                out[k] = cleaned

    if tname in {"write_file", "read_file"}:
        _move("file_path", "path")
        _move("filePath", "path")
        _move("filepath", "path")
        _move("filename", "path")

    if tname == "exec_shell":
        # Model sometimes emits {name: "command", ...} or {cmd: ...}
        _move("name", "command")
        _move("cmd", "command")
        _move("shell_command", "command")

    if tname == "run_skill":
        _move("name", "skill_name")

    if tname == "run_routine":
        _move("name", "routine_name")

    if tname == "web_search":
        _move("q", "query")
        _move("search_query", "query")
        _move("name", "query")

    if tname == "browser_use":
        _move("objective", "task")
        _move("goal", "task")
        _move("prompt", "task")
        _move("start_url", "url")
        _move("steps", "max_steps")

    if tname == "sensors":
        _move("intent", "action")
        # NOTE: `command` and `device` are now real control-action parameters
        # (actuator + command), so we do NOT alias them to `action`/`target`.
        # Use `which` as an alias for `target` (the sensor device id).
        _move("which", "target")

    if tname == "http_get":
        _move("uri", "url")
        _move("name", "url")

    return out


def rewrite_date_tokens(text: str) -> str:
    """Replace shell date substitution tokens with today's actual date.

    Local models sometimes emit literal '$(date +%Y-%m-%d)' instead of
    computing the date themselves.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return text.replace("$(date +%Y-%m-%d)", today)
