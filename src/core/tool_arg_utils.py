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
        # Keep the last line that has real content and isn't a leaked XML/`>`
        # closing remnant. Guard against a value whose *every* line is filtered
        # out (all end with '>' or all blank) — previously `[...][-1]` blew up
        # with IndexError. Fall back to the raw (stripped) value in that case so
        # the tool loop never crashes on a malformed arg.
        _lines = [p.strip() for p in val.split("\n") if p.strip() and not p.strip().endswith(">")]
        if _lines:
            val = _lines[-1]
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


def get_tool_schema(tool_name: str) -> dict | None:
    """Return the tool's JSON-Schema parameters block, or None if unknown."""
    try:
        from core.tools import TOOLS
        for t in TOOLS:
            fn = t.get("function", {})
            if fn.get("name") == tool_name:
                return fn.get("parameters") or {}
    except Exception:
        return None
    return None


def validate_tool_args(tool_name: str, args: dict) -> list:
    """Validate args against the real tool schema (T1 in the critique design).

    Returns a list of problem descriptors (strings). Empty list == schema-valid.
    Checks:
      - every schema-required key present and non-empty
      - value type matches the schema (string/integer/number/boolean/array)

    A schema-required key with a missing or empty value is a hard problem
    (the call will fail). A type mismatch is also a problem. Optional keys are
    never required.
    """
    if not isinstance(args, dict):
        return [f"args must be an object for {tool_name}, got {type(args).__name__}"]
    schema = get_tool_schema(tool_name)
    if not schema:
        # Unknown tool — can't validate. Treat as no problems (fail open).
        return []
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    problems = []
    # Presence: every schema-required key must be present and non-empty.
    for key in required:
        val = args.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            problems.append(f"missing required key '{key}'")
    # Types: validate EVERY present key against its schema type (optional too),
    # so a wrong-typed optional arg (e.g. max_steps="15") is still caught.
    for key, val in args.items():
        if val is None:
            continue
        prop = props.get(key, {})
        ptype = prop.get("type")
        if not ptype:
            continue
        if ptype == "string" and not isinstance(val, str):
            problems.append(f"key '{key}' should be string, got {type(val).__name__}")
        elif ptype == "integer" and isinstance(val, bool):
            problems.append(f"key '{key}' should be integer, got bool")
        elif ptype == "integer" and not isinstance(val, int):
            problems.append(f"key '{key}' should be integer, got {type(val).__name__}")
        elif ptype == "number" and not isinstance(val, (int, float)):
            problems.append(f"key '{key}' should be number, got {type(val).__name__}")
        elif ptype == "boolean" and not isinstance(val, bool):
            problems.append(f"key '{key}' should be boolean, got {type(val).__name__}")
        elif ptype == "array" and not isinstance(val, list):
            problems.append(f"key '{key}' should be array, got {type(val).__name__}")
    return problems
