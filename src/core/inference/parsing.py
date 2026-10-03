"""Stateless parsing helpers for model-server tool-call extraction.

Extracted from the original model_server.py monolith (refactor). These
functions only parse raw model output into tool_calls-format structures and
carry no shared state, so they are a leaf module — safe to import from anywhere
without circular-dependency risk.
"""

import json


def _sanitize_text(val: str) -> str:
    from core.tool_arg_utils import sanitize_text
    return sanitize_text(val)


def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    from core.tool_arg_utils import normalize_tool_args
    return normalize_tool_args(tool_name, args)


def _parse_function_eq_xml(raw_content: str) -> list:
    """Parse Nemotron's `function=` XML into tool_calls-format list.

    Accepts BOTH the alt form `<parameter=key>value</parameter>` AND the native
    attribute form `<parameter name="key">value</parameter>`. Previously only the
    alt form matched, so any native-form call silently dropped every argument
    (write_file/read_file/web_search/etc. all came back `{}` despite the model
    emitting correct args).
    """
    import re as _re
    fn_blocks = _re.findall(
        r'<function=([A-Za-z_][\w-]*)>(.*?)(?:</function>|(?=<function=)|$)',
        raw_content, _re.DOTALL,
    )
    if not fn_blocks:
        return []
    calls = []
    for tool_name, fn_body in fn_blocks:
        params = _re.findall(
            r'<parameter(?:=([A-Za-z_][\w-]*)|\s+name=["\']([A-Za-z_][\w-]*)["\'])>(.*?)</parameter>',
            fn_body, _re.DOTALL,
        )
        args = {}
        for p0, p1, val in params:
            key = p0 if p0 else p1
            if key:
                args[key] = val.strip()
        args = _normalize_tool_args(tool_name.strip(), args)
        calls.append({"function": {"name": tool_name.strip(), "arguments": args}})
    return calls


def _parse_tag_fallback(raw_content: str) -> list:
    """Parse the generic `<tool>name(args)</tool>` form into tool_calls-format list."""
    import re as _re
    tag_matches = _re.findall(r'<tool>(\w+)\((.*)\)</tool>', raw_content, _re.DOTALL)
    if not tag_matches:
        return []
    calls = []
    for tool_name, args_str in tag_matches:
        args_str = args_str.strip()
        try:
            args = json.loads(args_str) if args_str.startswith('{') else {}
            if not args:
                kv = _re.findall(r'(\w+)=["\']([^"\']*)["\']?', args_str)
                args = dict(kv)
        except Exception:
            args = {}
        calls.append({"function": {"name": tool_name, "arguments": args}})
    return calls
