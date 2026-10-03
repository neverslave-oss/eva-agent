#!/usr/bin/env python3
"""
fc_recovery_parser.py
=====================
Dedicated recovery parser for malformed-but-recoverable Nemotron function-call
XML. The strict harness parser (`_parse_function_eq_xml`) only accepts two
forms (``<parameter=key>value</parameter>`` and
``<parameter name="key">value</parameter>``); the Nemotron adapter (v8-v11)
frequently emits a small set of broken variants that still contain the real
argument values. The strict parser drops them all -> every call returns ``{}``
-> schema validation fails -> the battery fails on those tools.

This module recovers the values at the harness level. Each strategy is written
against a REAL emission captured verbatim from the model-server log
(``[tool_loop/raw ...] RAW=...``), so no invented shapes.

Return shape identical to `_parse_function_eq_xml`:
  [{"function": {"name": <tool>, "arguments": {<key>: <value>}}}]
"""
from __future__ import annotations

import re
from typing import Any

_FN_BLOCK_RE = re.compile(
    r"<function=([A-Za-z_][\w-]*)>(.*?)(?:</function>|(?=<function=)|$)",
    re.DOTALL,
)

# Tag names that are schema scaffolding, never argument keys.
_RESERVED = {
    "parameter", "function", "name", "type", "description",
    "required", "parameters", "tool_name",
}


def _fn_blocks(raw: str) -> list[tuple[str, str]]:
    return [(n.strip(), b) for n, b in _FN_BLOCK_RE.findall(raw)]


def _clean_val(v: str) -> str:
    """Strip literal \n escapes, surrounding whitespace/quotes, and any stray
    leftover closing tags so the value is the bare argument string."""
    v = v.replace("\\n", "\n")
    v = re.sub(r"</?parameter>", "", v)
    v = re.sub(r"</?name>", "", v)
    v = v.strip()
    v = v.strip(' \'"')
    return v.strip()


def _recover_clean(body: str) -> dict[str, str]:
    """The two canonical forms (keep working): <parameter=key>v</parameter> and
    <parameter name="key">v</parameter>."""
    args: dict[str, str] = {}
    for m in re.finditer(
        r"<parameter\s*=\s*([A-Za-z_][\w-]*)\s*>(.*?)</parameter>", body, re.DOTALL
    ):
        v = _clean_val(m.group(2))
        if v:
            args[m.group(1)] = v
    for m in re.finditer(
        r'<parameter\s+name\s*=\s*["\']([A-Za-z_][\w-]*)["\']\s*>(.*?)</parameter>',
        body, re.DOTALL,
    ):
        v = _clean_val(m.group(2))
        if v:
            args[m.group(1)] = v
    return args


def _recover_schema_echo(body: str) -> dict[str, str]:
    """C-form: schema echo where the value is inside <description>.
    REAL: <parameter><name>skill_name</name><type>string</type>
              <description>security-scanner</description></parameter>
    -> {skill_name: "security-scanner"}"""
    args: dict[str, str] = {}
    names = list(re.finditer(r"<name>\s*([A-Za-z_][\w-]*)\s*</name>", body, re.DOTALL))
    for i, m in enumerate(names):
        key = m.group(1)
        seg = body[m.end(): names[i + 1].start() if i + 1 < len(names) else len(body)]
        dm = re.search(r"<description>(.*?)</description>", seg, re.DOTALL)
        if dm:
            v = _clean_val(dm.group(1))
            if v:
                args[key] = v
    return args


def _recover_param_as_content(body: str) -> dict[str, str]:
    """A-form: key appears as bare <parameter>content</name>, value follows.
    REAL: <parameter>query</name>pdf</parameter>  -> {query: "pdf"}
          <parameter>query</parameter>value_pdf</parameter> -> {query: "value_pdf"}
          <parameter>query</name>value="pdf"</parameter> -> {query: "pdf"}
    NOT (B-form): when the body is a bare key-list
    (<parameter>url</parameter><parameter>timeout</parameter>...), the
    `KEY</parameter>VALUE</parameter>` pattern would wrongly match the last key
    plus the trailing value-list. Detect that and bail on the key-then-value
    arm (the value="..." arm stays safe)."""
    args: dict[str, str] = {}
    # B-form guard: 2+ successive bare <parameter>key</parameter> tags => this is
    # a key-list, not key-as-content. Skip the key-then-value patterns.
    bare_keys = re.findall(r"<parameter>\s*([A-Za-z_][\w-]*)\s*</parameter>", body)
    is_b_form = len(bare_keys) >= 2
    # <parameter>KEY</name>value="v"</parameter> (safe in both A and B forms)
    for m in re.finditer(
        r"<parameter>\s*([A-Za-z_][\w-]*)\s*</name>\s*value\s*=\s*\"([^\"]*)\"",
        body, re.DOTALL,
    ):
        args[m.group(1)] = _clean_val(m.group(2))
    if is_b_form:
        return args
    # <parameter>KEY</name>VALUE</parameter>
    for m in re.finditer(
        r"<parameter>\s*([A-Za-z_][\w-]*)\s*</name>\s*([^<]*)</parameter>",
        body, re.DOTALL,
    ):
        v = _clean_val(m.group(2))
        if v and m.group(1) not in args:
            args[m.group(1)] = v
    # <parameter>KEY</parameter>VALUE</parameter> (only when not a B-form key-list)
    for m in re.finditer(
        r"<parameter>\s*([A-Za-z_][\w-]*)\s*</parameter>\s*([^<]*)</parameter>",
        body, re.DOTALL,
    ):
        v = _clean_val(m.group(2))
        if v and m.group(1) not in args:
            args[m.group(1)] = v
    return args


def _recover_named_tag(body: str) -> dict[str, str]:
    """D-form: the parameter tag name IS the key, wrapping its own value.
    REAL: <query>pdf</query>  -> {query: "pdf"}"""
    args: dict[str, str] = {}
    for m in re.finditer(
        r"<([A-Za-z_][\w-]*)\s*>\s*([^<]*?)\s*</\1>", body, re.DOTALL
    ):
        key, val = m.group(1), _clean_val(m.group(2))
        if key not in _RESERVED and val:
            args[key] = val
    return args


def _recover_key_value_list(body: str) -> dict[str, str]:
    """B-form: a list of bare <parameter>key</parameter> tags followed by a list
    of bare values, closed by a single stray </parameter>.
    REAL: <parameter>url</parameter><parameter>timeout</parameter>
          https://example.com/api/status\n5\n</parameter>
    -> {url: "...", timeout: "5"}"""
    keys = re.findall(r"<parameter>\s*([A-Za-z_][\w-]*)\s*</parameter>", body)
    if not keys:
        return {}
    rest = re.sub(r"<parameter>\s*[A-Za-z_][\w-]*\s*</parameter>", "", body)
    rest = rest.replace("</parameter>", "")
    values = [_clean_val(x) for x in rest.split("\n")]
    values = [v for v in values if v]
    out: dict[str, str] = {}
    for i, k in enumerate(keys):
        if i < len(values):
            out[k] = values[i]
    return out


def recover_function_calls(raw_content: str) -> list[dict[str, Any]]:
    """Parse a raw Nemotron function-call XML string, recovering the malformed
    but recoverable forms. Returns [] if nothing recoverable."""
    if not raw_content:
        return []
    blocks = _fn_blocks(raw_content)
    if not blocks:
        return []
    calls = []
    for tool, body in blocks:
        args: dict[str, str] = {}
        # Clean canonical forms take priority (exact, trustworthy).
        args.update(_recover_clean(body))
        # Then recover the corrupted variants (only fill missing keys).
        for strat in (
            _recover_schema_echo,
            _recover_param_as_content,
            _recover_named_tag,
            _recover_key_value_list,
        ):
            for k, v in strat(body).items():
                if k not in args and v:
                    args[k] = v
        if args:
            calls.append({"function": {"name": tool, "arguments": args}})
    return calls
