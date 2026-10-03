"""
test_tool_arg_parse_sweep.py — Arg-preservation sweep across every registered tool.

Regression test for the empty-args bug: the Nemotron `function=` fallback parser
only matched the alt form `<parameter=key>value</parameter>`, so any call using the
native attribute form `<parameter name="key">value</parameter>` silently dropped
every argument (write_file/read_file/web_search/etc. all came back `{}` despite
the model emitting correct args).

This test drives the REAL parser (`_parse_function_eq_xml` in model_server.py)
— not a reimplementation — with representative emissions for every tool schema,
in BOTH forms, and asserts every schema-required key survives intact.

Rules (per AGENTS.md):
  - Unit test, no model loaded, no real HTTP, no Telegram.
  - Uses the extracted pure parser; no inference, no server socket.
"""

import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from core.inference.model_server import _parse_function_eq_xml
from core.tools import TOOLS


# ── Build the tool map: name -> (required_keys, sample_values) ──────────────
def _tool_map():
    m = {}
    for t in TOOLS:
        fn = t.get("function", {})
        name = fn.get("name")
        params = fn.get("parameters") or {}
        required = params.get("required", []) or []
        props = params.get("properties", {}) or {}
        # Representative sample value per key (type-safely).
        samples = {}
        for k in required:
            p = props.get(k, {})
            ptype = p.get("type", "string")
            if ptype == "integer":
                samples[k] = "42"
            elif ptype == "number":
                samples[k] = "3.14"
            elif ptype == "boolean":
                samples[k] = "true"
            else:
                samples[k] = f"sample-{k}"
        # ask_questions.options is an array — give a JSON snippet.
        if name == "ask_questions" and "options" in samples:
            samples["options"] = '["opt-a", "opt-b"]'
        m[name] = (required, samples)
    return m


TOOLS_BY_NAME = _tool_map()
ALL_TOOL_NAMES = sorted(TOOLS_BY_NAME.keys())


def _emit_alt_form(name, samples):
    """<function=name> <parameter=key>value</parameter> ... </function>"""
    body = "\n".join(f'<parameter={k}>{v}</parameter>' for k, v in samples.items())
    return f"<function={name}>  \n{body}  \n</function>"


def _emit_native_form(name, samples):
    """<function=name> <parameter name="key">value</parameter> ... </function>"""
    body = "\n".join(f'<parameter name="{k}">{v}</parameter>' for k, v in samples.items())
    return f"<function={name}>  \n{body}  \n</function>"


def _parse_and_get_args(raw, expected_name):
    calls = _parse_function_eq_xml(raw)
    assert calls, f"parser returned no calls for {expected_name}"
    # find the matching call (allow normalization reordering of multiple)
    for c in calls:
        if c["function"]["name"] == expected_name:
            return c["function"]["arguments"]
    raise AssertionError(f"no call found for expected tool {expected_name}; got {calls}")


# ── Sweep: every tool, both emission forms, every required key survives ─────
@pytest.mark.parametrize("form", ["alt", "native"], scope="module")
@pytest.mark.parametrize("name", ALL_TOOL_NAMES, scope="module")
def test_required_keys_survive_parsing(name, form):
    required, samples = TOOLS_BY_NAME[name]
    emit = _emit_alt_form if form == "alt" else _emit_native_form
    raw = emit(name, samples)
    args = _parse_and_get_args(raw, name)
    for k in required:
        assert k in args, (
            f"[{form} form] tool '{name}' lost required key '{k}': "
            f"parsed args={args!r} from raw={raw!r}"
        )
        assert str(args[k]) == str(samples[k]), (
            f"[{form} form] tool '{name}' key '{k}' wrong value: "
            f"expected {samples[k]!r}, got {args[k]!r}"
        )


# ── No-required-args tools still resolve to a call with the right name ───────
@pytest.mark.parametrize("name", ALL_TOOL_NAMES)
def test_tool_resolves_to_named_call(name):
    required, samples = TOOLS_BY_NAME[name]
    if required:
        return  # covered by the sweep above
    args = _parse_and_get_args(_emit_native_form(name, samples), name)
    assert isinstance(args, dict)


# ── The exact regression that triggered this bug ─────────────────────────────
def test_write_file_native_form_preserves_path_and_content():
    raw = (
        '<function=write_file>  \n'
        '<parameter name="path">/tmp/eva_raw_t1.txt</parameter>  \n'
        '<parameter name="content">hello raw probe</parameter>  \n'
        '</function>'
    )
    args = _parse_and_get_args(raw, "write_file")
    assert args.get("path") == "/tmp/eva_raw_t1.txt"
    assert args.get("content") == "hello raw probe"


def test_read_file_native_form_preserves_path():
    raw = '<function=read_file><parameter name="path">/tmp/x.txt</parameter></function>'
    args = _parse_and_get_args(raw, "read_file")
    assert args.get("path") == "/tmp/x.txt"


def test_exec_shell_alt_form_still_works():
    raw = '<function=exec_shell><parameter=command>ls -la</parameter></function>'
    args = _parse_and_get_args(raw, "exec_shell")
    assert args.get("command") == "ls -la"
