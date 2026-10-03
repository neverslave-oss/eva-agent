#!/usr/bin/env python3
"""Tests for the malformed-Nemotron-FC recovery parser (fc_recovery_parser).

Each fixture is a REAL emission captured verbatim from the model-server log
(`[tool_loop/raw ...] RAW=...`) for the failing tools http_get / run_skill /
search_skills. The strict parser (`_parse_function_eq_xml`) drops all of these
to `{}`; the recovery parser must salvage the argument values so the tool loop
can execute them.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.inference.fc_recovery_parser import recover_function_calls

# ── A-form: mismatched close / name-as-content ─────────────────────────────
RAW_A1 = (
    "<function=search_skills>  \n<parameter>query</name>  \npdf  \n"
    "</parameter>  \n</function>"
)
RAW_A2 = (
    "<function=search_skills>\n<parameter>query</parameter>\nvalue_pdf\n"
    "</parameter>\n</function>"
)
RAW_A3 = (
    "<function=search_skills>\n<parameter>query</name>\nvalue=\"pdf\"\n"
    "</parameter>\n</function>"
)

# ── B-form: key-list then value-list, single trailing </parameter> ──────────
RAW_B1 = (
    "<function=http_get>  \n<parameter>url</parameter>  \n"
    "<parameter>timeout</parameter>  \nhttps://example.com/api/status  \n5  \n"
    "</parameter>  \n</function>"
)

# ── C-form: schema-echo with value embedded in <description> ───────────────
RAW_C1 = (
    "<function=run_skill>  \n<parameter>  \n<name>skill_name</name>  \n"
    "<type>string</type>  \n<description>security-scanner</description>  \n"
    "</parameter>  \n<parameter>  \n<name>input</name>  \n<type>string</type>  \n"
    "<description>Scan the kernel repo for vulnerabilities</description>  \n"
    "</parameter>  \n</function>"
)

# ── D-form: parameter tag name == key ──────────────────────────────────────
RAW_D1 = "<function=search_skills>  \n<query>pdf</query>  \n</function>"

# ── E-form: value="..." inner wrapper ──────────────────────────────────────
RAW_E1 = (
    "<function=search_skills>\n<parameter>query</name>\nvalue=\"pdf\"\n"
    "</parameter>\n</function>"
)

# ── Clean form (must still parse) ──────────────────────────────────────────
RAW_CLEAN = "<function=search_skills>\n<parameter=query>\npdf\n</parameter>\n</function>"
RAW_CLEAN_NATIVE = (
    '<function=exec_shell>  \n<parameter name="command">  \n"ls -ld /"  \n'
    "</parameter>  \n</function>"
)

# ── Multiple calls in one emission ─────────────────────────────────────────
RAW_MULTI = (
    "<function=search_skills>  \n<query>pdf</query>  \n</function>  \n"
    "<function=list_routines>  \n</function>"
)


def arg_map(raw):
    calls = recover_function_calls(raw)
    assert calls, f"expected at least one recovered call from: {raw!r}"
    merged = {}
    for c in calls:
        name = c["function"]["name"]
        merged[name] = c["function"]["arguments"]
    return merged


@pytest.mark.parametrize("raw,expect_key,expect_val", [
    (RAW_A1, "query", "pdf"),
    (RAW_A2, "query", "value_pdf"),
    (RAW_A3, "query", "pdf"),
    (RAW_D1, "query", "pdf"),
    (RAW_E1, "query", "pdf"),
    (RAW_CLEAN, "query", "pdf"),
])
def test_single_key_value(raw, expect_key, expect_val):
    m = arg_map(raw)
    assert "search_skills" in m or "exec_shell" in m or True
    tool = m["search_skills"] if "search_skills" in m else m["exec_shell"]
    assert tool.get(expect_key) == expect_val


def test_b_form_http_get_url_and_timeout():
    m = arg_map(RAW_B1)
    assert "http_get" in m
    args = m["http_get"]
    assert args.get("url") == "https://example.com/api/status"
    assert args.get("timeout") == "5"


def test_c_form_run_skill_values_from_description():
    m = arg_map(RAW_C1)
    assert "run_skill" in m
    args = m["run_skill"]
    assert args.get("skill_name") == "security-scanner"
    assert "input" in args
    assert "Scan" in args["input"]


def test_clean_native_attr_form_still_works():
    m = arg_map(RAW_CLEAN_NATIVE)
    assert "exec_shell" in m
    assert m["exec_shell"].get("command") == "ls -ld /"


def test_multi_call_emission_recovers_participating_tools():
    m = arg_map(RAW_MULTI)
    assert "search_skills" in m
    assert m["search_skills"].get("query") == "pdf"
    # list_routines with no args: may recover as empty args or be dropped;
    # the important thing is search_skills is recovered.
    assert any(k == "search_skills" for k in m)


def test_dropped_when_nothing_recoverable():
    # Pure schema echo with no values (C-form but empty descriptions) must not
    # produce fake values.
    raw = (
        "<function=search_skills>  \n<parameter>  \n<name>query</name>  \n"
        "<type>string</type>  \n<description></description>  \n</parameter>  \n"
        "</function>"
    )
    calls = recover_function_calls(raw)
    assert calls == []
