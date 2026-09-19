"""tests/test_map_trajectories_to_nemotron.py — mapper conversion tests.

Tests the pure conversion logic in map_trajectories_to_nemotron.py (no tokenizer,
no model, no network). The optional tokenizer render is exercised via a mocked
tokenizer to keep tests hermetic.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from map_trajectories_to_nemotron import (
    convert_record,
    collect_tool_schemas,
    TOOL_SCHEMAS,
    _as_args_dict,
)


def _sample_record(**overrides):
    rec = {
        "id": "7",
        "ts": "2026-09-15T17:56:39.963105",
        "task": "write a hello world script to /tmp/hello.py",
        "provider": "deepseek-ai",
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "call_type": "task_inference",
        "messages": [],
        "artifacts": ["/tmp/hello.py"],
        "critic_score": 0.9,
        "verdict": "PASS",
        "tool_calls": [
            {
                "tool": "write_file",
                "args": {"path": "/tmp/hello.py", "content": "print('hi')\n"},
                "result": "Written to /tmp/hello.py",
            }
        ],
        "final_reply": "Created the script. Run it with python3 /tmp/hello.py",
    }
    rec.update(overrides)
    return rec


def test_convert_record_builds_user_tool_assistant_sequence():
    out = convert_record(_sample_record())
    roles = [m["role"] for m in out["messages"]]
    # user -> assistant(tool_calls) -> tool -> assistant(final)
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert out["messages"][0]["content"] == "write a hello world script to /tmp/hello.py"
    # assistant tool call shape (HF/Nemotron-consumable)
    tc = out["messages"][1]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "write_file"
    assert tc["function"]["arguments"] == {"path": "/tmp/hello.py", "content": "print('hi')\n"}
    # tool response
    assert out["messages"][2]["role"] == "tool"
    assert out["messages"][2]["name"] == "write_file"
    assert out["messages"][2]["content"] == "Written to /tmp/hello.py"
    # final reply preserved
    assert out["messages"][3]["content"].startswith("Created the script")


def test_convert_record_multiple_tool_calls_in_order():
    rec = _sample_record()
    rec["tool_calls"] = [
        {"tool": "exec_shell", "args": {"command": "ls"}, "result": "a.txt"},
        {"tool": "read_file", "args": {"path": "a.txt"}, "result": "contents"},
    ]
    out = convert_record(rec)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "exec_shell"
    assert out["messages"][3]["tool_calls"][0]["function"]["name"] == "read_file"


def test_convert_record_drops_empty_tool_names_and_messages():
    rec = _sample_record()
    rec["tool_calls"] = [
        {"tool": "", "args": {}, "result": "x"},
        {"args": {}, "result": "y"},  # no name at all
    ]
    out = convert_record(rec)
    # both malformed tool_calls dropped -> only user + final assistant
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant"]


def test_convert_record_skips_missing_final_reply():
    rec = _sample_record()
    rec["final_reply"] = ""
    out = convert_record(rec)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool"]


def test_convert_record_string_arguments_normalised():
    rec = _sample_record()
    rec["tool_calls"] = [
        {"tool": "write_file", "args": '{"path": "/x", "content": "y"}', "result": "ok"}
    ]
    out = convert_record(rec)
    args = out["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args == {"path": "/x", "content": "y"}


def test_collect_tool_schemas_dedupes_and_maps_known():
    records = [
        _sample_record(),
        _sample_record(id="8", tool_calls=[{"tool": "write_file", "args": {}, "result": "x"}]),
    ]
    schemas = collect_tool_schemas(records)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["write_file"]
    assert schemas[0] == TOOL_SCHEMAS["write_file"]


def test_collect_tool_schemas_covers_multiple_names():
    records = [
        _sample_record(
            tool_calls=[
                {"tool": "exec_shell", "args": {"command": "ls"}, "result": "a"},
                {"tool": "read_file", "args": {"path": "a"}, "result": "b"},
            ]
        )
    ]
    schemas = collect_tool_schemas(records)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["exec_shell", "read_file"]


def test_collect_tool_schemas_synthesises_unknown():
    records = [
        _sample_record(
            tool_calls=[{"tool": "custom_tool_x", "args": {"url": "http://x"}, "result": "ok"}]
        )
    ]
    schemas = collect_tool_schemas(records)
    fn = schemas[0]["function"]
    assert fn["name"] == "custom_tool_x"
    assert "url" in fn["parameters"]["properties"]


def test_as_args_dict_handles_dict_string_and_garbage():
    assert _as_args_dict({"a": 1}) == {"a": 1}
    assert _as_args_dict('{"a": 1}') == {"a": 1}
    out = _as_args_dict("not-json")
    assert "_raw" in out
    assert _as_args_dict(None) == {}
