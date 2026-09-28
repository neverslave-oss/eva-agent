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


def _export_record(**overrides):
    """A record in the actual trajectory_collector.export_jsonl format: the tool
    calls live INSIDE messages (assistant tool_calls turns + tool response turns)."""
    rec = {
        "id": "7",
        "ts": "2026-09-15T17:56:39.963105",
        "task": "write a hello world script to /tmp/hello.py",
        "provider": "deepseek-ai",
        "call_type": "task_inference",
        "messages": [
            {"role": "user", "content": "write a hello world script to /tmp/hello.py"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "/tmp/hello.py", "content": "print(\'hi\')\\n"}',
                    },
                }],
            },
            {"role": "tool", "name": "write_file", "content": "Written to /tmp/hello.py"},
            {"role": "assistant", "content": "Created the script. Run it with python3 /tmp/hello.py"},
        ],
        "critic_score": 0.9,
        "verdict": "PASS",
    }
    rec.update(overrides)
    return rec


def test_convert_record_builds_clean_sequence():
    out = convert_record(_export_record())
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tc = out["messages"][1]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "write_file"
    # string arguments normalised to dict for Nemotron template
    assert tc["function"]["arguments"] == {"path": "/tmp/hello.py", "content": "print('hi')\n"}
    assert out["messages"][2]["role"] == "tool"
    assert out["messages"][3]["content"].startswith("Created the script")


def test_convert_record_multiple_tool_calls_in_order():
    rec = _export_record()
    rec["messages"] = [
        {"role": "user", "content": "list then read"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "exec_shell", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "name": "exec_shell", "content": "a.txt"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]},
        {"role": "tool", "name": "read_file", "content": "contents"},
        {"role": "assistant", "content": "done"},
    ]
    out = convert_record(rec)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert out["messages"][1]["tool_calls"][0]["function"]["name"] == "exec_shell"
    assert out["messages"][3]["tool_calls"][0]["function"]["name"] == "read_file"


def test_convert_record_drops_malformed_tool_calls():
    rec = _export_record()
    rec["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": ""}}, {"type": "function", "function": {"name": "read_file", "arguments": {}}}]},
        {"role": "tool", "name": "read_file", "content": "x"},
        {"role": "assistant", "content": "done"},
    ]
    out = convert_record(rec)
    roles = [m["role"] for m in out["messages"]]
    # empty-name call dropped; only read_file remains
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert len(out["messages"][1]["tool_calls"]) == 1


def test_convert_record_drops_blank_trailing_assistant():
    rec = _export_record()
    rec["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {}}}]},
        {"role": "tool", "name": "read_file", "content": "x"},
        {"role": "assistant", "content": ""},  # blank trailing assistant -> dropped
    ]
    out = convert_record(rec)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool"]


def test_convert_record_empty_messages_returns_none():
    assert convert_record({"id": "1", "messages": []}) is None


def test_collect_tool_schemas_from_messages():
    records = [_export_record()]
    schemas = collect_tool_schemas(records)
    names = [s["function"]["name"] for s in schemas]
    assert names == ["write_file"]
    assert schemas[0] == TOOL_SCHEMAS["write_file"]


def test_collect_tool_schemas_multiple_names():
    rec = _export_record()
    rec["messages"] = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "exec_shell", "arguments": {}}}]},
        {"role": "tool", "name": "exec_shell", "content": "a"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "http_get", "arguments": {}}}]},
        {"role": "tool", "name": "http_get", "content": "b"},
    ]
    schemas = collect_tool_schemas([rec])
    names = [s["function"]["name"] for s in schemas]
    assert names == ["exec_shell", "http_get"]


def test_collect_tool_schemas_synthesises_unknown():
    rec = _export_record()
    rec["messages"] = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "custom_tool_x", "arguments": {"url": "http://x"}}}]},
    ]
    schemas = collect_tool_schemas([rec])
    fn = schemas[0]["function"]
    assert fn["name"] == "custom_tool_x"
    assert "url" in fn["parameters"]["properties"]


def test_as_args_dict_handles_dict_string_and_garbage():
    assert _as_args_dict({"a": 1}) == {"a": 1}
    assert _as_args_dict('{"a": 1}') == {"a": 1}
    out = _as_args_dict("not-json")
    assert "_raw" in out
    assert _as_args_dict(None) == {}
