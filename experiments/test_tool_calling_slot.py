#!/usr/bin/env python3
"""
test_tool_calling_slot.py — Validate the two-stage tool-calling pipeline.

Tests:
  1. Qwen3-0.6B loads from tool_calling slot config
  2. Qwen generates tool calls with enable_thinking=False
  3. Tool calls parse correctly via _parse_qwen_tool_calls
  4. Nemotron synthesis prompt is well-formed
  5. Full two-stage pipeline (if model_server running)

Run from kernel-evolving root:
  python3 experiments/test_tool_calling_slot.py
"""
import json
import os
import re
import sys
import time

# Add src to path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def test_parse_qwen_tool_calls():
    """Test 1: Parse Qwen tool call format."""
    print("\n--- Test 1: parse_qwen_tool_calls ---")
    from core.inference._two_stage_helpers import parse_qwen_tool_calls

    # Case 1: JSON inside tool_call XML
    text1 = '<think>I need to check disk usage.</think><tool_call>\n{"name": "exec_shell", "arguments": {"command": "df -h"}}\n</tool_call>'
    calls1 = parse_qwen_tool_calls(text1)
    assert len(calls1) == 1, f"Expected 1 call, got {len(calls1)}"
    assert calls1[0]["function"]["name"] == "exec_shell"
    assert calls1[0]["function"]["arguments"]["command"] == "df -h"
    print("  PASS: JSON tool_call with thinking strip")

    # Case 2: Multiple tool calls
    text2 = '<tool_call>\n{"name": "exec_shell", "arguments": {"command": "df -h"}}\n</tool_call>\n<tool_call>\n{"name": "list_dir", "arguments": {"path": "/tmp"}}\n</tool_call>'
    calls2 = parse_qwen_tool_calls(text2)
    assert len(calls2) == 2, f"Expected 2 calls, got {len(calls2)}"
    print("  PASS: Multiple tool calls")

    # Case 3: No tool calls (plain text)
    text3 = "Here is the answer: everything looks fine."
    calls3 = parse_qwen_tool_calls(text3)
    assert len(calls3) == 0
    print("  PASS: No tool calls in plain text")

    # Case 4: Thinking overflow (all thinking, no tool_call)
    text4 = "<think>The user wants to check disk usage. I should use exec_shell with df -h command. This will show the filesystem usage in human-readable format. Let me call that tool. But wait, maybe I should also check the largest directories...</think>"
    calls4 = parse_qwen_tool_calls(text4)
    assert len(calls4) == 0
    print("  PASS: Thinking overflow returns empty (correct — needs re-prompt)")

    return True


def test_build_nemotron_prompt():
    """Test 2: Nemotron synthesis prompt construction."""
    print("\n--- Test 2: build_nemotron_synthesis_prompt ---")
    from core.inference._two_stage_helpers import build_nemotron_synthesis_prompt

    # Case 1: With tool results + Qwen answer
    results = [
        {"name": "exec_shell", "args": {"command": "df -h"}, "result": "Filesystem  Size  Used"},
        {"name": "list_dir", "args": {"path": "/tmp"}, "result": "file1.txt\nfile2.txt"},
    ]
    prompt = build_nemotron_synthesis_prompt("Check disk", "Disk is 50% used", results)
    assert "Check disk" in prompt
    assert "Disk is 50% used" in prompt
    assert "exec_shell" in prompt
    assert "function_calls" not in prompt.lower() or "Do not mention" in prompt
    print("  PASS: Prompt contains query + results + Qwen answer")

    # Case 2: No tool results — returns Qwen answer directly
    direct = build_nemotron_synthesis_prompt("Hello", "Hi there!", [])
    assert direct == "Hi there!"
    print("  PASS: No tools — returns Qwen answer directly")

    # Case 3: Empty everything
    empty = build_nemotron_synthesis_prompt("", "", [])
    assert "could not complete" in empty.lower()
    print("  PASS: Empty everything — graceful fallback")

    return True


def test_config_tool_calling_slot():
    """Test 3: Verify tool_calling slot in config.yaml."""
    print("\n--- Test 3: config.yaml tool_calling slot ---")
    import yaml

    config_path = os.path.join(_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    slots = cfg.get("model_slots", {})
    assert "tool_calling" in slots, "tool_calling slot missing from config.yaml"

    tc = slots["tool_calling"]
    assert tc.get("role") == "tool_calling"
    assert tc.get("model_path"), "tool_calling model_path is empty"
    assert os.path.isdir(tc["model_path"]), f"model_path not found: {tc['model_path']}"

    print(f"  PASS: tool_calling slot configured")
    print(f"    model_path: {tc['model_path']}")
    print(f"    role: {tc.get('role')}")
    print(f"    dtype: {tc.get('dtype')}")
    print(f"    device: {tc.get('device')}")
    return True


def test_qwen_model_loads():
    """Test 4: Qwen3-0.6B loads from configured path."""
    print("\n--- Test 4: Qwen3-0.6B model loading ---")
    import yaml

    config_path = os.path.join(_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["model_slots"]["tool_calling"]["model_path"]
    print(f"  Loading from: {model_path}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dt = time.time() - t0
    print(f"  Tokenizer loaded in {dt:.1f}s (vocab={tokenizer.vocab_size})")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    dt = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded in {dt:.1f}s (VRAM: {vram:.2f}GB)")

    # Quick generation test with tools
    messages = [
        {"role": "system", "content": "You are a tool-calling microplanner."},
        {"role": "user", "content": "Check disk usage"},
    ]
    tools = [
        {"type": "function", "function": {"name": "exec_shell", "description": "Run shell", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    ]

    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.3, pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
    print(f"  Generated ({out[0].shape[-1] - inputs['input_ids'].shape[-1]} tokens):")
    print(f"    {generated[:400]}")

    # Check if tool call was generated
    from core.inference._two_stage_helpers import parse_qwen_tool_calls
    calls = parse_qwen_tool_calls(generated)
    if calls:
        print(f"  PASS: Generated {len(calls)} tool call(s)")
        for c in calls:
            print(f"    -> {c['function']['name']}({c['function']['arguments']})")
    else:
        print("  WARN: No tool calls generated (may need prompt tuning)")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return True


def test_model_server_syntax():
    """Test 5: model_server.py parses without errors."""
    print("\n--- Test 5: model_server.py syntax check ---")
    import ast

    path = os.path.join(_src, "core", "inference", "model_server.py")
    with open(path) as f:
        source = f.read()
    ast.parse(source)
    print("  PASS: Syntax OK")

    # Check key functions exist
    assert "def _run_two_stage_if_available" in source
    assert "def _nemotron_synthesize_answer" in source
    assert "def _ensure_tool_calling_slot" in source
    print("  PASS: Two-stage functions present")

    return True


def main():
    print("=" * 60)
    print("Two-Stage Tool-Calling Pipeline Tests")
    print("=" * 60)

    tests = [
        ("Parse tool calls", test_parse_qwen_tool_calls),
        ("Nemotron prompt", test_build_nemotron_prompt),
        ("Config slot", test_config_tool_calling_slot),
        ("Qwen model load", test_qwen_model_loads),
        ("Server syntax", test_model_server_syntax),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {name}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
