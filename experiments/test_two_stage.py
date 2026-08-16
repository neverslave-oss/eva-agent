#!/usr/bin/env python3
"""
Two-stage architecture test:
  Stage 1: Qwen3-0.6B (microplanner) -> generates tool calls in native format
  Stage 2: Nemotron-Labs-Diffusion-3B (orchestrator) -> final answer

Models available:
  - Qwen3-0.6B: 0.6B params, ~1.5GB VRAM, native Qwen tool-calling format
  - Nemotron-Labs-Diffusion-3B: running on kernel-evolving port 8779

Run from kernel-evolving root:
  python3 experiments/test_two_stage.py
"""

import json, os, re, subprocess, sys, time
import requests, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -- Config --
QWEN_MODEL = "Qwen/Qwen3-0.6B"
NEM_URL = "http://localhost:8779/message"

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "exec_shell", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def exec_tool(name, args):
    try:
        if name == "read_file":
            p = args.get("path", "")
            if not os.path.exists(p): return f"Error: not found: {p}"
            return open(p, errors="replace").read(4096)
        elif name == "list_dir":
            p = args.get("path", "")
            if not os.path.isdir(p): return f"Error: not a dir: {p}"
            return "\n".join(os.listdir(p)[:50])
        elif name == "exec_shell":
            c = args.get("command", "")
            r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=15)
            return (r.stdout + r.stderr)[:4096] or "(no output)"
        return f"Error: unknown tool '{name}'"
    except Exception as e:
        return f"Error: {e}"


def parse_tool_calls(text):
    """Parse Qwen native tool call format. Handles both XML and JSON variants."""
    calls = []

    # Pattern 1: <function=name><parameter=key>value</parameter></function>
    # Pattern 2: {"name": "...", "arguments": {...}} inside <tool_call>
    tc_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for match in tc_re.finditer(text):
        block = match.group(1).strip()

        # Try JSON format first (Qwen3-0.6B uses this)
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "name" in obj:
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
                continue
        except (json.JSONDecodeError, TypeError):
            pass

        # Try XML format
        fn_re = r"<function=(\w+)>"
        param_re = r"<parameter=(\w+)>\s*(.*?)\s*</parameter>"
        fn = re.search(fn_re, block)
        if fn:
            params = {}
            for pm in re.finditer(param_re, block, re.DOTALL):
                params[pm.group(1)] = pm.group(2).strip()
            calls.append({"name": fn.group(1), "arguments": params})

    return calls


def stage1_qwen(user_msg, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a microplanner. Given the user's request, break it into concrete tool calls. Use the available tools step by step. Keep your plan minimal -- only call tools that are actually needed."},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True,
        enable_thinking=True, tokenize=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.3,
            top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )

    dt = time.time() - t0
    new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=False)

    thinking = ""
    if " thinking" in generated:
        parts = generated.split(" thinking", 1)
        if len(parts) > 1:
            rest = parts[1]
            if " response" in rest:
                thinking = rest.split(" response", 1)[0].strip()

    tool_calls = parse_tool_calls(generated)

    return {
        "tool_calls": tool_calls,
        "thinking": thinking,
        "raw": generated,
        "elapsed": dt,
        "tokens": int(new_ids.shape[0]),
    }


def stage2_nemotron(user_msg, thinking, tool_results):
    results_text = ""
    for r in tool_results:
        results_text += (
            "\n### Tool: " + r["name"] + "(" + json.dumps(r["args"]) + ")\n"
            "```\n" + r["result"][:2000] + "\n```\n"
        )

    prompt = (
        "User asked: " + user_msg + "\n\n"
        "Microplanner reasoning: " + thinking + "\n\n"
        "Tool execution results:" + results_text + "\n"
        "Synthesize a clear, concise answer for the user based on these results."
    )

    t0 = time.time()
    try:
        resp = requests.post(NEM_URL, json={"message": prompt}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", data.get("response", str(data)))
        return reply, time.time() - t0
    except Exception as e:
        return f"[Nemotron error: {e}]", time.time() - t0


def main():
    print("=" * 60)
    print("Two-Stage Architecture Test")
    print("  Stage 1: Qwen3-0.6B (microplanner)")
    print("  Stage 2: Nemotron-Labs-Diffusion-3B (orchestrator)")
    print("=" * 60)

    # Check Nemotron
    print("\n[Nemotron] Checking health...")
    try:
        r = requests.get("http://localhost:8779/health", timeout=5)
        h = r.json()
        print(f"  Status: {h.get('status')}")
        print(f"  VRAM free: {h.get('vram_free_mb')}MB")
    except Exception as e:
        print(f"  WARN: {e}")

    # Load Qwen
    print(f"\n[Stage 1] Loading {QWEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    print(f"  Vocab: {tokenizer.vocab_size}, Tool support: YES")

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    print(f"  Device: {model.device}, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    tests = [
        "What files are in /tmp? Read any .txt files you find.",
        "Check disk usage with df -h and list top 5 largest dirs in ~.",
    ]

    for i, prompt in enumerate(tests, 1):
        print(f"\n{'='*60}\nTEST {i}: {prompt}\n{'='*60}")

        # Stage 1
        print("\n--- Stage 1: Qwen microplanner ---")
        plan = stage1_qwen(prompt, tokenizer, model)
        print(f"  Time: {plan['elapsed']:.1f}s")
        print(f"  Tokens: {plan['tokens']}")
        print(f"  Thinking: {plan['thinking'][:300]}")
        print(f"  Tool calls: {len(plan['tool_calls'])}")
        for tc in plan['tool_calls']:
            print(f"    -> {tc['name']}({tc['arguments']})")
        print(f"  Raw output:\n    {plan['raw'][:600]}")

        if not plan['tool_calls']:
            print("\n  [WARN] No tool calls! Falling back...")
            ans, dt = stage2_nemotron(prompt, plan['thinking'], [])
            print(f"\n--- Stage 2: Nemotron ({dt:.1f}s) ---\n  {ans[:600]}")
            continue

        # Execute
        print("\n--- Tool Execution ---")
        results = []
        for tc in plan['tool_calls']:
            print(f"\n  [{tc['name']}({tc['arguments']})]")
            res = exec_tool(tc['name'], tc['arguments'])
            print(f"  Result ({len(res)} chars):")
            for line in res.split("\n")[:3]:
                print(f"    {line[:120]}")
            results.append({"name": tc['name'], "args": tc['arguments'], "result": res})

        # Stage 2
        print("\n--- Stage 2: Nemotron orchestrator ---")
        ans, dt = stage2_nemotron(prompt, plan['thinking'], results)
        print(f"\n  Time: {dt:.1f}s")
        print(f"\n{'='*40}\nFINAL ANSWER:\n{'='*40}\n{ans}\n{'='*40}")

    print(f"\n{'='*60}\nDone!\n{'='*60}")

if __name__ == "__main__":
    main()
