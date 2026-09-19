#!/usr/bin/env python3
"""
map_trajectories_to_nemotron.py
===============================
Map kernel-evolving trajectory exports (TRL-style JSONL from
`trajectory_collector.export_jsonl`) into training data for the **Nemotron**
function-calling adapter, rendered with Nemotron's own chat template so the model
learns its native `<tool_call><function=...>` format.

Export format (from trajectory_collector.export_jsonl) puts the tool calls INSIDE
the `messages` array — assistant turns carry `tool_calls`, followed by `tool`
response turns. There is NO top-level `tool_calls` key on the record. This mapper
therefore derives everything from `messages`, so it works on real exports.

Nemotron chat-template facts (from the model's `chat_template.jinja`):
  - Tools are declared in a "# Tools\n\n<tools>...</tools>" system block, injected
    when `apply_chat_template(..., tools=[...])` is used.
  - Assistant tool calls render as:
      <tool_call>
      <function=NAME>
      <parameter=KEY>
      value
      </parameter>
      </function>
      </tool_call>
  - Tool results render as a `user` turn wrapped in `<tool_response>...</tool_response>`.
  - Generation prompt: `<|im_start|>assistant\n thinking response`

The mapper emits a **standard messages list** and hands it to Nemotron's tokenizer
with `tools=`, so the output `text` is what Nemotron sees at inference.

Usage:
  python3 scripts/map_trajectories_to_nemotron.py \
      --input ~/.kernel-evolving/workspace/artifacts/trajectories/export.jsonl \
      --output ~/.kernel-evolving/workspace/artifacts/trajectories/sft_nemotron_fc.jsonl \
      --model /mnt/e/models/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B
      --min-score 0.7

  # Raw conversion only (no tokenizer / no render) — for tests and inspection:
  python3 scripts/map_trajectories_to_nemotron.py \
      --input export.jsonl --output messages_only.jsonl --no-render
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Known tool schemas (subset of what eva actually calls) ──────────────────
TOOL_SCHEMAS = {
    "exec_shell": {
        "type": "function",
        "function": {
            "name": "exec_shell",
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "offset": {"type": "integer", "description": "Line offset."},
                    "limit": {"type": "integer", "description": "Max lines."},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target path."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "http_get": {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Make an HTTP GET request and return the response body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds."},
                },
                "required": ["url"],
            },
        },
    },
    "run_skill": {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Run a named skill with natural-language input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "input": {"type": "string"},
                },
                "required": ["skill_name"],
            },
        },
    },
}


def _schema_for_unknown(name: str, observed_args) -> dict:
    """Build a function schema from observed argument keys (best-effort)."""
    props = {}
    required = []
    if isinstance(observed_args, dict):
        for key in observed_args:
            if key in ("path", "content", "command", "url", "skill_name", "input"):
                props[key] = {"type": "string"}
                required.append(key)
            else:
                props[key] = {"type": "string"}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool.",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


def _extract_tool_calls(messages: list) -> list:
    """Yield (name, args) pairs from assistant tool_calls turns in a messages list."""
    out = []
    for m in messages or []:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if name:
                    out.append((name, fn.get("arguments", {})))
    return out


def collect_tool_schemas(records: list) -> list:
    """De-duplicated tool list for apply_chat_template(tools=...), from messages."""
    by_name: dict[str, dict] = {}
    for rec in records:
        for name, args in _extract_tool_calls(rec.get("messages", [])):
            if name in TOOL_SCHEMAS:
                by_name[name] = TOOL_SCHEMAS[name]
            elif name not in by_name:
                by_name[name] = _schema_for_unknown(name, args)
    return [by_name[n] for n in sorted(by_name)]


def _as_args_dict(args):
    """Arguments may be a dict or a JSON string — normalise to dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"_raw": args}
        except Exception:
            return {"_raw": args}
    return {}


def convert_record(rec: dict) -> dict | None:
    """
    Convert one export record into a clean messages list. Pure function
    (no tokenizer) — unit-testable.

    - Keeps user/system/tool/assistant turns, normalising assistant tool_calls
      arguments from JSON string to dict (what Nemotron's template expects).
    - Drops malformed tool_calls (no name) and strips empty assistant turns that
      follow a tool call (they carry no learnable content).
    """
    messages = rec.get("messages", [])
    if not messages:
        task = rec.get("task")
        messages = [{"role": "user", "content": task}] if task else []

    clean = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if not name:
                    continue
                tcs.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _as_args_dict(fn.get("arguments", {})),
                    },
                })
            if tcs:
                clean.append({"role": "assistant", "tool_calls": tcs})
        elif role in ("user", "system", "tool", "assistant"):
            clean.append(m)

    # Drop a trailing empty assistant content turn that just says "Done"-style
    # wrappers after a tool result — keeps FC signal crisp.
    if clean and clean[-1].get("role") == "assistant":
        content = (clean[-1].get("content") or "").strip()
        if not content:
            clean.pop()

    if not clean:
        return None

    return {
        "id": rec.get("id"),
        "task": rec.get("task"),
        "messages": clean,
        "critic_score": rec.get("critic_score"),
        "verdict": rec.get("verdict"),
    }


def render_nemotron(rec: dict, tools: list, tokenizer):
    """Apply Nemotron's chat template to a converted record. Returns text or None."""
    try:
        return tokenizer.apply_chat_template(
            rec["messages"],
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"  [skip] render error for id={rec.get('id')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Map trajectories to Nemotron FC training data.")
    parser.add_argument("--input", required=True, help="JSONL export from trajectory_collector.export_jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", default=None, help="Nemotron model path or repo (for tokenizer)")
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument("--no-render", action="store_true", help="Write messages only (skip tokenizer render)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"  [skip] bad JSON line: {e}")
                continue
            if (rec.get("critic_score") or 0) >= args.min_score:
                records.append(rec)
    print(f"Loaded {len(records)} records (score>={args.min_score})")

    converted = [r for r in (convert_record(r) for r in records) if r]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    if args.no_render:
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in converted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(converted)} message-only records to {args.output}")
        return

    if not args.model:
        print("ERROR: --model required when rendering (Nemotron path/repo)", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer
    print(f"Loading Nemotron tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    tools = collect_tool_schemas(records)
    print(f"Using {len(tools)} tool schemas: {[t['function']['name'] for t in tools]}")

    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in converted:
            text = render_nemotron(rec, tools, tokenizer)
            if text is None:
                continue
            row = {
                "text": text,
                "task": rec.get("task"),
                "critic_score": rec.get("critic_score"),
                "id": rec.get("id"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} rendered SFT records to {args.output}")


if __name__ == "__main__":
    main()
