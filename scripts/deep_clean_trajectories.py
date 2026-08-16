#!/usr/bin/env python3
"""
deep_clean_trajectories.py
===========================
Final consistency pass for SFT readiness:

  1. Drop error records (max steps reached, inference errors)
  2. Synthesise missing final assistant confirmation from tool result
  3. Normalise tool_call format to OpenAI standard {type, id, function}
  4. Ensure every tool_call has a matching tool result message
  5. Truncate very long system prompts to a consistent max
  6. Verify path normalisation ($HOME, no pacificDev)
  7. Emit a clean JSONL in HuggingFace SFT chat format

Output format per record:
  {id, task, provider, model, messages, artifacts, critic_score, verdict}
  messages: [{role, content}] or [{role, tool_calls}] or [{role, tool_call_id, content}]
"""
import argparse, json, re, uuid
from pathlib import Path

# ── Error markers ────────────────────────────────────────────────────────────
_ERROR_MARKERS = [
    "(max steps reached)", "[model_server error]", "[model_client error]",
    "CUDA error", "out of memory", "(inference unavailable)", "(timed out",
    "(error: web_search", "(error: exec_shell",
]

def _is_error_reply(text: str) -> bool:
    return any(m in text for m in _ERROR_MARKERS)

# ── Tool call format normalisation ───────────────────────────────────────────
def _normalise_tool_call(tc: dict) -> dict:
    """Ensure {type, id, function: {name, arguments}} format."""
    if "function" in tc:
        fn = tc["function"]
        args = fn.get("arguments", {})
        if isinstance(args, dict):
            args = json.dumps(args)
        return {
            "type": "function",
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "function": {"name": fn.get("name",""), "arguments": args}
        }
    # Flat format: {name, args/arguments}
    args = tc.get("args") or tc.get("arguments") or {}
    if isinstance(args, dict):
        args = json.dumps(args)
    return {
        "type": "function",
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "function": {"name": tc.get("name",""), "arguments": args}
    }

# ── Path normalisation ────────────────────────────────────────────────────────
_PATH_PATS = [
    (re.compile(r'/home/\w+/\.kernel-evolving/workspace'), '$KERNEL_WORKSPACE'),
    (re.compile(r'/home/\w+/'), '$HOME/'),
    (re.compile(r'/home/\w+'), '$HOME'),
    (re.compile(r'\bpacificDev\b'), '<user>'),
    (re.compile(r'\bMSI\b'), '<hostname>'),
]
def _norm(text: str) -> str:
    for pat, repl in _PATH_PATS:
        text = pat.sub(repl, text)
    return text

def _norm_val(v):
    if isinstance(v, str): return _norm(v)
    if isinstance(v, list): return [_norm_val(i) for i in v]
    if isinstance(v, dict): return {k: _norm_val(val) for k, val in v.items()}
    return v

# ── Synthesise missing final reply ───────────────────────────────────────────
def _synthesise_reply(msgs: list) -> str | None:
    """If last non-empty message is a tool result, infer the confirmation reply."""
    last_tool = None
    for m in reversed(msgs):
        if m["role"] == "tool":
            last_tool = m
            break
    if not last_tool:
        return None
    content = str(last_tool.get("content",""))
    # Extract path if written
    path_match = re.search(r'Written to ([\S]+)', content)
    if path_match:
        return f"Done — written to {path_match.group(1)}."
    if "Written" in content:
        return "Done — file written."
    if content.startswith("{") or content.startswith("["):
        return "Done."
    return f"Done — {content[:60].rstrip('.')}."

# ── Message normaliser ────────────────────────────────────────────────────────
def _clean_messages(msgs: list, max_sys_chars: int = 3000) -> list | None:
    """
    Returns cleaned messages list or None if record should be dropped.
    """
    result = []
    sys_seen = False
    
    for i, m in enumerate(msgs):
        role = m.get("role","")
        content = m.get("content","")
        tool_calls = m.get("tool_calls")
        
        # System prompt
        if role == "system":
            if sys_seen:
                continue  # drop duplicate system prompts
            sys_seen = True
            content = _norm(content or "")
            if len(content) > max_sys_chars:
                content = content[:max_sys_chars] + "\n...[truncated for training]"
            result.append({"role": "system", "content": content})
            continue
        
        # User
        if role == "user":
            result.append({"role": "user", "content": _norm(str(content or ""))})
            continue
        
        # Assistant
        if role == "assistant":
            msg = {"role": "assistant"}
            if tool_calls:
                norm_tcs = [_normalise_tool_call(tc) for tc in tool_calls]
                msg["tool_calls"] = norm_tcs
                if content:
                    msg["content"] = _norm(str(content))
            else:
                c = _norm(str(content or ""))
                if _is_error_reply(c):
                    return None  # drop the whole record
                msg["content"] = c
            result.append(msg)
            continue
        
        # Tool result
        if role == "tool":
            msg = {"role": "tool", "content": _norm(str(content or ""))}
            tool_call_id = m.get("tool_call_id","")
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            # Backfill tool_call_id from preceding assistant if missing
            elif result and result[-1]["role"] == "assistant" and result[-1].get("tool_calls"):
                tcs = result[-1]["tool_calls"]
                if tcs:
                    msg["tool_call_id"] = tcs[-1].get("id","")
            result.append(msg)
            continue
    
    if not result:
        return None
    
    # Ensure final message is a non-empty assistant reply
    last = result[-1]
    if last["role"] != "assistant" or not last.get("content"):
        reply = _synthesise_reply(result)
        if reply:
            result.append({"role": "assistant", "content": reply})
        else:
            # Check if last assistant message just has tool_calls but no content
            last_asst = next((m for m in reversed(result) if m["role"]=="assistant"), None)
            if last_asst and last_asst.get("tool_calls") and not last_asst.get("content"):
                result.append({"role": "assistant", "content": "Done."})
    
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = [json.loads(l) for l in f if l.strip()]
    
    print(f"Input: {len(raw)} records")
    
    kept, dropped_error, dropped_format, fixed_reply = [], 0, 0, 0
    
    for r in raw:
        msgs = r.get("messages", [])
        
        # Check if final content is an error before full clean
        for m in msgs:
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                if _is_error_reply(str(m.get("content",""))):
                    dropped_error += 1
                    break
        else:
            clean_msgs = _clean_messages(msgs)
            if clean_msgs is None:
                dropped_format += 1
                continue
            
            # Check if we synthesised a reply
            orig_last = next((m for m in reversed(msgs) if m.get("role")=="assistant" and m.get("content")), None)
            new_last = next((m for m in reversed(clean_msgs) if m.get("role")=="assistant" and m.get("content")), None)
            if (not orig_last or not orig_last.get("content")) and new_last and new_last.get("content"):
                fixed_reply += 1
            
            r2 = dict(r)
            r2["messages"] = clean_msgs
            r2["model"] = "<teacher>"  # anonymise
            r2.pop("ts", None)
            # Final path normalise on task/artifacts
            r2["task"] = _norm(r.get("task",""))
            r2["artifacts"] = _norm_val(r.get("artifacts",[]))
            kept.append(r2)
    
    with open(args.output, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nResults:")
    print(f"  Dropped (error reply):   {dropped_error}")
    print(f"  Dropped (format issue):  {dropped_format}")
    print(f"  Fixed (reply added):     {fixed_reply}")
    print(f"  Final clean records:     {len(kept)}")
    print(f"\nOutput: {args.output}")
    
    if args.stats:
        import collections, statistics
        scores = [r.get("critic_score",0) for r in kept]
        depths = collections.Counter(
            sum(1 for m in r.get("messages",[]) for tc in m.get("tool_calls",[]))
            for r in kept
        )
        print(f"\nScore: mean={statistics.mean(scores):.3f} min={min(scores):.2f}")
        print(f"Depth: {dict(sorted(depths.items()))}")
        
        # Verify zero remaining issues
        path_leaks = sum(1 for r in kept if 'pacificDev' in json.dumps(r) or re.search(r'/home/\w+/', json.dumps(r)))
        error_replies = sum(1 for r in kept for m in r.get("messages",[])
                          if m.get("role")=="assistant" and _is_error_reply(str(m.get("content",""))))
        no_final = sum(1 for r in kept
                      if not any(m.get("role")=="assistant" and m.get("content")
                                for m in r.get("messages",[])))
        print(f"\nVerification:")
        print(f"  Path leaks:    {path_leaks}")
        print(f"  Error replies: {error_replies}")
        print(f"  No final reply:{no_final}")

if __name__ == "__main__":
    main()
