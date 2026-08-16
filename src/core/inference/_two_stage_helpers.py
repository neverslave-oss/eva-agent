"""
_two_stage_helpers.py - Two-stage tool-calling pipeline helpers.

Stage 1: Qwen3.5-0.8B (tool_calling slot) - microplans + tool loop, non-thinking mode
         Template injects tool definitions via `tools=` param. Model outputs
         <tool_call><function=name>...</function></tool_call> XML natively.
Stage 2: Nemotron-Labs-Diffusion-3B (primary slot) - conversation synthesis, no tools

Qwen3.5 native tool format (from chat template):
  <tool_call>
  <function=name>
  <parameter=key>
  value
  </parameter>
  </function>
  </tool_call>
"""
import json
import re


def parse_qwen_tool_calls(text: str) -> list:
    """Parse tool calls from Qwen3.5 output.

    Qwen3.5 emits <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    XML blocks natively (from chat template). Pattern 1 handles JSON format for
    backward compat, Pattern 2 handles the native <function=name> XML format.
    Strips  thinking... response blocks if present.
    Returns list of function call dicts.
    """
    calls = []
    raw = text
    raw = re.sub(r' thinking.*? response', '', raw, flags=re.DOTALL).strip()

    # Pattern 1: tool_call XML with JSON
    tc_re = r'<tool_call>(.*?)</tool_call>'
    for m in re.finditer(tc_re, raw, re.DOTALL):
        block = m.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and 'name' in obj:
                calls.append({
                    'function': {
                        'name': obj['name'],
                        'arguments': obj.get('arguments', {}),
                    }
                })
        except (json.JSONDecodeError, TypeError):
            pass

    # Pattern 2: <function=name> XML (Qwen3.5 native format from chat template)
    if not calls:
        fn_re = r'<function=([A-Za-z_][\w-]*)(.*?)(?:</function>|(?=<function=)|$)'
        fn_blocks = re.findall(fn_re, raw, re.DOTALL)
        for tool_name, fn_body in fn_blocks:
            param_re = r'<parameter=([A-Za-z_][\w-]*)(.*?)</parameter>'
            params = re.findall(param_re, fn_body, re.DOTALL)
            args = dict((k, v.strip()) for k, v in params)
            if tool_name.strip():
                if args:
                    calls.append({
                        'function': {'name': tool_name.strip(), 'arguments': args}
                    })
                else:
                    calls.append({
                        'function': {'name': tool_name.strip(), 'arguments': {}}
                    })

    return calls


def build_nemotron_synthesis_prompt(original_query, qwen_answer, tool_results, max_history_turns: int = 6):
    """Build Nemotron synthesis prompt - no tools, no function_calls.
    
    Note: qwen_answer is deliberately NOT included in the prompt when tool_results
    exist, because Qwen is a tool-calling microplanner and its final text output
    is often unreliable (e.g. 'these tools are not available'). Nemotron should
    synthesize purely from the actual tool results.

    max_history_turns: reserved for callers that pass history into this prompt;
    not used directly here (history injection happens in model_server.py), but
    accepted so the signature is consistent with the ADR-022 spec.
    """
    results_text = ""
    for r in tool_results:
        args_str = json.dumps(r.get("args", {}))[:200] if r.get("args") else ""
        result_preview = str(r.get("result", ""))[:2000]
        results_text += "\n### " + r["name"] + "(" + args_str + ")\n" + result_preview + "\n"

    if tool_results:
        results_str = ""
        for r in tool_results:
            args_str = json.dumps(r.get("args", {}))[:200] if r.get("args") else ""
            result_preview = str(r.get("result", ""))[:2000]
            results_str += "\n### " + r["name"] + "(" + args_str + ")\n" + result_preview + "\n"

        return (
            'The user asked: "' + original_query + '"\n\n'
            "Here are the results from the tools that were executed:\n" + results_str + "\n"
            "Now answer the user's question based on these results. Be conversational and informative."
        )
    return qwen_answer or "I could not complete that request."


def build_qwen_system_prompt():
    """Build system message for Qwen3.5 microplanner.

    Template with `tools=` injects tool definitions + format instructions natively.
    We keep only a minimal, English-only role prompt — no format overrides,
    since the chat template already handles the tool-call format.
    """
    return {
        "role": "system",
        "content": "You are a helpful tool-calling assistant. Call the right tool to answer the user.",
    }
