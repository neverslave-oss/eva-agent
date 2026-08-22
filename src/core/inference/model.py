"""
model.py — Load Gemma 4 E2B-it and expose inference + audio.
One model loaded once. All agents share it.
"""
import os
import torch
import yaml
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from core.tools import WORKSPACE as _DEFAULT_WORKSPACE, execute_tool, execute_tool_with_meta


def _build_failure_reprompt_simple(tool_name: str, status: str, reason: str, available_tools: list[str]) -> str:
    """Lightweight failure re-prompt for in-process Gemma loop (Plan 007)."""
    tool_list = ", ".join(sorted(available_tools)) if available_tools else "no alternative tools available"
    hints = {
        "empty": "The tool returned no output — it may need different parameters.",
        "timeout": "The tool timed out. Try breaking the task into smaller steps.",
        "error": f"Error: {reason[:150]}",
        "skill_not_found": f"Skill not found. Available tools: {tool_list}",
        "routine_not_found": f"Routine not found. Available tools: {tool_list}",
        "no_results": "No results — try a different query.",
        "skill_execution_failed": "Skill execution failed — try a different approach.",
        "routine_execution_failed": "Routine failed — try a manual approach.",
        "permission_denied": "Permission denied — use a different path or request approval.",
    }
    hint = hints.get(status, f"Tool call failed: {reason[:120]}")
    return f"⚠️ Tool `{tool_name}` failed: {hint}\nAvailable tools: {tool_list}\nPlease try a different approach or tool."

_model = None
_processor = None
_config = None


def load_config(path="config.yaml"):
    global _config
    from runtime_paths import load_config as _load_expanded
    _config = _load_expanded(path)
    return _config


def load(config_path="config.yaml"):
    global _model, _processor, _config
    if _model:
        return _model, _processor

    # If model server is running, skip in-process load entirely
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            print("[model] Model server detected — skipping in-process load")
            return None, None
    except Exception:
        pass

    # If task_inference is routed to a cloud provider, skip local GPU load
    try:
        import yaml as _yaml
        _cfg_check = _yaml.safe_load(open(config_path))
        _task_provider = _cfg_check.get("providers", {}).get("task_inference", "local")
        if _task_provider != "local":
            print(f"[model] task_inference='{_task_provider}' — skipping local model load")
            return None, None
    except Exception:
        pass

    cfg = load_config(config_path)
    model_source = os.environ.get("MODEL_SOURCE", "local")

    if model_source == "docker-hub":
        model_path = os.environ.get("MODEL_ID", cfg["model"]["name"])
        print(f"[model] Docker Hub mode — using {model_path}")
    else:
        model_path = cfg["model"].get("path") or cfg["model"]["name"]

    device = cfg["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    print(f"[model] Loading {model_path} on {device} ({dtype})")
    _processor = AutoProcessor.from_pretrained(model_path)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, device_map="auto"
    )
    print(f"[model] Ready")
    return _model, _processor


def infer(messages: list, max_new_tokens=8192, adapter_path: str | None = None) -> str:
    """Run a chat completion. messages = OpenAI-style list with string content."""
    # Try model server first (persistent process)
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            import core.inference.model_client as _client
            return _client.infer(messages, max_new_tokens=max_new_tokens, adapter_path=adapter_path)
    except Exception:
        pass
    # Fallback: in-process loading (original behaviour)
    # Gemma 4 uses 'model' not 'assistant' for assistant role
    formatted = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        formatted.append({"role": role, "content": content})

    text = _processor.apply_chat_template(
        formatted,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = _processor(text=text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        out = _model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, do_sample=False)
    return _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def infer_with_tools(
    messages: list,
    tools: list,
    workspace: str = _DEFAULT_WORKSPACE,
    max_steps: int = 15,
    step_callback=None,
    adapter_path: str | None = None,
    chat_id: str = "",
) -> str:
    """
    Agentic inference loop with native function calling.
    Gemma emits tool calls → we execute them → feed results back → repeat until final answer.
    Returns the final text response.
    """
    # Try model server first (persistent process)
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            import core.inference.model_client as _client
            return _client.infer_with_tools(
                messages,
                tools,
                workspace=workspace,
                max_steps=max_steps,
                step_callback=step_callback,
                adapter_path=adapter_path,
                chat_id=chat_id,
            )
    except Exception:
        pass
    # Fallback: in-process loading (original behaviour)
    import json

    current_messages = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        current_messages.append({"role": role, "content": content})

    for step in range(max_steps):
        text = _processor.apply_chat_template(
            current_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = _processor(text=text, return_tensors="pt").to(_model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
            )

        # Decode both ways — use raw for tool call parsing, clean for display
        response_raw = _processor.decode(out[0][input_len:], skip_special_tokens=False)
        response_clean = _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

        tool_call = _parse_tool_call(response_raw)
        if tool_call is None:
            print(f"[tool] final answer after {step} steps", flush=True)
            return response_clean

        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
        tool_args = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments", {})
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                tool_args = {}
        tool_args = _normalize_tool_args(tool_name, tool_args)

        print(f"[tool] step {step}: {tool_name}({json.dumps(tool_args)[:80]})", flush=True)
        _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace)
        result = _meta.get("result", "")
        _ok = bool(_meta.get("ok", False))
        _status = str(_meta.get("status", "unknown"))
        _reason = str(_meta.get("failure_reason", ""))
        print(f"[tool] result: {result[:100]}", flush=True)

        if step_callback:
            step_callback(step + 1, tool_name, tool_args, result)

        # Feed assistant turn + tool result using user message format
        current_messages.append({
            "role": "model",
            "content": [{"type": "text", "text": response_clean or f"[called {tool_name}]"}]
        })
        current_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"Tool `{tool_name}` result:\n{result}"}]
        })

        # Plan 007: Structured failure re-prompt
        if not _ok and step < max_steps - 2:
            _tool_names = [t["function"]["name"] for t in tools if "function" in t and "name" in t["function"]]
            _fail_msg = _build_failure_reprompt_simple(tool_name, _status, _reason, _tool_names)
            print(f"[tool] injecting failure re-prompt: {_status}", flush=True)
            current_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": _fail_msg}]
            })

    return "(max steps reached)"


def _parse_tool_call(text: str) -> "dict | None":
    """Try to extract a tool call from model output (raw, with special tokens)."""
    import json
    import re

    # Pattern 0: call:name\n{...} — newline-separated JSON block (Gemma 4 chat format)
    # Uses json.raw_decode to correctly handle nested braces in content fields
    _m0 = re.search(r'call:(\w+)\s*\n', text)
    if _m0:
        _tname = _m0.group(1)
        _json_start = _m0.end()
        _brace = text.find('{', _json_start)
        if _brace >= 0:
            try:
                _args, _ = json.JSONDecoder().raw_decode(text, _brace)
                if isinstance(_args, dict):
                    return {"name": _tname, "arguments": _args}
            except Exception:
                pass

    # Pattern 1: Gemma 4 with special tokens preserved
    # <|tool_call>call:exec_shell{command:<|"|>date<|"|>}<tool_call|>
    # The <|"|> tokens are string delimiters
    m = re.search(r'call:(\w+)\{([^}]*)\}', text)
    if m:
        tool_name = m.group(1)
        raw_args = m.group(2)
        # Remove Gemma string delimiter tokens <|"|>
        raw_args = re.sub(r'<\|"\|>', '"', raw_args)
        raw_args = raw_args.strip()
        args = {}
        # Try JSON parse of args as object
        try:
            args = json.loads('{' + raw_args + '}')
        except Exception:
            # Fall back to key:value parsing
            for pair in re.findall(r'(\w+):\s*"([^"]*)"', raw_args):
                args[pair[0]] = pair[1]
            if not args:
                for pair in re.findall(r'(\w+):\s*([^,}]+)', raw_args):
                    args[pair[0].strip()] = pair[1].strip().strip('"')
        if tool_name and args:
            return {"name": tool_name, "arguments": args}

    # Pattern 2: JSON code block
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if 'name' in data or 'function' in data:
                return data
        except Exception:
            pass

    # Pattern 3: <tool_call> tags
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Pattern 4: OpenAI-style {"name": ..., "arguments": {...}}
    m = re.search(r'(\{"name":\s*"[^"]+",\s*"arguments":\s*\{.*?\}\s*\})', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Pattern 5: Raw JSON object with name/function key
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            data = json.loads(stripped)
            if 'name' in data or 'function' in data:
                return data
        except Exception:
            pass

    # Pattern 6: Prose tool call — model describes the call instead of emitting structured format.
    # e.g. 'Executing write_file(path="foo.md", content="...")'
    # e.g. 'Executing `write_file(path=..., content=...)`'
    # Only match known tool names to avoid false positives.
    _KNOWN_TOOLS = {"write_file", "read_file", "exec_shell", "http_get", "run_skill", "run_routine"}
    _prose = re.search(
        r'(?:Executing|Calling|Running|Invoking)\s+`?(\w+)\s*\(([^)]{0,2000})\)',
        text, re.DOTALL | re.IGNORECASE
    )
    if _prose:
        _tname = _prose.group(1).strip()
        if _tname in _KNOWN_TOOLS:
            _raw_args = _prose.group(2).strip()
            _args = {}
            # Try key=value pairs (handles both quoted and unquoted values)
            for _m in re.finditer(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'|([^,)]+))', _raw_args):
                _key = _m.group(1)
                _val = _m.group(2) if _m.group(2) is not None else (_m.group(3) if _m.group(3) is not None else (_m.group(4) or "").strip())
                _args[_key] = _val
            if _tname and _args:
                return {"name": _tname, "arguments": _args}

    # Pattern 7: bare call:name with no args block (model forgot to include arguments).
    # Only match known tool names to avoid false positives.
    _m7 = re.search(r'call:(\w+)\s*$', text.strip())
    if _m7:
        _tname7 = _m7.group(1)
        _KNOWN_TOOLS_BARE = {"write_file", "read_file", "exec_shell", "http_get", "run_skill", "run_routine"}
        if _tname7 in _KNOWN_TOOLS_BARE:
            # Can't execute without args — return a sentinel so the loop re-prompts
            return {"name": _tname7, "arguments": {"_missing_args": True}}

    # Pattern 8: XML-ish function tags
    # Supports both:
    #   <function=write_file> ... </function>
    # and unterminated variants that rely on end-of-text:
    #   <function=write_file> ...
    _fn_blocks = re.findall(r'<function=([A-Za-z_][\w-]*)>(.*?)(?:</function>|(?=<function=)|$)', text, re.DOTALL)
    if _fn_blocks:
        tool_name, fn_body = _fn_blocks[0]
        _params = re.findall(r'<parameter=([A-Za-z_][\w-]*)>(.*?)</parameter>', fn_body, re.DOTALL)
        args = {k: v.strip() for k, v in _params}
        args = _normalize_tool_args(tool_name.strip(), args)
        if tool_name.strip() and args:
            return {"name": tool_name.strip(), "arguments": args}

    return None


def _sanitize_text(val: str) -> str:
    from core.tool_arg_utils import sanitize_text
    return sanitize_text(val)


def _normalize_tool_args(tool_name: str, args: dict) -> dict:
    from core.tool_arg_utils import normalize_tool_args
    return normalize_tool_args(tool_name, args)


def infer_with_image(image_path: str, prompt: str, max_new_tokens: int = 8192) -> str:
    """Run multimodal inference with an image. Returns text response."""
    # Try model server first (persistent process)
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            import core.inference.model_client as _client
            return _client.infer_with_image(image_path, prompt, max_new_tokens=max_new_tokens)
    except Exception:
        pass
    # Fallback: in-process loading (original behaviour)
    from PIL import Image
    import torch

    # Ensure the model/processor are loaded before touching them (vision may be
    # the very first request, before any text inference). If load is skipped
    # (model server present / cloud-routed task) and still no processor, return
    # a clean error instead of crashing on apply_chat_template.
    if _processor is None or _model is None:
        load()
    if _processor is None or _model is None:
        return "Vision unavailable: no in-process model loaded (model server running or task routed to cloud)."

    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}
    ]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _processor(text=text, images=[img], return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def infer_with_audio(audio_path: str, prompt: str = "The user sent you a voice message. Listen and reply naturally.", max_new_tokens: int = 8192, history: list = None, mode: str = "stt") -> str:
    """Run multimodal inference with audio. mode='stt' = transcribe only, mode='chat' = conversational.
    Raises RuntimeError on model server errors so callers get a clean exception instead of
    an error string silently propagating as user content.
    """
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            import core.inference.model_client as _client
            result = _client.infer_with_audio(audio_path, prompt, max_new_tokens=max_new_tokens, history=history, mode=mode)
            # Raise on error strings so voice handler catches them cleanly
            if isinstance(result, str) and result.startswith(("[model_server error]", "[model_client error]", "[model_server timeout]")):
                raise RuntimeError(result)
            return result
    except RuntimeError:
        raise
    except Exception:
        pass
    # Fallback: in-process loading (original behaviour)
    import torch
    import soundfile as sf
    import numpy as np

    # Ensure the model/processor are loaded before touching them (audio may be
    # the very first request, before any text inference). If load is skipped
    # (model server present / cloud-routed task) and still no processor, return
    # a clean error instead of crashing on apply_chat_template.
    if _processor is None or _model is None:
        load()
    if _processor is None or _model is None:
        return "Audio unavailable: no in-process model loaded (model server running or task routed to cloud)."

    # Load audio — convert OGG/MP3/etc to PCM via soundfile or ffmpeg fallback
    try:
        audio_array, sample_rate = sf.read(audio_path, dtype="float32")
    except Exception:
        # ffmpeg fallback for OGG (Telegram voice notes)
        import subprocess, tempfile, os
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", tmp],
                       capture_output=True, timeout=30)
        audio_array, sample_rate = sf.read(tmp, dtype="float32")
        os.unlink(tmp)

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)  # stereo → mono
    # Ensure float32 contiguous — processor's feature_extractor needs this
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)

    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio": {"array": audio_array, "sampling_rate": sample_rate}},
            {"type": "text",  "text": prompt},
        ]}
    ]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _processor(
        text=text,
        audio=[audio_array],
        sampling_rate=16000,
        return_tensors="pt"
    ).to(_model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def vram_free_mb() -> int:
    """Return free VRAM in MB, or RAM if CPU."""
    # Try model server first (persistent process)
    try:
        from core.inference.model_client import is_server_running
        if is_server_running():
            import core.inference.model_client as _client
            return _client.vram_free_mb()
    except Exception:
        pass
    # Fallback: in-process loading (original behaviour)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free // (1024 * 1024)
    import psutil
    return psutil.virtual_memory().available // (1024 * 1024)
