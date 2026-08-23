"""
provider.py — ADR-013: Multi-provider inference wrapper.

Routes infer() and infer_with_tools() calls to configured provider per call type.
Providers: local (Gemma 4 via model_server), openai, anthropic, hf, copilot, olly (OpenClaw/claude-sonnet-4.6)

Local-first routing:
  - Primary: local Gemma 4 E2B-it via model_server socket.
  - Fallback to cloud ONLY when:
      (a) GPU temp ≥ providers.gpu_temp_critical_c (configurable, default 85°C), or
      (b) local model_server socket is unreachable (crash/OOM), or
      (c) user explicitly sets task_inference to a cloud provider in config.
  - Cloud fallback is temporary: resumes local when temp drops below gpu_temp_resume_c.
  - All fallback transitions are logged with a [provider] FALLBACK line.
"""

import os
import json
import logging
import time
import urllib.request
import urllib.error
from core.tools import WORKSPACE as _DEFAULT_WORKSPACE
from core.voice_activity import is_voice_active

logger = logging.getLogger(__name__)


# ── GPU temperature monitor ────────────────────────────────────────────────────────

_gpu_temp_cache: dict = {"temp": None, "ts": 0.0}  # module-level cache, refreshed every 30s
_TEMP_CACHE_TTL = 30.0  # seconds between nvidia-smi calls


def _gpu_temp_celsius() -> int | None:
    """Return current GPU temperature in °C, or None if unavailable."""
    global _gpu_temp_cache
    now = time.monotonic()
    if now - _gpu_temp_cache["ts"] < _TEMP_CACHE_TTL and _gpu_temp_cache["temp"] is not None:
        return _gpu_temp_cache["temp"]
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        temp = int(result.stdout.strip().split("\n")[0])
        _gpu_temp_cache = {"temp": temp, "ts": now}
        return temp
    except Exception:
        return None


class InferenceProvider:
    def __init__(self, config: dict):
        self._cfg = config.get("providers", {})
        self._models = self._cfg.get("models", {})
        # Build routing dict: call_type -> provider name
        self._routing = {}
        for call_type in ("task_inference", "synthesis", "critic", "planning", "trajectory_teacher"):
            val = self._cfg.get(call_type)
            if val:
                self._routing[call_type] = val
        self._streaming = self._cfg.get("streaming", {})
        # Thermal fallback config
        self._temp_critical = int(self._cfg.get("gpu_temp_critical_c", 85))
        self._temp_resume = int(self._cfg.get("gpu_temp_resume_c", 70))
        self._fallback_provider = self._cfg.get("fallback_provider", "openai")
        self._temp_fallback_active = False

    def get_provider(self, call_type: str) -> str:
        """Return effective provider for call_type, applying thermal fallback if needed."""
        env_key = "PROVIDER_" + call_type.upper().replace("-", "_")
        configured = os.environ.get(env_key) or self._routing.get(call_type, "local")

        # Voice-priority guard: while STT/TTS/clone is active, keep synthesis/critic off local GPU.
        if configured == "local" and call_type in ("synthesis", "critic"):
            try:
                if is_voice_active():
                    logger.info(
                        f"[provider] VOICE GUARD: voice activity detected — routing {call_type} to {self._fallback_provider}"
                    )
                    return self._fallback_provider
            except Exception:
                pass

        # Only apply thermal fallback to local task_inference — never override explicit cloud config
        if configured == "local" and call_type == "task_inference" and self._temp_critical > 0:
            temp = _gpu_temp_celsius()
            if temp is not None:
                if not self._temp_fallback_active and temp >= self._temp_critical:
                    self._temp_fallback_active = True
                    logger.warning(
                        f"[provider] THERMAL FALLBACK: GPU temp {temp}°C ≥ critical {self._temp_critical}°C "
                        f"— routing task_inference to {self._fallback_provider} until temp ≤ {self._temp_resume}°C"
                    )
                elif self._temp_fallback_active and temp <= self._temp_resume:
                    self._temp_fallback_active = False
                    logger.info(
                        f"[provider] THERMAL RECOVERY: GPU temp {temp}°C ≤ resume {self._temp_resume}°C "
                        f"— resuming local inference"
                    )
                if self._temp_fallback_active:
                    return self._fallback_provider

        return configured

    def get_model(self, provider: str, call_type: str = None) -> str | None:
        """Return model name for provider. call_type overrides take priority."""
        overrides = self._cfg.get("model_overrides", {})
        if call_type and overrides.get(call_type):
            return overrides[call_type]
        return self._models.get(provider)

    def infer(self, messages: list, max_new_tokens: int = 8192, call_type: str = "task_inference") -> str:
        """Route a standard (non-tool) inference call with ordered fallback chain."""
        provider = self.get_provider(call_type)
        # Build fallback chain: primary first, then all others in priority order, excluding primary
        _CHAIN = ["local", "openai", "anthropic", "openrouter", "copilot", "hf"]
        chain = [provider] + [p for p in _CHAIN if p != provider]
        # Skip cloud providers without API keys — faster than waiting for network timeout
        _keyless = {
            "openai": (os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY"),
            "hf": os.environ.get("HF_TOKEN"),
            "copilot": (os.environ.get("GITHUB_COPILOT_TOKEN") or os.environ.get("GITHUB_TOKEN")),
        }
        last_err = ""
        for p in chain:
            if p != "local" and not _keyless.get(p):
                logger.debug(f"[provider] skip {p} — no API key configured")
                continue
            model = self.get_model(p, call_type)
            try:
                if p == "local":
                    from core.inference.model_client import infer as _local_infer
                    result = _local_infer(messages, max_new_tokens=max_new_tokens)
                    # local provider returns error strings instead of raising —
                    # check and continue chain instead of returning garbage
                    if isinstance(result, str) and any(
                        m in result for m in ("[model_server error]", "[model_client error]",
                                              "[model_server timeout]", "CUDA error", "out of memory")
                    ):
                        last_err = result
                        logger.warning(f"[provider] local infer returned error — trying next in chain: {result[:120]}")
                        continue
                    return result
                elif p == "openai":
                    return self._call_openai(messages, model)
                elif p == "anthropic":
                    return self._call_anthropic(messages, model)
                elif p == "openrouter":
                    return self._call_openrouter(messages, model)
                elif p == "hf":
                    return self._call_hf(messages, model)
                elif p == "copilot":
                    return self._call_copilot(messages, model)
                else:
                    continue
            except Exception as e:
                last_err = str(e)
                logger.warning(f"[provider] {p} failed for {call_type}: {e} — trying next in chain")
                continue
        logger.error(f"[provider] all providers exhausted for {call_type}. Last error: {last_err}")
        return "(inference unavailable)"

    def infer_with_tools(self, messages: list, tools: list, workspace: str = _DEFAULT_WORKSPACE,
                         max_steps: int = 60, step_callback=None, call_type: str = "task_inference",
                         chunk_callback=None, chat_id: str = "") -> str:
        """Route an agentic tool-calling inference call with ordered fallback chain.
        chunk_callback(text: str): called with each streamed text chunk on the final answer.
        chat_id: only needed for the "local" provider — model_server runs in a separate
        process and can't see core.tools._current_chat_id (R5/T4). Cloud provider tool
        loops execute tools in-process here, where the module global already works.
        """
        provider = self.get_provider(call_type)
        _CHAIN = ["local", "openai", "anthropic", "openrouter", "copilot", "hf"]
        chain = [provider] + [p for p in _CHAIN if p != provider]
        # Skip cloud providers without API keys — faster than waiting for network timeout
        _keyless = {
            "openai": (os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")),
            "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY"),
            "hf": os.environ.get("HF_TOKEN"),
            "copilot": (os.environ.get("GITHUB_COPILOT_TOKEN") or os.environ.get("GITHUB_TOKEN")),
        }
        last_err = ""
        for p in chain:
            if p != "local" and not _keyless.get(p):
                logger.debug(f"[provider] skip {p} — no API key configured")
                continue
            model = self.get_model(p, call_type)
            try:
                if p == "local":
                    from core.inference.model_client import infer_with_tools as _local_iwt
                    result = _local_iwt(
                        messages, tools, workspace=workspace,
                        max_steps=max_steps, step_callback=step_callback,
                        chunk_callback=chunk_callback, chat_id=chat_id,
                    )
                    # local provider returns error strings instead of raising —
                    # check and continue chain instead of returning garbage
                    if isinstance(result, str) and any(
                        m in result for m in ("[model_server error]", "[model_client error]",
                                              "[model_server timeout]", "CUDA error", "out of memory")
                    ):
                        last_err = result
                        logger.warning(f"[provider] local infer_with_tools returned error — trying next in chain: {result[:120]}")
                        continue
                    return result
                elif p in ("openai", "copilot", "openrouter"):
                    return self._openai_tool_loop(messages, tools, workspace, max_steps, step_callback, model, p, chunk_callback=chunk_callback)
                elif p == "anthropic":
                    return self._anthropic_tool_loop(messages, tools, workspace, max_steps, step_callback, model)
                elif p == "hf":
                    return self._hf_tool_loop(messages, tools, workspace, max_steps, step_callback, model)
                else:
                    continue
            except Exception as e:
                last_err = str(e)
                logger.warning(f"[provider] {p} infer_with_tools failed for {call_type}: {e} — trying next in chain")
                continue
        logger.error(f"[provider] all providers exhausted for infer_with_tools/{call_type}. Last error: {last_err}")
        return "(inference unavailable)"

    # ── Simple (non-tool) provider calls ─────────────────────────────────────

    def _call_openai(self, messages: list, model: str | None) -> str:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY")
        payload = json.dumps({
            "model": model or "gpt-5.4",
            "messages": messages,
            "max_completion_tokens": 4096,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, messages: list, model: str | None) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("No ANTHROPIC_API_KEY")
        payload = json.dumps({
            "model": model or "claude-sonnet-4-6",
            "max_tokens": 8192,
            "messages": messages,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["content"][0]["text"]

    def _hf_router_provider(self) -> str:
        """Resolve the Hugging Face Router inference-provider name.

        The HF OpenAI-compatible endpoint is https://router.huggingface.co/v1 with
        the provider selected by a ":{provider}" suffix on the model id, e.g.
        "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra". Providers: deepinfra,
        together, novita, fireworks-ai, hf-inference, ... Not every provider serves
        every model. Priority: env HF_ROUTER_PROVIDER -> config providers.hf_provider
        -> 'deepinfra' (cheapest).
        """
        env_val = os.environ.get("HF_ROUTER_PROVIDER")
        if env_val:
            return env_val.strip().rstrip("/")
        cfg_val = self._cfg.get("hf_provider")
        if cfg_val:
            return str(cfg_val).strip().rstrip("/")
        return "deepinfra"

    def _hf_model(self, model: str) -> str:
        """Append the ":{provider}" suffix to the model id for the HF Router."""
        provider = self._hf_router_provider()
        if provider and not model.endswith(f":{provider}"):
            return f"{model}:{provider}"
        return model

    def _hf_base_url(self) -> str:
        """Return the HF Router OpenAI-compatible base URL (router root)."""
        return "https://router.huggingface.co/v1"

    def _call_hf(self, messages: list, model: str | None) -> str:
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise RuntimeError("No HF_TOKEN")
        if not model:
            raise RuntimeError("No HF model configured")
        payload = json.dumps({
            "model": self._hf_model(model),
            "messages": messages,
            "max_tokens": 8192,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # urllib's default User-Agent is blocked by the HF Router (403); send a real one.
            "User-Agent": "kernel-evolving/1.0",
        }
        req = urllib.request.Request(
            f"{self._hf_base_url()}/chat/completions",
            data=payload, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _call_copilot(self, messages: list, model: str | None) -> str:
        """GitHub Copilot API — uses GITHUB_COPILOT_TOKEN or GITHUB_TOKEN."""
        api_key = os.environ.get("GITHUB_COPILOT_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not api_key:
            raise RuntimeError("No GITHUB_COPILOT_TOKEN or GITHUB_TOKEN")
        payload = json.dumps({
            "model": model or "gpt-5.4",
            "messages": messages,
            "max_completion_tokens": 4096,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Copilot-Integration-Id": "vscode-chat",
        }
        req = urllib.request.Request(
            "https://api.githubcopilot.com/chat/completions",
            data=payload, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def _call_openrouter(self, messages: list, model: str | None) -> str:
        """OpenRouter API — OpenAI-compatible endpoint with model routing."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("No OPENROUTER_API_KEY")
        resolved_model = model or "google/gemma-3-27b-it"
        payload = json.dumps({
            "model": resolved_model,
            "messages": messages,
            "max_tokens": 4096,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8779",
            "X-Title": "kernel-evolving",
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload, headers=headers, method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    # ── Tool-loop implementations ─────────────────────────────────────────────

    def _openai_tool_loop(self, messages, tools, workspace, max_steps, step_callback, model, provider, chunk_callback=None):
        from core.tools import execute_tool_with_meta

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TMP_OPEN_AI_API_KEY")
        if provider == "copilot":
            api_key = os.environ.get("GITHUB_TOKEN")
        if provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(f"No API key for {provider}")

        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "copilot":
            base_url = "https://api.githubcopilot.com"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"
        workspace = os.path.expanduser(workspace)
        # Sanitise: drop any message missing a role — corrupt history entries cause HTTP 400
        current_messages = [
            m for m in messages
            if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool")
        ]

        for step in range(max_steps):
            # Use streaming on the final answer step when chunk_callback is set
            use_stream = bool(chunk_callback)
            payload = json.dumps({
                "model": model or "gpt-5.4",
                "messages": current_messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_completion_tokens": 2048,
                "stream": use_stream,
            }).encode()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            if provider == "copilot":
                headers["Copilot-Integration-Id"] = "vscode-chat"
            if provider == "openrouter":
                headers["HTTP-Referer"] = "http://localhost:8779"
                headers["X-Title"] = "kernel-evolving"

            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload, headers=headers, method="POST"
            )
            try:
                resp = urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as _http_err:
                _err_body = ""
                try:
                    _err_body = _http_err.read(1000).decode(errors="replace")
                except Exception:
                    pass
                logger.error(f"[provider] OpenAI HTTP {_http_err.code} on step {step}: {_err_body[:400]}")
                raise

            if use_stream:
                # Stream SSE lines, accumulate text + tool_calls
                accumulated_text = []
                tool_calls_raw = {}
                finish_reason = None
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    delta = chunk["choices"][0].get("delta", {})
                    finish_reason = chunk["choices"][0].get("finish_reason") or finish_reason
                    # Text chunk
                    txt = delta.get("content") or ""
                    if txt:
                        accumulated_text.append(txt)
                        try:
                            chunk_callback(txt)
                        except Exception:
                            pass
                    # Tool call chunks
                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_raw:
                            tool_calls_raw[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                        tool_calls_raw[idx]["id"] += tc.get("id") or ""
                        fn = tc.get("function") or {}
                        tool_calls_raw[idx]["function"]["name"] += fn.get("name") or ""
                        tool_calls_raw[idx]["function"]["arguments"] += fn.get("arguments") or ""

                final_text = "".join(accumulated_text)
                tool_calls_list = [tool_calls_raw[i] for i in sorted(tool_calls_raw)]

                # Final answer — no tool calls
                if not tool_calls_list:
                    return final_text

                # Has tool calls — build msg dict and continue loop
                msg = {"content": final_text, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls_list
                ]}
            else:
                data = json.loads(resp.read())
                choice = data["choices"][0]
                msg = choice["message"]

                # No tool call — final answer
                if not msg.get("tool_calls"):
                    return msg.get("content", "")

            # Execute each tool call
            current_messages.append(msg)
            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])
                _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace)
                result_str = _meta.get("result", "")
                _status = _meta.get("status", "unknown")
                _reason = _meta.get("failure_reason", "")
                _step_result = result_str
                if not _meta.get("ok", False):
                    _step_result = f"{result_str}\n[status={_status}] {_reason}".strip()

                if step_callback:
                    step_callback(step + 1, tool_name, tool_args, _step_result)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            # Keep model in tool-calling mode — same pattern as model_server continuation
            remaining = max_steps - step - 1
            if remaining > 0:
                current_messages.append({
                    "role": "user",
                    "content": (
                        "Tool execution complete. Continue — call the next required tool "
                        "to complete the original task. Do NOT summarise yet. "
                        f"({remaining} steps remaining)"
                    )
                })

        return "(max steps reached)"

    def _anthropic_tool_loop(self, messages, tools, workspace, max_steps, step_callback, model):
        from core.tools import execute_tool_with_meta

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("No ANTHROPIC_API_KEY")

        # Convert OpenAI tool format to Anthropic format
        def _convert_tools(tools):
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function":
                    fn = t["function"]
                    anthropic_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
            return anthropic_tools

        workspace = os.path.expanduser(workspace)
        current_messages = list(messages)
        anthropic_tools = _convert_tools(tools)

        for step in range(max_steps):
            # Anthropic doesn't accept system messages inside messages array
            system_content = None
            api_messages = []
            for m in current_messages:
                if m["role"] == "system":
                    system_content = m["content"]
                else:
                    api_messages.append(m)

            payload_dict = {
                "model": model or "claude-sonnet-4-6",
                "max_tokens": 8192,
                "messages": api_messages,
                "tools": anthropic_tools,
            }
            if system_content:
                payload_dict["system"] = system_content

            payload = json.dumps(payload_dict).encode()
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload, headers=headers, method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())

            # Check for tool use blocks
            tool_uses = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
            text_blocks = [b for b in data.get("content", []) if b.get("type") == "text"]

            if not tool_uses:
                return text_blocks[0]["text"] if text_blocks else ""

            # Add assistant message
            current_messages.append({"role": "assistant", "content": data["content"]})

            # Execute each tool use
            tool_results = []
            for tu in tool_uses:
                tool_name = tu["name"]
                tool_args = tu["input"]
                _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace)
                result_str = _meta.get("result", "")
                _status = _meta.get("status", "unknown")
                _reason = _meta.get("failure_reason", "")
                _step_result = result_str
                if not _meta.get("ok", False):
                    _step_result = f"{result_str}\n[status={_status}] {_reason}".strip()

                if step_callback:
                    step_callback(step + 1, tool_name, tool_args, _step_result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result_str,
                })

            current_messages.append({"role": "user", "content": tool_results})

        return "(max steps reached)"

    def _hf_tool_loop(self, messages, tools, workspace, max_steps, step_callback, model):
        """HuggingFace uses OpenAI-compatible format — reuse _openai_tool_loop with HF config."""
        api_key = os.environ.get("HF_TOKEN")
        if not api_key:
            raise RuntimeError("No HF_TOKEN")
        if not model:
            raise RuntimeError("No HF model configured")

        from core.tools import execute_tool_with_meta
        workspace = os.path.expanduser(workspace)
        current_messages = list(messages)

        for step in range(max_steps):
            payload = json.dumps({
                "model": self._hf_model(model),
                "messages": current_messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 8192,
            }).encode()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                # urllib's default User-Agent is blocked by the HF Router (403); send a real one.
                "User-Agent": "kernel-evolving/1.0",
            }
            req = urllib.request.Request(
                f"{self._hf_base_url()}/chat/completions",
                data=payload, headers=headers, method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            choice = data["choices"][0]
            msg = choice["message"]

            if not msg.get("tool_calls"):
                return msg.get("content", "")

            current_messages.append(msg)
            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])
                _meta = execute_tool_with_meta(tool_name, tool_args, workspace=workspace)
                result_str = _meta.get("result", "")
                _status = _meta.get("status", "unknown")
                _reason = _meta.get("failure_reason", "")
                _step_result = result_str
                if not _meta.get("ok", False):
                    _step_result = f"{result_str}\n[status={_status}] {_reason}".strip()

                if step_callback:
                    step_callback(step + 1, tool_name, tool_args, _step_result)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        return "(max steps reached)"


# ── Module-level singleton ────────────────────────────────────────────────────

_provider_instance: InferenceProvider | None = None


def get_provider(config: dict = None) -> InferenceProvider:
    """Get or create the module-level provider singleton."""
    global _provider_instance
    if _provider_instance is None or config is not None:
        import yaml
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.yaml')
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        _provider_instance = InferenceProvider(config)
    return _provider_instance
