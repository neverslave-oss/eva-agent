"""
tools.py — Kernel Evo tool registry + executor.
Provides 4 tools for native function-calling: exec_shell, read_file, http_get, write_file.
"""
import subprocess
import json
import os
import re
import time
from pathlib import Path
from runtime_paths import WORKSPACE_ROOT

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

KERNEL_WORKSPACE = str(WORKSPACE_ROOT)
WORKSPACE = os.environ.get("KERNEL_WORKSPACE", KERNEL_WORKSPACE)

# Identity files the model must never overwrite via write_file. These carry the
# agent's core persona/architecture and are managed by setup.py (template copy +
# self-healing repair) and the identity consolidator (append-only learnings).
_PROTECTED_IDENTITY_FILES = (
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
)

# ── Authorization gate: current chat_id for exec_shell auth ──────────────
# Only a same-process fallback for in-process callers (e.g. model.py's
# in-process tool loop, which shares memory with agent.py's triage()).
# The model_server process is multi-threaded (ThreadingMixIn) and runs in a
# separate process from the API/agent.py — this global is never set there.
# Cross-process callers (model_server.py) must pass chat_id explicitly to
# execute_tool_with_meta()/execute_tool() instead of relying on this global.
_current_chat_id: str = ""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_shell",
            "description": "Execute a shell command and return stdout/stderr. Use for running scripts, checking service health, git operations, file operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use for reading logs, configs, memory files, ROUTINE.md steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or workspace-relative file path"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Make an HTTP GET request to a URL and return the response body. Use ONLY for http:// or https:// URLs. For local files use read_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 5)"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Execute a named skill from the Kernel skill ecosystem. Use this when the user's request matches a skill's purpose (e.g. browser search, image generation, GitHub operations, security scan). Pass the user's original request as 'input'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The skill name (e.g. 'browser-automation', 'github', 'security-scanner')"
                    },
                    "input": {
                        "type": "string",
                        "description": "The user's request or task to pass to the skill"
                    }
                },
                "required": ["skill_name", "input"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_routine",
            "description": "Execute a named routine from the Kernel routine library. Routines are multi-step procedures (e.g. morning-briefing, security-check, deploy, end-of-session).",
            "parameters": {
                "type": "object",
                "properties": {
                    "routine_name": {
                        "type": "string",
                        "description": "The routine name (e.g. 'morning-briefing', 'security-check', 'deploy', 'end-of-session')"
                    }
                },
                "required": ["routine_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_file",
            "description": "Send a file from the local workspace to the user via Telegram. Use this when the user asks to receive, download, or get a file, report, export, or any workspace document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path to the file to send (e.g. ~/.kernel-evolving/workspace/documents/report.md)"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption shown with the file in Telegram"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web or fetch a URL using the browser-automation skill (Puppeteer + Chromium). Use when you need fresh information, documentation, news, research papers, or anything not in local files. Returns page content or search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g. 'python asyncio tutorial 2026') or full URL to fetch (e.g. 'https://docs.python.org/3/library/asyncio.html')"
                    },
                    "save_to": {
                        "type": "string",
                        "description": "Optional file path to save the result to (e.g. ~/evo_research.md)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_use",
            "description": "Perform an agentic multi-step web task using browser-use (navigate, click, fill forms, log in, extract data, complete workflows in a real browser). Use when a task requires INTERACTING with a web app or multiple coordinated steps (e.g. 'log into X and download the report', 'search for the top result and extract its title'). For simple lookups or single-page reads, prefer web_search instead. Returns a summary of actions taken + extracted data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The natural-language web task to complete (e.g. 'Search for the latest news about AI, open the top result, and extract the title')"
                    },
                    "url": {
                        "type": "string",
                        "description": "Optional starting URL to navigate to first (e.g. 'https://example.com')"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Optional max agent steps (default from config, e.g. 15). Caps runaway loops."
                    },
                    "save_screenshot": {
                        "type": "boolean",
                        "description": "Optional: save a final screenshot to the configured screenshot_dir"
                    }
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": "Search installed skills by name or description keyword. Use this to find the right skill before calling run_skill. Returns a list of matching skills with their names, commands, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (e.g. 'browser', 'image', 'github', 'security'). Pass empty string to list all skills."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_routines",
            "description": "List all available routines. Use this to find the correct routine name before calling run_routine. Returns each routine's name, trigger, and description.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Semantic search of long-term memory for past conversations, notes, or facts. Pass a topic or question as query — NOT a filename or date. Example: query='voice clone setup' not query='2026-06-07.md'. Use when the user references something from a past session or you need context about prior work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term or topic to look up in memory"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look",
            "description": "Perceive the world through connected cameras (eyes). Acts by intent and opens one eye or all. Intents: 'what's there' (objects), 'who is it' (faces, gated), 'plant health' (leaf detection), 'describe' (semantic scene description via local Gemma E2B / Ollama brain), 'scan' (all eyes). Endpoints are config-driven via the eye registry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["what's there", "who is it", "plant health", "describe", "scan"],
                        "description": "What the agent wants to perceive."
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional eye id hint (e.g. 'left', 'right'). Router falls back to intent mapping if omitted."
                    }
                },
                "required": ["intent"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sensors",
            "description": "Read environmental sensor data from connected devices (e.g. the Pi: temperature, humidity, soil moisture). Actions: 'read' returns the latest sensor readings (temp/humi/moisture/moisture_percent). 'water on' / 'water off' (pump override) are reserved for a later phase and currently return a 'deferred' note. Endpoints are config-driven via the sensor registry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "water on", "water off"],
                        "description": "What to do: 'read' fetches current sensor readings."
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional device id hint (default 'pi'). Router falls back to the default device if omitted."
                    }
                },
                "required": ["action"]
            }
        }
    }
]


def _rewrite_date_tokens(text: str) -> str:
    from core.tool_arg_utils import rewrite_date_tokens
    return rewrite_date_tokens(text)


def _extract_backend_hint(result_text: str) -> str:
    m = re.search(r"\[backend=([^\]]+)\]", result_text or "")
    return (m.group(1).strip() if m else "")


def classify_tool_result(name: str, result_text: str) -> dict:
    """Classify a tool result into explicit success/failure metadata."""
    text = (result_text or "").strip()
    low = text.lower()

    if not text:
        return {
            "ok": False,
            "status": "empty",
            "failure_reason": "tool returned empty output",
            "backend": "",
        }

    if low.startswith("(timeout") or "timeout" in low:
        return {
            "ok": False,
            "status": "timeout",
            "failure_reason": text[:240],
            "backend": _extract_backend_hint(text),
        }

    if low.startswith("(error") or "[model_server error]" in low or "[model_client error]" in low:
        status = "error"
        if name == "run_skill" and "not found" in low:
            status = "skill_not_found"
        elif name == "run_routine" and "not found" in low:
            status = "routine_not_found"
        return {
            "ok": False,
            "status": status,
            "failure_reason": text[:240],
            "backend": _extract_backend_hint(text),
        }

    # Informational "not found" results from read_file and recall_memory are
    # valid answers — treat as ok so the model moves on instead of retrying.
    informational_markers = (
        "no memory entries found",
        "no such file",
        "file not found",
        "does not exist",
    )
    if name in ("read_file", "recall_memory") and any(m in low for m in informational_markers):
        return {
            "ok": True,
            "status": "ok",
            "failure_reason": "",
            "backend": _extract_backend_hint(text),
        }

    # Generic failure markers for shell/skill/routine dispatchers only.
    failure_markers = (
        "unknown command",
        "command not found",
        "no such file or directory",
        "traceback (most recent call last)",
        "exception:",
        "failed:",
        "permission denied",
    )
    if any(m in low for m in failure_markers):
        status = "error"
        if name == "run_skill":
            status = "skill_execution_failed"
        elif name == "run_routine":
            status = "routine_execution_failed"
        return {
            "ok": False,
            "status": status,
            "failure_reason": text[:240],
            "backend": _extract_backend_hint(text),
        }

    # "(no results for:" from web_search etc. — still a soft failure worth retrying
    if "(no results for:" in low:
        return {
            "ok": False,
            "status": "no_results",
            "failure_reason": text[:240],
            "backend": _extract_backend_hint(text),
        }

    return {
        "ok": True,
        "status": "ok",
        "failure_reason": "",
        "backend": _extract_backend_hint(text),
    }


def _run_look(arguments: dict) -> str:
    """Run the unified `look` tool: route an intent to the configured eyes.

    Endpoints/IPs are config-driven (config.yaml `vision.eyes`), never hardcoded,
    so EVA is portable across installs. Returns a compact JSON envelope.
    """
    intent = arguments.get("intent")
    if not intent:
        return "(error: look requires 'intent' argument)"
    target = arguments.get("target") or None
    try:
        from core.vision.registry import EyeRegistry
        from core.vision.router import route_look
        registry = EyeRegistry()
        result = route_look(intent, registry, target=target)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return f"(error: look failed: {exc})"


def _run_sensors(arguments: dict) -> str:
    """Run the unified `sensors` tool: route an action to the configured devices.

    Endpoints/IPs are config-driven (config.yaml `sensors`), never hardcoded, so
    EVA is portable across installs. Returns a compact JSON envelope.
    """
    action = arguments.get("action")
    if not action:
        return "(error: sensors requires 'action' argument)"
    target = arguments.get("target") or None
    try:
        from core.sensors.registry import SensorRegistry
        from core.sensors.router import route_sensors
        registry = SensorRegistry()
        result = route_sensors(action, registry, target=target)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return f"(error: sensors failed: {exc})"


def execute_tool_with_meta(name: str, arguments: dict, workspace: str = WORKSPACE, chat_id: str = "") -> dict:
    """Execute tool and return structured metadata for observability."""
    started = time.monotonic()
    result = execute_tool(name, arguments, workspace=workspace, chat_id=chat_id)
    meta = classify_tool_result(name, result)
    meta.update({
        "tool": name,
        "result": result,
        "duration_ms": int((time.monotonic() - started) * 1000),
    })
    return meta


def execute_tool(name: str, arguments: dict, workspace: str = WORKSPACE, chat_id: str = "") -> str:
    """Execute a tool call and return the result as a string."""
    import os
    # Always expand ~ so subprocess.run cwd never gets a literal tilde
    workspace = os.path.expanduser(workspace)
    if name == "exec_shell":
        cmd = arguments.get("command")
        if not cmd:
            return "(error: exec_shell requires 'command' argument)"
        cmd = _rewrite_date_tokens(str(cmd))
        timeout = arguments.get("timeout", 30)
        # ── Authorization gate ─────────────────────────────────────────
        # Check if the command requires user approval via Telegram.
        # Prefer the explicitly-passed chat_id (required for cross-process
        # callers like model_server.py); fall back to the module global for
        # same-process callers that still rely on it.
        _chat_id = chat_id or _current_chat_id
        if _chat_id:
            try:
                from core.auth_gate import request_auth
                auth_result = request_auth(_chat_id, cmd)
                if auth_result == "deny":
                    return "(authorization denied — command blocked)"
                elif auth_result == "timeout":
                    return "(authorization timed out — command blocked)"
                elif auth_result.startswith("deny"):
                    return f"(authorization failed: {auth_result})"
                # auth_result == "allow" → proceed
            except ImportError:
                pass  # auth_gate not available (tests) → proceed without gate
            except Exception as e:
                print(f"[tools] auth gate error (proceeding anyway): {e}", flush=True)
        # ── Execute ────────────────────────────────────────────────────
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=workspace
            )
            out = result.stdout.strip() or result.stderr.strip() or "(no output)"
            return out[:10000]
        except subprocess.TimeoutExpired:
            return f"(timeout after {timeout}s)"
        except Exception as e:
            return f"(error: {e})"

    elif name == "read_file":
        path = arguments.get("path")
        if not path:
            return "(error: read_file requires 'path' argument)"
        import os
        path = _rewrite_date_tokens(str(path))
        # Sanitize: strip shell metacharacters and newlines the model may emit
        # (e.g. "/home/user/workspace/>\n~/workspace/file.md")
        path = path.strip()
        # If path contains newlines, take the last line (often the actual path)
        if "\n" in path:
            path = [p.strip() for p in path.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
        # Strip shell redirect characters that should never be in a file path
        path = path.lstrip(">").rstrip("<")
        path = os.path.expanduser(path)
        if not path.startswith("/"):
            path = f"{workspace}/{path}"
        try:
            with open(path) as f:
                return f.read()[:3000]
        except Exception as e:
            return f"(error reading {path}: {e})"

    elif name == "http_get":
        if not _HAS_REQUESTS:
            return "(error: requests library not available)"
        url = arguments.get("url")
        if not url:
            return "(error: http_get requires 'url' argument)"
        # Guard: if the model passes a local path instead of a URL, reroute to read_file
        url_str = str(url)
        if url_str.startswith("/") or url_str.startswith("~") or not (url_str.startswith("http://") or url_str.startswith("https://")):
            # Sanitize the same way read_file does
            local_path = url_str.strip()
            if "\n" in local_path:
                local_path = [p.strip() for p in local_path.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
            local_path = local_path.lstrip(">").rstrip("<")
            local_path = os.path.expanduser(local_path)
            if not local_path.startswith("/"):
                local_path = os.path.join(workspace, local_path)
            try:
                with open(local_path) as _f:
                    return _f.read()[:3000]
            except Exception as _e:
                return f"(error reading local path {local_path}: {_e})"
        timeout = arguments.get("timeout", 5)
        try:
            r = _requests.get(url_str, timeout=timeout)
            return r.text[:1000]
        except Exception as e:
            return f"(error: {e})"

    elif name == "write_file":
        path = arguments.get("path")
        if not path:
            return "(error: write_file requires 'path' argument)"
        content = arguments.get("content")
        if content is None:
            return "(error: write_file requires 'content' argument)"
        # Sanitize: strip shell metacharacters and newlines the model may emit
        path = str(path).strip()
        if "\n" in path:
            path = [p.strip() for p in path.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
        path = path.lstrip(">").rstrip("<")
        # Expand ~ first, then fall back to workspace-relative only if truly relative
        path = os.path.expanduser(path)
        if not path.startswith("/"):
            path = os.path.join(workspace, path)
        # Hard workspace guard: redirect /tmp, /var, /root, or any path outside
        # ~/.kernel-evolving to the kernel-evolving workspace tmp dir.
        # The model must not write outside its own workspace.
        _allowed_prefix = os.path.expanduser("~/.kernel-evolving")
        _tmp_dir = os.path.join(_allowed_prefix, "workspace", "tmp")
        if not path.startswith(_allowed_prefix):
            _basename = os.path.basename(path)
            path = os.path.join(_tmp_dir, _basename)
            import logging as _log_wf
            _log_wf.getLogger(__name__).warning(
                f"[write_file] path outside workspace — redirected to {path}"
            )
        # Protected identity files: refuse to overwrite AGENTS.md / SOUL.md /
        # IDENTITY.md. These carry the agent's core persona and are managed by
        # setup.py + the identity consolidator — the model must never clobber them.
        if os.path.basename(path) in _PROTECTED_IDENTITY_FILES:
            return (
                f"(error: '{os.path.basename(path)}' is a protected identity file and "
                f"cannot be overwritten via write_file. If you want to record a session "
                f"learning, mention it and it will be appended to the Session learnings section.)"
            )
        # Strip any trailing tool-error lines that the model may have accidentally
        # appended to the content (e.g. "Error: URL must start with http://")
        import re as _re_wf
        _error_tail = _re_wf.compile(
            r'(\s*(?:Error:\s+URL must start with|\(error[:\s]|error reading|\[model_server error\])[^\n]*)$',
            _re_wf.IGNORECASE | _re_wf.MULTILINE,
        )
        content = _error_tail.sub('', str(content)).rstrip()
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Written to {path}"
        except Exception as e:
            return f"(error: {e})"

    elif name == "search_skills":
        # Sanitize: strip newlines and shell redirects Nemotron may emit
        query = (arguments.get("query") or "").lower().strip()
        if "\n" in query:
            query = [p.strip() for p in query.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
        query = query.lstrip(">").rstrip("<")
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            from core.skills import load_all as _load_skills
            config_path = str(_Path(__file__).parent.parent.parent / "config.yaml")
            cfg = _yaml.safe_load(open(config_path))
            skills_dir = cfg.get("skills_dir", "./skills")
            import os as _os
            skills_dir = _os.path.expanduser(skills_dir)
            all_skills = _load_skills(skills_dir)
            if query:
                matches = [
                    s for s in all_skills
                    if query in s.get("name", "").lower()
                    or query in s.get("description", "").lower()
                    or any(query in str(c).lower() for c in s.get("commands", []))
                ]
            else:
                matches = all_skills
            # Expertise-field bias (ADR-015/ADR-022): re-order matches so the
            # active field's domain skills float to top. "Feed, don't bypass" —
            # never drops non-field matches, only reorders.
            if matches:
                try:
                    from core.expansions.expertise_field_bridge import reorder_matches as _reorder
                    matches = _reorder(query, matches, chat_id=_current_chat_id)
                except Exception:
                    pass
            if not matches:
                return f"No skills match '{query}'. Try a broader term or empty string to list all."
            lines = [f"Found {len(matches)} skill(s) matching '{query}':" if query else f"{len(matches)} skills installed:"]
            for s in matches[:30]:
                cmds = s.get("commands", [])
                cmd_str = f" [{', '.join(str(c) for c in cmds)}]" if cmds else ""
                desc = s.get("description", "")[:120]
                lines.append(f"- **{s['name']}**{cmd_str}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"(error searching skills: {e})"

    elif name == "list_routines":
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            from core.routines import load_all as _load_routines
            config_path = str(_Path(__file__).parent.parent.parent / "config.yaml")
            cfg = _yaml.safe_load(open(config_path))
            routines_dir = cfg.get("routines_dir", "./routines")
            import os as _os
            routines_dir = _os.path.expanduser(routines_dir)
            all_routines = _load_routines(routines_dir)
            if not all_routines:
                return "No routines installed."
            lines = [f"{len(all_routines)} routines available:"]
            for r in all_routines:
                trigger = r.get("trigger", {})
                t_also = trigger.get("also", "") if isinstance(trigger, dict) else ""
                t_cron = trigger.get("cron", "") if isinstance(trigger, dict) else ""
                trigger_str = f" (cmd: {t_also})" if t_also else (f" (cron: {t_cron})" if t_cron else "")
                desc = r.get("description", "")[:120]
                lines.append(f"- **{r['name']}**{trigger_str}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"(error listing routines: {e})"

    elif name == "recall_memory":
        query = (arguments.get("query") or "").strip()
        if not query:
            return "(error: recall_memory requires a 'query' argument)"
        try:
            from long_term_memory import search_memory
            results = search_memory(query, limit=8, max_chars=400)
            if not results:
                return f"No memory entries found for '{query}'."
            lines = [f"Memory results for '{query}':"]
            for r in results:
                lines.append(f"- {r}")
            return "\n".join(lines)
        except ImportError:
            # Fallback: scan memory files for keyword
            try:
                import glob as _glob
                mem_dir = Path.home() / ".kernel-evolving" / "workspace" / "memory"
                results = []
                q_lower = query.lower()
                for f in sorted(_glob.glob(str(mem_dir / "*.md")))[-30:]:
                    try:
                        text = open(f).read()
                        if q_lower in text.lower():
                            # Extract surrounding context
                            idx = text.lower().find(q_lower)
                            snippet = text[max(0, idx-80):idx+200].strip()
                            results.append(f"[{Path(f).name}] ...{snippet}...")
                    except Exception:
                        pass
                if not results:
                    return f"No memory entries found for '{query}'."
                return f"Memory results for '{query}':\n" + "\n".join(results[:8])
            except Exception as e:
                return f"(error searching memory: {e})"
        except Exception as e:
            return f"(error recalling memory: {e})"

    elif name == "run_skill":
        # Accept both 'skill_name' (canonical) and 'name' (model alias)
        skill_name = arguments.get("skill_name") or arguments.get("name")
        input_text = arguments.get("input")
        if not skill_name:
            return "(error: run_skill requires 'skill_name' argument)"
        # Sanitize: strip newlines and shell redirects Nemotron may emit
        skill_name = skill_name.strip()
        if "\n" in skill_name:
            skill_name = [p.strip() for p in skill_name.split("\n") if p.strip() and not p.strip().endswith(">")][-1]
        skill_name = skill_name.lstrip(">").rstrip("<")
        if not input_text:
            return "(error: run_skill requires 'input' argument)"
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            from core.skills import load_all as _load_skills, find as _find_skill, run as _run_skill
            from core.inference.model import infer as _infer
            config_path = str(_Path(__file__).parent.parent.parent / "config.yaml")
            cfg = _yaml.safe_load(open(config_path))
            skills_dir = cfg.get("skills_dir", "./skills")
            import os as _os
            skills_dir = _os.path.expanduser(skills_dir)
            all_skills = _load_skills(skills_dir)
            skill = _find_skill(skill_name, all_skills)
            if not skill:
                # Try partial match
                skill = next((s for s in all_skills if skill_name.lower() in s["name"].lower()), None)
            if not skill:
                available = [s["name"] for s in all_skills]
                return f"(error: skill '{skill_name}' not found. Available: {', '.join(available)})"
            return _run_skill(skill, input_text, _infer)
        except Exception as e:
            return f"(error running skill '{skill_name}': {e})"

    elif name == "run_routine":
        routine_name = arguments.get("routine_name")
        if not routine_name:
            return "(error: run_routine requires 'routine_name' argument)"
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            from core.routines import load_all as _load_routines, find as _find_routine, run as _run_routine
            from core.inference.model import infer as _infer
            config_path = str(_Path(__file__).parent.parent.parent / "config.yaml")
            cfg = _yaml.safe_load(open(config_path))
            routines_dir = cfg.get("routines_dir", "./routines")
            import os as _os
            routines_dir = _os.path.expanduser(routines_dir)
            all_routines = _load_routines(routines_dir)
            routine = _find_routine(routine_name, all_routines)
            if not routine:
                routine = next((r for r in all_routines if routine_name.lower() in r["name"].lower()), None)
            if not routine:
                available = [r["name"] for r in all_routines]
                return f"(error: routine '{routine_name}' not found. Available: {', '.join(available)})"
            return _run_routine(routine, _infer)
        except Exception as e:
            return f"(error running routine '{routine_name}': {e})"

    elif name == "send_file":
        file_path = arguments.get("file_path", "")
        caption = arguments.get("caption", "")
        if not file_path:
            return "(error: send_file requires 'file_path' argument)"
        import os as _os
        file_path = _os.path.expanduser(file_path)
        if not _os.path.isfile(file_path):
            return f"(error: file not found: {file_path})"
        try:
            import services.channels.telegram_bot as _tb
            chat_id = _tb.ALLOWED_CHAT_ID
            if not chat_id:
                return "(error: no Telegram chat_id configured)"
            ok = _tb.send_file(str(chat_id), file_path, caption=caption)
            return f"✅ File sent: {_os.path.basename(file_path)}" if ok else "(error: Telegram sendDocument failed)"
        except Exception as e:
            return f"(error sending file: {e})"

    elif name == "web_search":
        query = arguments.get("query", "")
        save_to = arguments.get("save_to", "")
        if not query:
            return "(error: web_search requires 'query' argument)"
        try:
            # Use browse.py from the browser-automation skill directly — it's the
            # canonical search/fetch implementation for this system (DDG → Bing fallback).
            import importlib.util as _ilu, os as _os
            _browse_path = _os.path.expanduser(
                "~/.kernel-evolving/ecosystem/private/skills/browser-automation/browse.py"
            )
            _spec = _ilu.spec_from_file_location("browse", _browse_path)
            _browse = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_browse)

            if query.startswith("http"):
                result = _browse.fetch_url(query)
            else:
                result = _browse.ddg_search(query)  # ddg_search auto-falls-back to Bing

            if save_to:
                p = _os.path.expanduser(save_to)
                _os.makedirs(_os.path.dirname(p) or ".", exist_ok=True)
                open(p, "w").write(result)
                return f"Search result saved to {p}\n\n{result[:500]}"
            return result[:3000]
        except Exception as e:
            return f"(error: web_search failed: {e})"

    elif name == "browser_use":
        return _run_browser_use(arguments)

    elif name == "look":
        return _run_look(arguments)

    elif name == "sensors":
        return _run_sensors(arguments)

    return f"Unknown tool: {name}"


def _load_browser_config() -> dict:
    """Load the `browser` section from config.yaml (empty dict on any failure)."""
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        cfg = _yaml.safe_load(open(str(_Path(__file__).parent.parent.parent / "config.yaml")))
        return cfg.get("browser", {}) or {}
    except Exception:
        return {}


def _run_browser_use(arguments: dict) -> str:
    """Run an agentic multi-step browser task via browser-use.

    Bridges the synchronous tool loop to browser-use's async Agent.run() with
    asyncio.run(). Builds the driving LLM from the configured inference provider
    (reuses task_inference routing so it works in cloud/HF-Router mode).
    """
    import asyncio
    task = arguments.get("task")
    if not task:
        return "(error: browser_use requires 'task' argument)"
    task = str(task).strip()
    url = arguments.get("url") or ""
    url = str(url).strip()

    browser_cfg = _load_browser_config()
    if not browser_cfg.get("enabled", True):
        return "(error: browser_use is disabled in config — use web_search instead)"
    headless = bool(browser_cfg.get("headless", True))
    cfg_max_steps = int(browser_cfg.get("max_steps", 60) or 60)
    timeout_s = float(browser_cfg.get("timeout_s", 120) or 120)
    screenshot_dir = browser_cfg.get("screenshot_dir", "") or ""
    import os as _os
    screenshot_dir = _os.path.expanduser(screenshot_dir)

    # Defensive headless enforcement: browser-use reads BROWSER_USE_HEADLESS from
    # the environment and applies it to the browser profile, guaranteeing the
    # browser never opens a visible window on the host desktop regardless of how
    # the Browser instance is constructed.
    _os.environ["BROWSER_USE_HEADLESS"] = "true" if headless else "false"

    max_steps = arguments.get("max_steps")
    if max_steps is None:
        max_steps = cfg_max_steps
    try:
        max_steps = int(max_steps)
    except (TypeError, ValueError):
        max_steps = cfg_max_steps
    max_steps = max(1, min(max_steps, 50))  # hard cap against runaway loops

    save_screenshot = bool(arguments.get("save_screenshot", False))

    try:
        from browser_use import Agent, Browser, ChatOpenAI
    except ImportError as e:
        return (
            f"(error: browser-use not installed — run `pip install browser-use playwright` "
            f"and `playwright install chromium`. Detail: {e})"
        )

    # ── Build the driving LLM from the configured inference provider ──────
    try:
        from core.inference.provider import get_provider as _get_prov
        prov = _get_prov()
        provider_name = prov.get_provider("task_inference")
        model = prov.get_model(provider_name, "task_inference")
        hf_token = _os.environ.get("HF_TOKEN")
        # The HF Router (OpenAI-compatible) is used when task_inference routes to
        # a cloud provider OR when no local model is available. Only the HF
        # provider has a router-based model id (with ":provider" suffix).
        if provider_name == "hf" and hf_token:
            llm = ChatOpenAI(
                model=prov._hf_model(model or "deepseek-ai/DeepSeek-V4-Flash-0731"),
                base_url=prov._hf_base_url(),
                api_key=hf_token,
                temperature=0.1,
            )
        else:
            # Fallback: reuse the HF Router with the task_inference model (cloud
            # mode), since local model_server is not an OpenAI-compatible endpoint
            # browser-use can drive directly.
            if not hf_token:
                return "(error: browser_use needs HF_TOKEN to drive the browser agent)"
            llm = ChatOpenAI(
                model=prov._hf_model(model or "deepseek-ai/DeepSeek-V4-Flash-0731"),
                base_url=prov._hf_base_url(),
                api_key=hf_token,
                temperature=0.1,
            )
    except Exception as e:
        return f"(error: browser_use could not configure LLM: {e})"

    full_task = f"Navigate to {url} and: {task}" if url else task

    async def _run() -> dict:
        browser = Browser(headless=headless)
        agent = Agent(task=full_task, llm=llm, browser=browser, max_steps=max_steps)
        try:
            history = await agent.run(max_steps=max_steps)
            return await _summarize(history, save_screenshot, screenshot_dir)
        finally:
            try:
                await browser.close()
            except Exception:
                pass

    try:
        result = asyncio.run(_run())
    except Exception as e:
        return f"(error: browser_use failed: {e})"

    final = (result.get("final") or "").strip()
    urls = result.get("urls", [])
    steps = result.get("steps", 0)
    errs = result.get("errors", [])
    lines = []
    if final:
        lines.append(f"Result: {final[:2000]}")
    else:
        lines.append("(browser_use completed but returned no final result)")
    if urls:
        lines.append(f"Visited: {', '.join(urls[:5])}")
    if steps:
        lines.append(f"Steps: {steps}")
    if errs:
        lines.append(f"Errors: {'; '.join(str(e)[:200] for e in errs[:3])}")
    if result.get("screenshot"):
        lines.append(f"Screenshot: {result['screenshot']}")
    return "\n".join(lines)[:3000]


async def _summarize(history, save_screenshot: bool, screenshot_dir: str) -> dict:
    """Extract a concise summary + optional screenshot from browser-use history."""
    out = {"final": "", "urls": [], "steps": 0, "errors": [], "screenshot": ""}
    try:
        out["final"] = history.final_result() or ""
    except Exception:
        pass
    try:
        out["urls"] = history.urls() or []
    except Exception:
        pass
    try:
        out["steps"] = history.number_of_steps() or 0
    except Exception:
        pass
    try:
        out["errors"] = [e for e in (history.errors() or []) if e]
    except Exception:
        pass
    if save_screenshot and screenshot_dir:
        try:
            import os as _os
            from datetime import datetime as _dt
            _os.makedirs(screenshot_dir, exist_ok=True)
            path = _os.path.join(
                screenshot_dir, f"browser_use_{_dt.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            # history.screenshot_paths() returns paths of saved screenshots (if
            # save_screenshots enabled); otherwise take the last one via browser.
            paths = history.screenshot_paths() or []
            if paths:
                _os.rename(str(paths[-1]), path)
                out["screenshot"] = path
        except Exception:
            pass
    return out
