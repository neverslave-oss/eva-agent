"""
replica.py — Spawn specialist sub-agents within VRAM limits.
Shared model weights. Isolated context and conversation history per replica.
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from core.inference.model import vram_free_mb, infer, infer_with_tools as _model_infer_with_tools

CONTEXT_BUDGET_MB = 512
MAX_REPLICAS = 4  # increased from 3 to support named meeting agents
MAX_PERSISTENT_REPLICAS = 32

_replicas: dict[str, "Replica"] = {}  # name → Replica (was a list)
_lock = threading.Lock()


def _provider_infer(messages, max_new_tokens=8192, adapter_path=None):
    """Route replica inference through the configured provider.

    In cloud mode (task_inference != local) the local model is not loaded, so
    calling core.inference.model.infer() directly crashes ('NoneType' processor).
    Routing through the provider sends the call to the configured cloud provider
    (e.g. HF Router) instead. Falls back to the local wrapper if the provider is
    unavailable.
    """
    try:
        from core.inference.provider import get_provider as _gp
        _prov = _gp()
        return _prov.infer(messages, max_new_tokens=max_new_tokens, call_type="task_inference")
    except Exception:
        return infer(messages, max_new_tokens=max_new_tokens, adapter_path=adapter_path)


def _provider_infer_with_tools(messages, tools, workspace, adapter_path=None):
    """Route replica tool-calling inference through the configured provider.

    Same rationale as _provider_infer: works in cloud mode where the local model
    is not loaded. Falls back to the local model wrapper if the provider is
    unavailable.
    """
    try:
        from core.inference.provider import get_provider as _gp
        _prov = _gp()
        return _prov.infer_with_tools(
            messages, tools, workspace=workspace, call_type="task_inference"
        )
    except Exception:
        return _model_infer_with_tools(messages, tools, workspace=workspace, adapter_path=adapter_path)

BUILTIN_ROLES = {
    "researcher": "You are a research specialist. Gather information, search for facts, and synthesise findings clearly.",
    "coder":      "You are a coding specialist. Write clean, working code based on the specification provided.",
    "reviewer":   "You are a code reviewer. Validate output for correctness, security, and quality.",
    "reporter":   "You are a reporting specialist. Summarise results concisely and deliver them clearly.",
    # "agent" role: matches the training system prompt so the LoRA adapter activates tool use correctly.
    # The adapter was trained on trajectories that always started with this exact header.
    "agent": (
        "You are Evo (kernel-evolving) \u2014 a self-evolving local AI agent.\n\n"
        "You have access to these tools:\n"
        "  \u2022 exec_shell(command)           \u2014 run shell commands\n"
        "  \u2022 read_file(path)               \u2014 read a file\n"
        "  \u2022 write_file(path, content)     \u2014 write content to a file\n"
        "  \u2022 http_get(url)                 \u2014 make an HTTP GET request\n"
        "  \u2022 run_skill(skill_name, input)  \u2014 execute an installed skill\n"
        "  \u2022 run_routine(routine_name)     \u2014 execute a routine\n\n"
        "Rules:\n"
        "- TOOL FIRST: always call the tool before claiming completion.\n"
        "- Never say 'I have written...' without a preceding write_file tool call.\n"
        "- Use ~ for home paths (e.g. ~/evo_notes.txt not absolute paths).\n"
        "- Be concise. One-line confirmation after task completion."
    ),
}

# Keep backward-compat alias
ROLES = BUILTIN_ROLES


@dataclass
class Replica:
    name: str                        # unique identifier e.g. "lawy", "marty"
    role: str                        # role key or "custom"
    system_prompt: str               # full system prompt (built from role + brief)
    task: str = ""                   # initial task (empty for persistent replicas)
    persistent: bool = False         # if True, stays alive after first response
    brief_path: Optional[str] = None # path to brief file (loaded into system prompt)
    workspace: Optional[str] = None  # scoped workspace path (read-only)
    tools_enabled: bool = False      # ADR-008 Change 1: use infer_with_tools instead of infer
    output_path: Optional[str] = None  # write final response to this path when done
    input_path: Optional[str] = None   # read this file and prepend to first message
    adapter_path: Optional[str] = None # optional LoRA adapter path for this replica
    slot: str = "primary"              # named model slot to use ("primary", "audio", etc.)
    result: str = ""
    done: bool = False
    history: list = field(default_factory=list)  # conversation history for persistent mode
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def start(self, callback=None):
        """Start replica. For persistent replicas, just mark as ready. For task replicas, run the task."""
        if self.persistent:
            self.done = False  # persistent = never done until explicitly stopped
            return
        self._thread = threading.Thread(target=self._run_task, daemon=True)
        self._thread.start()

    def _run_task(self):
        """Fire-and-forget task execution (backward compat)."""
        # Ensure the named slot is loaded before inference
        if self.slot and self.slot != "primary":
            try:
                import core.inference.model_client as model_client
                model_client.load_slot(self.slot)
            except Exception as _se:
                pass  # best effort; fallback to legacy path
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.task},
        ]
        if self.tools_enabled:
            from core.tools import TOOLS
            ws = self.workspace or "~/.openclaw/workspace"
            self.result = _provider_infer_with_tools(messages, TOOLS, workspace=ws, adapter_path=self.adapter_path)
        else:
            self.result = _provider_infer(messages, max_new_tokens=8192, adapter_path=self.adapter_path)
        if self.output_path:
            try:
                p = Path(self.output_path).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.result, encoding="utf-8")
            except Exception as e:
                import logging; logging.getLogger(__name__).warning(f"[replica] output_path write failed: {e}")
        self.done = True
        with _lock:
            _replicas.pop(self.name, None)

    def message(self, user_text: str) -> str:
        """Send a message to a persistent replica and get a response."""
        if not self.persistent:
            return "This replica is not in persistent mode."
        # Prepend input_path content to first user message if set
        if self.input_path and not self.history:
            try:
                content = Path(self.input_path).expanduser().read_text(encoding="utf-8")
                user_text = f"{user_text}\n\n## Context\n{content}"
            except Exception:
                pass
        self.history.append({"role": "user", "content": user_text})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history[-20:]
        if self.tools_enabled:
            from core.tools import TOOLS
            ws = self.workspace or "~/.openclaw/workspace"
            reply = _provider_infer_with_tools(messages, TOOLS, workspace=ws, adapter_path=self.adapter_path)
        else:
            reply = _provider_infer(messages, max_new_tokens=8192, adapter_path=self.adapter_path)
        self.history.append({"role": "assistant", "content": reply})
        if self.output_path:
            try:
                p = Path(self.output_path).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(reply, encoding="utf-8")
            except Exception:
                pass
        return reply

    def stop(self):
        """Shut down a persistent replica."""
        with _lock:
            _replicas.pop(self.name, None)
        self.done = True


def _build_system_prompt(role: str, brief_path: Optional[str] = None, custom_prompt: Optional[str] = None) -> str:
    """Build system prompt from role + optional brief file."""
    base = custom_prompt or BUILTIN_ROLES.get(role, f"You are a {role} specialist.")
    if brief_path:
        try:
            path = Path(brief_path).expanduser()
            if path.exists():
                brief_content = path.read_text(encoding="utf-8")
                base = f"{base}\n\n## Your Brief\n\n{brief_content}"
        except Exception:
            pass
    return base


def can_spawn() -> bool:
    free = vram_free_mb()
    # If free == 0, assume CPU mode (no GPU) — no VRAM constraint applies
    vram_ok = (free == 0) or (free > CONTEXT_BUDGET_MB)
    return len(_replicas) < MAX_REPLICAS and vram_ok


def spawn(
    role: str,
    task: str,
    name: Optional[str] = None,
    adapter_path: Optional[str] = None,
) -> Optional["Replica"]:
    """Backward-compat: spawn a fire-and-forget task replica."""
    _name = name or f"{role}_{len(_replicas)}"
    with _lock:
        if not can_spawn():
            return None
        if _name in _replicas:
            return _replicas[_name]  # already running
        system = _build_system_prompt(role)
        r = Replica(name=_name, role=role, system_prompt=system, task=task, persistent=False, adapter_path=adapter_path)
        _replicas[_name] = r
    r.start()
    return r


def spawn_named(
    name: str,
    role: str = "custom",
    brief_path: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    workspace: Optional[str] = None,
    tools_enabled: bool = False,
    output_path: Optional[str] = None,
    input_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    slot: Optional[str] = None,
) -> Optional["Replica"]:
    """Spawn a named persistent replica (for client-facing meetings)."""
    with _lock:
        if name in _replicas:
            return _replicas[name]  # already running, return existing

        # Named replicas are long-lived conversation agents; keep a separate cap so
        # transient workers do not block them in live-server scenarios.
        persistent_count = sum(1 for rp in _replicas.values() if rp.persistent)
        if persistent_count >= MAX_PERSISTENT_REPLICAS:
            return None

        free = vram_free_mb()
        vram_ok = (free == 0) or (free > CONTEXT_BUDGET_MB)
        if not vram_ok:
            return None

        system = _build_system_prompt(role, brief_path, custom_prompt)
        r = Replica(
            name=name,
            role=role,
            system_prompt=system,
            persistent=True,
            brief_path=brief_path,
            workspace=workspace,
            tools_enabled=tools_enabled,
            output_path=output_path,
            input_path=input_path,
            adapter_path=adapter_path,
            slot=slot or "primary",
        )
        _replicas[name] = r
    r.start()
    return r


def get(name: str) -> Optional["Replica"]:
    return _replicas.get(name)


def stop(name: str) -> bool:
    r = _replicas.get(name)
    if r:
        r.stop()
        return True
    return False


def active() -> list["Replica"]:
    return list(_replicas.values())


def pipeline(stages: list, config: dict = None) -> dict:
    """ADR-010: Programmatic pipeline — run sequential stages without HTTP.

    Each stage is a dict with keys matching PipelineStage model fields:
      name, role, brief, task, tools, workspace, input_from, output_path
    Optionally: draft_first (bool) — use drafter for initial stage generation.

    Returns dict of {stage_name: output_text}.
    """
    results: dict[str, str] = {}
    stage_outputs: dict[str, str] = {}

    for stage in stages:
        # Support both dict and dataclass-like objects
        if hasattr(stage, "__dict__"):
            s = stage.__dict__
        else:
            s = stage

        name = s.get("name", "stage")
        role = s.get("role", "custom")
        brief = s.get("brief", "")
        task_text = s.get("task", "Begin your work.")
        tools_enabled = s.get("tools", False)
        workspace = s.get("workspace")
        input_from = s.get("input_from")
        output_path = s.get("output_path")
        draft_first = s.get("draft_first", False)

        # Inject previous stage output if input_from specified
        if input_from and input_from in stage_outputs:
            task_text = (
                f"{task_text}\n\n"
                f"## Output from '{input_from}' stage\n"
                f"{stage_outputs[input_from]}"
            )

        system_prompt = _build_system_prompt(role, custom_prompt=brief)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_text},
        ]

        reply = ""
        if draft_first:
            # Stage 1 drafter-first: fast draft, then main model completes
            try:
                from core.inference.model_client import infer_draft
                # Build simple prompt for drafter
                draft_prompt = f"{brief}\n\nTask: {task_text}"
                reply = infer_draft(draft_prompt, max_new_tokens=8192)
                if reply.startswith("[model_server") or reply.startswith("[model_client"):
                    reply = ""
            except Exception:
                reply = ""

        if not reply:
            if tools_enabled:
                from core.tools import TOOLS
                ws = workspace or "~/.openclaw/workspace"
                reply = _provider_infer_with_tools(messages, TOOLS, workspace=ws, adapter_path=s.get("adapter_path"))
            else:
                reply = _provider_infer(messages, max_new_tokens=8192, adapter_path=s.get("adapter_path"))

        if output_path:
            try:
                p = Path(output_path).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(reply, encoding="utf-8")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[replica.pipeline] output_path write failed: {e}")

        stage_outputs[name] = reply
        results[name] = reply

    return results
