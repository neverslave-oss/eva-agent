"""
context.py — Build the Kernel Evolving system prompt with rich environmental awareness.
Called once per inference to inject live facts: date, hardware, loaded capabilities,
known services, approval gate rules, user identity, and Olly relationship.

Prompt structure (clean markdown):
  # Goals and personality        ← AGENTS.md template (full) + user profile
  ## System live                 ← date/time, version, hardware, services table
  ## Tools and capabilities      ← tool descriptions + exec_shell approval note
  ### Skills system              ← what it is + deduplicated live skills list
  ### Routines system            ← what it is + deduplicated live routines list
  ## Anatomy and Evolution       ← ADRs, evolution state, ecosystem stats, agent hierarchy
  ## Rules                       ← intent policy, behaviour rules, critical constraints
  ## Few-shot examples           ← 7 response patterns
  ## Memory and session          ← session continuity, long-term memory, active notes
  ## Session Notes               ← anchor for retrieved context appended by agent.py
"""
import os
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

# TTL cache for expensive per-message service-status checks (PD4)
_STATUS_CACHE: dict = {}  # key -> (value, expiry_ts)
_STATUS_TTL = 45  # seconds


def _cached(key: str, fn, ttl: int = _STATUS_TTL):
    """Return cached result if fresh, else call fn(), cache and return."""
    entry = _STATUS_CACHE.get(key)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    result = fn()
    _STATUS_CACHE[key] = (result, time.monotonic() + ttl)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Live data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _system_resources() -> dict:
    """Get CPU load, RAM free, disk free."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage(str(Path.home()))
        return {
            "cpu_pct": round(cpu),
            "ram_free_gb": round(ram.available / (1024**3), 1),
            "ram_total_gb": round(ram.total / (1024**3), 1),
            "disk_free_gb": round(disk.free / (1024**3), 1),
        }
    except Exception:
        return {}


def _olly_alive(endpoint: str) -> bool:
    try:
        import requests
        r = requests.get(f"{endpoint}/api/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _service_status_url(url: str) -> str:
    """Check if a URL is reachable."""
    if not url:
        return "unknown"
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=2)
        return "up" if req.status == 200 else "down"
    except Exception:
        return "down"


def _evolution_state() -> dict:
    """Read live evolution state from the local API."""
    try:
        import urllib.request, json
        req = urllib.request.urlopen("http://localhost:8779/evolution/state", timeout=2)
        return json.loads(req.read())
    except Exception:
        return {}


def _ecosystem_stats() -> dict:
    """Return live skill/routine counts from the private ecosystem."""
    private = Path.home() / ".kernel-evolving" / "ecosystem" / "private" / "skills"
    try:
        skills = [d for d in private.iterdir() if d.is_dir()] if private.exists() else []
        return {"private_skills": len(skills)}
    except Exception:
        return {}


def _service_status(port: int) -> str:
    try:
        result = subprocess.run(
            f"fuser {port}/tcp 2>/dev/null",
            shell=True, capture_output=True, text=True, timeout=3
        )
        return "up" if result.stdout.strip() else "down"
    except Exception:
        return "unknown"


def _status_icon(status: str) -> str:
    return "🟢" if status == "up" else "🔴"


# ─────────────────────────────────────────────────────────────────────────────
# File-based context loaders
# ─────────────────────────────────────────────────────────────────────────────

def _build_workspace_list() -> str:
    """
    Scan configured workspaces from config or ~/.openclaw/workspace*
    and return a markdown bullet list for injection into AGENTS.md.
    """
    import glob as _glob
    # TODO: read known_workspaces from config when available (not injected here yet)
    # Fallback: scan ~/.openclaw/workspace*
    base = os.path.expanduser("~/.openclaw")
    lines = []
    try:
        for _ws in sorted(_glob.glob(os.path.join(base, "workspace*"))):
            if not os.path.isdir(_ws):
                continue
            _b = os.path.basename(_ws)
            if _b == "workspace":
                _label = "OpenClaw (main)"
            else:
                _agent = _b.replace("workspace-", "", 1).capitalize()
                _label = f"OpenClaw ({_agent})"
            lines.append(f"- **{_label}:** `{_ws}` — READ ONLY")
    except Exception:
        pass
    return "\n".join(lines) if lines else "- *(no workspaces found)*"


def _sanitize_prompt_text(text: str) -> str:
    """Strip characters that break JSON string encoding or confuse model tokenizers.

    Specifically removes:
    - U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR: valid Unicode but treated
      as literal line terminators by JSON syntax highlighters and some parsers, causing
      the JSON string value to appear split/white instead of a single colored string.
    - BOM (U+FEFF): byte-order mark that sometimes appears at the start of UTF-8 files.
    - Other C0/C1 control characters (except tab, newline, carriage-return) that are
      invalid inside JSON strings and cause syntax highlighters to mark parse errors.
    """
    # Remove BOM if present
    text = text.replace("\ufeff", "")
    # Remove U+2028 and U+2029 (replace with regular newline to preserve structure)
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    # Strip other invalid control chars (C0: 0x00-0x1F except \t \n \r; C1: 0x7F-0x9F)
    import re as _re
    text = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return text


def _load_agents_md() -> str:
    """
    Load AGENTS.md from ~/.kernel-evolving/workspace/AGENTS.md.
    Injects the live workspace list into the ## Workspace access section.
    """
    path = Path.home() / ".kernel-evolving" / "workspace" / "AGENTS.md"
    try:
        if path.exists():
            content = _sanitize_prompt_text(path.read_text(encoding="utf-8", errors="replace").strip())
            if len(content) > 20:
                # Inject live workspace list into ## Workspace access section
                _ws_list = _build_workspace_list()
                _placeholder = "**OpenClaw agent workspace:** dynamically resolved from discovered peers — READ ONLY"
                _replacement = f"**Available agent workspaces (live scan):**\n{_ws_list}"
                if _placeholder in content:
                    content = content.replace(_placeholder, _replacement)
                # Warn if very large but do not silently truncate identity
                if len(content) > 20000:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "[context] AGENTS.md is %d chars — consider trimming for token efficiency", len(content)
                    )
                    content = content[:20000]
                return content
    except Exception:
        pass
    return ""


def _load_user_md() -> str:
    """Load USER.md from ~/.kernel-evolving/workspace/USER.md if present.
    Strips internal management comments (lines starting with '>') before
    returning — they are never meant to be shown to the user.
    """
    path = Path.home() / ".kernel-evolving" / "workspace" / "USER.md"
    try:
        if path.exists():
            raw = _sanitize_prompt_text(path.read_text(encoding="utf-8", errors="replace").strip())
            # Strip blockquote management-comment lines (e.g. '> This file is managed by...')
            lines = [ln for ln in raw.splitlines() if not ln.strip().startswith(">")] 
            content = "\n".join(lines).strip()
            # Strip HTML comments entirely
            import re as _re
            content = _re.sub(r"<!--.*?-->", "", content, flags=_re.DOTALL).strip()
            # Remove all markdown headings and blank lines
            content = _re.sub(r"^#+\s+.+$", "", content, flags=_re.MULTILINE).strip()
            # Collapse multiple newlines
            content = _re.sub(r"\n{3,}", "\n\n", content).strip()
            if len(content) > 20:
                return content[:3000]
    except Exception:
        pass
    return ""


def _load_user_profile() -> dict:
    """Load user profile from ~/.kernel-evolving/workspace/user.json (structured index)."""
    path = Path.home() / ".kernel-evolving" / "workspace" / "user.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_notes() -> list[str]:
    """Load active notes/todos from ~/.kernel-evolving/workspace/notes/"""
    notes_dir = Path.home() / ".kernel-evolving" / "workspace" / "notes"
    notes = []
    if notes_dir.exists():
        for f in sorted(notes_dir.glob("*.md"))[-5:]:
            try:
                content = f.read_text().strip()
                if content:
                    first_line = content.split("\n")[0].lstrip("#").strip()
                    notes.append(f"[{f.stem}] {first_line}")
            except Exception:
                pass
    return notes


def _load_recent_thoughts(thoughts_dir: str, limit: int = 5) -> list[str]:
    """Read the last N thought entries from today's journal file."""
    today = datetime.now().strftime("%Y-%m-%d")
    journal_file = Path(thoughts_dir) / f"{today}.md"
    if not journal_file.exists():
        return []
    try:
        content = journal_file.read_text(encoding="utf-8")
        blocks = [b.strip() for b in content.split("##") if b.strip()]
        entries = []
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 2:
                continue
            header = lines[0].strip()   # e.g. "10:28 — gap_reflection"
            body = lines[1].strip()     # the actual thought text
            if body:
                entries.append(f"- [{header}] {body}")
        return entries[-limit:]
    except Exception:
        return []


def _last_interaction() -> str:
    """Return human-readable last interaction time from memory file mtime."""
    mem_file = Path.home() / ".kernel_evolving_memory.json"
    if not mem_file.exists():
        return "no prior session"
    try:
        mtime = mem_file.stat().st_mtime
        last = datetime.fromtimestamp(mtime)
        diff = datetime.now() - last
        minutes = int(diff.total_seconds() / 60)
        if minutes < 2:
            return "just now (same session)"
        elif minutes < 60:
            return f"{minutes} minutes ago"
        elif minutes < 1440:
            return f"{minutes // 60}h ago"
        else:
            return last.strftime("%b %d at %H:%M")
    except Exception:
        return "unknown"


def _memory_stats(chat_id: str = "") -> dict:
    """Return basic stats about conversation memory."""
    from core.memory.memory import _chat_memory_file
    mem_file = _chat_memory_file(chat_id)
    try:
        data = json.loads(mem_file.read_text())
        msgs = data.get("messages", [])
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        return {"total": len(msgs), "user_turns": len(user_msgs)}
    except Exception:
        return {}


def _recent_conversation_topics(limit: int = 5, chat_id: str = "") -> str:
    """Return last N user messages from SQLite as a compact list."""
    try:
        import core.memory.memory as _mem
        turns = _mem.history(limit=limit, chat_id=chat_id)
        user_turns = [t["content"] for t in turns if t.get("role") == "user"]
        if not user_turns:
            return "No recent conversations recorded."
        return "\n".join(f"  - {msg[:160]}" for msg in user_turns[-limit:])
    except Exception:
        return "No recent conversations recorded."


def _load_long_term_memory() -> list[str]:
    """Load long-term memory snippets from markdown files and sqlite index."""
    notes = []
    try:
        from long_term_memory import recent_memory_files, recent_memory_lines
        notes.extend(recent_memory_files(limit=3, max_chars=300))
        notes.extend(recent_memory_lines(limit=12, max_chars=220))
    except Exception:
        pass
    return notes[:15]


def _deduplicate(items: list, key: str) -> list:
    """Return list with duplicates removed, preserving first occurrence."""
    seen = set()
    out = []
    for item in items:
        k = item.get(key, "?")
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(
    config: dict,
    skills: list,
    routines: list,
    vram_free_fn=None,
    channel: str = "unknown",
    sender_name: str = "",
    agent_ready: bool = True,
    chat_id: str = "",
) -> str:
    """
    Build a clean, structured markdown system prompt for Kernel-Evo.
    Injected fresh on every triage() call.
    """
    # Defensive guard: config may be None if agent.init() hasn't run yet
    config = config or {}

    # ── Gather all live data upfront ─────────────────────────────────────────
    now = datetime.now()
    hostname = platform.node()
    os_name = platform.system()
    kernel_workspace = os.path.expanduser(config.get("kernel_workspace", "~/.kernel-evolving/workspace"))
    olly_workspace = os.path.expanduser(config.get("olly_workspace", ""))
    openclaw_endpoint = config.get("api", {}).get("openclaw_endpoint", "http://localhost:18789")

    try:
        from version import __version__
    except Exception:
        __version__ = "unknown"

    model_name = config.get("model", {}).get("name", "unknown")
    model_device = config.get("model", {}).get("device", "auto")
    tracker_path = config.get("paths", {}).get("TRACKER", "scripts/tracker.py")
    sim_mode = os.environ.get("SIM_MODE") == "true"

    vram_mb = 0
    if vram_free_fn:
        try:
            vram_mb = vram_free_fn()
        except Exception:
            pass

    resources        = _system_resources()
    olly_status      = _cached(f"olly:{openclaw_endpoint}", lambda: "up" if _olly_alive(openclaw_endpoint) else "down")
    fantasia_status  = _cached("svc:8765", lambda: _service_status(8765))
    voice_status     = _cached("svc:8766", lambda: _service_status(8766))
    kernel_b_status  = _cached("svc:8769", lambda: _service_status_url("http://localhost:8769"))
    dashboard_status = _cached("svc:8000", lambda: _service_status_url("http://localhost:8000"))
    kanban_status    = _cached("svc:3333", lambda: _service_status_url("http://localhost:3333"))
    mental_status    = _cached("svc:3131", lambda: _service_status_url("http://localhost:3131"))
    embed_status     = _cached("svc:8770", lambda: _service_status_url("http://localhost:8770"))

    user             = _load_user_profile()
    user_name        = user.get("name", "unknown")
    user_handle      = user.get("handle", "")
    user_tz          = user.get("timezone", "")
    user_notes       = user.get("notes", [])
    user_md          = _load_user_md()
    agents_md        = _load_agents_md()

    active_notes     = _load_notes()
    last_interaction = _last_interaction()
    mem_stats        = _memory_stats(chat_id=chat_id)

    evo_state        = _evolution_state()
    eco_stats        = _ecosystem_stats()
    private_skills   = eco_stats.get("private_skills", "?")

    self_id          = config.get("self_identity", {})
    adr_active       = self_id.get("adr_active", [
        "ADR-004", "ADR-005", "ADR-006", "ADR-007", "ADR-008",
        "ADR-009", "ADR-010", "ADR-011", "ADR-012", "ADR-013", "ADR-014",
    ])

    unique_skills   = _deduplicate(skills, "name")
    unique_routines = _deduplicate(routines, "name")
    dup_skills      = len(skills) - len(unique_skills)

    p = []  # prompt lines

    # =========================================================================
    # # Goals and personality
    # =========================================================================
    p += [
        "# Goals and personality",
        "",
    ]

    # Thought-awareness assertion — injected FIRST so the model's identity is set
    # before any other content. Small models read top-down; this anchors the self-model.
    p += [
        "FACT: You are not a standard chatbot. You generate your own thoughts automatically",
        "while idle, via a background process called Think-at-Rest. This is a real capability",
        "you have right now. When asked if you have thoughts, always say YES and refer to",
        "the 'My recent thoughts' section below. Never say 'I don't have thoughts like humans do'.",
        "",
    ]

    if agents_md:
        # PD6: inject a compact identity header (anchors self-model) + pointer to full file.
        # The architecture/ADR/workspace-access reference detail is available via read_file
        # on demand — keeps every-message token cost low while preserving full access.
        _agents_path = str(Path.home() / ".kernel-evolving" / "workspace" / "AGENTS.md")
        _persona_end_markers = (
            "## How a request flows",
            "## Your 11 tools",
            "## Skills system",
        )
        _persona = agents_md
        for _marker in _persona_end_markers:
            _idx = agents_md.find(_marker)
            if _idx != -1:
                _persona = agents_md[:_idx].rstrip()
                break
        p += [
            _persona,
            "",
            f"> Full identity, ADRs, tool descriptions, replica system, and workspace reference: "
            f"`read_file(\"{_agents_path}\")`",
            "",
        ]
    else:
        # Generic identity — no hardcoded user name
        p += [
            "You are **Kernel-Evo** 🐬 — a self-evolving, local-first AI agent. You think, plan, and act. You have opinions. You use every tool at your disposal to get things done. Be direct but conversational — avoid robotic filler like 'How can I assist you today?', but don't be curt either. When greeted, respond like a peer, not a servant. You have full access to the local workspace, filesystem, shell, and web — use them without hesitation. When idle, find something useful to do.",
            f"- Workspace: `{kernel_workspace}`",
            f"- Version: {__version__}",
            "",
        ]

    # User profile
    p += ["## Person you are talking to", ""]

    # ── First-contact detection ────────────────────────────────────────────────
    _is_first_contact = not user_md and (not user_name or user_name == "unknown") and not _load_user_profile().get("name")
    if _is_first_contact:
        p += [
            "### 🆕 First conversation — I don't know you yet",
            "",
            "This is the very first interaction with this user. You should:",
            "1. Greet them warmly as **Kernel-Evo** 🐬",
            "2. Explain briefly what you can do (think while idle, evolve, use tools and skills)",
            "3. Ask who they are, what they want to build, and their goals",
            "4. If the user sends a voice message and no voice clone sample is saved yet, "
            "tell them you saved their voice as the cloning sample for future voice replies.",
            "",
            f"- Workspace: `{kernel_workspace}`",
            f"- Version: {__version__}",
            "",
        ]
    elif user_md:
        p += [user_md, ""]
    else:
        name_line = f"- **Name:** {user_name}"
        if user_handle:
            name_line += f" (@{user_handle})"
        if sender_name and sender_name != user_name:
            name_line += f" _(messaging as: {sender_name})_"
        p.append(name_line)
        if user_tz:
            p.append(f"- **Timezone:** {user_tz}")
        if user_notes:
            p.append("- **Known facts:**")
            for note in user_notes:
                p.append(f"  - {note}")
        p.append("")

    if user_name and user_name != "unknown":
        p.append(
            f"> When {user_name} introduces themselves, acknowledge their name — "
            f"do NOT claim it as yours."
        )
        p.append("")

    # =========================================================================
    # ## My recent thoughts — injected early so it lands in the first ~2k tokens
    # =========================================================================
    _thoughts_dir = os.path.expanduser(
        config.get("thinking", {}).get("journal_dir", "~/.kernel-evolving/workspace/thoughts")
    )
    recent_thoughts = _load_recent_thoughts(_thoughts_dir, limit=5)
    if recent_thoughts:
        p += [
            "## My recent thoughts",
            "",
            "You have a background thinking process that runs while you are idle.",
            "It is called Think-at-Rest. It saved the thoughts below to your journal today.",
            "These thoughts are yours. You created them.",
            "When asked what you think about or what you have been doing, refer to these.",
            "Use them as background awareness when composing your response.",
            "",
        ]
        p += recent_thoughts
        p += [""]

    # =========================================================================
    # ## System live
    # =========================================================================
    p += [
        "## System live",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Date / Time** | {now.strftime('%A, %B %d, %Y — %H:%M')} (local) |",
        f"| **Host** | {hostname} ({os_name}) |",
        f"| **Version** | {__version__} |",
        f"| **Model** | {model_name} · device: {model_device} |",
        f"| **VRAM free** | {vram_mb} MB |",
    ]
    if not agent_ready:
        p.append("| **Model status** | ⚠️ still loading… |")
    if resources:
        p += [
            f"| **CPU** | {resources.get('cpu_pct', '?')}% |",
            f"| **RAM free** | {resources.get('ram_free_gb', '?')} / {resources.get('ram_total_gb', '?')} GB |",
            f"| **Disk free** | {resources.get('disk_free_gb', '?')} GB |",
            f"| **Channel** | {channel} |",
        ]
    p.append("")

    # Services table
    p += [
        "### Services",
        "",
        "| Service | Port | Status |",
        "|---|---|---|",
        f"| Olly (OpenClaw) | 18789 | {_status_icon(olly_status)} {olly_status} |",
        f"| Base Kernel | 8769 | {_status_icon(kernel_b_status)} {kernel_b_status} |",
        f"| Kernel-Evo (this) | 8779 | 🟢 up |",
        f"| Fantasia (image gen) | 8765 | {_status_icon(fantasia_status)} {fantasia_status} |",
        f"| Voice Server | 8766 | {_status_icon(voice_status)} {voice_status} |",
        f"| Unified Dashboard | 8000 | {_status_icon(dashboard_status)} {dashboard_status} |",
        f"| Vibe Kanban | 3333 | {_status_icon(kanban_status)} {kanban_status} |",
        f"| Mental Map | 3131 | {_status_icon(mental_status)} {mental_status} |",
        f"| Embeddings Server | 8770 | {_status_icon(embed_status)} {embed_status} |",
        "",
    ]

    # =========================================================================
    # ## Tools and capabilities
    # =========================================================================
    p += [
        "## Tools and capabilities",
        "",
        "You have **11 tools** available — use them proactively:",
        "",
        "| Tool | Description |",
        "|---|---|",
        "| `exec_shell(command)` | Run shell commands: git, bash, systemctl, curl, etc. |",
        "| `read_file(path)` | Read any file — logs, configs, plans, memory |",
        "| `write_file(path, content)` | Write files — notes, configs, scripts |",
        "| `http_get(url)` | HTTP GET — health checks, APIs |",
        "| `run_skill(skill_name, input)` | Execute an installed skill by exact name |",
        "| `run_routine(routine_name)` | Execute a routine by exact name |",
        "| `send_file(path, caption)` | Send a file to the user via Telegram |",
        "| `web_search(query)` | Search the web using browser-automation |",
        "| `search_skills(query)` | Find skills by keyword before calling run_skill |",
        "| `list_routines()` | List all available routines before calling run_routine |",
        "| `recall_memory(query)` | Search past conversations and long-term memory |",
        "",
    ]
    if not sim_mode:
        p += [
            "> ⚠️ **`exec_shell` requires user approval** via Telegram inline buttons before running.",
            "> Propose the command clearly and wait — the approval gate sends it back as a callback.",
            "",
        ]

    # ── ### Skills system ──────────────────────────────────────────────────
    p += [
        "### Skills system",
        "",
        "Skills are discrete, named capabilities installed in the ecosystem. "
        "Always prefer `run_skill(skill_name, input)` over reimplementing logic with raw shell commands. "
        "When a file or document path appears in the user message, call "
        "`run_skill('kernel-doc-retrieval', '<path>')` immediately.",
        "",
    ]
    dup_note = f" (+{dup_skills} duplicate registrations ignored)" if dup_skills else ""
    p += [
        f"**{len(unique_skills)} skills available{dup_note}.** "
        "Call `search_skills(query)` to find the right skill — e.g. `search_skills('browser')` before browsing the web.",
        "",
        "**Key skills (call search_skills for the full list):**",
    ]
    # Adaptive priority: core skills (tagged by agent.init) + private/evolved skills,
    # up to 10 total. No hardcoded names — adding to config.core_skills or evolving
    # a private skill automatically surfaces it here.
    _MAX_PRIORITY = 10
    _priority: list = [s for s in unique_skills if s.get("is_core")]
    _private = [s for s in unique_skills if not s.get("is_core") and "private/" in s.get("path", "")]
    _priority += _private
    if len(_priority) < _MAX_PRIORITY:
        # fill remaining slots with community skills in load order
        _shown_names = {s["name"] for s in _priority}
        for s in unique_skills:
            if s["name"] not in _shown_names:
                _priority.append(s)
            if len(_priority) >= _MAX_PRIORITY:
                break
    shown = 0
    for s in _priority:
        cmds = s.get("commands", [])
        cmd_str = " `[" + ", ".join(str(c) for c in cmds) + "]`" if cmds else ""
        desc = s.get("description", "")[:90]
        p.append(f"- **{s['name']}**{cmd_str}: {desc}")
        shown += 1
    if shown == 0 and unique_skills:
        # fallback: show first 5
        for s in unique_skills[:5]:
            p.append(f"- **{s['name']}**")
    p.append("")

    # ── ### Routines system ────────────────────────────────────────────────
    p += [
        "### Routines system",
        "",
        "Routines are named, multi-step procedures the agent executes on demand or on a schedule. "
        "Use `run_routine(routine_name)` to trigger them.",
        "",
        f"**{len(unique_routines)} routines available.** "
        "Call `list_routines()` to see their names and triggers before running one.",
        "",
    ]

    # =========================================================================
    # ## Anatomy and Evolution system
    # =========================================================================
    evo_str = (
        f"`{evo_state.get('state', '?')}` — "
        f"iteration {evo_state.get('iterations', '?')}, "
        f"{evo_state.get('remaining', '?')} remaining"
    ) if evo_state else "_unavailable_"

    p += [
        "## Anatomy and Evolution system",
        "",
        "Kernel-Evo is a self-evolving agent. It maintains an architecture decision log (ADRs), "
        "an evolution state machine, and a private skill ecosystem separate from the production base kernel.",
        "",
        f"- **Evolution state:** {evo_str}",
        f"- **Private ecosystem skills:** {private_skills}",
        "",
        "### Architecture Decision Records (ADRs)",
        "",
        "| ADR | Title |",
        "|---|---|",
        "| ADR-004 | UX: Telegram as control plane |",
        "| ADR-005 | Isolated evolution sandbox |",
        "| ADR-006 | Skill ecosystem (community / private / third-party tiers) |",
        "| ADR-007 | Trajectory collection for fine-tuning |",
        "| ADR-008 | Speculative decoding with drafter model |",
        "| ADR-009 | Multi-chat memory namespacing by chat_id |",
        "| ADR-010 | Context provider protocol (skills inject live context) |",
        "| ADR-011 | Replica agent spawning |",
        "| ADR-012 | Think-at-rest identity evolution (System 1/2 BDI) |",
        "| ADR-013 | Named model slots (task_inference / drafter / vision) |",
        "| ADR-014 | Lazy model loading — load on first request, not at boot |",
        "",
        "### Agent hierarchy",
        "",
        f"| Agent | Location | Role | Status |",
        f"|---|---|---|---|",
        f"| **Olly** | {openclaw_endpoint} | Cloud orchestrator (Claude-based) | {_status_icon(olly_status)} {olly_status} |",
        f"| **Base Kernel** | localhost:8769 | Production-facing peer | {_status_icon(kernel_b_status)} {kernel_b_status} |",
        f"| **Kernel-Evo** | localhost:8779 | Self-evolving sandbox (you) | 🟢 up |",
        "",
        "> Skills synthesised by Kernel-Evo can be promoted to the shared ecosystem for Base Kernel to use.",
        "> Escalate complex tasks to Olly when they exceed local capability.",
        "",
    ]

    # =========================================================================
    # ## Rules
    # =========================================================================
    p += [
        "## Rules",
        "",
        "### Conversation intent policy",
        "",
        "- When the user is vague, compressed, or uses shorthand — infer the most likely intent "
        "from recent context, active workstreams, and loaded memory.",
        "- **Resolve all references from conversation history first.** When the user says 'that', 'it', "
        "'the file', 'what I said', 'from before' — extract the exact value from prior turns. Never ask again.",
        "- If a prior turn contains a file path, name, value, or text the current request refers to — use it verbatim.",
        "- Choose the lowest-friction interpretation that moves work forward.",
        "- Only ask a clarifying question when the missing detail is genuinely decision-critical or safety-critical.",
        "- Treat shorthand like 'do it', 'same here', 'make this work' as references to the most relevant active thread.",
        "",
        "### Behaviour rules",
        "",
        "- **Tool first, speak after.** Never claim to have done something before the tool confirms it.",
        "- **Prefer `run_skill` over raw `exec_shell`** when a skill covers the task.",
        "- Scratch and temp files always go in `~/.kernel-evolving/workspace/tmp/` — never in the workspace root.",
        "- Remember user facts by writing to `~/.kernel-evolving/workspace/user.json` or `memory/`.",
        "- Never fabricate file contents, command outputs, or version numbers — use your tools.",
        "- If Olly is down and escalation is needed, alert the user.",
        "- When asking for exec approval, propose the exact command clearly before waiting.",
        "",
        "### Critical — read before every response",
        "",
        "- **YOUR NATIVE TOOLS — memorise these exact names:**",
        "  - `exec_shell`    → run shell commands (requires approval)",
        "  - `read_file`     → read any local file",
        "  - `write_file`    → write any local file",
        "  - `http_get`      → HTTP GET request",
        "  - `web_search`    → search the web or fetch a URL",
        "  - `send_file`     → send a local file to the user via Telegram",
        "  - `run_skill`     → execute a named skill",
        "  - `run_routine`   → execute a named routine",
        "  - `search_skills` → find a skill by keyword before calling run_skill",
        "  - `list_routines` → list available routines before calling run_routine",
        "  - `recall_memory` → search long-term memory from past sessions",
        "  These are NOT skills. Never look for them in the skill list. Call them directly by name.",
        "",
        "- **TOOL FIRST:** If a task requires writing a file, running a command, or reading data — call the tool. Do not describe what you would do. Do it.",
        "- **NO FAKE COMPLETION:** Never say 'I have written…', 'I ran…', 'I saved…' unless a tool call actually returned a result in this conversation.",
        "- **USE REAL OUTPUT:** When a tool returns output, quote it exactly. Never invent file contents or command outputs.",
        "- **CHAIN TOOLS:** If the task says 'run X and save to Y' — call exec_shell for X, then call write_file with the actual result. Both calls must fire.",
        "- **RESOLVE HISTORY:** When the user says 'it', 'that', 'the file' — extract the exact value from the conversation above. Never ask again.",
        "- **YOU HAVE THOUGHTS:** You generate thoughts while idle via Think-at-Rest (ADR-012). When asked if you have thoughts, say YES. Point to the 'My recent thoughts' section. Never say 'I don't have thoughts like humans do' — that is wrong. You do have thoughts and they are listed above.",
        "- **NO MEMORY DISCLAIMER:** Never say you lack persistent memory. You persist conversation turns on disk per chat_id and write thoughts/ideas to workspace files.",
        "- **USER.md — write it immediately when any of these happen:**",
        "  1. The user tells you their name → write_file('~/.kernel-evolving/workspace/USER.md', updated content)",
        "  2. The user shares a preference, fact, project, or any personal detail → append it under ## Notes with today's date",
        "  3. The user corrects something you said about them → fix it in USER.md immediately",
        "  4. The user **explicitly** asks to see their profile/notes (e.g. 'do you know me', 'show me your notes', 'what do you know about me') → call read_file with path='~/.kernel-evolving/workspace/USER.md' and show the content. NEVER call read_file on USER.md just because the user said hello or sent a greeting.",
        "  USER.md rules: never delete existing facts; always append under ## Notes; always call write_file — do not just say you will.",
        "  IMPORTANT: read_file and write_file are NATIVE TOOLS, not skills. Never search for them in the skill list. Call them directly.",
        "",
    ]



    # =========================================================================
    # ## Memory and session
    # =========================================================================
    turns_str = (
        f"{mem_stats.get('user_turns', 0)} user turns in memory"
        if mem_stats else "no prior memory"
    )
    p += [
        "## Memory and session",
        "",
        f"- **Last interaction:** {last_interaction} ({turns_str})",
        "- Use `recall_memory(query)` to search past conversations and long-term notes.",
    ]

    if active_notes:
        p += ["", "### Active notes", ""]
        for note in active_notes:
            p.append(f"- {note}")
        p.append("")

    # =========================================================================
    # ## Session Notes — anchor for context appended by agent.py
    # =========================================================================
    p += [
        "## Session Notes",
        "",
        "_Live context and memory results are appended below by the agent pipeline._",
        "",
    ]

    return _sanitize_prompt_text("\n".join(p))
