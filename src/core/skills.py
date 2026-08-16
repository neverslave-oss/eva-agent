"""
skills.py — Load and execute SKILL.md files.
Reads frontmatter, exposes skill list, runs via model.infer().
"""
import os
import re
import yaml
from pathlib import Path
from core.memory.embedding_client import EmbeddingClient


def _parse_skill(path: Path) -> dict | None:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---(?:\n(.*))?", text, re.DOTALL)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except Exception:
        return None
    body_instructions = (m.group(2) or "").strip()
    frontmatter_instructions = str(fm.get("instructions", "") or "").strip()
    instructions = body_instructions or frontmatter_instructions
    return {
        "name": fm.get("name", path.parent.name),
        "description": fm.get("description", ""),
        "commands": fm.get("commands", fm.get("metadata", {}).get("commands", [])),
        "instructions": instructions,
        "path": str(path),
        "exec": fm.get("exec", None),  # direct script dispatch, bypasses LLM
        "intents": fm.get("intents", []),  # natural language intent phrases
        "command_only": fm.get("command_only", False),  # skip name/semantic matching if True
        "context_provider": bool(fm.get("context_provider", False)),  # auto-inject context before inference
        "context_search_cmd": fm.get("context_search_cmd", None),  # cmd template with {query} placeholder
    }


def _rglob_follow(base: Path, filename: str):
    """rglob that follows symlinks (Path.rglob doesn't in Python 3.12+)."""
    import os
    for root, dirs, files in os.walk(str(base), followlinks=True):
        if filename in files:
            yield Path(root) / filename


def _skill_priority(path_str: str, skills_dir: str) -> int:
    """Lower number = higher priority (wins deduplication).
    private/skills/ > community top-level copies > everything else.
    """
    rel = Path(path_str).relative_to(Path(skills_dir).expanduser()).as_posix()
    if rel.startswith("private/"):
        return 0
    return 1


def load_all(skills_dir="./skills", core_skills: list[str] | None = None) -> list[dict]:
    """Load all skills from skills_dir.
    If core_skills is provided, those skill names are sorted to the front
    so they are always matched first and never shadowed by evolved/Tier-2 skills.

    Deduplication: when the same skill name appears in multiple tiers,
    private/skills/ wins over top-level community copies. This ensures
    that kernel-specific evolutions of a skill always take precedence over
    the original workspace copy.
    """
    raw: list[dict] = []
    for skill_md in _rglob_follow(Path(skills_dir).expanduser(), "SKILL.md"):
        s = _parse_skill(skill_md)
        if s:
            raw.append(s)

    # Deduplicate by name: keep highest-priority copy (lowest priority number)
    seen: dict[str, dict] = {}
    for s in raw:
        name = s["name"].lower()
        prio = _skill_priority(s["path"], skills_dir)
        if name not in seen or prio < _skill_priority(seen[name]["path"], skills_dir):
            seen[name] = s
    skills = list(seen.values())

    if core_skills:
        _core_set = [n.lower() for n in core_skills]
        skills.sort(key=lambda s: (_core_set.index(s["name"].lower()) if s["name"].lower() in _core_set else len(_core_set)))
        # Tag core skills for system prompt context
        for s in skills:
            s["is_core"] = s["name"].lower() in _core_set
    return skills


def find(name: str, skills: list[dict]) -> dict | None:
    name = name.lower().strip()
    return next((s for s in skills if s["name"].lower() == name), None)


def find_semantic(query: str, skills: list[dict], embedding_client: EmbeddingClient) -> dict | None:
    """Find best matching skill using semantic similarity. Falls back to exact find()."""
    exact = find(query, skills)
    if exact:
        return exact

    best_skill = None
    best_score = 0.0
    THRESHOLD = 0.5  # minimum cosine similarity to consider a match

    for skill in skills:
        name = skill.get("name", "")
        desc = skill.get("instructions", "")[:200]  # first 200 chars of instructions
        commands = " ".join(str(c) for c in skill.get("commands", []))
        skill_text = f"{name} {desc} {commands}".strip()

        sim = embedding_client.similarity(query, skill_text)
        if sim is not None and sim > best_score:
            best_score = sim
            best_skill = skill

    if best_score >= THRESHOLD:
        return best_skill
    return None


def run(skill: dict, user_input: str, infer_fn) -> str:
    """Execute a skill using native function-calling when model is loaded."""
    import subprocess
    from core.inference.model import infer_with_tools, _model
    from core.tools import TOOLS

    # --- Direct exec dispatch (script-backed skills, no LLM) ---
    exec_template = skill.get("exec", None)
    if exec_template:
        # Substitute {args} with everything after the command trigger
        # e.g. "/markdown /path/to/file.pdf" → args = "/path/to/file.pdf"
        import shlex, re as _re
        args = user_input.strip()
        # Strip leading slash-command word if present
        parts = args.split(None, 1)
        if len(parts) > 1 and parts[0].startswith("/"):
            args = parts[1]
        elif len(parts) == 1 and parts[0].startswith("/"):
            args = ""
        # For natural-language inputs (from run_skill tool call), extract URL if present
        # so exec scripts receive a clean URL rather than a full sentence
        if not args.startswith("/") and "{args}" in exec_template:
            url_match = _re.search(r"https?://\S+", args)
            if url_match:
                args = url_match.group(0).rstrip(".,)")
        cmd = exec_template.replace("{args}", args).replace("{input}", user_input)
        skill_dir = str(Path(skill["path"]).parent)
        cmd = cmd.replace("{skill_dir}", skill_dir)
        # If the exec template has no placeholders, append args explicitly
        if args and "{args}" not in exec_template and "{input}" not in exec_template:
            import shlex as _shlex
            cmd = f"{cmd} {_shlex.quote(args)}"
        env = os.environ.copy()
        # Always inject KERNEL_SRC so scripts can find model_client
        env["KERNEL_SRC"] = str(Path(__file__).parent)
        # Load Kernel's own .env first (Telegram creds etc.)
        kernel_env = Path(skill["path"]).parent
        for _ in range(5):  # walk up max 5 levels looking for kernel .env
            kernel_env = kernel_env.parent
            candidate = kernel_env / ".env"
            if candidate.exists() and "KERNEL_EVO_TELEGRAM" in candidate.read_text():
                for line in candidate.read_text().splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        env.setdefault(k.strip(), v.strip())  # don't override existing
                break
        # Also load well-known Kernel-evolving .env path directly
        kernel_default_env = Path(".") / ".env"
        if kernel_default_env.exists():
            for line in kernel_default_env.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
        # Load skill .env if present (can override kernel env)
        env_path = Path(skill_dir) / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
        print(f"[skills] Direct exec: {cmd}")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=300, env=env, cwd=skill_dir
            )
            output = (result.stdout + result.stderr).strip()
            return output if output else "✅ Done."
        except subprocess.TimeoutExpired:
            return "❌ Script timed out after 300s."
        except Exception as e:
            return f"❌ Exec error: {e}"

    instructions = skill.get("instructions", "")
    name = skill.get("name", "skill")

    # Always use plain infer_fn when provided — avoids nested socket deadlock when
    # run() is called from inside model_server's own infer_with_tools loop.
    # infer_with_tools is only safe to call from the *top-level* agent path (not tool callbacks).
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
    ]
    return infer_fn(messages)
