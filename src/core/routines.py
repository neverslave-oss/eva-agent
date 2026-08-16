"""routines.py — Load and execute ROUTINE.md files.
Hybrid execution engine:
  1. Inline shell blocks (```bash) → run directly via subprocess (no model, no deadlock)
  2. Inline commands in bullet steps (Run: `cmd`) → same
  3. Skill references from frontmatter skills: [...] → tools.execute_tool('run_skill')
  4. Remaining prose → summarised via infer_fn (plain text, safe inside tool loop)
  5. Final reply assembled from all step results
"""
import re
import os
import yaml
import subprocess
from pathlib import Path


def _parse_routine(path: Path) -> dict | None:
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except Exception:
        return None
    return {
        "name": fm.get("name", path.parent.name),
        "description": fm.get("description", ""),
        "trigger": fm.get("trigger", {}),
        "skills": fm.get("skills", []),
        "body": m.group(2).strip(),
        "path": str(path),
    }


def load_all(routines_dir="./routines") -> list[dict]:
    routines = []
    for r_md in Path(routines_dir).rglob("ROUTINE.md"):
        r = _parse_routine(r_md)
        if r:
            routines.append(r)
    return routines


def find(name: str, routines: list[dict]) -> dict | None:
    q = name.lower().strip()
    # 1. exact
    for r in routines:
        if r["name"].lower() == q:
            return r
    # 2. prefix (either direction)
    for r in routines:
        rn = r["name"].lower()
        if rn.startswith(q) or q.startswith(rn):
            return r
    # 3. substring
    for r in routines:
        rn = r["name"].lower()
        if q in rn or rn in q:
            return r
    # 4. slug normalisation (strip hyphens + underscores)
    q_slug = q.replace("-", "").replace("_", "")
    for r in routines:
        if r["name"].lower().replace("-", "").replace("_", "") == q_slug:
            return r
    return None


def _extract_shell_blocks(body: str) -> list[str]:
    """Extract bash/shell code blocks from routine body."""
    pattern = r'```(?:bash|shell)\n(.*?)```'
    return re.findall(pattern, body, re.DOTALL)


_CHECK_SERVICE_MAP: dict[str, str] = {
    "fantasia": "http://localhost:8765/health",
    "kernel": "http://localhost:8769/health",
    "kernel-evolving": "http://localhost:8779/health",
    "voice": "http://localhost:8766/health",
    "olly": "http://localhost:18789/api/health",
}


def _extract_check_patterns(body: str) -> list[str]:
    """Extract ``- check: <service_or_url>`` lines and map to curl commands.

    Known services are mapped to their health-check URLs. Values that already
    start with ``http`` are curled directly. Lines where ``check:`` appears
    mid-sentence (prose) are skipped.
    """
    pattern = r'^\s*[-*]\s*check:\s*(.+)$'
    cmds: list[str] = []
    for match in re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE):
        value = match.group(1).strip()
        # Skip prose: if there are spaces *before* a colon continuation, it's
        # likely a sentence (e.g. "routine check: send note").
        # We accept only single-token names or URLs (no embedded spaces).
        if ' ' in value:
            continue
        if value.lower().startswith('http'):
            url = value
            label = url.upper()
            cmds.append(f"curl -sf {url} && echo '{label} UP' || echo '{label} DOWN'")
        else:
            mapped = _CHECK_SERVICE_MAP.get(value.lower())
            if mapped:
                label = value.upper()
                cmds.append(f"curl -sf {mapped} && echo '{label} UP' || echo '{label} DOWN'")
    return cmds


def _extract_inline_commands(body: str) -> list[str]:
    """Extract inline shell commands from routine bullet steps.

    Matches:
      - Run: `cmd`
      - Execute: `cmd`
      - Check Fantasia: `curl ...`   (label between keyword and backtick)

    Skips non-command file references:
      - Check if `some/file.md` was updated  (file check, not a shell cmd)
    Skips check: patterns (handled separately by _extract_check_patterns).
    """
    # Match bullet lines starting with run/execute/check, capturing backtick content
    # Exclude lines that are purely a ``check: <value>`` pattern (no backticks)
    pattern = r'^\s*[-*]\s*(?:run|execute|check)[^\n]*?`([^`\n]+)`'
    candidates = re.findall(pattern, body, re.IGNORECASE | re.MULTILINE)

    out: list[str] = []
    seen: set[str] = set()
    for cmd in candidates:
        cmd = cmd.strip()
        # Skip obvious file/path references that aren't shell commands
        if not cmd:
            continue
        if cmd.startswith(('memory/', '{{DIGEST', '~/.openclaw/media/')):
            continue
        # Skip bare file paths (no spaces, ends with an extension) — not a shell command
        if ' ' not in cmd and ('.' in cmd.split('/')[-1]):
            continue
        # Skip token-only references like {{DIGEST_HTML}} (no spaces = not a command)
        if cmd.startswith('{{') and ' ' not in cmd:
            continue
        if cmd in seen:
            continue
        seen.add(cmd)
        out.append(cmd)
    return out


def _extract_context(body: str) -> str:
    """Extract non-code instruction text from routine body (for LLM summary)."""
    steps_match = re.search(r'## Steps(.*?)(?=\n## |\Z)', body, re.DOTALL)
    if not steps_match:
        return body
    steps = steps_match.group(1)
    clean = re.sub(r'```.*?```', '', steps, flags=re.DOTALL)
    clean = re.sub(r'#{1,4} ', '', clean)
    return clean.strip()


def _run_shell(cmd: str, timeout: int = 30) -> str:
    """Run a shell command safely, return stdout+stderr truncated.
    Injects path registry values as environment variables so commands
    like `python $TRACKER todo list` work without hardcoded paths.
    """
    _load_path_registry()
    # Resolve {{VAR}} tokens in the command itself
    cmd = _resolve_tokens(cmd)
    env = {**os.environ, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(_PATH_REGISTRY)  # inject all paths as env vars
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        combined = out
        if err and not out:
            combined = err
        elif err:
            combined = out + "\n[stderr] " + err[:200]
        return combined[:1200] if combined else "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as e:
        return f"(error: {e})"


from core.tools import WORKSPACE as _DEFAULT_WORKSPACE, execute_tool

KERNEL_WORKSPACE = _DEFAULT_WORKSPACE

# ── Path registry (loaded once from config) ───────────────────────────────────
_PATH_REGISTRY: dict[str, str] = {}


def _load_path_registry():
    """Load paths block from config.yaml into the module-level registry."""
    global _PATH_REGISTRY
    if _PATH_REGISTRY:
        return  # already loaded
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.yaml')
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get('paths', {})
        _PATH_REGISTRY = {k: os.path.expanduser(str(v)) for k, v in raw.items()}
        # Also expose olly_workspace under a canonical key
        if 'OLLY_WORKSPACE' not in _PATH_REGISTRY and cfg.get('olly_workspace'):
            _PATH_REGISTRY['OLLY_WORKSPACE'] = os.path.expanduser(cfg['olly_workspace'])
    except Exception:
        pass


def _resolve_tokens(text: str) -> str:
    """Replace {{VAR}} tokens in text with values from the path registry."""
    _load_path_registry()
    for key, val in _PATH_REGISTRY.items():
        text = text.replace('{{' + key + '}}', val)
    return text


def run(routine: dict, infer_fn, workspace: str = None, step_callback=None) -> str:
    """Hybrid routine executor.

    Priority order per step type:
      1. Shell blocks (```bash) and inline Run: cmds → direct subprocess
      2. Skill references from frontmatter → tools.execute_tool('run_skill')
      3. Context/prose → brief LLM summary via infer_fn (plain, no tools, no deadlock)

    step_callback(step_num, label, result): called after each shell step completes
      so the caller can stream progress back to the user without waiting for the
      full routine to finish. If None, all output is batched to the final summary.
    """
    if workspace is None:
        workspace = KERNEL_WORKSPACE

    body = routine.get("body", routine.get("instructions", ""))
    name = routine.get("name", "routine")
    frontmatter_skills = routine.get("skills", [])

    # Resolve {{VAR}} tokens in the body so commands contain real paths
    body = _resolve_tokens(body)
    step_results: list[str] = []

    # ── 1. Run all shell code blocks directly ─────────────────────────────────
    shell_blocks = _extract_shell_blocks(body)
    for i, cmd in enumerate(shell_blocks, 1):
        cmd = cmd.strip()
        if not cmd:
            continue
        out = _run_shell(cmd)
        label = f"{cmd[:50]}.." if len(cmd) > 50 else cmd
        step_results.append(f"**Shell step {i}** (`{cmd[:60]}`...):\n{out}")
        if step_callback:
            step_callback(i, label, out)

    # ── 2. Run inline Run: `cmd` patterns ─────────────────────────────────────
    inline_cmds = _extract_inline_commands(body)
    block_text = " ".join(shell_blocks)
    _step_offset = len(shell_blocks)
    for j, cmd in enumerate(inline_cmds, 1):
        cmd = cmd.strip()
        if not cmd or cmd in block_text:  # skip duplicates already in shell blocks
            continue
        out = _run_shell(cmd)
        label = f"{cmd[:50]}.." if len(cmd) > 50 else cmd
        step_results.append(f"**`{cmd[:60]}`**:\n{out}")
        if step_callback:
            step_callback(_step_offset + j, label, out)

    # ── 2b. Run check: patterns ──────────────────────────────────────────────
    check_cmds = _extract_check_patterns(body)
    _check_offset = _step_offset + len(inline_cmds)
    for jj, cmd in enumerate(check_cmds, 1):
        cmd = cmd.strip()
        if not cmd:
            continue
        out = _run_shell(cmd)
        label = f"{cmd[:50]}.." if len(cmd) > 50 else cmd
        step_results.append(f"**check** (`{cmd[:60]}`):\n{out}")
        if step_callback:
            step_callback(_check_offset + jj, label, out)

    # ── 3. Dispatch declared skills from frontmatter ───────────────────────────
    _skill_offset = _check_offset + len(check_cmds)
    for k, skill_name in enumerate(frontmatter_skills, 1):
        try:
            out = execute_tool("run_skill", {"skill_name": skill_name, "input": f"Run for {name} routine"})
            step_results.append(f"**Skill `{skill_name}`**:\n{out[:600]}")
            if step_callback:
                step_callback(_skill_offset + k, f"skill:{skill_name}", out[:200])
        except Exception as e:
            step_results.append(f"**Skill `{skill_name}`**: error — {e}")
            if step_callback:
                step_callback(_skill_offset + k, f"skill:{skill_name}", f"error: {e}")

    # ── 4. LLM summary over collected results + prose context ──────────────────
    context = _extract_context(body)
    collected = "\n\n".join(step_results) if step_results else "(no shell output — prose-only routine)"

    summary_prompt = [
        {
            "role": "system",
            "content": (
                f"You are Kernel. You just executed the '{name}' routine. "
                "Format the results as a clean, concise briefing. "
                "Use the actual command outputs below — do not simulate or invent data. "
                "If a section had no output, say 'unavailable' and move on."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Routine goal:\n{context[:800]}\n\n"
                f"Actual outputs from execution:\n{collected[:2000]}\n\n"
                "Summarise the results in the output format specified in the routine. "
                "Be concise. Use real data only."
            ),
        },
    ]
    try:
        result = infer_fn(summary_prompt)
    except Exception as e:
        result = f"(summary failed: {e})\n\nRaw outputs:\n{collected}"

    # ── 5. Log to evolution DB ─────────────────────────────────────────────────
    try:
        from core.evolution.evolution_log import EvolutionLog
        EvolutionLog().record_activity(
            event_type="routine_execution",
            task=f"Routine executed: {name}",
            entity_name=name,
            provider_used="hybrid",
            found=True,
            confidence="EXEC",
            meta={
                "routine_path": routine.get("path", ""),
                "shell_steps": len(shell_blocks),
                "inline_steps": len(inline_cmds),
                "skills": frontmatter_skills,
            },
        )
    except Exception:
        pass

    return result
