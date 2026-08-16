"""
setup.py — First-boot workspace setup for Kernel-Evolving.
Run once when ~/.kernel-evolving/workspace doesn't exist,
and refresh_identity_files() on every boot to sync AGENTS.md / SOUL.md from repo.
"""
import json
import logging
from pathlib import Path
from runtime_paths import ensure_runtime_dirs

logger = logging.getLogger(__name__)

KERNEL_HOME = Path.home() / ".kernel-evolving"
WORKSPACE   = KERNEL_HOME / "workspace"
MEMORY_FILE = KERNEL_HOME / "memory.json"
CONFIG_DIRS = [
    "memory",
    "tmp",
    "notes",
    "scripts",
    "thoughts",
    "thoughts/ideas",
    "data",
    "artifacts",
    "artifacts/eval_runs",
    "artifacts/eval_results",
    "artifacts/trajectories",
    "artifacts/finetune",
    "artifacts/adapter_active",
    "logs",
    "runtime",
    "documents",
]

# Repo root — two levels up from this file (src/setup.py → repo root)
_REPO_ROOT = Path(__file__).parent.parent.parent


def setup_workspace() -> bool:
    """Create the workspace on first boot. Idempotent — returns False if already exists."""
    if WORKSPACE.exists():
        return False  # already set up

    print("[setup] First boot — creating Kernel workspace...")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dirs()
    for d in CONFIG_DIRS:
        (WORKSPACE / d).mkdir(parents=True, exist_ok=True)

    # Write deprecated IDENTITY.md redirect notice (not used at runtime)
    (WORKSPACE / "IDENTITY.md").write_text(
        "This file is deprecated. See AGENTS.md for agent identity.\n",
        encoding="utf-8",
    )

    # Write README
    (WORKSPACE / "README.md").write_text(
        "# Kernel Evolving Workspace\nKernel's private working directory.\n",
        encoding="utf-8",
    )

    print(f"[setup] Workspace created at {WORKSPACE}")
    return True


def _ensure_tmp_dir() -> None:
    """Ensure tmp/ sub-directory exists."""
    tmp = WORKSPACE / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)


def _copy_file_if_absent(src: Path, dst: Path) -> bool:
    """Copy src → dst only if dst is absent or empty. Returns True if copied."""
    if dst.exists():
        try:
            content = dst.read_text(encoding="utf-8").strip()
            if len(content) > 20:
                return False  # already present and non-trivial
        except Exception:
            return False
    if src.exists():
        try:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return True
        except Exception as exc:
            logger.warning("[setup] Could not copy %s → %s: %s", src, dst, exc)
    return False


def _fix_stale_identity_md() -> None:
    """Replace stale IDENTITY.md (pointing to localhost:8769 / base Kernel) with redirect notice."""
    identity_path = WORKSPACE / "IDENTITY.md"
    if not identity_path.exists():
        return
    try:
        text = identity_path.read_text(encoding="utf-8")
        stale = "localhost:8769" in text or (
            "Name: Kernel\n" in text and "Evolving" not in text
        )
        if stale:
            identity_path.write_text(
                "This file is deprecated. See AGENTS.md for agent identity.\n",
                encoding="utf-8",
            )
            logger.info("[setup] Replaced stale IDENTITY.md with redirect notice")
    except Exception as exc:
        logger.warning("[setup] Could not check IDENTITY.md: %s", exc)


def _ensure_user_profile() -> None:
    """Non-destructively upgrade user.json with missing fields (defaults).

    Schema is designed to be general enough to work across user setups while
    giving the agent structured fields it can read and update as it learns from
    interactions. All fields are optional — agents should treat unknown keys as
    opaque extensions.
    """
    path = WORKSPACE / "user.json"
    defaults: dict = {
        # ── Identity ───────────────────────────────────────────────────────────
        "name": "",                     # full name or preferred name
        "handle": "",                   # username / alias
        "timezone": "UTC",              # IANA timezone, e.g. 'Europe/Berlin'
        "language": "en",               # ISO 639-1 primary language
        "locale": "",                   # e.g. 'en-GB' — for formatting / date style

        # ── Communication preferences ──────────────────────────────────────────
        "preferred_channel": "",        # e.g. 'telegram', 'discord', 'cli'
        "reply_style": "balanced",       # 'concise' | 'balanced' | 'verbose' | 'technical'
        "proactive": True,              # agent may send unsolicited messages/insights

        # ── Domain context (agent fills as it learns) ──────────────────────────
        "occupation": "",               # e.g. 'software developer', 'designer'
        "projects": [],                 # list of active project names/slugs
        "interests": [],                # topics the user cares about
        "skills": [],                   # known technical/professional skills

        # ── Learning journal (agent appends; human may edit) ──────────────────
        "notes": [],                    # free-form facts the agent has observed
        "preferences": {},              # key→value map of learned preferences
        "corrections": [],              # [{"wrong": "...", "right": "..."}] — agent mistakes corrected by user

        # ── Environment ───────────────────────────────────────────────────────
        "devices": [],                  # e.g. ['MSI laptop (WSL2)', 'Android phone']
        "os": "",                       # primary OS
        "shell": "",                    # preferred shell (bash/zsh/fish)

        # ── Schema metadata ───────────────────────────────────────────────────
        "_schema_version": 2,           # bump when adding breaking schema changes
        "_updated_at": "",              # ISO timestamp — agent sets on write
    }

    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    changed = False
    for key, default_val in defaults.items():
        if key not in existing:
            existing[key] = default_val
            changed = True
        elif key == "notes":
            # Merge: keep existing notes, add defaults not already present
            existing_set = set(existing["notes"])
            for n in default_val:
                if n not in existing_set:
                    existing["notes"].append(n)
                    changed = True

    if changed:
        from datetime import datetime, timezone
        existing["_updated_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("[setup] user.json updated with missing fields")


def refresh_identity_files() -> None:
    """
    Called on every boot — sync AGENTS.md and SOUL.md from the repo into the workspace.
    Also repairs stale files and ensures user profile is up-to-date.
    Always safe to call; never overwrites non-empty workspace copies.
    """
    # Ensure workspace + tmp exist first
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    ensure_runtime_dirs()
    for d in CONFIG_DIRS:
        (WORKSPACE / d).mkdir(parents=True, exist_ok=True)

    # Fix stale IDENTITY.md if present
    _fix_stale_identity_md()

    # Copy AGENTS.md from repo root → workspace (first boot only)
    agents_src = _REPO_ROOT / "AGENTS.md"
    agents_dst = WORKSPACE / "AGENTS.md"
    copied = _copy_file_if_absent(agents_src, agents_dst)
    if copied:
        logger.info("[setup] Copied AGENTS.md from repo to workspace")

    # Copy SOUL.md from repo root → workspace (first boot only)
    soul_src = _REPO_ROOT / "SOUL.md"
    soul_dst = WORKSPACE / "SOUL.md"
    if soul_src.exists():
        copied_soul = _copy_file_if_absent(soul_src, soul_dst)
        if copied_soul:
            logger.info("[setup] Copied SOUL.md from repo to workspace")
    else:
        # Write minimal stub if no repo SOUL.md and workspace copy absent
        if not soul_dst.exists() or soul_dst.stat().st_size < 20:
            soul_dst.write_text(
                "# Kernel-Evolving Soul\n\nYou are a self-evolving local AI agent.\n",
                encoding="utf-8",
            )

    # Ensure user profile has all expected fields
    _ensure_user_profile()

    # Create USER.md if absent — blank template; agent fills it in over time via write_file
    user_md_dst = WORKSPACE / "USER.md"
    if not user_md_dst.exists() or user_md_dst.stat().st_size < 20:
        user_md_dst.write_text(
            "# USER.md — About the user\n\n"
            "<!-- kernel-evolving fills this in as it learns things about the user -->\n"
            "<!-- Use write_file to append facts here when the user shares them -->\n"
            "<!-- context.py injects this file verbatim into every system prompt -->\n\n"
            "## Known facts\n"
            "<!-- Add facts below as you learn them -->\n\n"
            "## Preferences\n"
            "<!-- Add preferences below as you learn them -->\n\n"
            "## Notes\n"
            "<!-- Freeform notes -->\n",
            encoding="utf-8",
        )
        logger.info("[setup] Created USER.md blank template in workspace")

    # Ensure tmp/ dir exists
    _ensure_tmp_dir()


if __name__ == "__main__":
    setup_workspace()
    refresh_identity_files()
