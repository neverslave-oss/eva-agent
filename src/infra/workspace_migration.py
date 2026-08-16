"""Workspace migration helpers for runtime layout consolidation.

Idempotent migration from legacy root-level runtime files into canonical folders.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from runtime_paths import (
    WORKSPACE_ROOT,
    DATA_DIR,
    MEMORY_DIR,
    ARTIFACTS_DIR,
    THOUGHTS_DIR,
    THOUGHTS_IDEAS_DIR,
    ARTIFACTS_TRAJECTORIES_DIR,
    ARTIFACTS_EVAL_RUNS_DIR,
    ARTIFACTS_EVAL_RESULTS_DIR,
    ARTIFACTS_FINETUNE_DIR,
    ARTIFACTS_ADAPTER_ACTIVE_DIR,
    CHAT_HISTORY_DB,
    LONG_TERM_MEMORY_DB,
    SYSTEM_DEBUG_LOG_DB,
    EVOLUTION_DB,
    PROMOTED_SIGNALS_DB,
    FAILED_REQUESTS_DB,
    PROBES_DB,
    CONVERSATIONS_DB,
    ensure_runtime_dirs,
)


def _same_file(src: Path, dst: Path) -> bool:
    try:
        if src.stat().st_size != dst.stat().st_size:
            return False
        return int(src.stat().st_mtime) == int(dst.stat().st_mtime)
    except Exception:
        return False


def _archive_conflict(dst: Path) -> Path:
    archive_root = DATA_DIR / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    candidate = archive_root / dst.name
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        c = archive_root / f"{dst.stem}-{i}{dst.suffix}"
        if not c.exists():
            return c
        i += 1


def _move_file(src: Path, dst: Path, report: dict) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        report["moved"].append({"src": str(src), "dst": str(dst)})
        return
    if _same_file(src, dst):
        src.unlink(missing_ok=True)
        report["skipped"].append({"src": str(src), "dst": str(dst), "reason": "already_migrated"})
        return
    archived = _archive_conflict(dst)
    shutil.move(str(src), str(archived))
    report["conflicts"].append({"src": str(src), "existing": str(dst), "archived": str(archived)})


def _move_dir_contents(src_dir: Path, dst_dir: Path, report: dict) -> None:
    if not src_dir.exists() or not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in list(src_dir.iterdir()):
        dst_child = dst_dir / child.name
        if child.is_dir() and dst_child.exists() and dst_child.is_dir():
            _move_dir_contents(child, dst_child, report)
            try:
                child.rmdir()
            except OSError:
                pass
            continue
        if dst_child.exists():
            archived = _archive_conflict(dst_child)
            shutil.move(str(child), str(archived))
            report["conflicts"].append({"src": str(child), "existing": str(dst_child), "archived": str(archived)})
            continue
        shutil.move(str(child), str(dst_child))
        report["moved"].append({"src": str(child), "dst": str(dst_child)})
    try:
        src_dir.rmdir()
    except OSError:
        pass


def migrate_runtime_workspace() -> dict:
    """Migrate legacy runtime files/dirs to canonical layout.

    Safe to run repeatedly.
    """
    ensure_runtime_dirs()
    report = {"moved": [], "skipped": [], "conflicts": [], "errors": []}

    file_moves = {
        WORKSPACE_ROOT / "chat_history_evolving.db": CHAT_HISTORY_DB,
        WORKSPACE_ROOT / "memory.db": LONG_TERM_MEMORY_DB,
        WORKSPACE_ROOT / "system_debug_log.db": SYSTEM_DEBUG_LOG_DB,
        WORKSPACE_ROOT / "evolution.db": EVOLUTION_DB,
        WORKSPACE_ROOT / "promoted_signals.db": PROMOTED_SIGNALS_DB,
        WORKSPACE_ROOT / "failed_requests.db": FAILED_REQUESTS_DB,
        WORKSPACE_ROOT / "probes.db": PROBES_DB,
        WORKSPACE_ROOT / "conversations.db": CONVERSATIONS_DB,
    }

    dir_moves = {
        WORKSPACE_ROOT / "trajectories": ARTIFACTS_TRAJECTORIES_DIR,
        WORKSPACE_ROOT / "eval_runs": ARTIFACTS_EVAL_RUNS_DIR,
        WORKSPACE_ROOT / "eval_results": ARTIFACTS_EVAL_RESULTS_DIR,
        WORKSPACE_ROOT / "ideas": THOUGHTS_IDEAS_DIR,
    }

    try:
        for src, dst in file_moves.items():
            _move_file(src, dst, report)

        for src, dst in dir_moves.items():
            _move_dir_contents(src, dst, report)

        # Move finetune/adapters that were created as versioned root directories.
        for child in WORKSPACE_ROOT.glob("finetune_output*"):
            if child.is_dir():
                _move_dir_contents(child, ARTIFACTS_FINETUNE_DIR / child.name, report)
        for child in WORKSPACE_ROOT.glob("adapter_active*"):
            if child.is_dir():
                _move_dir_contents(child, ARTIFACTS_ADAPTER_ACTIVE_DIR / child.name, report)

        # Ensure thoughts root exists for journal writer users.
        THOUGHTS_DIR.mkdir(parents=True, exist_ok=True)

    except Exception as exc:
        report["errors"].append(str(exc))

    return report
