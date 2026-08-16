"""
fresh.py — Kernel-evolving fresh reset logic.
"""

import os
import shutil
from pathlib import Path
import sqlite3
from runtime_paths import (
    WORKSPACE_ROOT,
    CHAT_HISTORY_DB,
    EVOLUTION_DB,
    PROMOTED_SIGNALS_DB,
    FAILED_REQUESTS_DB,
    PROBES_DB,
    SYSTEM_DEBUG_LOG_DB,
    CONVERSATIONS_DB,
)

HOME = Path.home()
WORKSPACE_DIR = WORKSPACE_ROOT
BACKUP_DIR = HOME / ".kernel-evolving" / "backups"
ECOSYSTEM_DIR = HOME / ".kernel-evolving" / "ecosystem"

def list_reset_targets() -> dict:
    """
    Return dict of files and directories that will be deleted during fresh reset.
    """
    targets = {}
    
    # Database files
    db_files = [
        CHAT_HISTORY_DB,
        EVOLUTION_DB,
        PROMOTED_SIGNALS_DB,
        FAILED_REQUESTS_DB,
        PROBES_DB,
        SYSTEM_DEBUG_LOG_DB,
        CONVERSATIONS_DB,
    ]
    targets["databases"] = [str(p) for p in db_files if p.exists()]
    
    # Single config files
    single_files = [
        WORKSPACE_DIR / "user.json",
        WORKSPACE_DIR / "todos.md",
        WORKSPACE_DIR / "self_evolving_agent_concept.txt",
        WORKSPACE_DIR / "skills_summary.md",
    ]
    targets["files"] = [str(p) for p in single_files if p.exists()]
    
    # Directories to clear (not delete)
    clear_dirs = [
        WORKSPACE_DIR / "thoughts",
        WORKSPACE_DIR / "notes",
        WORKSPACE_DIR / "memory",
        WORKSPACE_DIR / "data",
        WORKSPACE_DIR / "artifacts",
        WORKSPACE_DIR / "logs",
        WORKSPACE_DIR / "tmp",
    ]
    targets["directories"] = [str(p) for p in clear_dirs if p.exists()]
    
    # Directories to keep (backups)
    keep_dirs = [
        BACKUP_DIR,
        ECOSYSTEM_DIR,
    ]
    targets["keep"] = [str(p) for p in keep_dirs if p.exists()]
    
    return targets

def dry_run_fresh() -> dict:
    """
    Simulate fresh reset and return what would be deleted.
    """
    targets = list_reset_targets()
    total_files = len(targets["databases"]) + len(targets["files"])
    total_dirs = len(targets["directories"])
    return {
        "status": "dry_run",
        "targets": targets,
        "summary": {
            "databases": len(targets["databases"]),
            "files": len(targets["files"]),
            "directories_to_clear": total_dirs,
            "total_items": total_files + total_dirs,
        },
        "message": "This is a dry run. No files have been deleted.",
    }

def execute_fresh() -> dict:
    """
    Perform actual fresh reset:
    - Delete database files
    - Delete config files
    - Clear directories (remove contents, keep empty dir)
    - Keep backups and ecosystem directories.
    """
    targets = list_reset_targets()
    deleted = []
    cleared = []
    errors = []
    
    # Delete database files
    for path_str in targets["databases"]:
        try:
            os.remove(path_str)
            deleted.append(path_str)
        except Exception as e:
            errors.append(f"{path_str}: {e}")
    
    # Delete single files
    for path_str in targets["files"]:
        try:
            os.remove(path_str)
            deleted.append(path_str)
        except Exception as e:
            errors.append(f"{path_str}: {e}")
    
    # Clear directories (remove all contents, keep empty directory)
    for path_str in targets["directories"]:
        p = Path(path_str)
        try:
            if p.exists():
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                cleared.append(path_str)
        except Exception as e:
            errors.append(f"{path_str}: {e}")
    
    # Reinitialize databases via init module
    import init
    init_result = init.initialize(force=True)
    
    return {
        "status": "fresh_executed",
        "deleted": deleted,
        "cleared": cleared,
        "errors": errors,
        "init_result": init_result,
        "summary": {
            "deleted_count": len(deleted),
            "cleared_count": len(cleared),
            "error_count": len(errors),
        },
        "message": "Fresh reset completed. Databases reinitialized. Restart kernel-evolving API to ensure clean state.",
    }


if __name__ == "__main__":
    # Quick test
    print("Dry run:")
    print(dry_run_fresh())
    # print("Execute (uncomment):")
    # print(execute_fresh())