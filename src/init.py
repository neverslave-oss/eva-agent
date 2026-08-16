"""
init.py — Kernel-evolving initialization and workspace setup.
"""

import os
import sqlite3
from pathlib import Path
import yaml
from runtime_paths import (
    WORKSPACE_ROOT,
    DATA_DIR,
    MEMORY_DIR,
    ARTIFACTS_DIR,
    THOUGHTS_DIR,
    LOGS_DIR,
    CHAT_HISTORY_DB,
    PROMOTED_SIGNALS_DB,
    ensure_runtime_dirs,
)
from database.agent import PromotedSignalsRepository as _PromotedSignalsRepo

HOME = Path.home()
WORKSPACE_DIR = WORKSPACE_ROOT
BACKUP_DIR = HOME / ".kernel-evolving" / "backups"
ECOSYSTEM_DIR = HOME / ".kernel-evolving" / "ecosystem"
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

def ensure_directories() -> dict:
    """
    Create essential directories if missing.
    Returns dict of paths created.
    """
    created = []
    ensure_runtime_dirs()
    for d in [WORKSPACE_DIR, BACKUP_DIR, ECOSYSTEM_DIR, DATA_DIR, MEMORY_DIR, ARTIFACTS_DIR, THOUGHTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        created.append(str(d))
    return {"created": created}

def ensure_database_schema() -> dict:
    """
    Ensure chat_history_evolving.db and promoted_signals.db have correct schema.
    Returns dict with table counts.
    """
    db_files = [
        CHAT_HISTORY_DB,
        PROMOTED_SIGNALS_DB,
    ]
    results = {}
    for db_path in db_files:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # chat_history_evolving schema
        if db_path.name == "chat_history_evolving.db":
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot        TEXT    NOT NULL DEFAULT 'kernel-evolving',
                    session_id TEXT    NOT NULL,
                    role       TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
                    content    TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
            """)
            cursor.execute("SELECT COUNT(*) FROM messages")
            count = cursor.fetchone()[0]
            results["chat_history_messages"] = count
        elif db_path.name == "promoted_signals.db":
            # Schema managed by PromotedSignalsRepository — just count
            repo = _PromotedSignalsRepo(db_path)
            count = repo.count()
            results["promoted_signals"] = count
        conn.commit()
        conn.close()
    return results

def check_kernel_base() -> dict:
    """
    Check if kernel base (port 8769) is reachable.
    """
    import requests
    try:
        r = requests.get("http://localhost:8769/health", timeout=2)
        return {"reachable": r.ok, "status": r.json() if r.ok else None}
    except Exception:
        return {"reachable": False, "status": None}

def initialize(force: bool = False) -> dict:
    """
    Initialize kernel-evolving workspace.
    If force=False and workspace already initialized, returns status without changes.
    """
    # Check if workspace already has databases
    existing = CHAT_HISTORY_DB.exists()
    runtime_paths = {
        "workspace": str(WORKSPACE_DIR),
        "data_dir": str(DATA_DIR),
        "memory_dir": str(MEMORY_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "thoughts_dir": str(THOUGHTS_DIR),
        "logs_dir": str(LOGS_DIR),
        "chat_history_db": str(CHAT_HISTORY_DB),
        "promoted_signals_db": str(PROMOTED_SIGNALS_DB),
    }

    if existing and not force:
        return {
            "status": "already_initialized",
            "workspace": str(WORKSPACE_DIR),
            "databases": ensure_database_schema(),
            "paths": runtime_paths,
        }
    
    # Ensure directories
    dirs = ensure_directories()
    # Ensure schema
    dbs = ensure_database_schema()
    # Check kernel base
    kernel = check_kernel_base()
    
    # Seed with a default thought if databases empty
    if dbs.get("chat_history_messages", 0) == 0:
        # Optionally add a welcome message
        pass
    
    return {
        "status": "initialized",
        "workspace": str(WORKSPACE_DIR),
        "directories": dirs,
        "databases": dbs,
        "kernel_base": kernel,
        "paths": runtime_paths,
    }


if __name__ == "__main__":
    # Quick test
    result = initialize()
    print(result)