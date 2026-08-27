"""
backup.py — Backup and restore functionality for kernel-evolving.

Features:
- Create timestamped .tar.gz archives of workspace, databases, ecosystem, config
- Compute SHA256 checksum
- Store backups in ~/.kernel-evolving/backups/
- List existing backups
- Restore from backup (future)
"""

import tarfile
import os
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from runtime_paths import CHAT_HISTORY_DB, PROMOTED_SIGNALS_DB, EVOLUTION_DB

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME = Path.home()
BACKUP_ROOT = HOME / ".kernel-evolving" / "backups"
WORKSPACE_DIR = HOME / ".kernel-evolving" / "workspace"
ECOSYSTEM_DIR = HOME / ".kernel-evolving" / "ecosystem"
# Honor KERNEL_EVO_CONFIG (set by start.sh --config=...) when present.
_CONFIG_OVERRIDE = os.environ.get("KERNEL_EVO_CONFIG", "").strip()
CONFIG_PATH = (
    Path(os.path.abspath(os.path.expanduser(_CONFIG_OVERRIDE)))
    if _CONFIG_OVERRIDE
    else Path(__file__).parent.parent.parent / "config.yaml"
)
ENV_PATH = Path(__file__).parent.parent / ".env"

# Ensure backup root exists
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)


def create_backup(description: str = "", full: bool = True) -> Dict[str, str]:
    """
    Create a timestamped backup archive.
    
    Args:
        description: Optional user description for manifest.
        full: If True, include ecosystem directory (may be large).
    
    Returns:
        Dict with backup_path, size_mb, sha256, manifest_path.
    
    Raises:
        FileNotFoundError if critical paths missing.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect items to backup
    items = []
    
    # 1. Databases
    db_files = [
        CHAT_HISTORY_DB,
        PROMOTED_SIGNALS_DB,
        EVOLUTION_DB,
    ]
    for db in db_files:
        if db.exists():
            items.append(("databases", db))
    
    # 2. Workspace (excluding large caches)
    workspace_exclude = {
        "*.log",
        "*.tmp",
        "*.cache",
        "__pycache__",
        "node_modules",
        ".git",
        "*.tar.gz",
        "*.zip",
    }
    if WORKSPACE_DIR.exists():
        for root, dirs, files in os.walk(WORKSPACE_DIR):
            root_path = Path(root)
            # Skip excluded patterns
            rel_path = root_path.relative_to(WORKSPACE_DIR)
            if any(rel_path.match(pattern) for pattern in workspace_exclude):
                continue
            for f in files:
                file_path = root_path / f
                if not any(file_path.match(pattern) for pattern in workspace_exclude):
                    items.append(("workspace", file_path))
    
    # 3. Ecosystem (skills/routines) if full=True
    if full and ECOSYSTEM_DIR.exists():
        # Follow symlink if it's a symlink
        target = ECOSYSTEM_DIR.resolve() if ECOSYSTEM_DIR.is_symlink() else ECOSYSTEM_DIR
        for root, dirs, files in os.walk(target):
            root_path = Path(root)
            # Skip .git directories to keep size manageable
            if ".git" in root_path.parts:
                continue
            for f in files:
                file_path = root_path / f
                # Skip large binary files
                if file_path.suffix in {".bin", ".model", ".pth", ".ckpt", ".h5"}:
                    continue
                items.append(("ecosystem", file_path))
    
    # 4. Config files
    if CONFIG_PATH.exists():
        items.append(("config", CONFIG_PATH))
    if ENV_PATH.exists():
        items.append(("config", ENV_PATH))
    
    # Create manifest
    manifest = {
        "timestamp": timestamp,
        "description": description,
        "full": full,
        "items": [],
        "counts": {
            "databases": 0,
            "workspace": 0,
            "ecosystem": 0,
            "config": 0,
        }
    }
    
    # Create tar.gz archive
    archive_name = f"kernel-evolving-backup-{timestamp}.tar.gz"
    archive_path = backup_dir / archive_name
    sha256_hash = hashlib.sha256()
    
    with tarfile.open(archive_path, "w:gz") as tar:
        for category, item_path in items:
            # Ensure relative path inside archive
            arcname = f"{category}/{item_path.relative_to(item_path.parents[-2])}"
            tar.add(item_path, arcname=arcname)
            manifest["counts"][category] += 1
            manifest["items"].append({
                "category": category,
                "path": str(item_path),
                "size": item_path.stat().st_size,
            })
    
    # Compute checksum
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    checksum = sha256_hash.hexdigest()
    
    # Write manifest
    manifest_path = backup_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Symlink latest
    latest_link = BACKUP_ROOT / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(timestamp)
    
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    
    return {
        "backup_path": str(archive_path),
        "backup_dir": str(backup_dir),
        "size_mb": round(size_mb, 2),
        "sha256": checksum,
        "manifest_path": str(manifest_path),
        "timestamp": timestamp,
    }


def list_backups(limit: int = 20) -> List[Dict]:
    """
    List existing backups sorted by timestamp (newest first).
    """
    backups = []
    if not BACKUP_ROOT.exists():
        return backups
    
    for timestamp_dir in BACKUP_ROOT.iterdir():
        if not timestamp_dir.is_dir():
            continue
        manifest_path = timestamp_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        
        # Find tar.gz archive
        archives = list(timestamp_dir.glob("*.tar.gz"))
        if not archives:
            continue
        
        archive_path = archives[0]
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        backups.append({
            "timestamp": timestamp_dir.name,
            "archive_path": str(archive_path),
            "size_mb": round(archive_path.stat().st_size / (1024 * 1024), 2),
            "description": manifest.get("description", ""),
            "full": manifest.get("full", True),
            "counts": manifest.get("counts", {}),
        })
    
    # Sort descending by timestamp
    backups.sort(key=lambda b: b["timestamp"], reverse=True)
    return backups[:limit]


def get_backup_details(timestamp: str) -> Optional[Dict]:
    """
    Get details for a specific backup by timestamp.
    """
    backup_dir = BACKUP_ROOT / timestamp
    if not backup_dir.exists():
        return None
    
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    archives = list(backup_dir.glob("*.tar.gz"))
    archive_path = archives[0] if archives else None
    
    return {
        "timestamp": timestamp,
        "archive_path": str(archive_path) if archive_path else None,
        "manifest": manifest,
    }


def delete_backup(timestamp: str) -> bool:
    """
    Delete a backup by timestamp.
    """
    backup_dir = BACKUP_ROOT / timestamp
    if not backup_dir.exists():
        return False
    shutil.rmtree(backup_dir)
    
    # Remove latest symlink if pointing to deleted backup
    latest_link = BACKUP_ROOT / "latest"
    if latest_link.is_symlink() and latest_link.resolve() == backup_dir:
        latest_link.unlink()
    
    return True


def restore_backup(timestamp: str, target_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Restore a backup (future implementation).
    For now, raise NotImplementedError.
    """
    raise NotImplementedError("Restore functionality will be implemented in phase 2.")


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        result = create_backup("Test backup", full=False)
        print(json.dumps(result, indent=2))
        backups = list_backups()
        print(f"\nTotal backups: {len(backups)}")
        for b in backups:
            print(f"  {b['timestamp']}: {b['size_mb']} MB - {b['description']}")