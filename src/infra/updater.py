"""
updater.py — Shared self-update logic for Kernel Evo (CLI + Telegram bot).

Used by:
  - kernel_evo (CLI)   → /update command
  - telegram_bot.py   → /update command + background version check loop
"""
import os
import subprocess
import importlib.util
from pathlib import Path
from typing import Callable, Optional

REPO_DIR = str(Path(__file__).parent.parent)

# GitHub API URLs — configurable via env vars so forks/self-hosted instances
# can point to their own repos without code changes.
_default_gh_repo = os.environ.get("KERNEL_EVO_GITHUB_REPO", "fabiopacifici-bot/kernel-evolving")
_GITHUB_RELEASES_URL = os.environ.get(
    "KERNEL_EVO_RELEASES_URL",
    f"https://api.github.com/repos/{_default_gh_repo}/releases/latest",
)
_GITHUB_TAGS_URL = os.environ.get(
    "KERNEL_EVO_TAGS_URL",
    f"https://api.github.com/repos/{_default_gh_repo}/tags",
)


def get_current_version() -> str:
    """Return runtime version metadata, preferring git tags (dynamic), fallback to src/version.py."""
    # 1) Exact tag on current commit (best for release builds)
    try:
        exact = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if exact.returncode == 0:
            tag = exact.stdout.strip()
            if tag:
                return tag.lstrip("v")
    except Exception:
        pass

    # 2) Nearest reachable tag (useful on dev commits after a release)
    try:
        nearest = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if nearest.returncode == 0:
            tag = nearest.stdout.strip()
            if tag:
                return tag.lstrip("v")
    except Exception:
        pass

    # 3) Fallback to source constant
    ver_path = Path(REPO_DIR) / "src" / "version.py"
    spec = importlib.util.spec_from_file_location("_kernel_version", ver_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.__version__


def fetch_latest_version() -> str:
    """Fetch latest remote version by combining release + tags + git remote tags."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    api_headers = {"Accept": "application/vnd.github+json"}
    if token:
        api_headers["Authorization"] = f"Bearer {token}"

    candidates: list[str] = []

    # 1) GitHub Releases API
    try:
        import requests
        resp = requests.get(_GITHUB_RELEASES_URL, timeout=10, headers=api_headers)
        if resp.status_code == 200:
            tag = (resp.json() or {}).get("tag_name", "")
            if tag:
                candidates.append(tag.lstrip("v"))

        # 2) GitHub Tags API
        resp = requests.get(_GITHUB_TAGS_URL, params={"per_page": 100}, timeout=10, headers=api_headers)
        if resp.status_code == 200:
            tags = resp.json() or []
            for t in tags:
                name = t.get("name", "") if isinstance(t, dict) else ""
                if name:
                    candidates.append(name.lstrip("v"))
    except Exception as e:
        print(f"[updater] github api version-check error: {e}")

    # 3) Git remote fallback (works for private repos with git credentials configured)
    try:
        ls = subprocess.run(
            ["git", "ls-remote", "--tags", "--refs", "origin"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if ls.returncode == 0 and ls.stdout.strip():
            for line in ls.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].startswith("refs/tags/"):
                    candidates.append(parts[1].replace("refs/tags/", "").lstrip("v"))
    except Exception as e:
        print(f"[updater] git remote version-check error: {e}")

    if not candidates:
        return ""

    # Deduplicate and select highest semantic-like version, tolerating suffixes
    candidates = list(dict.fromkeys(candidates))

    def _version_key(v: str):
        import re
        m = re.match(r"^(\d+(?:\.\d+)*)", v)
        if m:
            parts = tuple(int(x) for x in m.group(1).split('.'))
            # Prefer clean numeric tags over suffixed variants at same numeric level
            has_suffix = 1 if len(v) > len(m.group(1)) else 0
            return (parts, -has_suffix, v)
        return ((0,), -1, v)

    candidates.sort(key=_version_key, reverse=True)
    return candidates[0]


def do_update(
    notify: Callable[[str], None],
    restart_fn: Optional[Callable[[], None]] = None,
) -> bool:
    """
    Pull latest from git and restart.

    Args:
        notify:     function(str) called with status messages (print for CLI, send_message for bot)
        restart_fn: called after successful pull to restart the process.
                    If None, uses os._exit(0) (relies on start.sh to relaunch).

    Returns True if update was applied, False otherwise.
    """
    current = get_current_version()
    notify("🔍 Checking for updates...")

    latest = fetch_latest_version()
    if not latest:
        notify("❌ Could not reach GitHub to check for updates.")
        return False

    if latest == current:
        notify(f"✅ Already on latest version (v{current}). Nothing to update.")
        return False

    notify(f"🔄 Updating v{current} → v{latest}... pulling main")

    pull = subprocess.run(
        ["git", "pull", "origin", "main"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if pull.returncode != 0:
        notify(f"❌ git pull failed:\n{pull.stderr[:300]}")
        return False

    # Read version from disk after pull
    try:
        disk_version = get_current_version()
    except Exception:
        disk_version = latest

    notify(f"✅ Updated to v{disk_version}. Restarting...")

    if restart_fn:
        restart_fn()
    else:
        # Default: exit and let start.sh relaunch
        os._exit(0)

    return True


def check_update_available(current_version: str = "") -> Optional[str]:
    """Return latest version string if an update is available, else None."""
    if not current_version:
        current_version = get_current_version()
    latest = fetch_latest_version()
    if latest and latest != current_version:
        return latest
    return None
