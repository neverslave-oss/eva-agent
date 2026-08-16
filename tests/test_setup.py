"""test_setup.py — Tests for setup.py workspace init and refresh logic."""
import json
import sys
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _patch_workspace(tmp_path: Path):
    """Return a context manager that redirects setup.WORKSPACE/KERNEL_HOME to tmp_path."""
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    kernel_home = tmp_path
    repo_root = Path(__file__).parent.parent
    return (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', kernel_home),
        patch.object(_setup, '_REPO_ROOT', repo_root),
    )


def test_setup_workspace_creates_dirs(tmp_path):
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', Path(__file__).parent.parent),
    ):
        result = _setup.setup_workspace()
    assert result is True
    assert workspace.exists()
    for d in _setup.CONFIG_DIRS:
        assert (workspace / d).exists(), f"Missing subdir: {d}"


def test_refresh_copies_agents_md(tmp_path):
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    repo_root = Path(__file__).parent.parent
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', repo_root),
    ):
        _setup.refresh_identity_files()
    agents_dst = workspace / "AGENTS.md"
    assert agents_dst.exists(), "AGENTS.md should have been copied to workspace"
    if (repo_root / "AGENTS.md").exists():
        repo_content = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
        assert agents_dst.read_text(encoding="utf-8") == repo_content


def test_setup_idempotent(tmp_path):
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', Path(__file__).parent.parent),
    ):
        r1 = _setup.setup_workspace()
        r2 = _setup.setup_workspace()
    assert r1 is True
    assert r2 is False  # second call is a no-op


def test_ensure_user_profile_merges(tmp_path):
    """User preferences/memories are non-destructively merged.

    The test verifies the mechanism generically:
    - existing values are preserved (not overwritten by defaults)
    - existing notes are kept (notes list is merged, not replaced)
    - missing schema fields are added with their default values
    - a second call is idempotent (no duplicate notes, no data loss)
    """
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    # Pre-seed with a partial profile: custom name, extra note, a preference
    existing = {
        "name": "Alice",
        "notes": ["Prefers dark mode.", "Has a dog named Rex."],
        "preferences": {"theme": "dark"},
    }
    (workspace / "user.json").write_text(json.dumps(existing), encoding="utf-8")

    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', Path(__file__).parent.parent),
    ):
        _setup._ensure_user_profile()

    result = json.loads((workspace / "user.json").read_text(encoding="utf-8"))

    # --- Existing values must be preserved ---
    assert result["name"] == "Alice", "Existing name must not be overwritten"
    assert result["preferences"] == {"theme": "dark"}, "Existing preferences must be preserved"

    # --- Existing notes must survive the merge ---
    assert "Prefers dark mode." in result["notes"], "Existing note 1 must be preserved"
    assert "Has a dog named Rex." in result["notes"], "Existing note 2 must be preserved"

    # --- Missing schema fields must be added with defaults ---
    assert "handle" in result, "Missing field 'handle' should be added"
    assert "timezone" in result, "Missing field 'timezone' should be added"
    assert "language" in result, "Missing field 'language' should be added"
    assert "projects" in result, "Missing field 'projects' should be added"
    assert "_schema_version" in result, "Schema version field should be added"

    # --- Idempotency: calling again must not duplicate notes or change values ---
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', Path(__file__).parent.parent),
    ):
        _setup._ensure_user_profile()

    result2 = json.loads((workspace / "user.json").read_text(encoding="utf-8"))
    assert result2["notes"].count("Prefers dark mode.") == 1, "Notes must not be duplicated on re-run"
    assert result2["name"] == "Alice", "Name must not change on re-run"
