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
    # Source is now the canonical template dir, not the repo root.
    template = repo_root / "src" / "assets" / "agent-templates" / "AGENTS.md"
    if template.exists():
        assert agents_dst.read_text(encoding="utf-8") == template.read_text(encoding="utf-8")


def test_refresh_repairs_corrupted_agents_md_preserving_learnings(tmp_path):
    """A corrupted AGENTS.md (learnings only, no template) is repaired on refresh,
    and existing session learnings are preserved."""
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    repo_root = Path(__file__).parent.parent
    agents_dst = workspace / "AGENTS.md"
    # Corrupted file: only session learnings, no identity template markers.
    agents_dst.write_text(
        "## Session learnings (2026-08-06)\n- Always call tools directly\n",
        encoding="utf-8",
    )
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', repo_root),
    ):
        _setup.refresh_identity_files()
    content = agents_dst.read_text(encoding="utf-8")
    # Template restored
    assert "## Who you are" in content
    assert "## How a request flows" in content
    # Learnings preserved
    assert "Session learnings (2026-08-06)" in content
    assert "Always call tools directly" in content


def test_refresh_does_not_rewrite_valid_agents_md(tmp_path):
    """A valid AGENTS.md (with template + learnings) is left untouched on refresh."""
    import infra.setup as _setup
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    repo_root = Path(__file__).parent.parent
    agents_dst = workspace / "AGENTS.md"
    agents_dst.write_text(
        "# AGENTS.md\n\n## Who you are\nKernel-Evo.\n\n"
        "## Session learnings (2026-08-06)\n- Always call tools directly\n",
        encoding="utf-8",
    )
    original = agents_dst.read_text(encoding="utf-8")
    with (
        patch.object(_setup, 'WORKSPACE', workspace),
        patch.object(_setup, 'KERNEL_HOME', tmp_path),
        patch.object(_setup, '_REPO_ROOT', repo_root),
    ):
        _setup.refresh_identity_files()
    assert agents_dst.read_text(encoding="utf-8") == original, (
        "Valid AGENTS.md must not be rewritten on refresh"
    )


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
