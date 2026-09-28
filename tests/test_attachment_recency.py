"""
Regression tests for stale-attachment context injection.

Bug: attachment_context_block() surfaces the most recent attachment records
with NO age limit. A photo uploaded days ago is still treated as "recent", so a
later plain-text message that merely contains a common keyword ("photo", "image",
"file", "document", ...) gets the stale attachment context injected AND the
attachment guard forces the reply to reference it — making the agent believe the
user just sent an attachment they never sent in this turn.

Fix: recent_attachments()/attachment_context_block() accept a max_age_seconds
window so stale attachments are excluded. When no recent attachment exists, no
context block is produced and the guard does not fire.
"""

import os
import sys
import importlib
import tempfile
import time
from datetime import datetime, timedelta, timezone

import pytest


def _make_memory_module(db_path: str):
    """Import memory.py with a patched DB_FILE so each test gets an isolated DB."""
    spec = importlib.util.spec_from_file_location(
        "memory_test_recency",
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "memory", "memory.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.DB_FILE = __import__("pathlib").Path(db_path)
    spec.loader.exec_module(mod)
    return mod


def _record_attachment(mem, chat_id, *, kind="photo", original_name="photo.jpg",
                        created_at=None):
    """Record an attachment, optionally back-dating its created_at in the DB."""
    row_id = mem.record_attachment(
        kind=kind,
        local_path="(temp-deleted)",
        original_name=original_name,
        mime_type="image/jpeg",
        caption="Describe this image.",
        chat_id=chat_id,
    )
    if created_at is not None:
        import sqlite3
        conn = mem._get_conn()
        conn.execute(
            "UPDATE attachments SET created_at=? WHERE id=?",
            (created_at.isoformat(), row_id),
        )
        conn.commit()
        conn.close()
    return row_id


def test_stale_attachment_excluded_from_context_block():
    """An attachment older than the recency window must NOT appear in the context block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")
        mem = _make_memory_module(db_path)
        chat_id = "chat_stale"

        # Record an attachment 4 days ago (the observed bug: a Sep 1 photo
        # surfaced on Sep 5).
        _record_attachment(
            mem, chat_id,
            created_at=datetime.now(timezone.utc) - timedelta(days=4),
        )

        block = mem.attachment_context_block(chat_id=chat_id, limit=3,
                                            max_age_seconds=86400)
        assert not block, "stale attachment must not be injected as recent context"


def test_fresh_attachment_included_in_context_block():
    """A recent attachment within the window IS surfaced."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")
        mem = _make_memory_module(db_path)
        chat_id = "chat_fresh"

        _record_attachment(mem, chat_id, created_at=datetime.now(timezone.utc))

        block = mem.attachment_context_block(chat_id=chat_id, limit=3,
                                              max_age_seconds=86400)
        assert block, "fresh attachment should be injected as recent context"
        assert "photo.jpg" in block


def test_guard_does_not_fire_on_stale_attachment():
    """attachment_guard must not force a reply to reference a stale attachment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")
        mem = _make_memory_module(db_path)
        chat_id = "chat_guard_stale"

        _record_attachment(
            mem, chat_id,
            created_at=datetime.now(timezone.utc) - timedelta(days=4),
        )

        # User message merely contains the word "file" but no recent attachment
        # was sent in this turn.
        guard = mem.attachment_guard(
            "Please fix the file parsing issue",
            "I'll fix the parsing logic.",
            chat_id=chat_id,
            max_age_seconds=86400,
        )
        assert guard["ok"], "guard must not fire for a stale attachment"


def test_guard_fires_on_fresh_attachment():
    """attachment_guard still fires when a genuinely recent attachment exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")
        mem = _make_memory_module(db_path)
        chat_id = "chat_guard_fresh"

        _record_attachment(mem, chat_id, created_at=datetime.now(timezone.utc))

        guard = mem.attachment_guard(
            "What did you think of the file I sent?",
            "It looks good.",
            chat_id=chat_id,
            max_age_seconds=86400,
        )
        assert not guard["ok"], "guard should fire for a genuinely recent attachment"
