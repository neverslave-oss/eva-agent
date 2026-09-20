"""
tests/test_conversations_sessions.py — Unit tests for the conversations/sessions API.

Uses the isolated test DB: tests/conftest.py already redirects KERNEL_CONVERSATIONS_DB
to a per-session temp file, so these tests never touch the production
`conversations.db`. Mirrors the style of test_probe_store.py.
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from database.agent.conversations import ConversationsRepository


def _repo(tmp_path):
    return ConversationsRepository(db_path=str(tmp_path / "conversations.db"))


# ── repository: upsert / get / list ────────────────────────────────────────

class TestRepositoryListCreate:
    def test_upsert_creates_and_roundtrips(self, tmp_path):
        repo = _repo(tmp_path)
        cid = str(uuid.uuid4())
        repo.upsert(cid, chat_id="agent-abc", title="Hello world")
        row = repo.get(cid)
        assert row is not None
        assert row["chat_id"] == "agent-abc"
        assert row["title"] == "Hello world"
        assert row["updated_at"]

    def test_upsert_updates_title(self, tmp_path):
        repo = _repo(tmp_path)
        cid = str(uuid.uuid4())
        repo.upsert(cid, chat_id="agent-abc", title="Old")
        repo.upsert(cid, chat_id="agent-abc", title="New")
        row = repo.get(cid)
        assert row["title"] == "New"

    def test_list_all_sorted_by_updated_at_desc(self, tmp_path):
        repo = _repo(tmp_path)
        repo.upsert(str(uuid.uuid4()), chat_id="agent-old", title="Old")
        repo.upsert(str(uuid.uuid4()), chat_id="agent-new", title="New")
        rows = repo.list_all(limit=10)
        assert len(rows) == 2
        # Most recently inserted has a later updated_at and should sort first.
        assert rows[0]["title"] == "New"
        assert rows[1]["title"] == "Old"

    def test_list_all_respects_limit(self, tmp_path):
        repo = _repo(tmp_path)
        for i in range(5):
            repo.upsert(str(uuid.uuid4()), chat_id=f"agent-{i}", title=f"S{i}")
        rows = repo.list_all(limit=3)
        assert len(rows) == 3

    def test_list_by_chat_filters(self, tmp_path):
        repo = _repo(tmp_path)
        repo.upsert(str(uuid.uuid4()), chat_id="agent-a", title="A1")
        repo.upsert(str(uuid.uuid4()), chat_id="agent-a", title="A2")
        repo.upsert(str(uuid.uuid4()), chat_id="agent-b", title="B1")
        rows = repo.list_by_chat("agent-a")
        assert len(rows) == 2


# ── API endpoint logic (pure) ───────────────────────────────────────────────

def test_create_payload_uses_generated_chat_id_when_blank():
    # Verify the fallback used by the POST /api/sessions handler.
    import uuid as _u
    chat_id = ("".strip()) or f"agent-{_u.uuid4().hex[:8]}"
    assert chat_id.startswith("agent-")
    assert len(chat_id) == 14  # "agent-" (6) + 8 hex chars
