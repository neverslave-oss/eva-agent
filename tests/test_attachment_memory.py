"""
Tests for attachment memory (record_attachment, recent_attachments,
attachment_context_block, attachment_guard) added in memory.py.
"""
import os
import sys
import importlib
import tempfile
import sqlite3
import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Redirect DB and JSON paths to a temp dir so tests are fully isolated."""
    import core.memory.memory as mem
    monkeypatch.setattr(mem, "DB_DIR", tmp_path)
    monkeypatch.setattr(mem, "DB_FILE", tmp_path / "chat_history_evolving.db")
    monkeypatch.setattr(mem, "MEMORY_FILE", tmp_path / "core.memory.memory.json")
    mem._SESSION_ID = None  # reset session id
    tmp_path.mkdir(parents=True, exist_ok=True)
    yield
    # Clean up connections (SQLite locks)


def _mem():
    import core.memory.memory as memory
    importlib.reload(memory)  # pick up monkeypatched paths
    return memory


# ── record_attachment ─────────────────────────────────────────────────────────

def test_record_attachment_returns_id(isolated_memory):
    import core.memory.memory as mem
    row_id = mem.record_attachment(
        kind="document",
        local_path="/tmp/docs/20240101_test.pdf",
        original_name="test.pdf",
        mime_type="application/pdf",
        caption="Please summarise this.",
        chat_id="12345",
    )
    assert isinstance(row_id, int) and row_id > 0


def test_record_multiple_attachments(isolated_memory):
    import core.memory.memory as mem
    for i in range(3):
        mem.record_attachment(
            kind="document",
            local_path=f"/tmp/docs/file{i}.pdf",
            original_name=f"file{i}.pdf",
            mime_type="application/pdf",
            caption=f"Caption {i}",
            chat_id="42",
        )
    rows = mem.recent_attachments(limit=10, chat_id="42")
    assert len(rows) == 3


# ── recent_attachments ────────────────────────────────────────────────────────

def test_recent_attachments_order(isolated_memory):
    import core.memory.memory as mem
    for i in range(5):
        mem.record_attachment(
            kind="document",
            local_path=f"/tmp/f{i}.pdf",
            original_name=f"f{i}.pdf",
            mime_type="application/pdf",
            chat_id="77",
        )
    rows = mem.recent_attachments(limit=3, chat_id="77")
    assert len(rows) == 3
    # Most recent first
    names = [r["original_name"] for r in rows]
    assert names == ["f4.pdf", "f3.pdf", "f2.pdf"]


def test_recent_attachments_chat_filter(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(kind="document", local_path="/tmp/a.pdf", original_name="a.pdf", chat_id="111")
    mem.record_attachment(kind="document", local_path="/tmp/b.pdf", original_name="b.pdf", chat_id="222")
    rows_111 = mem.recent_attachments(chat_id="111")
    rows_222 = mem.recent_attachments(chat_id="222")
    assert len(rows_111) == 1 and rows_111[0]["original_name"] == "a.pdf"
    assert len(rows_222) == 1 and rows_222[0]["original_name"] == "b.pdf"


def test_recent_attachments_no_chat_filter(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(kind="document", local_path="/tmp/a.pdf", original_name="a.pdf", chat_id="111")
    mem.record_attachment(kind="photo", local_path="(temp-deleted)", original_name="photo.jpg", chat_id="222")
    rows = mem.recent_attachments(limit=10)
    assert len(rows) == 2


# ── attachment_context_block ──────────────────────────────────────────────────

def test_attachment_context_block_empty(isolated_memory):
    import core.memory.memory as mem
    block = mem.attachment_context_block(chat_id="999")
    assert block == ""


def test_attachment_context_block_content(isolated_memory, tmp_path):
    import core.memory.memory as mem
    real_file = tmp_path / "report.pdf"
    real_file.write_bytes(b"%PDF-1.4")
    mem.record_attachment(
        kind="document",
        local_path=str(real_file),
        original_name="report.pdf",
        mime_type="application/pdf",
        caption="Summarise page 2",
        chat_id="55",
    )
    block = mem.attachment_context_block(chat_id="55")
    assert "report.pdf" in block
    assert str(real_file) in block
    assert "✓" in block  # file exists
    assert "Summarise page 2" in block


def test_attachment_context_block_missing_file(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(
        kind="document",
        local_path="/tmp/nonexistent_xyz.pdf",
        original_name="ghost.pdf",
        chat_id="55",
    )
    block = mem.attachment_context_block(chat_id="55")
    assert "✗" in block


# ── attachment_guard ──────────────────────────────────────────────────────────

def test_guard_no_attachment_keyword(isolated_memory):
    import core.memory.memory as mem
    # No keyword → always OK
    result = mem.attachment_guard("What is 2+2?", "The answer is 4.")
    assert result["ok"] is True


def test_guard_no_recent_attachments(isolated_memory):
    import core.memory.memory as mem
    # Keyword present but nothing uploaded yet
    result = mem.attachment_guard("Summarise the PDF I sent.", "Here is a summary.")
    assert result["ok"] is True


def test_guard_reply_references_file(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(
        kind="document", local_path="/tmp/docs/myreport.pdf",
        original_name="myreport.pdf", chat_id="99"
    )
    # Reply mentions the filename
    result = mem.attachment_guard(
        "Can you summarise the PDF?",
        "Sure! Based on myreport.pdf, the key points are…",
        chat_id="99",
    )
    assert result["ok"] is True


def test_guard_reply_references_path(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(
        kind="document", local_path="/home/user/.kernel-evolving/workspace/documents/2024_report.pdf",
        original_name="2024_report.pdf", chat_id="99"
    )
    result = mem.attachment_guard(
        "Analyse the document I uploaded.",
        "I analysed /home/user/.kernel-evolving/workspace/documents/2024_report.pdf and found…",
        chat_id="99",
    )
    assert result["ok"] is True


def test_guard_reply_ignored_file(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(
        kind="document", local_path="/tmp/docs/invoice.pdf",
        original_name="invoice.pdf", chat_id="88"
    )
    result = mem.attachment_guard(
        "Please process the document.",
        "I don't see any files in the workspace.",
        chat_id="88",
    )
    assert result["ok"] is False
    assert "recent" in result
    assert result["recent"][0]["original_name"] == "invoice.pdf"


def test_guard_reply_mentions_documents_folder(isolated_memory):
    import core.memory.memory as mem
    mem.record_attachment(
        kind="document", local_path="/tmp/docs/x.pdf",
        original_name="x.pdf", chat_id="100"
    )
    # Even if the filename isn't mentioned, mentioning 'documents' is enough
    result = mem.attachment_guard(
        "Translate the file I sent.",
        "I found the file in your documents folder and translated it.",
        chat_id="100",
    )
    assert result["ok"] is True


# ── Schema migration safety ───────────────────────────────────────────────────

def test_existing_messages_table_not_broken(isolated_memory, tmp_path):
    """Creating the attachments table on a DB that already has messages must not break reads."""
    import core.memory.memory as mem

    # Seed a message via normal API
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    mem.save(msgs)

    # Now record an attachment (triggers _ensure_schema which now also creates attachments)
    mem.record_attachment(kind="document", local_path="/tmp/a.pdf", original_name="a.pdf")

    # Load messages must still work
    loaded = mem.load()
    assert any(m["content"] == "hello" for m in loaded)
