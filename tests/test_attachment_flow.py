"""
Regression tests for attachment context injection and guard-triggered retry.

Covers:
  1. upload → follow-up: attachment context block is injected into normal chat turns
  2. guard fail → retry: when first triage ignores the file, a second call is made
     with explicit file-use instruction; if second also fails, ⚠️ warning appended
  3. guard pass on retry: when retry references the file, no warning is appended
"""

import sys
import os
import importlib
import tempfile
import types
import sqlite3

import pytest

# ---------------------------------------------------------------------------
# Helpers to build a minimal in-memory memory module backed by a temp DB
# ---------------------------------------------------------------------------

def _make_memory_module(db_path: str):
    """Import memory.py with a patched DB_PATH so each test gets an isolated DB."""
    spec = importlib.util.spec_from_file_location(
        "memory_test_attach_flow",
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "memory", "memory.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.DB_PATH = db_path
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1 — attachment_context_block injected in follow-up turns
# ---------------------------------------------------------------------------

def test_context_block_injected_on_followup():
    """After a document upload, attachment_context_block returns a non-empty string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem.db")
        # Create a dummy file to attach
        dummy = os.path.join(tmpdir, "report.pdf")
        with open(dummy, "w") as f:
            f.write("dummy pdf content")

        mem = _make_memory_module(db_path)
        chat_id = "chat_99"

        # Simulate document upload recording
        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="report.pdf",
            mime_type="application/pdf",
            caption="Analyse this report",
            chat_id=chat_id,
        )

        # Follow-up turn: should get context block
        block = mem.attachment_context_block(chat_id=chat_id, limit=3)
        assert block, "Expected non-empty attachment context block after upload"
        assert "report.pdf" in block


def test_context_block_empty_when_no_attachment():
    """No attachments → context block is empty / falsy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem2.db")
        mem = _make_memory_module(db_path)
        block = mem.attachment_context_block(chat_id="chat_empty", limit=3)
        assert not block


def test_context_block_isolated_by_chat_id():
    """Attachment uploaded in chat A must not appear in chat B's context block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem3.db")
        dummy = os.path.join(tmpdir, "secret.pdf")
        with open(dummy, "w") as f:
            f.write("secret")
        mem = _make_memory_module(db_path)

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="secret.pdf",
            mime_type="application/pdf",
            caption="private",
            chat_id="chat_A",
        )

        block_b = mem.attachment_context_block(chat_id="chat_B", limit=3)
        assert not block_b, "chat_B must not see chat_A's attachment context"


# ---------------------------------------------------------------------------
# Test 2 — guard-triggered retry logic (unit-level simulation)
# ---------------------------------------------------------------------------

def test_guard_retry_called_when_first_triage_ignores_file():
    """
    Simulate the document triage path:
    - First triage returns a reply that ignores the file
    - Guard fires → retry triage called with augmented prompt
    - Retry returns a compliant reply → no ⚠️ appended
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem4.db")
        dummy = os.path.join(tmpdir, "doc.pdf")
        with open(dummy, "w") as f:
            f.write("important document")

        mem = _make_memory_module(db_path)
        chat_id = "chat_retry"

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="doc.pdf",
            mime_type="application/pdf",
            caption="Summarise this document",
            chat_id=chat_id,
        )

        att_ctx = mem.attachment_context_block(chat_id=chat_id, limit=3)
        user_caption = "Summarise this document"
        full_prompt = f"{att_ctx}\n\n{user_caption}\n\n[File saved to: {dummy}]"

        call_log = []

        def mock_triage(prompt, *args, **kwargs):
            call_log.append(prompt)
            if len(call_log) == 1:
                # First call: ignores the file
                return "Sure, here is a general summary of machine learning."
            else:
                # Retry call: references the file
                return f"Based on the file doc.pdf at {dummy}: it says 'important document'."

        # ---- replicate the bot logic inline ----
        reply = mock_triage(full_prompt)
        guard = mem.attachment_guard(user_caption, reply, chat_id=chat_id)

        if not guard["ok"]:
            retry_prompt = (
                f"{att_ctx}\n\n"
                f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                f"{full_prompt}"
            )
            reply = mock_triage(retry_prompt)
            guard2 = mem.attachment_guard(user_caption, reply, chat_id=chat_id)
            if not guard2["ok"]:
                reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

        assert len(call_log) == 2, f"Expected 2 triage calls (initial + retry), got {len(call_log)}"
        assert "IMPORTANT" in call_log[1], "Retry prompt must contain explicit file-use instruction"
        assert "⚠️" not in reply, "Warning must NOT be appended when retry references the file"
        assert "doc.pdf" in reply or dummy in reply


def test_guard_warning_appended_when_both_triage_calls_ignore_file():
    """
    When both the initial and retry triage calls ignore the file,
    the ⚠️ warning must be appended to the final reply.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem5.db")
        dummy = os.path.join(tmpdir, "data.csv")
        with open(dummy, "w") as f:
            f.write("col1,col2\n1,2")

        mem = _make_memory_module(db_path)
        chat_id = "chat_double_fail"

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="data.csv",
            mime_type="text/csv",
            caption="Analyse this file",
            chat_id=chat_id,
        )

        att_ctx = mem.attachment_context_block(chat_id=chat_id, limit=3)
        user_caption = "Analyse this file"
        full_prompt = f"{att_ctx}\n\n{user_caption}\n\n[File saved to: {dummy}]"

        def mock_triage_both_ignore(prompt, *args, **kwargs):
            return "Here is a general analysis of CSV files in Python."

        reply = mock_triage_both_ignore(full_prompt)
        guard = mem.attachment_guard(user_caption, reply, chat_id=chat_id)

        if not guard["ok"]:
            retry_prompt = (
                f"{att_ctx}\n\n"
                f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                f"{full_prompt}"
            )
            reply = mock_triage_both_ignore(retry_prompt)
            guard2 = mem.attachment_guard(user_caption, reply, chat_id=chat_id)
            if not guard2["ok"]:
                reply = reply + "\n\n⚠️ (Note: I may not have fully used your uploaded file — please confirm or re-ask if needed.)"

        assert "⚠️" in reply, "Warning must be appended when both triage calls ignore the file"


def test_no_retry_when_first_triage_references_file():
    """
    When the first triage already references the file, guard passes →
    no retry should occur.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "mem6.db")
        dummy = os.path.join(tmpdir, "notes.txt")
        with open(dummy, "w") as f:
            f.write("meeting notes")

        mem = _make_memory_module(db_path)
        chat_id = "chat_pass"

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="notes.txt",
            mime_type="text/plain",
            caption="Summarise these notes",
            chat_id=chat_id,
        )

        att_ctx = mem.attachment_context_block(chat_id=chat_id, limit=3)
        user_caption = "Summarise these notes"
        full_prompt = f"{att_ctx}\n\n{user_caption}\n\n[File saved to: {dummy}]"

        call_log = []

        def mock_triage_pass(prompt, *args, **kwargs):
            call_log.append(prompt)
            return f"From notes.txt at {dummy}: the file contains meeting notes."

        reply = mock_triage_pass(full_prompt)
        guard = mem.attachment_guard(user_caption, reply, chat_id=chat_id)

        if not guard["ok"]:
            reply = mock_triage_pass(full_prompt + " [retry]")

        assert len(call_log) == 1, "Must not retry when first triage passes guard"
        assert "⚠️" not in reply
