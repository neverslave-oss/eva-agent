"""
Tests for artifact_check() and the normal-chat completion gate logic.

Covers:
  1. artifact_check: no artifact request → ok=True
  2. artifact_check: request but no file produced → ok=False
  3. artifact_check: request, file written to workspace → ok=True
  4. artifact_check: request, reply claims 'saved to …' → ok=True (trust claim)
  5. Normal-chat guard+retry: guard fail triggers retry with explicit instruction
  6. Normal-chat guard+retry: guard pass on first attempt → no retry
"""

import sys
import os
import importlib
import tempfile
import time
import types

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_module(db_path: str):
    spec = importlib.util.spec_from_file_location(
        "memory_test_completion",
        os.path.join(os.path.dirname(__file__), "..", "src", "core", "memory", "memory.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    mod.DB_PATH = db_path
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# artifact_check tests
# ---------------------------------------------------------------------------

def test_artifact_check_no_request():
    """Messages without artifact-request keywords always pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = _make_memory_module(os.path.join(tmpdir, "m.db"))
        r = mem.artifact_check("What is the capital of France?", "Paris.", workspace=tmpdir)
        assert r["ok"]


def test_artifact_check_request_no_file():
    """User asked to generate something but no new file exists → fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = _make_memory_module(os.path.join(tmpdir, "m.db"))
        r = mem.artifact_check(
            "Generate a summary report",
            "Here is a nice summary of the topic.",
            workspace=tmpdir,
            since_seconds=30,
        )
        assert not r["ok"]
        assert "reason" in r


def test_artifact_check_file_written():
    """A file written to workspace within the window satisfies the check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = _make_memory_module(os.path.join(tmpdir, "m.db"))
        # Write a file inside the workspace
        out = os.path.join(tmpdir, "report.md")
        with open(out, "w") as f:
            f.write("# Report\n\nContent here.")
        r = mem.artifact_check(
            "Write a report on AI trends",
            "I've written the report.",
            workspace=tmpdir,
            since_seconds=30,
        )
        assert r["ok"]


def test_artifact_check_reply_claims_saved():
    """If reply says 'saved to …', trust it even without a detected file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = _make_memory_module(os.path.join(tmpdir, "m.db"))
        r = mem.artifact_check(
            "Export this as a JSON file",
            "Done — saved to /tmp/output.json",
            workspace=tmpdir,
            since_seconds=5,
        )
        assert r["ok"]


def test_artifact_check_summarise_keyword():
    """'summarise' is a recognised artifact keyword."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = _make_memory_module(os.path.join(tmpdir, "m.db"))
        r = mem.artifact_check(
            "Summarise this document into a file",
            "Here is a text summary. No file.",
            workspace=tmpdir,
            since_seconds=5,
        )
        assert not r["ok"]


# ---------------------------------------------------------------------------
# Normal-chat guard + retry simulation tests
# ---------------------------------------------------------------------------

def test_normal_chat_guard_retry_fires_on_fail():
    """
    When guard fails on normal chat triage, retry must be called with
    explicit file-use instruction prepended.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "m.db")
        dummy = os.path.join(tmpdir, "notes.pdf")
        with open(dummy, "w") as f:
            f.write("meeting notes")

        mem = _make_memory_module(db_path)
        chat_id = "chat_normal_guard"

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="notes.pdf",
            mime_type="application/pdf",
            caption="What's in the file?",
            chat_id=chat_id,
        )

        att_ctx = mem.attachment_context_block(chat_id=chat_id, limit=3)
        user_text = "What's in the file?"
        calls = []

        def mock_triage(prompt, *args, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return "Files are containers for data."  # ignores specific file
            return f"The file notes.pdf at {dummy} contains meeting notes."

        # --- replicate bot normal-chat gate ---
        triage_text = f"{att_ctx}\n\n{user_text}" if att_ctx else user_text
        reply = mock_triage(triage_text)

        if att_ctx:
            guard = mem.attachment_guard(user_text, reply, chat_id=chat_id)
            if not guard["ok"]:
                retry_text = (
                    f"{att_ctx}\n\n"
                    f"IMPORTANT: The user sent a file. You MUST reference and use it.\n\n"
                    f"{user_text}"
                )
                reply = mock_triage(retry_text)
                guard2 = mem.attachment_guard(user_text, reply, chat_id=chat_id)
                if not guard2["ok"]:
                    reply = reply + "\n\n⚠️ (Note: ...)"

        assert len(calls) == 2, f"Expected 2 triage calls; got {len(calls)}"
        assert "IMPORTANT" in calls[1]
        assert "⚠️" not in reply
        assert "notes.pdf" in reply


def test_normal_chat_guard_no_retry_when_pass():
    """When first triage already references the file, no retry is made."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "m2.db")
        dummy = os.path.join(tmpdir, "spec.pdf")
        with open(dummy, "w") as f:
            f.write("spec content")

        mem = _make_memory_module(db_path)
        chat_id = "chat_normal_pass"

        mem.record_attachment(
            kind="document",
            local_path=dummy,
            original_name="spec.pdf",
            mime_type="application/pdf",
            caption="Summarise the file",
            chat_id=chat_id,
        )

        att_ctx = mem.attachment_context_block(chat_id=chat_id, limit=3)
        user_text = "Summarise the file"
        calls = []

        def mock_triage_pass(prompt, *args, **kwargs):
            calls.append(prompt)
            return f"From spec.pdf at {dummy}: spec content."

        triage_text = f"{att_ctx}\n\n{user_text}" if att_ctx else user_text
        reply = mock_triage_pass(triage_text)

        if att_ctx:
            guard = mem.attachment_guard(user_text, reply, chat_id=chat_id)
            if not guard["ok"]:
                reply = mock_triage_pass(triage_text + " [retry]")

        assert len(calls) == 1, "Should not retry when guard passes"
        assert "⚠️" not in reply
