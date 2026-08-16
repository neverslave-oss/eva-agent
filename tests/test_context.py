"""test_context.py — Tests for context.py grounding fixes (chat_id filtering)."""
import sys
import os
import json
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _seed_db(db_path: Path, messages_by_chat: dict) -> None:
    """Seed a SQLite DB with messages for multiple chat_ids (used as session_id)."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT    PRIMARY KEY,
            chat_id     TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL,
            updated_at  TEXT    NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            bot        TEXT    NOT NULL DEFAULT 'kernel-evolving',
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        );
    """)
    now = "2026-01-01T00:00:00Z"
    for chat_id, msgs in messages_by_chat.items():
        # Register session so get_history_by_chat_id can find it
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, chat_id, created_at, updated_at, message_count) VALUES (?,?,?,?,?)",
            (chat_id, chat_id, now, now, len(msgs))
        )
        for m in msgs:
            conn.execute(
                "INSERT INTO messages (bot, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
                ("kernel-evolving", chat_id, m["role"], m["content"], now)
            )
    conn.commit()
    conn.close()


def test_recent_topics_filters_by_chat_id(tmp_path, monkeypatch):
    """_recent_conversation_topics(chat_id='chat_a') should only return chat_a messages."""
    db_path = tmp_path / "chat_history_evolving.db"
    _seed_db(db_path, {
        "chat_a": [
            {"role": "user", "content": "chat_a message one"},
            {"role": "assistant", "content": "reply one"},
        ],
        "chat_b": [
            {"role": "user", "content": "chat_b unrelated message"},
        ],
    })

    import core.memory.memory as mem_mod
    monkeypatch.setattr(mem_mod, "DB_FILE", db_path)

    from core.memory.context import _recent_conversation_topics
    result = _recent_conversation_topics(limit=10, chat_id="chat_a")

    assert "chat_a message one" in result, f"Expected chat_a content in result, got: {result}"
    assert "chat_b unrelated message" not in result, \
        f"chat_b content should be filtered out, got: {result}"


def test_memory_stats_chat_specific(tmp_path, monkeypatch):
    """_memory_stats(chat_id='testchat') should return non-empty stats from chat-specific file."""
    # Create a chat-specific memory JSON file
    mem_data = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
    }
    # The _chat_memory_file logic: safe name from chat_id
    safe = "testchat"
    mem_file = tmp_path / f".kernel_evolving_memory_{safe}.json"
    mem_file.write_text(json.dumps(mem_data))

    import core.memory.memory as mem_mod
    # Monkeypatch MEMORY_FILE parent so _chat_memory_file resolves to tmp_path
    monkeypatch.setattr(mem_mod, "MEMORY_FILE", tmp_path / ".kernel_evolving_memory.json")

    from core.memory.context import _memory_stats
    result = _memory_stats(chat_id="testchat")

    assert result, f"Expected non-empty stats, got: {result}"
    assert result.get("total", 0) == 3, f"Expected 3 total messages, got: {result}"
    assert result.get("user_turns", 0) == 2, f"Expected 2 user turns, got: {result}"


def test_memory_load_returns_full_session_history(tmp_path, monkeypatch):
    """core.memory.memory.load(chat_id=X) should return the full stored session history, not just the hot cache window."""
    db_path = tmp_path / "chat_history_evolving.db"
    _seed_db(db_path, {
        "chat_full": [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "turn 2"},
            {"role": "user", "content": "turn 3"},
            {"role": "assistant", "content": "turn 4"},
            {"role": "user", "content": "turn 5"},
            {"role": "assistant", "content": "turn 6"},
        ],
    })

    import core.memory.memory as mem_mod
    monkeypatch.setattr(mem_mod, "DB_FILE", db_path)
    monkeypatch.setattr(mem_mod, "MEMORY_FILE", tmp_path / ".kernel_evolving_memory.json")
    monkeypatch.setattr(mem_mod, "MAX_TURNS", 1)

    loaded = mem_mod.load(chat_id="chat_full")

    assert len(loaded) == 6, f"Expected the full session history, got {len(loaded)} turns: {loaded}"
    assert loaded[0]["content"] == "turn 1"
    assert loaded[-1]["content"] == "turn 6"


# ── Identity clarity tests (v1.19.3) ────────────────────────────────────────

def _build_prompt(tmp_path, monkeypatch, user_name="Fabio", user_notes=None):
    """Helper: build a system prompt with a real user.json."""
    import json
    user_data = {"name": user_name, "timezone": "Europe/Berlin", "notes": user_notes or ["Fabio has 10 cats."]}
    user_file = tmp_path / "user.json"
    user_file.write_text(json.dumps(user_data))

    import core.memory.context as ctx
    monkeypatch.setattr(ctx, "_load_user_profile", lambda: user_data)
    monkeypatch.setattr(ctx, "_load_user_md", lambda: "")
    monkeypatch.setattr(ctx, "_load_agents_md", lambda: "")
    monkeypatch.setattr(ctx, "_load_long_term_memory", lambda: "")
    monkeypatch.setattr(ctx, "_load_notes", lambda: [])
    monkeypatch.setattr(ctx, "_last_interaction", lambda: None)
    monkeypatch.setattr(ctx, "_memory_stats", lambda chat_id=None: {})
    monkeypatch.setattr(ctx, "_recent_conversation_topics", lambda limit=5, chat_id="": "")
    monkeypatch.setattr(ctx, "_system_resources", lambda: {})
    monkeypatch.setattr(ctx, "_olly_alive", lambda url: True)
    monkeypatch.setattr(ctx, "_service_status", lambda port: "up")
    config = {"model": {"name": "test-model", "device": "cpu"}, "kernel_workspace": str(tmp_path)}
    return ctx.build_system_prompt(config=config, skills=[], routines=[], chat_id="test123")


def test_system_prompt_uses_person_you_are_talking_to(tmp_path, monkeypatch):
    """Plan 007: user is referenced in the prompt."""
    prompt = _build_prompt(tmp_path, monkeypatch)
    # With or without USER.md, the user's name should appear
    assert "Fabio" in prompt, (
        f"Expected 'Fabio' in prompt, got: {prompt[:800]}"
    )
    assert "## Your user" not in prompt, (
        "Old '## Your user' header should be replaced; found it in prompt"
    )


def test_system_prompt_known_facts_header(tmp_path, monkeypatch):
    """When USER.md contains facts, they appear in the prompt; fallback json notes also work."""
    # Scenario A: USER.md with known facts
    user_md = tmp_path / "USER.md"
    user_md.write_text("# USER.md\n\n## Known facts\n- Fabio has 10 cats.\n", encoding="utf-8")
    monkeypatch.setattr("core.memory.context._load_user_md", lambda: user_md.read_text())
    prompt_a = _build_prompt(tmp_path, monkeypatch)
    assert "10 cats" in prompt_a, "USER.md facts should appear in prompt when file has content"

    # Scenario B: USER.md absent — fallback to json notes
    monkeypatch.setattr("core.memory.context._load_user_md", lambda: "")
    prompt_b = _build_prompt(tmp_path, monkeypatch, user_notes=["Fabio has 10 cats."])
    assert "10 cats" in prompt_b, "json fallback notes should appear when USER.md is empty"


def test_system_prompt_anti_confusion_line(tmp_path, monkeypatch):
    """The user's name must appear in the prompt — identity clarity."""
    prompt = _build_prompt(tmp_path, monkeypatch)
    # Plan 007: user is identified inline; the model should know who it's talking to
    assert "Fabio" in prompt, (
        f"User name 'Fabio' missing from prompt. Got: {prompt[:800]}"
    )


def test_agents_md_injected_in_system_prompt(tmp_path, monkeypatch):
    """When AGENTS.md is present, its content appears verbatim in the system prompt.
    The section heading was removed (AGENTS.md IS the identity, no wrapper needed)."""
    fake_agents_content = "# Kernel Agent Rules\n\nBe concise. Do not leak data. Use tools first."

    import core.memory.context as ctx
    monkeypatch.setattr(ctx, "_load_agents_md", lambda: fake_agents_content)
    monkeypatch.setattr(ctx, "_load_user_profile", lambda: {"name": "Fabio", "timezone": "Europe/Berlin", "notes": []})
    monkeypatch.setattr(ctx, "_load_long_term_memory", lambda: "")
    monkeypatch.setattr(ctx, "_load_notes", lambda: [])
    monkeypatch.setattr(ctx, "_last_interaction", lambda: None)
    monkeypatch.setattr(ctx, "_memory_stats", lambda chat_id=None: {})
    monkeypatch.setattr(ctx, "_recent_conversation_topics", lambda limit=5, chat_id="": "")
    monkeypatch.setattr(ctx, "_system_resources", lambda: {})
    monkeypatch.setattr(ctx, "_olly_alive", lambda url: True)
    monkeypatch.setattr(ctx, "_service_status", lambda port: "up")
    config = {"model": {"name": "test-model", "device": "cpu"}, "kernel_workspace": str(tmp_path)}

    prompt = ctx.build_system_prompt(config=config, skills=[], routines=[], chat_id="test")

    assert fake_agents_content in prompt, "AGENTS.md content should appear verbatim in the system prompt"
    # The old '## Agent behaviour guidelines' wrapper heading was removed;
    # AGENTS.md content is injected directly — no section wrapper needed.
    assert "## Agent behaviour guidelines" not in prompt, (
        "Old wrapper heading should no longer exist — AGENTS.md is injected directly"
    )
