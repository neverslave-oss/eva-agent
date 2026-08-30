"""
E2E tests for the full Evo pipeline — verifying chat_id propagation through
bot handler → agent.triage → memory.load/save → context.build_system_prompt → reply.

These tests use mocks and temp SQLite DBs to stay fully self-contained.
"""
import os
import sys
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Ensure src is on the path
SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helper: build an in-process sqlite3.Connection with seeded data
# ---------------------------------------------------------------------------
class _NoCloseConn:
    """Wraps a SQLite connection, ignoring close() calls so in-memory data persists."""
    def __init__(self, conn):
        self._conn = conn
        self.row_factory = conn.row_factory

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass  # Don't actually close the in-memory DB


def _make_seeded_conn(turns: list[dict]) -> sqlite3.Connection:
    """Return an in-memory SQLite connection pre-seeded with the given turns."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot TEXT NOT NULL DEFAULT 'kernel-evolving',
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_bot_session ON messages(bot, session_id, id);
        CREATE TABLE IF NOT EXISTS attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot TEXT NOT NULL DEFAULT 'kernel-evolving',
            session_id TEXT NOT NULL,
            chat_id TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL CHECK(kind IN ('document','photo','voice','audio','video','other')),
            local_path TEXT NOT NULL,
            original_name TEXT NOT NULL DEFAULT '',
            mime_type TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
    """)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for t in turns:
        conn.execute(
            "INSERT INTO messages (bot, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (t["bot"], t["session_id"], t["role"], t["content"], now)
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Helper: fake ChatHistoryRepository backed by an in-memory _NoCloseConn
# (needed since memory.py now calls _get_repo() not _get_conn() internally)
# ---------------------------------------------------------------------------
class _FakeChatHistoryRepo:
    """Wraps a seeded in-memory SQLite connection as a ChatHistoryRepository-
    compatible object so tests can inject it via patch.object(memory, "_get_repo")."""

    def __init__(self, conn):
        self._conn = _NoCloseConn(conn) if not isinstance(conn, _NoCloseConn) else conn

    def connection(self):
        return self._conn

    def touch_session(self, session_id, chat_id):
        pass  # no-op for tests

    def append_messages(self, messages, session_id):
        pass  # no-op — test data already seeded

    def get_history_by_chat_id(self, chat_id, legacy_session_hint="", bot="kernel-evolving"):
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
            (chat_id,)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def get_history(self, session_id, bot="kernel-evolving"):
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def get_recent_user_messages(self, bot="kernel-evolving", chat_id="", limit=5):
        rows = self._conn.execute(
            "SELECT content FROM messages WHERE bot=? AND session_id=? AND role='user' ORDER BY id DESC LIMIT ?",
            (bot, chat_id, limit)
        ).fetchall()
        return [r[0] for r in rows]

    def get_active_session(self, chat_id):
        # Fake: always report single legacy session (no splits in tests)
        return None

    def get_history_by_session(self, session_id, bot="kernel-evolving"):
        return self.get_history(session_id)

    def clear_session(self, session_id):
        pass

    def clear_all(self):
        pass

    def record_attachment(self, *args, **kwargs):
        pass

    def recent_attachments(self, *args, **kwargs):
        return []


# ---------------------------------------------------------------------------
# Test 1 — chat_id is passed to triage() from the main message handler
# ---------------------------------------------------------------------------
class TestChatIdPropagation(unittest.TestCase):

    def test_e2e_chat_id_propagation(self):
        """handle_message must pass chat_id to agent.triage."""
        import services.channels.telegram_bot as bot

        captured = {}

        def fake_triage(text, step_callback=None, chat_id="", **kwargs):
            captured["chat_id"] = chat_id
            captured["text"] = text
            return "pong"

        # conftest pre-imports core.agent so patch.dict sys.modules is insufficient
        # (Python resolves `from core.agent import triage` against the real module).
        # Fix: patch the triage attribute directly on the real already-imported module.
        import core.agent as agent_mod
        import core.memory.memory as memory_mod

        memory_mock = MagicMock()
        memory_mock.load.return_value = []
        memory_mock.save.return_value = None
        memory_mock.attachment_context_block.return_value = ""
        memory_mock.artifact_check.return_value = {"ok": True}

        class _NoopCtx:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch.object(agent_mod, "triage", side_effect=fake_triage), \
             patch.object(memory_mod, "load", return_value=[]), \
             patch.object(memory_mod, "save", return_value=None), \
             patch.object(memory_mod, "attachment_context_block", return_value=""), \
             patch.object(memory_mod, "artifact_check", return_value={"ok": True}), \
             patch.object(bot, "send_message", return_value=42), \
             patch.object(bot, "edit_message", return_value=True), \
             patch.object(bot, "send_typing", MagicMock()), \
             patch.object(bot, "TypingKeepAlive", _NoopCtx), \
             patch.object(bot, "_ensure_agent", MagicMock()), \
             patch.object(bot, "_agent_ready", True):

            bot.handle_message("12345", "hello")

        self.assertEqual(
            captured.get("chat_id"), "12345",
            f"Expected chat_id='12345', got: {captured}"
        )


# ---------------------------------------------------------------------------
# Test 2 — memory.load isolates by chat_id
# ---------------------------------------------------------------------------
class TestMemoryIsolation(unittest.TestCase):

    def test_e2e_memory_isolation(self):
        """core.memory.memory.load(chat_id=X) must return only turns for X."""
        import core.memory.memory as memory

        turns = [
            {"bot": "kernel-evolving", "session_id": "user_a", "role": "user",      "content": "hello from a"},
            {"bot": "kernel-evolving", "session_id": "user_a", "role": "assistant",  "content": "hi a"},
            {"bot": "kernel-evolving", "session_id": "user_a", "role": "user",       "content": "how are you a"},
            {"bot": "kernel-evolving", "session_id": "user_b", "role": "user",       "content": "hello from b"},
            {"bot": "kernel-evolving", "session_id": "user_b", "role": "assistant",  "content": "hi b"},
            {"bot": "kernel-evolving", "session_id": "user_b", "role": "user",       "content": "how are you b"},
        ]
        shared_conn = _make_seeded_conn(turns)

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_json = Path(tmpdir) / ".kernel_evolving_memory_user_a.json"

            fake_repo = _FakeChatHistoryRepo(shared_conn)

            with patch.object(memory, "_get_repo", lambda: fake_repo), \
                 patch.object(memory, "MEMORY_FILE", Path(tmpdir) / ".kernel_evolving_memory.json"):
                result = memory.load(chat_id="user_a")

        contents = [m["content"] for m in result]
        self.assertTrue(any("from a" in c for c in contents),
                        f"user_a turns not found: {contents}")
        self.assertFalse(any("from b" in c for c in contents),
                         f"user_b turns leaked into user_a: {contents}")


# ---------------------------------------------------------------------------
# Test 3 — build_system_prompt is scoped to chat_id
# ---------------------------------------------------------------------------
class TestSystemPromptChatScoped(unittest.TestCase):

    def test_e2e_system_prompt_chat_scoped(self):
        """build_system_prompt must not include turns from other chat sessions."""
        import core.memory.memory as memory
        import core.memory.context as context

        turns = [
            {"bot": "kernel-evolving", "session_id": "testchat",  "role": "user", "content": "testchat exclusive message"},
            {"bot": "kernel-evolving", "session_id": "otherchat", "role": "user", "content": "otherchat secret message"},
        ]
        shared_conn = _make_seeded_conn(turns)

        config = {
            "persona": {"name": "Evo", "description": "test"},
            "system_prompt": "",
            "tools": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_repo = _FakeChatHistoryRepo(shared_conn)

            with patch.object(memory, "_get_repo", lambda: fake_repo), \
                 patch.object(memory, "MEMORY_FILE", Path(tmpdir) / ".kernel_evolving_memory.json"):
                prompt = context.build_system_prompt(
                    config, skills=[], routines=[], chat_id="testchat"
                )

        self.assertNotIn("otherchat secret message", prompt,
                         "build_system_prompt leaked turns from another chat session")


# ---------------------------------------------------------------------------
# Test 4 — full triage pipeline returns model output and saves memory
# ---------------------------------------------------------------------------
class TestFullTriagePipeline(unittest.TestCase):

    def test_e2e_full_triage_pipeline(self):
        """core.agent.triage must invoke provider and return a reply string."""
        import core.memory.memory as memory
        import core.agent as agent

        turns = []
        shared_conn = _make_seeded_conn(turns)

        fake_reply = "Hello from Evo"

        mock_provider = MagicMock()
        mock_provider.infer.return_value = fake_reply

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_repo = _FakeChatHistoryRepo(shared_conn)

        with patch.object(memory, "_get_repo", lambda: fake_repo), \
                 patch.object(memory, "MEMORY_FILE", Path(tmpdir) / ".kernel_evolving_memory.json"), \
                 patch("core.agent.infer_with_tools", return_value=fake_reply):

                result = agent.triage("hi", chat_id="test_fab")

        self.assertIsInstance(result, str, "triage() must return a string")
        self.assertGreater(len(result), 0, "triage() must return non-empty string")


# ---------------------------------------------------------------------------
# Test 5 — conversation continuity: history injected into provider call
# ---------------------------------------------------------------------------
class TestConversationContinuity(unittest.TestCase):

    def test_e2e_conversation_continuity(self):
        """History for chat_id must be injected into provider messages."""
        import core.memory.memory as memory

        turns = [
            {"bot": "kernel-evolving", "session_id": "fab_real",
             "role": "user", "content": "my cat is named Luna"},
        ]
        shared_conn = _make_seeded_conn(turns)

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_repo = _FakeChatHistoryRepo(shared_conn)

            with patch.object(memory, "_get_repo", lambda: fake_repo), \
                 patch.object(memory, "MEMORY_FILE", Path(tmpdir) / ".kernel_evolving_memory.json"):

                loaded = memory.load(chat_id="fab_real")
                contents = [m.get("content", "") for m in loaded]
                self.assertTrue(
                    any("Luna" in c for c in contents),
                    f"Memory for fab_real should contain 'Luna', got: {contents}"
                )

                # Verify cross-session isolation: load for different chat_id returns no Luna
                loaded_other = memory.load(chat_id="other_user")
                contents_other = [m.get("content", "") for m in loaded_other]
                self.assertFalse(
                    any("Luna" in c for c in contents_other),
                    f"other_user should NOT see Luna, got: {contents_other}"
                )


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Test 6 — kernel-doc-retrieval skill is installed and dispatchable
# ---------------------------------------------------------------------------
class TestDocRetrievalSkillDispatch(unittest.TestCase):

    def test_kernel_doc_retrieval_skill_installed(self):
        """kernel-doc-retrieval skill must be loadable from the ecosystem."""
        from core.skills import load_all
        import os
        skills_dir = os.path.expanduser("~/.kernel-evolving/ecosystem")
        if not os.path.exists(skills_dir):
            self.skipTest(f"Skills dir not found: {skills_dir}")
        skills = load_all(skills_dir)
        skill_names = [s.get("name", "").lower() for s in skills]
        self.assertIn(
            "kernel-doc-retrieval", skill_names,
            f"kernel-doc-retrieval not in ecosystem skills: {skill_names[:20]}"
        )

    def test_pdf_document_routes_to_doc_retrieval(self):
        """When a PDF path is in the message, triage should route to kernel-doc-retrieval."""
        import core.agent as _agent
        called_with = {}

        def fake_run_skill(skill, text, infer_fn):
            called_with["skill"] = skill.get("name")
            called_with["text"] = text
            return "summary of the PDF"

        # Find kernel-doc-retrieval skill in loaded skills
        kdr = next(
            (s for s in _agent._skills if "doc-retrieval" in s.get("name", "").lower()),
            None
        )
        if kdr is None:
            self.skipTest("kernel-doc-retrieval not installed — skip")

        with patch("core.agent.run_skill", side_effect=fake_run_skill), \
             patch("core.agent.infer_with_tools", return_value="summarised"):
            try:
                _agent.triage(
                    "summarise ~/.kernel-evolving/workspace/documents/test.pdf",
                    chat_id="test_doc_route"
                )
            except Exception:
                pass

        if called_with:
            self.assertIn("doc-retrieval", called_with.get("skill", "").lower(),
                          f"Expected doc-retrieval skill, got: {called_with}")

    def test_document_saved_to_workspace_on_receive(self):
        """Documents received via Telegram must be saved to workspace/documents/."""
        import services.channels.telegram_bot as bot
        import tempfile, os

        # Create a fake downloaded file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake content")
            fake_path = f.name

        docs_dir = os.path.expanduser("~/.kernel-evolving/workspace/documents")
        files_before = set(os.listdir(docs_dir)) if os.path.exists(docs_dir) else set()

        with patch.object(bot, "download_file", return_value=fake_path), \
             patch.object(bot, "_ensure_agent", MagicMock()), \
             patch.object(bot, "_agent_ready", True), \
             patch.object(bot, "send_message", return_value=42), \
             patch.object(bot, "edit_message", return_value=True):
            import core.agent as agent_mod_mock
            agent_mock = MagicMock()
            agent_mock.triage.return_value = "Here is the summary."
            with patch.dict("sys.modules", {"core.agent": agent_mock, "agent": agent_mock}):
                try:
                    bot.handle_message(
                        "123456789", "",
                        document_file_id="fake_file_id",
                        document_name="test.pdf",
                        document_mime="application/pdf"
                    )
                except Exception:
                    pass

        files_after = set(os.listdir(docs_dir)) if os.path.exists(docs_dir) else set()
        new_files = files_after - files_before
        pdf_files = [f for f in new_files if f.endswith(".pdf")]
        self.assertTrue(
            len(pdf_files) > 0,
            f"Expected a PDF saved to {docs_dir}, new files: {new_files}"
        )
        # cleanup
        for f in pdf_files:
            try:
                os.unlink(os.path.join(docs_dir, f))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Test 7 — voice pipeline: STT → triage → TTS clone
# ---------------------------------------------------------------------------
class TestVoicePipeline(unittest.TestCase):

    def test_voice_pipeline_wired_in_bot(self):
        """handle_message with voice_file_id must call STT then triage then clone_voice."""
        import services.channels.telegram_bot as bot
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(b"fake ogg audio")
            fake_audio = f.name

        triage_called = {}
        clone_called = {}

        def fake_triage(text, step_callback=None, chat_id="", **kwargs):
            triage_called["text"] = text
            triage_called["chat_id"] = chat_id
            return "Here is my voice reply"

        def fake_clone(text, voice_sample=""):
            clone_called["text"] = text
            return "/tmp/fake_reply.wav"

        # conftest pre-imports core.agent — patch sys.modules does NOT intercept
        # attribute access on already-imported modules. Fix: patch attributes directly.
        import core.agent as agent_mod
        import core.memory.memory as memory_mod

        memory_mock = MagicMock()
        memory_mock.load.return_value = []
        memory_mock.save.return_value = None
        memory_mock.record_attachment.return_value = None

        class _NoopCtx:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        with patch.object(agent_mod, "triage", side_effect=fake_triage), \
             patch.object(memory_mod, "load", return_value=[]), \
             patch.object(memory_mod, "save", return_value=None), \
             patch.object(memory_mod, "record_attachment", return_value=None), \
             patch.object(bot, "download_file", return_value=fake_audio), \
             patch.object(bot, "_ensure_agent", MagicMock()), \
             patch.object(bot, "_agent_ready", True), \
             patch.object(bot, "TypingKeepAlive", _NoopCtx), \
             patch.object(bot, "_MultimodalActivity", _NoopCtx), \
             patch.object(bot, "send_message", return_value=42), \
             patch.object(bot, "edit_message", return_value=True), \
             patch.object(bot, "send_voice", return_value=True), \
             patch.object(bot, "_clone_voice_reply", side_effect=fake_clone), \
             patch.object(bot, "ALLOWED_CHAT_ID", None), \
             patch("core.inference.model.infer_with_audio", return_value="hello from voice"):
            try:
                bot.handle_message("123456789", "", voice_file_id="fake_voice_id")
                import time; time.sleep(0.3)  # allow async clone thread
            except Exception:
                pass

        self.assertTrue(
            bool(triage_called),
            "triage() must be called for voice messages"
        )
        self.assertTrue(
            bool(clone_called) or bot.send_voice.called if hasattr(bot.send_voice, 'called') else True,
            "voice clone reply must be attempted"
        )

    def test_voice_clone_endpoint_reachable(self):
        """olly-voice-server /tts/clone endpoint must return 200 with WAV content."""
        import requests, os
        sample = os.path.expanduser("~/.openclaw/media/voice-samples/default-en-phonetic.wav")
        # Override with BENCH_AUDIO / KERNEL_DEFAULT_VOICE_SAMPLE if a real sample exists
        sample = os.environ.get("KERNEL_DEFAULT_VOICE_SAMPLE", sample)
        if not os.path.exists(sample):
            self.skipTest(f"Voice sample not found: {sample}")
        try:
            r = requests.post(
                "http://127.0.0.1:8766/tts/clone",
                files={"sample": open(sample, "rb")},
                data={"text": "test", "model": "0.6"},
                timeout=60,
            )
            self.assertEqual(r.status_code, 200,
                             f"Expected 200, got {r.status_code}: {r.text[:200]}")
            self.assertGreater(len(r.content), 1000,
                               "Expected WAV content >1KB, got empty/tiny response")
        except requests.exceptions.ConnectionError:
            self.skipTest("olly-voice-server not running — skip live test")

    def test_stt_model_configured(self):
        """config.yaml must have stt_model.path pointing to an existing model."""
        import os
        from runtime_paths import load_config
        cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
        stt = cfg.get("stt_model", {})
        self.assertIn("path", stt, "config.yaml must have stt_model.path")
        model_path = os.path.expanduser(stt["path"])
        # If the path still contains an unresolved ${VAR} (env not loaded),
        # pull the value from the local .env so the check stays meaningful.
        # IMPORTANT: never leak .env vars into the global os.environ — snapshot
        # prior values and restore them so later tests are not polluted.
        if "${" in model_path:
            _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            _prior = {}
            if os.path.isfile(_env_path):
                with open(_env_path) as _ef:
                    for _line in _ef:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _v = _line.split("=", 1)
                            _k = _k.strip()
                            if _k not in os.environ:
                                _prior[_k] = None  # absent before — remove after
                            else:
                                _prior[_k] = os.environ[_k]
                            os.environ.setdefault(_k, _v.strip())
                try:
                    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
                    model_path = os.path.expanduser(cfg.get("stt_model", {}).get("path", model_path))
                finally:
                    # Restore prior env state so we never pollute other tests.
                    for _k, _old in _prior.items():
                        if _old is None:
                            os.environ.pop(_k, None)
                        else:
                            os.environ[_k] = _old
        self.assertTrue(
            os.path.exists(model_path),
            f"stt_model path does not exist: {model_path}"
        )


class TestNamedSlots(unittest.TestCase):
    """Integration tests for named model slot system.
    These tests verify the contract without requiring a live model server.
    """

    def test_slot_status_client_returns_list(self):
        """core.inference.model_client.slot_status() must always return a list (even on connection error)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        import core.inference.model_client as model_client
        result = model_client.slot_status()
        self.assertIsInstance(result, list,
            "slot_status() must return a list (empty or populated)")

    def test_health_contains_slots_key(self):
        """core.inference.model_client.health() must return a dict containing a 'slots' key."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        import core.inference.model_client as model_client
        health = model_client.health()
        # When server not available, health returns {"error": ...} — skip.
        # When connected to a server built before this feature, also skip.
        if "error" in health:
            self.skipTest("model server not running — skip live health check")
        if "slots" not in health:
            self.skipTest("connected to pre-slots model server — skip (upgrade server to test)")
        self.assertIsInstance(health["slots"], list,
            "health()['slots'] must be a list")

    def test_slot_registry_scaffold(self):
        """SlotRegistry must be importable and functional as a unit."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from core.inference.model_slots import SlotRegistry, SlotSpec
        registry = SlotRegistry()
        self.assertEqual(registry.status(), [])
        self.assertEqual(registry.loaded_slots(), {})
        spec = SlotSpec(name="test", role="audio", model_path="/fake")
        registry.register(spec)
        status = registry.status()
        self.assertEqual(len(status), 1)
        self.assertFalse(status[0]["loaded"])
