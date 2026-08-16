"""
tests/test_telegram_missing_coverage.py - Missing test coverage requested by Fabio.

Covers four explicit gaps:
  1. "/local" command -> sets task_inference to "local"/Nemotron routing (provider/set payload).
  2. Model picker flow for capabilities/pipelines (vision, stt, tts):
       cmprov_* callback dispatches to modal model picker.
       cmm_* callback dispatches to _apply_modal_model.
  3. Tool-calling + streaming output path: chunk_callback and step_callback are
     wired through triage and reflected via send_message / edit_message behavior.
  4. Sending attachments via Telegram chat: send_file() - the sendDocument success path.

Design
------
All stubs follow the same pattern as test_inline_buttons.py:
  * Build a minimal stub dict for sys.modules.
  * Use patch.dict(sys.modules, ...) + importlib.util.spec_from_file_location to
    load telegram_bot without real I/O.
  * No external network calls; requests.post / requests.get are MagicMock.
  * No DB access; all memory helpers are stubbed.
  * Tests are grouped by feature area into separate TestCase classes.
"""

import os
import sys
import json
import types
import unittest
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Pre-load real core submodules BEFORE any sys.modules patching occurs.
# This avoids segfaults from importing C-extension-heavy libraries (datasets,
# torch/arrow) inside a patch.dict context that partially replaces sys.modules.
# TestStreamingAndToolCallOutput.setUp monkeypatches these module objects directly.
# ---------------------------------------------------------------------------
import importlib as _importlib
_REAL_MEM = _importlib.import_module('core.memory.memory')
_REAL_CTX = _importlib.import_module('core.memory.context')
_REAL_MODEL = _importlib.import_module('core.inference.model')
_REAL_AGENT = _importlib.import_module('core.agent')


# ---------------------------------------------------------------------------
# Shared stub factory (mirrors pattern in test_inline_buttons.py)
# ---------------------------------------------------------------------------

def _make_mod(name):
    m = types.ModuleType(name)
    m.__spec__ = None
    return m


def _bot_stubs():
    """Return a sys.modules stub dict that lets telegram_bot be loaded cleanly.

    Key design: do NOT stub top-level 'core', 'core.memory', 'core.inference', or
    'core.pipelines' packages — doing so prevents Python from finding the real
    core.voice_activity submodule (src/core/voice_activity.py) via __path__.
    Only stub leaf submodules like core.agent, core.memory.memory, etc.
    """
    req = _make_mod("requests")
    req.post = MagicMock(
        return_value=MagicMock(
            status_code=200,
            ok=True,
            json=lambda: {"ok": True, "result": {"message_id": 42}},
        )
    )
    req.get = MagicMock(
        return_value=MagicMock(status_code=200, ok=True, json=lambda: {"ok": True})
    )
    exc = _make_mod("requests.exceptions")
    exc.ConnectionError = ConnectionError
    exc.Timeout = TimeoutError
    exc.RequestException = Exception
    req.exceptions = exc

    mem_mem = _make_mod("core.memory.memory")
    mem_mem.load = MagicMock(return_value=[])
    mem_mem.save = MagicMock()
    mem_mem.record_attachment = MagicMock()
    mem_mem.attachment_context_block = MagicMock(return_value="")
    mem_mem.attachment_guard = MagicMock(return_value={"ok": True, "reason": ""})
    mem_mem.artifact_check = MagicMock(return_value={"ok": True, "reason": ""})

    agent = _make_mod("core.agent")
    agent.triage = MagicMock(return_value="ok")
    agent.init = MagicMock()
    agent._config = {}
    agent._skills = []

    model = _make_mod("core.inference.model")
    model._model = None
    model.load = MagicMock()
    model.infer = MagicMock(return_value="test reply")
    model.vram_free_mb = MagicMock(return_value=8000)

    model_client = _make_mod("core.inference.model_client")
    model_client.is_server_running = MagicMock(return_value=True)

    ctx = _make_mod("core.memory.context")
    ctx.build_system_prompt = MagicMock(return_value="sys")

    skills_mod = _make_mod("core.skills")
    skills_mod.load_all = MagicMock(return_value=[])
    skills_mod.find = MagicMock(return_value=None)

    routines_mod = _make_mod("core.routines")
    routines_mod.load_all = MagicMock(return_value=[])
    routines_mod.find = MagicMock(return_value=None)

    bootstrap = _make_mod("bootstrap")
    bootstrap.install = MagicMock(return_value={"message": "installed"})

    svc_disc = _make_mod("services.discovery")
    svc_ch = _make_mod("services.channels")

    yaml_mod = _make_mod("yaml")
    yaml_mod.safe_load = MagicMock(
        return_value={
            "skills_dir": "/tmp",
            "routines_dir": "/tmp",
            "model": {"name": "stub-model"},
            "api": {"port": 8779},
        }
    )
    yaml_mod.dump = MagicMock(return_value="")
    yaml_mod.safe_dump = MagicMock(return_value="")

    failed_req = _make_mod("database.agent.failed_requests")
    failed_req.detect_failure = MagicMock(return_value=None)
    failed_req.record = MagicMock()

    # infra.updater is imported at module level in telegram_bot.py (line ~2739)
    infra_mod = _make_mod("infra")
    infra_mod.__path__ = []
    updater_mod = _make_mod("infra.updater")
    updater_mod.get_current_version = MagicMock(return_value="test")
    updater_mod.fetch_latest_version = MagicMock(return_value="test")
    updater_mod.check_update_available = MagicMock(return_value=False)
    updater_mod.do_update = MagicMock(return_value={"ok": True})
    infra_mod.updater = updater_mod

    return {
        "requests": req,
        "requests.exceptions": exc,
        "yaml": yaml_mod,
        "bootstrap": bootstrap,
        "core.agent": agent,
        "core.memory.memory": mem_mem,
        "core.memory.context": ctx,
        "core.inference.model": model,
        "core.inference.model_client": model_client,
        "core.skills": skills_mod,
        "core.routines": routines_mod,
        "core.pipelines.micro_planner": _make_mod("core.pipelines.micro_planner"),
        "database.agent.failed_requests": failed_req,
        "services.discovery": svc_disc,
        "services.channels": svc_ch,
        "infra": infra_mod,
        "infra.updater": updater_mod,
    }

@contextmanager
def _bot_context():
    """Load telegram_bot via file path inside a sys.modules stub patch."""
    import pathlib as _pl
    import importlib.util as _ilu

    stubs = _bot_stubs()
    _bot_path = str(
        _pl.Path(__file__).parent.parent / "src/services/channels/telegram_bot.py"
    )

    with patch.dict(sys.modules, stubs):
        _spec = _ilu.spec_from_file_location("services.channels.telegram_bot", _bot_path)
        bot = _ilu.module_from_spec(_spec)
        sys.modules["services.channels.telegram_bot"] = bot
        stubs["services.channels"].telegram_bot = bot
        _spec.loader.exec_module(bot)
        bot._agent_ready = True
        yield bot, stubs


# ---------------------------------------------------------------------------
# Helper: spy on urllib.request.urlopen to capture POST bodies
# ---------------------------------------------------------------------------

class _UrlopenSpy:
    """Context manager that captures all urlopen POST calls without real HTTP."""

    def __init__(self, response_json):
        self._response_json = response_json
        self.calls = []
        self._patcher = None

    def __enter__(self):
        calls = self.calls
        response_json = self._response_json

        class _FakeResponse:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        def _fake_urlopen(req_or_url, *a, **kw):
            if hasattr(req_or_url, "data") and req_or_url.data is not None:
                calls.append(
                    {
                        "url": (
                            req_or_url.full_url
                            if hasattr(req_or_url, "full_url")
                            else str(req_or_url)
                        ),
                        "body": json.loads(req_or_url.data),
                        "method": getattr(req_or_url, "method", "POST"),
                    }
                )
            return _FakeResponse(json.dumps(response_json).encode())

        self._patcher = patch("urllib.request.urlopen", side_effect=_fake_urlopen)
        self._patcher.start()
        return self

    def __exit__(self, *a):
        if self._patcher:
            self._patcher.stop()


# ---------------------------------------------------------------------------
# 1) /local command -> provider/set payload with task_inference = "local"
# ---------------------------------------------------------------------------


class TestLocalCommandRouting(unittest.TestCase):
    """
    Item 1: "/local" must POST to /provider/set with task_inference="local"
    (Nemotron routing) and send a confirmation message.
    """

    def setUp(self):
        self._ctx = _bot_context()
        self.bot, self.stubs = self._ctx.__enter__()
        self.messages_sent = []
        self.bot.send_message = lambda cid, text, *a, **kw: self.messages_sent.append(text) or 42

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_local_command_posts_to_provider_set(self):
        """/local must make a POST to the /provider/set endpoint."""
        with _UrlopenSpy({"ok": True, "updated": ["task_inference"]}) as spy:
            self.bot.handle_message("123", "/local")

        provider_set_calls = [c for c in spy.calls if "/provider/set" in c["url"]]
        self.assertTrue(
            provider_set_calls,
            "/local must POST to /provider/set (no provider/set call captured)",
        )

    def test_local_command_sets_task_inference_to_local(self):
        """/local must set task_inference to 'local' in the provider/set payload."""
        with _UrlopenSpy({"ok": True}) as spy:
            self.bot.handle_message("123", "/local")

        provider_set_calls = [c for c in spy.calls if "/provider/set" in c["url"]]
        self.assertTrue(provider_set_calls, "No /provider/set call made by /local")
        payload = provider_set_calls[0]["body"]
        self.assertIn(
            "task_inference", payload, "provider/set body must include 'task_inference' key"
        )
        self.assertEqual(
            payload["task_inference"],
            "local",
            "task_inference must be 'local', got %r" % payload["task_inference"],
        )

    def test_local_command_payload_persist_is_true(self):
        """/local must send persist=True so the setting survives restart."""
        with _UrlopenSpy({"ok": True}) as spy:
            self.bot.handle_message("123", "/local")

        provider_set_calls = [c for c in spy.calls if "/provider/set" in c["url"]]
        self.assertTrue(provider_set_calls, "No /provider/set call made by /local")
        payload = provider_set_calls[0]["body"]
        self.assertTrue(
            payload.get("persist"),
            "provider/set body must have persist=True for /local command",
        )

    def test_local_command_sends_confirmation_message(self):
        """/local must send a Telegram message confirming local/Nemotron activation."""
        with _UrlopenSpy({"ok": True}):
            self.bot.handle_message("123", "/local")

        self.assertTrue(
            self.messages_sent,
            "/local must send a confirmation message to the chat",
        )
        full_text = " ".join(self.messages_sent).lower()
        self.assertTrue(
            "local" in full_text or "nemotron" in full_text,
            "Confirmation must reference 'local' or 'nemotron', got: %s" % self.messages_sent,
        )

    def test_local_command_synthesis_not_local(self):
        """/local keeps synthesis/critic/planning on cloud providers (not local)."""
        with _UrlopenSpy({"ok": True}) as spy:
            self.bot.handle_message("123", "/local")

        provider_set_calls = [c for c in spy.calls if "/provider/set" in c["url"]]
        self.assertTrue(provider_set_calls, "No /provider/set call made by /local")
        payload = provider_set_calls[0]["body"]
        synthesis = payload.get("synthesis", "")
        self.assertNotEqual(
            synthesis,
            "local",
            "synthesis must NOT be 'local' in /local mode (Fabio expects cloud synthesis)",
        )


# ---------------------------------------------------------------------------
# 2) Modal capability picker: cmprov_* and cmm_* callbacks
# ---------------------------------------------------------------------------


class TestModalCapabilityPickerCallbacks(unittest.TestCase):
    """
    Item 2: cmprov_* and cmm_* callbacks for vision/stt/tts pipelines.

    cmprov_{call_type}|{provider} -> shows model picker (or applies immediately for "local")
    cmm_{provider}|{call_type}|{model} -> calls _apply_modal_model
    """

    def setUp(self):
        self._ctx = _bot_context()
        self.bot, self.stubs = self._ctx.__enter__()
        self.bot.send_message = MagicMock(return_value=42)
        self.bot.send_buttons = MagicMock()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    # -- cmprov_* -----------------------------------------------------------

    def test_cmprov_vision_openai_shows_model_picker(self):
        """cmprov_vision|openai must call send_buttons showing a model picker."""
        self.bot._fetch_models_for_ct = MagicMock(return_value=["gpt-4o", "gpt-4-vision"])
        self.bot.handle_callback("123", "cmprov_vision|openai", 999)
        self.bot.send_buttons.assert_called()
        text_arg = self.bot.send_buttons.call_args[0][1]
        self.assertIn("Vision", text_arg, "Model picker text must mention 'Vision'")

    def test_cmprov_stt_openrouter_shows_model_picker(self):
        """cmprov_stt|openrouter must call send_buttons showing an STT model picker."""
        self.bot._fetch_models_for_ct = MagicMock(return_value=["whisper-1", "deepgram"])
        self.bot.handle_callback("123", "cmprov_stt|openrouter", 999)
        self.bot.send_buttons.assert_called()
        text_arg = self.bot.send_buttons.call_args[0][1]
        self.assertIn("STT", text_arg, "Model picker text must mention 'STT' for stt call type")

    def test_cmprov_tts_anthropic_shows_model_picker(self):
        """cmprov_tts|anthropic must call send_buttons showing a TTS model picker."""
        self.bot._fetch_models_for_ct = MagicMock(return_value=["claude-tts"])
        self.bot.handle_callback("123", "cmprov_tts|anthropic", 999)
        self.bot.send_buttons.assert_called()
        text_arg = self.bot.send_buttons.call_args[0][1]
        self.assertIn("TTS", text_arg, "Model picker text must mention 'TTS' for tts call type")

    def test_cmprov_local_calls_apply_without_model_picker(self):
        """cmprov_vision|local must call _apply_modal_model directly (no model picker)."""
        apply_calls = []

        def _spy(chat_id, provider, ct, model, idx):
            apply_calls.append({"provider": provider, "ct": ct, "model": model})

        with patch.object(self.bot, "_apply_modal_model", side_effect=_spy):
            self.bot.handle_callback("123", "cmprov_vision|local", 999)

        self.assertTrue(apply_calls, "_apply_modal_model must be called for 'local' provider")
        self.assertEqual(apply_calls[0]["provider"], "local")
        self.assertEqual(apply_calls[0]["ct"], "vision")
        # send_buttons must NOT have shown a model picker
        self.bot.send_buttons.assert_not_called()

    def test_cmprov_buttons_contain_cmm_callbacks_for_provider(self):
        """Buttons from cmprov_stt|openai must contain cmm_openai|stt|... callback_data."""
        self.bot._fetch_models_for_ct = MagicMock(return_value=["whisper-1", "whisper-large"])
        self.bot.handle_callback("123", "cmprov_stt|openai", 999)
        self.bot.send_buttons.assert_called()
        grid = self.bot.send_buttons.call_args[0][2]
        all_cbs = [btn["callback_data"] for row in grid for btn in row]
        model_cbs = [cb for cb in all_cbs if cb.startswith("cmm_openai|stt|")]
        self.assertTrue(
            model_cbs,
            "Expected cmm_openai|stt|... buttons, got: %s" % all_cbs,
        )

    # -- cmm_* --------------------------------------------------------------

    def test_cmm_vision_openai_calls_apply_modal_model(self):
        """cmm_openai|vision|gpt-4o must call _apply_modal_model with correct args."""
        apply_calls = []

        def _spy(chat_id, provider, ct, model, idx):
            apply_calls.append(
                {"chat_id": chat_id, "provider": provider, "ct": ct, "model": model, "idx": idx}
            )

        with patch.object(self.bot, "_apply_modal_model", side_effect=_spy):
            self.bot.handle_callback("123", "cmm_openai|vision|gpt-4o", 999)

        self.assertTrue(apply_calls, "_apply_modal_model must be called")
        c = apply_calls[0]
        self.assertEqual(c["provider"], "openai")
        self.assertEqual(c["ct"], "vision")
        self.assertEqual(c["model"], "gpt-4o")

    def test_cmm_stt_hf_default_passes_none_model(self):
        """cmm_hf|stt|default must call _apply_modal_model with model=None."""
        apply_calls = []

        def _spy(chat_id, provider, ct, model, idx):
            apply_calls.append({"provider": provider, "ct": ct, "model": model})

        with patch.object(self.bot, "_apply_modal_model", side_effect=_spy):
            self.bot.handle_callback("123", "cmm_hf|stt|default", 999)

        self.assertTrue(apply_calls, "_apply_modal_model must be called for cmm_hf|stt|default")
        self.assertIsNone(
            apply_calls[0]["model"], "model must be None when callback has 'default'"
        )

    def test_cmm_tts_openrouter_correct_call_type_index(self):
        """cmm_openrouter|tts|... must pass idx=7 (tts is index 7 in _CLOUD_CALL_TYPES)."""
        apply_calls = []

        def _spy(chat_id, provider, ct, model, idx):
            apply_calls.append({"idx": idx, "ct": ct})

        with patch.object(self.bot, "_apply_modal_model", side_effect=_spy):
            self.bot.handle_callback("123", "cmm_openrouter|tts|some-model", 999)

        self.assertTrue(apply_calls, "_apply_modal_model must be called")
        self.assertEqual(apply_calls[0]["ct"], "tts")
        self.assertEqual(apply_calls[0]["idx"], 7, "tts is index 7 in _CLOUD_CALL_TYPES")

    def test_cmm_unknown_call_type_does_not_crash(self):
        """cmm_ callback with an unknown call_type must not raise."""
        try:
            self.bot.handle_callback("123", "cmm_openai|unknown_ct|gpt-x", 999)
        except Exception as exc:
            self.fail("cmm_ with unknown call_type raised: %s" % exc)

    def test_cmprov_model_picker_buttons_within_64_byte_limit(self):
        """All buttons from cmprov_stt|openai must have callback_data <= 64 bytes (Telegram limit)."""
        self.bot._fetch_models_for_ct = MagicMock(return_value=["model-a", "model-b"])
        self.bot.handle_callback("123", "cmprov_stt|openai", 999)
        self.bot.send_buttons.assert_called()
        grid = self.bot.send_buttons.call_args[0][2]
        for row in grid:
            for btn in row:
                cb = btn.get("callback_data", "")
                self.assertLessEqual(
                    len(cb.encode("utf-8")),
                    64,
                    "callback_data exceeds 64 bytes: %r (%d bytes)" % (cb, len(cb.encode())),
                )


# ---------------------------------------------------------------------------
# 3) Tool-calling + streaming: chunk_callback / step_callback -> send/edit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3) Tool-calling + streaming: chunk_callback / step_callback -> send/edit
# ---------------------------------------------------------------------------


class TestStreamingAndToolCallOutput(unittest.TestCase):
    """
    Item 3: In the normal chat path, chunk_callback and step_callback are
    passed to triage(). Chunks trigger send_message / edit_message calls;
    step callbacks update the working message with tool names.

    Design note: handle_message does inline imports (import core.memory.memory
    as _memory_mod, from core.inference.model import infer, ...) that bypass
    sys.modules patch.dict because the real packages are pre-cached by conftest.
    We monkeypatch those real module attributes directly for the duration of each
    test, restoring them in tearDown.
    """

    def setUp(self):
        self._ctx = _bot_context()
        self.bot, self.stubs = self._ctx.__enter__()

        # Use pre-loaded real modules (loaded at module level before any stub
        # patching) to avoid segfaults from reimporting C-extension libs inside
        # a patch.dict context.
        _real_mem = _REAL_MEM
        _real_ctx_mod = _REAL_CTX
        _real_model = _REAL_MODEL
        _real_agent = _REAL_AGENT
        import core.agent as _real_agent

        self._real_mem = _real_mem
        self._real_agent = _real_agent
        self._real_model = _real_model
        self._real_ctx = _real_ctx_mod

        # Save originals for tearDown restoration
        self._orig_mem_load = _real_mem.load
        self._orig_mem_save = _real_mem.save
        self._orig_mem_att_ctx = getattr(_real_mem, "attachment_context_block", None)
        self._orig_mem_att_guard = getattr(_real_mem, "attachment_guard", None)
        self._orig_mem_artifact = getattr(_real_mem, "artifact_check", None)
        self._orig_triage = _real_agent.triage
        self._orig_infer = getattr(_real_model, "infer", None)
        self._orig_infer_wt = getattr(_real_model, "infer_with_tools", None)
        self._orig_vram = getattr(_real_model, "vram_free_mb", None)
        self._orig_build_prompt = getattr(_real_ctx_mod, "build_system_prompt", None)

        # Install safe no-op stubs on real modules
        _real_mem.load = MagicMock(return_value=[])
        _real_mem.save = MagicMock()
        _real_mem.attachment_context_block = MagicMock(return_value="")
        _real_mem.attachment_guard = MagicMock(return_value={"ok": True, "reason": ""})
        _real_mem.artifact_check = MagicMock(return_value={"ok": True, "reason": ""})
        _real_model.infer = MagicMock(return_value="stub reply")
        _real_model.infer_with_tools = MagicMock(return_value="stub reply with tools")
        _real_model.vram_free_mb = MagicMock(return_value=8000)
        _real_ctx_mod.build_system_prompt = MagicMock(return_value="system")

        self.sent = []
        self.edited = []
        self.bot.send_message = MagicMock(
            side_effect=lambda cid, txt, *a, **kw: (self.sent.append((cid, txt)), 42)[1]
        )
        self.bot.edit_message = MagicMock(
            side_effect=lambda cid, mid, txt, *a, **kw: (self.edited.append((cid, mid, txt)), True)[1]
        )

    def tearDown(self):
        # Restore original real-module attributes before exiting patch context
        self._real_mem.load = self._orig_mem_load
        self._real_mem.save = self._orig_mem_save
        if self._orig_mem_att_ctx is not None:
            self._real_mem.attachment_context_block = self._orig_mem_att_ctx
        if self._orig_mem_att_guard is not None:
            self._real_mem.attachment_guard = self._orig_mem_att_guard
        if self._orig_mem_artifact is not None:
            self._real_mem.artifact_check = self._orig_mem_artifact
        self._real_agent.triage = self._orig_triage
        if self._orig_infer is not None:
            self._real_model.infer = self._orig_infer
        if self._orig_infer_wt is not None:
            self._real_model.infer_with_tools = self._orig_infer_wt
        if self._orig_vram is not None:
            self._real_model.vram_free_mb = self._orig_vram
        if self._orig_build_prompt is not None:
            self._real_ctx.build_system_prompt = self._orig_build_prompt
        self._ctx.__exit__(None, None, None)

    def _run_with_streaming_triage(self, user_text, chunks, steps):
        """
        Replace agent.triage (on the real module) with a fake that fires
        chunk_callback and step_callback, then drives handle_message.
        Returns (captured_chunk_cbs, captured_step_cbs).
        """
        captured_chunk_cb = []
        captured_step_cb = []

        def _streaming_triage(
            prompt, step_callback=None, chunk_callback=None, chat_id=None, **kw
        ):
            captured_chunk_cb.append(chunk_callback)
            captured_step_cb.append(step_callback)
            for chunk in chunks:
                if chunk_callback:
                    chunk_callback(chunk)
            for s in steps:
                if step_callback:
                    step_callback(
                        s["n"], s["tool"], s.get("args", {}), s.get("result", "ok")
                    )
            return "Final synthesised answer."

        self._real_agent.triage = MagicMock(side_effect=_streaming_triage)
        self.bot.handle_message("123", user_text)
        return captured_chunk_cb, captured_step_cb

    def test_chunk_callback_passed_to_triage(self):
        """triage() must receive a non-None chunk_callback in the normal chat path."""
        cbs, _ = self._run_with_streaming_triage("Hello", [], [])
        self.assertTrue(cbs, "triage must be called at least once")
        self.assertIsNotNone(cbs[0], "chunk_callback passed to triage must not be None")

    def test_step_callback_passed_to_triage(self):
        """triage() must receive a non-None step_callback in the normal chat path."""
        _, step_cbs = self._run_with_streaming_triage("Hello", [], [])
        self.assertTrue(step_cbs, "triage must be called at least once")
        self.assertIsNotNone(step_cbs[0], "step_callback passed to triage must not be None")

    def test_chunks_trigger_send_or_edit_message(self):
        """
        When triage fires chunk_callback with enough total chars (>= _STREAM_EDIT_EVERY=40),
        send_message or edit_message must be called at least once with buffered text.
        """
        big_chunk = "A" * 200  # well above the 40-char internal throttle
        self._run_with_streaming_triage("Tell me something", [big_chunk], [])
        any_output = self.sent or self.edited
        self.assertTrue(
            any_output,
            "Streaming chunks must cause send_message or edit_message to be called",
        )

    def test_step_callback_updates_message_with_tool_name(self):
        """step_callback must cause send_message or edit_message containing the tool name."""
        steps = [{"n": 1, "tool": "read_file", "args": {}, "result": "content"}]
        self._run_with_streaming_triage("Use a tool please", [], steps)
        all_texts = [t for _, t in self.sent] + [t for _, _, t in self.edited]
        # Tool name may appear escaped (underscores -> \_) or unescaped
        tool_mentions = [t for t in all_texts if "read_file" in t or "read\\_file" in t]
        self.assertTrue(
            tool_mentions,
            "step_callback must cause a message containing the tool name to be sent or edited",
        )

    def test_final_reply_produces_send_or_edit(self):
        """After triage completes, the final answer must be delivered via send or edit."""
        self._run_with_streaming_triage("Give me an answer", [], [])
        any_output = self.sent or self.edited
        self.assertTrue(
            any_output, "Final reply must produce at least one send_message or edit_message"
        )

    def test_small_chunks_accumulate_before_triggering_output(self):
        """
        Multiple small chunks below _STREAM_EDIT_EVERY (40) must accumulate.
        After enough total chars exceed the threshold, output must be triggered.
        Total: 3 x 15 = 45 chars > 40 -> must trigger at least one send/edit.
        """
        call_count = [0]

        def _cnt_send(cid, txt, *a, **kw):
            call_count[0] += 1
            self.sent.append((cid, txt))
            return 42

        def _cnt_edit(cid, mid, txt, *a, **kw):
            call_count[0] += 1
            self.edited.append((cid, mid, txt))
            return True

        self.bot.send_message = _cnt_send
        self.bot.edit_message = _cnt_edit

        chunks = ["X" * 15, "X" * 15, "X" * 15]  # 45 chars total, crosses 40-char threshold
        self._run_with_streaming_triage("test accumulation", chunks, [])

        self.assertGreater(
            call_count[0],
            0,
            "After 45+ chars total (3x15), at least one send/edit must have occurred",
        )


class TestSendFileAttachment(unittest.TestCase):
    """
    Item 4: send_file() must POST to Telegram's sendDocument endpoint and return
    True on success. Covers the positive (ok=True), negative (ok=False), and
    error (network exception) paths, plus payload correctness.
    """

    def setUp(self):
        self._ctx = _bot_context()
        self.bot, self.stubs = self._ctx.__enter__()

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def _make_temp_file(self, suffix=".txt", content=b"test"):
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.write(content)
        f.close()
        return f.name

    def test_send_file_posts_to_sendDocument(self):
        """send_file() must POST to the sendDocument Telegram endpoint."""
        tmp = self._make_temp_file()
        try:
            self.bot.send_file("123", tmp, caption="Here is the file")
            self.stubs["requests"].post.assert_called()
            url = self.stubs["requests"].post.call_args[0][0]
            self.assertIn(
                "sendDocument", url,
                "send_file must POST to sendDocument endpoint, got URL: %s" % url,
            )
        finally:
            os.unlink(tmp)

    def test_send_file_returns_true_on_ok_response(self):
        """send_file() must return True when the Telegram API responds with ok=True."""
        tmp = self._make_temp_file(suffix=".pdf", content=b"%PDF content")
        try:
            mock_resp = MagicMock()
            mock_resp.ok = True
            self.stubs["requests"].post.return_value = mock_resp
            result = self.bot.send_file("123", tmp, caption="A PDF")
            self.assertTrue(result, "send_file must return True when Telegram responds ok=True")
        finally:
            os.unlink(tmp)

    def test_send_file_returns_false_on_api_error(self):
        """send_file() must return False when the Telegram API responds with ok=False."""
        tmp = self._make_temp_file(suffix=".pdf", content=b"%PDF content")
        try:
            mock_resp = MagicMock()
            mock_resp.ok = False
            self.stubs["requests"].post.return_value = mock_resp
            result = self.bot.send_file("123", tmp, caption="A PDF")
            self.assertFalse(
                result, "send_file must return False when Telegram responds ok=False"
            )
        finally:
            os.unlink(tmp)

    def test_send_file_includes_chat_id_in_payload(self):
        """send_file() must include the correct chat_id in the multipart data."""
        tmp = self._make_temp_file(content=b"hello")
        try:
            self.bot.send_file("456", tmp, caption="")
            call_kwargs = self.stubs["requests"].post.call_args
            # data is passed as kwarg 'data' in requests.post
            data_arg = call_kwargs[1].get("data", {})
            self.assertIn("chat_id", data_arg, "send_file payload must include chat_id")
            self.assertEqual(
                str(data_arg["chat_id"]),
                "456",
                "chat_id must be '456', got %r" % data_arg.get("chat_id"),
            )
        finally:
            os.unlink(tmp)

    def test_send_file_includes_caption(self):
        """send_file() must include the caption in the multipart data."""
        tmp = self._make_temp_file(content=b"content")
        try:
            self.bot.send_file("123", tmp, caption="Here is the report")
            call_kwargs = self.stubs["requests"].post.call_args
            data_arg = call_kwargs[1].get("data", {})
            self.assertIn("caption", data_arg, "send_file payload must include caption")
            self.assertEqual(
                data_arg["caption"],
                "Here is the report",
                "send_file caption must match the argument passed",
            )
        finally:
            os.unlink(tmp)

    def test_send_file_sends_file_bytes_under_document_key(self):
        """send_file() must include the file contents under 'document' key in files kwarg."""
        tmp = self._make_temp_file(suffix=".csv", content=b"col1,col2\n1,2")
        try:
            self.bot.send_file("123", tmp, caption="")
            call_kwargs = self.stubs["requests"].post.call_args
            files_arg = call_kwargs[1].get("files", {})
            self.assertIn(
                "document", files_arg,
                "send_file must pass file bytes under document key",
            )
        finally:
            os.unlink(tmp)

    def test_send_file_returns_false_on_network_exception(self):
        """send_file() must return False (not raise) on network errors."""
        tmp = self._make_temp_file(content=b"data")
        try:
            self.stubs["requests"].post.side_effect = Exception("network down")
            result = self.bot.send_file("123", tmp)
            self.assertFalse(
                result,
                "send_file must return False on network exception, not raise",
            )
        finally:
            self.stubs["requests"].post.side_effect = None
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
