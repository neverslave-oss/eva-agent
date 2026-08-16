"""
test_agent_triage.py — Unit tests for agent.py triage logic.
Mocks model inference — tests routing decisions only, not LLM output.

ADR-022: skill matching, semantic search, and evolution hook have been removed
from triage(). All free-text requests route to infer_with_tools directly.
These tests verify the simplified routing.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Patch heavy imports before importing agent.
# Must stay in sys.modules for the lifetime of the test session because
# patch("core.agent.vram_free_mb") calls pkgutil.resolve_name("core.agent")
# which re-executes core/__init__.py (via importlib) if core.inference.model
# hasn't been fully initialised yet — triggering `import torch` outside the
# patched context. Keeping the mock in sys.modules avoids the re-import.
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.cuda.mem_get_info.return_value = (4096 * 1024 * 1024, 8192 * 1024 * 1024)
_mock_torch.bfloat16 = "bfloat16"
_mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))

sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("requests", MagicMock())
sys.modules.setdefault("yaml", __import__("yaml"))

import core.agent as agent
import core.inference.model as kernel_model

# Provide a mock processor so infer() doesn't crash on _processor.apply_chat_template
_mock_processor = MagicMock()
_mock_processor.apply_chat_template.return_value = "<mock prompt>"
# Mock the tokenizer call: _processor(text=...) must return an object with .to()
_mock_tensor = MagicMock()
_mock_tensor.to.return_value = _mock_tensor
_mock_processor.return_value = _mock_tensor
_mock_processor.decode.return_value = "mock model response"
kernel_model._processor = _mock_processor

# Provide a mock model so generate() doesn't crash
_mock_llm = MagicMock()
_mock_llm.generate.return_value = [MagicMock()]  # list of token ids
_mock_llm.device = "cpu"
kernel_model._model = _mock_llm


def _reset_agent(skills=None, routines=None):
    agent._skills = skills or []
    agent._routines = routines or []
    agent._config = {"api": {"openclaw_endpoint": "http://localhost:18789"}}


def _make_provider_mock(return_value="mock infer_with_tools response"):
    """Build a mock provider that satisfies infer_with_tools calls in triage()."""
    prov = MagicMock()
    prov.infer_with_tools.return_value = return_value
    prov.get_provider.return_value = "local"
    prov.get_model.return_value = "mock-model"
    return prov


class TestTriageSlashCommands:
    def setup_method(self):
        _reset_agent(
            skills=[
                {"name": "weather", "description": "Get weather", "commands": ["/weather"]},
                {"name": "search", "description": "Search the web", "commands": ["/search"]},
            ],
            routines=[
                {"name": "daily-digest", "description": "Daily digest", "trigger": {}, "body": ""},
            ]
        )

    def test_skills_command_lists_skills(self):
        result = agent.triage("/skills")
        assert "weather" in result
        assert "search" in result

    def test_routines_command_lists_routines(self):
        result = agent.triage("/routines")
        assert "daily-digest" in result

    def test_skills_command_empty(self):
        _reset_agent()
        result = agent.triage("/skills")
        assert "No skills" in result

    def test_routines_command_empty(self):
        _reset_agent()
        result = agent.triage("/routines")
        assert "No routines" in result

    def test_run_nonexistent_returns_not_found(self):
        result = agent.triage("/run nonexistent_xyz")
        assert "not found" in result.lower() or "no" in result.lower()

    def test_status_returns_vram_info(self):
        with patch.object(agent, "vram_free_mb", return_value=4096), \
             patch.object(agent, "active_replicas", return_value=[]), \
             patch.object(agent, "can_spawn", return_value=True):
            result = agent.triage("/status")
            assert "4096" in result or "VRAM" in result


class TestTriageToolFirstPipeline:
    """ADR-022: all free-text requests (including those matching skill names/intents)
    must route directly to infer_with_tools — no skill dispatch, no semantic search.
    """

    def setup_method(self):
        _reset_agent(
            skills=[
                {"name": "weather", "description": "Get current weather", "commands": ["/weather"], "instructions": ""},
                {"name": "investor", "description": "Portfolio management", "commands": ["/investor"], "instructions": ""},
            ],
            routines=[]
        )

    def _run_with_mock_provider(self, text, return_value="tool-result"):
        """Helper: run triage() with all heavy deps mocked."""
        prov = _make_provider_mock(return_value)

        import core.inference.provider as _prov_mod
        import core.memory.memory as _mem_mod
        import core.memory.context as _ctx_mod

        mem_file_mock = MagicMock()
        mem_file_mock.write_text = MagicMock()

        with patch.object(_prov_mod, "get_provider", return_value=prov), \
             patch.object(_mem_mod, "load", return_value=[]), \
             patch.object(_mem_mod, "_chat_memory_file", return_value=mem_file_mock), \
             patch.object(_ctx_mod, "build_system_prompt", return_value="sys"), \
             patch.object(agent, "_embedding_client", None):
            result = agent.triage(text, chat_id="test-session")
        return result, prov

    def test_free_text_routes_to_infer_with_tools_not_skill(self):
        """'weather in Rome' must NOT dispatch the weather skill — it must call infer_with_tools."""
        result, prov = self._run_with_mock_provider("weather in Rome")
        # Verify infer_with_tools was called (tool-first pipeline)
        assert prov.infer_with_tools.called, "infer_with_tools must be called for free-text requests"

    def test_slash_non_builtin_routes_to_tools(self):
        """/investor (not a /skills /run /skill /routines /status) routes to infer_with_tools."""
        result, prov = self._run_with_mock_provider("/investor")
        assert prov.infer_with_tools.called, "infer_with_tools must be called for non-builtin /commands"

    def test_unknown_free_text_routes_to_infer_with_tools(self):
        """Any unknown message falls through to infer_with_tools."""
        result, prov = self._run_with_mock_provider("tell me a joke", return_value="I am a joke")
        assert prov.infer_with_tools.called
        assert result == "I am a joke"


class TestTriageRoutineMatching:
    def setup_method(self):
        _reset_agent(
            skills=[],
            routines=[
                {"name": "daily-digest", "description": "Portfolio digest", "trigger": {}, "body": ""},
            ]
        )

    def test_routine_triggered_by_name(self):
        result = agent.triage("daily-digest")
        assert isinstance(result, str)

    def test_run_slash_triggers_routine(self):
        result = agent.triage("/run daily-digest")
        assert isinstance(result, str)


class TestTriageStatusKeywords:
    def setup_method(self):
        _reset_agent()

    def test_vram_keyword(self):
        with patch.object(agent, "vram_free_mb", return_value=2048), \
             patch.object(agent, "active_replicas", return_value=[]), \
             patch.object(agent, "can_spawn", return_value=False):
            result = agent.triage("vram")
            assert "2048" in result or "VRAM" in result

    def test_health_keyword(self):
        with patch.object(agent, "vram_free_mb", return_value=2048), \
             patch.object(agent, "active_replicas", return_value=[]), \
             patch.object(agent, "can_spawn", return_value=True):
            result = agent.triage("health")
            assert isinstance(result, str)
            assert len(result) > 0
