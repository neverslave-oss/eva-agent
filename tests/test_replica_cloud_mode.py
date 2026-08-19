"""
test_replica_cloud_mode.py — Verify replicas route inference through the provider
in cloud mode (task_inference != local) instead of crashing on a None local
processor.

Regression for: AttributeError: 'NoneType' object has no attribute 'apply_chat_template'
(replica.py:92 → model.py:110) when the local model is not loaded because
task_inference is routed to a cloud provider.
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.replica.replica as replica


def test_provider_infer_routes_through_provider_in_cloud_mode():
    """In cloud mode, _provider_infer must call the provider, not local infer."""
    fake_prov = MagicMock()
    fake_prov.infer.return_value = "cloud provider reply"

    with patch("core.inference.provider.get_provider", return_value=fake_prov):
        result = replica._provider_infer([{"role": "user", "content": "hi"}])

    assert result == "cloud provider reply"
    fake_prov.infer.assert_called_once()


def test_provider_infer_falls_back_to_local_when_provider_unavailable():
    """If the provider is unavailable, _provider_infer falls back to local infer."""
    with patch("core.inference.provider.get_provider", side_effect=Exception("no provider")), \
         patch("core.replica.replica.infer", return_value="local reply") as mock_local:
        result = replica._provider_infer([{"role": "user", "content": "hi"}])

    assert result == "local reply"
    mock_local.assert_called_once()


def test_provider_infer_with_tools_routes_through_provider():
    """Tool-calling replica inference routes through the provider in cloud mode."""
    fake_prov = MagicMock()
    fake_prov.infer_with_tools.return_value = "cloud tools reply"

    with patch("core.inference.provider.get_provider", return_value=fake_prov):
        result = replica._provider_infer_with_tools(
            [{"role": "user", "content": "go"}], [], workspace="/tmp/ws"
        )

    assert result == "cloud tools reply"
    fake_prov.infer_with_tools.assert_called_once()


def test_pipeline_uses_provider_path():
    """pipeline() must call _provider_infer (provider-routed), not local infer directly."""
    def fake_infer(messages, max_new_tokens=1024, adapter_path=None, **kwargs):
        return "mocked reply"

    with patch.object(replica, "_provider_infer", fake_infer):
        stages = [
            {"name": "writer", "role": "custom", "brief": "Write stuff", "task": "Do task A"},
        ]
        results = replica.pipeline(stages)
    assert results["writer"] == "mocked reply"


# ---------------------------------------------------------------------------
# Config env expansion (collective_memory.url placeholder leak)
# ---------------------------------------------------------------------------

def test_agent_init_expands_collective_memory_env():
    """agent.init() must use the env-expanding loader so ${VAR} config values
    (e.g. collective_memory.url) resolve instead of leaking the placeholder."""
    import core.agent as agent

    fake_cfg = {
        "api": {"openclaw_endpoint": "http://localhost:18789"},
        "collective_memory": {"url": "http://192.168.1.113:8010"},  # already expanded
        "skills_dir": "./skills",
        "routines_dir": "./routines",
        "core_skills": [],
    }

    with patch("core.agent._load_config", return_value=fake_cfg) as mock_load, \
         patch("core.agent.load_skills", return_value=[]), \
         patch("core.agent.load_routines", return_value=[]), \
         patch("core.inference.provider.get_provider", return_value=MagicMock()):
        agent.init("config.yaml")

    mock_load.assert_called_once_with("config.yaml")
    assert agent._config["collective_memory"]["url"] == "http://192.168.1.113:8010"
    assert "${" not in agent._config["collective_memory"]["url"], (
        "collective_memory.url must not leak a ${VAR} placeholder"
    )
