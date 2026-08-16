"""
test_model_client_unit.py — Additional unit tests for model_client.py

Covers socket path resolution, is_server_running, and error-return paths
for infer(), infer_with_tools(), and health() — no real socket needed.
"""
import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import core.inference.model_client as model_client


# ---------------------------------------------------------------------------
# is_server_running — socket does not exist
# ---------------------------------------------------------------------------

class TestIsServerRunningUnit:

    def test_returns_false_when_socket_does_not_exist(self, tmp_path):
        nonexistent = str(tmp_path / "no_such.sock")
        with patch("core.inference.model_client._get_socket_path", return_value=nonexistent):
            assert model_client.is_server_running() is False


# ---------------------------------------------------------------------------
# _get_socket_path — env var and default
# ---------------------------------------------------------------------------

class TestGetSocketPath:

    def test_returns_default_socket_path_when_no_env_or_config(self, monkeypatch):
        """Without MODEL_SERVER_SOCKET env var and with a broken config, returns SOCKET_PATH."""
        monkeypatch.delenv("MODEL_SERVER_SOCKET", raising=False)
        # Make yaml/config load fail so we fall through to the default
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = model_client._get_socket_path()
        assert result == model_client.SOCKET_PATH

    def test_uses_env_var_when_set(self, monkeypatch, tmp_path):
        custom = str(tmp_path / "custom.sock")
        monkeypatch.setenv("MODEL_SERVER_SOCKET", custom)
        result = model_client._get_socket_path()
        assert result == custom


# ---------------------------------------------------------------------------
# infer() — connection failure → error string
# ---------------------------------------------------------------------------

class TestInferConnectionFailure:

    def test_infer_returns_error_string_on_connection_failure(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = ConnectionRefusedError("no server")
            mock_sock.close = MagicMock()
            mock_socket_cls.return_value = mock_sock

            with patch("core.inference.model_client._get_socket_path", return_value="/tmp/fake_test.sock"):
                result = model_client.infer(
                    messages=[{"role": "user", "content": "hello"}]
                )

        assert isinstance(result, str)
        assert "[model_client error]" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# infer_with_tools() — connection failure → error string
# ---------------------------------------------------------------------------

class TestInferWithToolsConnectionFailure:

    def test_infer_with_tools_returns_error_string_on_connection_failure(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = ConnectionRefusedError("no server")
            mock_sock.close = MagicMock()
            mock_socket_cls.return_value = mock_sock

            with patch("core.inference.model_client._get_socket_path", return_value="/tmp/fake_test.sock"):
                result = model_client.infer_with_tools(
                    messages=[{"role": "user", "content": "go"}],
                    tools=[],
                )

        assert isinstance(result, str)
        assert "[model_client error]" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# health() — connection failure → dict with "error" key
# ---------------------------------------------------------------------------

class TestHealthConnectionFailure:

    def test_health_returns_dict_with_error_key_on_failure(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.connect.side_effect = ConnectionRefusedError("no server")
            mock_sock.close = MagicMock()
            mock_socket_cls.return_value = mock_sock

            with patch("core.inference.model_client._get_socket_path", return_value="/tmp/fake_test.sock"):
                result = model_client.health()

        assert isinstance(result, dict)
        assert "error" in result
