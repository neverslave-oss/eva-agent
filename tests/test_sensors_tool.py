"""
test_sensors_tool.py — Validates the unified `sensors` tool.

Covers:
  - Sensor registry: config loading, env expansion
  - Router: action -> device mapping, read, offline degradation, unknown action
  - Pi client: read_sensors GET (mocked)
  - Tool dispatch: `sensors` registered in TOOLS + execute_tool dispatch

Rules:
  - No real HTTP, no model, no Telegram
  - All requests mocked; no production DBs (conftest redirects kernel DBs)
"""

import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.sensors.registry import SensorRegistry, SensorDevice
from core.sensors.router import route_sensors, VALID_ACTIONS
from core.sensors import pi as pi_client

import core.tools as tools_mod


# ─────────────────────────────────────────────────────────────────────────────
# Sensor registry
# ─────────────────────────────────────────────────────────────────────────────
class TestSensorRegistry:
    def test_empty_config_no_devices(self):
        reg = SensorRegistry(config={})
        assert reg.all() == []

    def test_loads_devices_from_dict(self):
        reg = SensorRegistry(config={"sensors": {"pi": {"base": "http://hub:5000", "timeout_s": 5.0}}})
        assert reg.ids() == ["pi"]
        dev = reg.get("pi")
        assert dev.base == "http://hub:5000"
        assert dev.timeout_s == 5.0

    def test_expands_env_var(self):
        with patch.dict(os.environ, {"KERNEL_EVO_SENSORS_PI_BASE": "http://192.0.2.120:5000"}):
            reg = SensorRegistry(config={"sensors": {"pi": {"base": "${KERNEL_EVO_SENSORS_PI_BASE:-}"}}})
        assert reg.get("pi").base == "http://192.0.2.120:5000"


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
class TestRouter:
    def test_unknown_action_returns_error(self):
        reg = SensorRegistry(config={"sensors": {"pi": {"base": "http://hub:5000"}}})
        result = route_sensors("teleport", reg)
        assert result["status"] == "error"

    def test_unknown_device_returns_error(self):
        reg = SensorRegistry(config={"sensors": {"pi": {"base": "http://hub:5000"}}})
        result = route_sensors("read", reg, target="ghost")
        assert result["status"] == "error"

    def test_read_returns_sensor_data(self):
        reg = SensorRegistry(config={"sensors": {"pi": {"base": "http://hub:5000"}}})
        with patch.object(pi_client, "read_sensors", return_value={
            "temp": 25.0, "humi": 60.0, "moisture": 0, "moisture_percent": 0.0
        }):
            result = route_sensors("read", reg, target="pi")
        assert result["status"] == "ok"
        assert result["sensor"] == "pi"
        assert result["action"] == "read"
        assert result["observations"]["temp"] == 25.0

    def test_read_offline_on_error(self):
        reg = SensorRegistry(config={"sensors": {"pi": {"base": "http://hub:5000"}}})
        with patch.object(pi_client, "read_sensors", return_value={"error": "conn refused"}):
            result = route_sensors("read", reg, target="pi")
        assert result["status"] == "offline"
        assert "conn refused" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Pi client
# ─────────────────────────────────────────────────────────────────────────────
class TestPiClient:
    def test_read_sensors_gets_correct_url(self):
        with patch("requests.get") as m_get:
            m_get.return_value = MagicMock(status_code=200, json=lambda: {
                "temp": 25.0, "humi": 60.0, "moisture": 0, "moisture_percent": 0.0
            })
            out = pi_client.read_sensors("http://hub:5000", timeout_s=5.0)
        m_get.assert_called_once_with("http://hub:5000/pico/sensors", timeout=5.0)
        assert out["temp"] == 25.0
        assert out["moisture"] == 0

    def test_read_sensors_error_on_http_500(self):
        with patch("requests.get") as m_get:
            m_get.return_value = MagicMock(status_code=500)
            out = pi_client.read_sensors("http://hub:5000")
        assert "error" in out

    def test_set_relay_posts_payload(self):
        with patch("requests.post") as m_post:
            m_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
            out = pi_client.set_relay("http://hub:5000", relay=True, watering=True)
        m_post.assert_called_once()
        assert m_post.call_args.kwargs["json"] == {"relay": True, "watering": True}
        assert out["ok"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────────────
class TestSensorsToolDispatch:
    def test_sensors_registered(self):
        names = {t["function"]["name"] for t in tools_mod.TOOLS}
        assert "sensors" in names

    def test_sensors_schema_has_read_action(self):
        entry = next(t for t in tools_mod.TOOLS if t["function"]["name"] == "sensors")
        enum = entry["function"]["parameters"]["properties"]["action"]["enum"]
        assert "read" in enum
        assert "action" in entry["function"]["parameters"]["required"]

    def test_missing_action_returns_error(self):
        result = tools_mod.execute_tool("sensors", {})
        assert "requires 'action'" in result

    def test_execute_tool_sensors_read(self):
        env = {"KERNEL_EVO_SENSORS_PI_BASE": "http://hub:5000"}
        with patch.dict(os.environ, env):
            with patch.object(pi_client, "read_sensors", return_value={
                "temp": 25.0, "humi": 60.0, "moisture": 0, "moisture_percent": 0.0
            }):
                result = tools_mod.execute_tool("sensors", {"action": "read"})
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["sensor"] == "pi"
        assert parsed["observations"]["temp"] == 25.0
