"""
test_look_tool.py — Validates the unified `look` tool (vision).

Covers:
  - Eye registry: config loading, health-check, plug-and-play status
  - Router: intent -> eye mapping, scan merge, offline degradation, unknown intent
  - Eye clients: object_face + plant_health endpoint calls (mocked)
  - Tool dispatch: `look` registered in TOOLS + execute_tool dispatch

Rules:
  - No real HTTP, no model, no Telegram
  - All requests mocked; no production DBs (conftest redirects kernel DBs)
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.vision.registry import EyeRegistry, Eye, EyeStatus
from core.vision.router import route_look, VALID_INTENTS
from core.vision import eyes as eyes_pkg
from core.vision.eyes import object_face, plant_health
from core.vision import capture as capture_mod
from core.vision import describe as describe_mod

import core.tools as tools_mod


# ─────────────────────────────────────────────────────────────────────────────
# EyeRegistry
# ─────────────────────────────────────────────────────────────────────────────
class TestEyeRegistry:
    def test_loads_from_dict(self):
        cfg = {
            "vision": {
                "eyes": {
                    "left": {"kind": "object_face", "base": "http://hub:5010"},
                    "right": {"kind": "plant_health", "base": "http://hub:5000"},
                }
            }
        }
        reg = EyeRegistry(config=cfg)
        assert set(reg.ids()) == {"left", "right"}
        assert reg.get("left").kind == "object_face"
        assert reg.get("right").kind == "plant_health"

    def test_empty_config_no_eyes(self):
        reg = EyeRegistry(config={})
        assert reg.all() == []

    def test_health_url_construction(self):
        eye = Eye(id="left", kind="object_face", base="http://hub:5010", health="/health")
        assert eye.health_url == "http://hub:5010/health"
        eye2 = Eye(id="right", kind="plant_health", base="http://hub:5000/", health="/")
        assert eye2.health_url == "http://hub:5000/"

    def test_refresh_marks_online_and_offline(self):
        cfg = {
            "vision": {
                "eyes": {
                    "left": {"kind": "object_face", "base": "http://hub:5010", "health": "/health"},
                    "right": {"kind": "plant_health", "base": "http://hub:5000", "health": "/"},
                }
            }
        }
        reg = EyeRegistry(config=cfg)
        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=200),  # left -> online
                Exception("conn refused"),    # right -> offline
            ]
            statuses = reg.refresh()
        assert statuses["left"] == EyeStatus.ONLINE
        assert statuses["right"] == EyeStatus.OFFLINE

    def test_offline_when_no_base(self):
        reg = EyeRegistry(config={"vision": {"eyes": {"x": {"kind": "object_face", "base": ""}}}})
        assert reg.is_online("x") is False

    def test_unknown_eye_not_online(self):
        reg = EyeRegistry(config={"vision": {"eyes": {}}})
        assert reg.is_online("nope") is False


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
class TestRouter:
    def _reg(self):
        cfg = {
            "vision": {
                "eyes": {
                    "left": {"kind": "object_face", "base": "http://hub:5010", "health": "/health"},
                    "right": {"kind": "plant_health", "base": "http://hub:5000", "health": "/"},
                }
            }
        }
        return EyeRegistry(config=cfg)

    def test_unknown_intent_returns_error(self):
        result = route_look("teleport", self._reg())
        assert result["status"] == "error"
        assert "unknown intent" in result["error"]

    def test_plant_health_routes_to_right_eye(self):
        reg = self._reg()
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch.object(plant_health, "capture_and_detect", return_value={
                "status": "ok", "num_crops": 4,
                "crops": ["/crops/a.jpg", "/crops/b.jpg"],
            }) as mock_cad:
                result = route_look("plant health", reg)
        mock_cad.assert_called_once()
        assert result["eye"] == "right"
        assert result["status"] == EyeStatus.ONLINE
        assert result["observations"]["num_crops"] == 4
        assert result["observations"]["crops"][0] == "http://hub:5000/crops/a.jpg"

    def test_whats_there_routes_to_left_eye(self):
        reg = self._reg()
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch.object(object_face, "detect_objects", return_value=[
                {"label": "person", "conf": 0.9, "box": [1, 2, 3, 4]}
            ]) as mock_detect:
                result = route_look("what's there", reg)
        mock_detect.assert_called_once()
        assert result["eye"] == "left"
        assert result["status"] == EyeStatus.ONLINE
        assert result["observations"][0]["label"] == "person"

    def test_target_override(self):
        reg = self._reg()
        # Force right eye online, route 'what's there' (normally left) with target=right
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch.object(plant_health, "capture_and_detect", return_value={
                "status": "ok", "num_crops": 2, "crops": []
            }):
                result = route_look("what's there", reg, target="right")
        assert result["eye"] == "right"

    def test_offline_eye_returns_offline(self):
        reg = self._reg()
        with patch("requests.get", side_effect=Exception("down")):
            result = route_look("plant health", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "no online eye" in result["error"]

    def test_scan_merges_all_online_eyes(self):
        reg = self._reg()
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch.object(object_face, "detect_objects", return_value=[]), \
                 patch.object(plant_health, "capture_and_detect", return_value={
                     "status": "ok", "num_crops": 1, "crops": ["/crops/c.jpg"]
                 }):
                result = route_look("scan", reg)
        assert result["eye"] == "all"
        assert result["status"] == EyeStatus.ONLINE
        assert len(result["observations"]) == 2  # both eyes online

    def test_scan_no_eyes_online(self):
        reg = self._reg()
        with patch("requests.get", side_effect=Exception("down")):
            result = route_look("scan", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "no eyes online" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Describe intent (semantic scene description)
# ─────────────────────────────────────────────────────────────────────────────
class TestDescribeRouting:
    def _reg(self, with_stream=True):
        cfg = {
            "vision": {
                "eyes": {
                    "right": {
                        "kind": "plant_health",
                        "base": "http://hub:5000",
                        "stream": "http://hub:5000/video_feed" if with_stream else "",
                        "health": "/",
                    }
                }
            }
        }
        return EyeRegistry(config=cfg)

    def test_describe_is_valid_intent(self):
        assert "describe" in VALID_INTENTS

    def test_describe_routes_to_online_eye_and_describes(self):
        reg = self._reg()
        fake_frame = b"\xff\xd8frame\xff\xd9"
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch("core.vision.router.grab_frame", return_value=fake_frame) as m_grab, \
                 patch("core.vision.router.describe_image", return_value={
                     "backend": "local_gemma_e2b", "description": "A tidy desk"
                 }) as m_desc:
                result = route_look("describe", reg)
        m_grab.assert_called_once()
        m_desc.assert_called_once()
        assert result["eye"] == "right"
        assert result["status"] == EyeStatus.ONLINE
        assert result["observations"]["description"] == "A tidy desk"

    def test_describe_no_stream_returns_offline(self):
        reg = self._reg(with_stream=False)
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch.object(capture_mod, "grab_frame") as m_grab:
                result = route_look("describe", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "no stream" in result["error"]

    def test_describe_frame_grab_failure_returns_offline(self):
        reg = self._reg()
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch("core.vision.router.grab_frame", return_value=None):
                result = route_look("describe", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "could not grab a frame" in result["error"]

    def test_describe_backend_error_returns_offline(self):
        reg = self._reg()
        fake_frame = b"\xff\xd8frame\xff\xd9"
        with patch("requests.get", return_value=MagicMock(status_code=200)):
            with patch("core.vision.router.grab_frame", return_value=fake_frame), \
                 patch("core.vision.router.describe_image", return_value={
                     "backend": "none", "error": "no describe backend available"
                 }):
                result = route_look("describe", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "no describe backend" in result["error"]

    def test_describe_offline_eye_returns_offline(self):
        reg = self._reg()
        with patch("requests.get", side_effect=Exception("down")):
            result = route_look("describe", reg)
        assert result["status"] == EyeStatus.OFFLINE
        assert "no online eye" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Eye clients
# ─────────────────────────────────────────────────────────────────────────────
class TestEyeClients:
    def test_object_face_detect_objects(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: [{"label": "cat"}])
            out = object_face.detect_objects("http://hub:5010")
        mock_post.assert_called_once_with("http://hub:5010/detect", timeout=5.0)
        assert out == [{"label": "cat"}]

    def test_object_face_detect_text(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, text="2 people")
            out = object_face.detect_text("http://hub:5010")
        mock_post.assert_called_once_with("http://hub:5010/detect_text", timeout=5.0)
        assert out == "2 people"

    def test_plant_health_capture_and_detect(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {"num_crops": 3})
            out = plant_health.capture_and_detect("http://hub:5000")
        mock_post.assert_called_once_with("http://hub:5000/plant_health/capture_and_detect", timeout=10.0)
        assert out == {"num_crops": 3}

    def test_crop_urls_resolves_relative(self):
        urls = plant_health.crop_urls("http://hub:5000", ["/crops/a.jpg", "http://x/y.jpg"])
        assert urls == ["http://hub:5000/crops/a.jpg", "http://x/y.jpg"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch
# ─────────────────────────────────────────────────────────────────────────────
class TestLookToolDispatch:
    def test_look_is_registered_in_tools(self):
        names = [t["function"]["name"] for t in tools_mod.TOOLS]
        assert "look" in names

    def test_look_schema_has_intent_enum(self):
        look = next(t for t in tools_mod.TOOLS if t["function"]["name"] == "look")
        intent = look["function"]["parameters"]["properties"]["intent"]
        assert set(intent["enum"]) == set(VALID_INTENTS)
        assert "intent" in look["function"]["parameters"]["required"]

    def test_execute_tool_look_missing_intent(self):
        result = tools_mod.execute_tool("look", {})
        assert "requires 'intent'" in result

    def test_execute_tool_look_unknown_intent(self):
        result = tools_mod.execute_tool("look", {"intent": "teleport"})
        parsed = json.loads(result)
        assert parsed["status"] == "error"

    def test_execute_tool_look_routes_plant_health(self):
        # _run_look builds EyeRegistry() from the real config.yaml, where the
        # eye bases are ${VAR} placeholders. Set the env vars so the config-
        # driven registry resolves the 'right' plant_health eye as online.
        env = {
            "KERNEL_EVO_EYE_LEFT_BASE": "http://hub:5010",
            "KERNEL_EVO_EYE_RIGHT_BASE": "http://hub:5000",
        }
        with patch.dict(os.environ, env):
            with patch("requests.get", return_value=MagicMock(status_code=200)):
                with patch.object(plant_health, "capture_and_detect", return_value={
                    "status": "ok", "num_crops": 1, "crops": ["/crops/a.jpg"]
                }):
                    result = tools_mod.execute_tool("look", {"intent": "plant health"})
        parsed = json.loads(result)
        assert parsed["eye"] == "right"
        assert parsed["status"] == EyeStatus.ONLINE
