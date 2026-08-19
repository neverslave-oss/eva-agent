"""
test_describe.py — Validates the semantic scene-description path for the `look`
tool: `describe_image` backend selection (local Gemma E2B native first, then
Ollama-compatible :8005 fallback) and the `grab_frame` MJPEG frame extractor.

Rules:
  - No real HTTP, no model server, no Telegram.
  - All network/model calls mocked; no production DBs (conftest redirects them).
"""

import os
import sys
import base64
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.vision import describe
from core.vision import capture


# ─────────────────────────────────────────────────────────────────────────────
# describe_image — backend selection
# ─────────────────────────────────────────────────────────────────────────────
class TestDescribeImage:
    def test_local_gemma_wins_when_available(self):
        with patch.object(describe, "_local_gemma_describe", return_value="A cat on a sofa") as m_local, \
             patch.object(describe, "_ollama_describe", return_value="should not be used") as m_remote:
            result = describe.describe_image("/tmp/frame.jpg")
        m_local.assert_called_once()
        m_remote.assert_not_called()
        assert result["backend"] == "local_gemma_e2b"
        assert result["description"] == "A cat on a sofa"

    def test_falls_back_to_ollama_when_local_unavailable(self):
        with patch.object(describe, "_local_gemma_describe", return_value=None), \
             patch.object(describe, "_ollama_describe", return_value="A plant on a desk") as m_remote:
            result = describe.describe_image("/tmp/frame.jpg")
        m_remote.assert_called_once()
        assert result["backend"] == "ollama"
        assert result["description"] == "A plant on a desk"

    def test_error_when_no_backend_available(self):
        with patch.object(describe, "_local_gemma_describe", return_value=None), \
             patch.object(describe, "_ollama_describe", return_value=None):
            result = describe.describe_image("/tmp/frame.jpg")
        assert result["backend"] == "none"
        assert "error" in result

    def test_uses_configured_base_and_model(self):
        with patch.object(describe, "_local_gemma_describe", return_value=None), \
             patch.object(describe, "_ollama_describe", return_value="x") as m_remote:
            describe.describe_image("/tmp/frame.jpg", base="http://custom:9999", model="my/model")
        args = m_remote.call_args.args
        assert args[0] == "http://custom:9999"
        assert args[1] == "my/model"


# ─────────────────────────────────────────────────────────────────────────────
# _local_gemma_describe
# ─────────────────────────────────────────────────────────────────────────────
class TestLocalGemmaDescribe:
    def test_skips_when_server_not_running(self):
        with patch("core.inference.model_client.is_server_running", return_value=False) as m_run, \
             patch("core.inference.model_client.infer_with_image") as m_infer:
            out = describe._local_gemma_describe("/tmp/f.jpg", "desc", 512)
        m_run.assert_called_once()
        m_infer.assert_not_called()
        assert out is None

    def test_returns_text_when_server_responds(self):
        with patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer_with_image", return_value="Two people walking") as m_infer:
            out = describe._local_gemma_describe("/tmp/f.jpg", "desc", 512)
        m_infer.assert_called_once_with("/tmp/f.jpg", "desc", max_new_tokens=512)
        assert out == "Two people walking"

    def test_rejects_model_server_error_prefix(self):
        with patch("core.inference.model_client.is_server_running", return_value=True), \
             patch("core.inference.model_client.infer_with_image", return_value="[model_server error] boom"):
            out = describe._local_gemma_describe("/tmp/f.jpg", "desc", 512)
        assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# _ollama_describe
# ─────────────────────────────────────────────────────────────────────────────
class TestOllamaDescribe:
    def test_posts_base64_image_to_generate(self):
        fake_jpeg = b"\xff\xd8fakejpeg\xff\xd9"
        with patch("builtins.open", MagicMock()) as m_open:
            m_open.return_value.__enter__.return_value.read.return_value = fake_jpeg
            with patch("requests.post") as m_post:
                m_post.return_value = MagicMock(status_code=200, json=lambda: {"response": "A room"})
                out = describe._ollama_describe("http://brain:8005", "m", "/tmp/f.jpg", "desc", 128)

        m_post.assert_called_once()
        url = m_post.call_args.args[0]
        assert url == "http://brain:8005/api/generate"
        payload = m_post.call_args.kwargs["json"]
        assert payload["model"] == "m"
        assert payload["images"] == [base64.b64encode(fake_jpeg).decode()]
        assert payload["options"] == {"num_predict": 128}
        assert out == "A room"

    def test_returns_none_on_http_error(self):
        with patch("builtins.open", MagicMock()) as m_open:
            m_open.return_value.__enter__.return_value.read.return_value = b"jpeg"
            with patch("requests.post", side_effect=Exception("conn refused")):
                out = describe._ollama_describe("http://brain:8005", "m", "/tmp/f.jpg", "desc", 128)
        assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# grab_frame — MJPEG frame extraction
# ─────────────────────────────────────────────────────────────────────────────
class TestGrabFrame:
    def test_returns_last_complete_jpeg(self):
        # Two complete JPEG frames in the stream; grab_frame returns the last.
        frame1 = b"\xff\xd8AAA\xff\xd9"
        frame2 = b"\xff\xd8BBB\xff\xd9"
        chunks = iter([frame1 + frame2])
        with patch("requests.get") as m_get:
            m_get.return_value = MagicMock(status_code=200, iter_content=lambda chunk_size: chunks)
            out = capture.grab_frame("http://eye/video_feed")
        assert out == frame2

    def test_returns_none_on_non_200(self):
        with patch("requests.get") as m_get:
            m_get.return_value = MagicMock(status_code=404)
            out = capture.grab_frame("http://eye/video_feed")
        assert out is None

    def test_returns_none_on_empty_stream(self):
        with patch("requests.get") as m_get:
            m_get.return_value = MagicMock(status_code=200, iter_content=lambda chunk_size: iter([]))
            out = capture.grab_frame("http://eye/video_feed")
        assert out is None

    def test_returns_none_for_empty_url(self):
        assert capture.grab_frame("") is None

    def test_returns_none_on_network_error(self):
        with patch("requests.get", side_effect=Exception("down")):
            out = capture.grab_frame("http://eye/video_feed")
        assert out is None
