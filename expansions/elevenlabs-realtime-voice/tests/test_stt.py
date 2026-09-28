"""
Tests for the ElevenLabs /stt endpoint.

These are isolated and need NO live network / ElevenLabs credentials. They mock
the upstream ElevenLabs Speech-to-Text HTTP call (app._elevenlabs_stt) so the
endpoint's request handling (multipart file + JSON base64) is exercised without
touching api.elevenlabs.io.
"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from elevenlabs_rt import app as app_module


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app_module.app)


@pytest.fixture(autouse=True)
def _fake_stt(monkeypatch):
    """Replace the upstream ElevenLabs STT call with a deterministic fake."""

    async def _fake(audio_bytes: bytes, filename: str = "audio.wav") -> str:
        # Echo a marker so the test can confirm the right bytes were forwarded.
        return f"transcribed:{len(audio_bytes)}:{filename}"

    monkeypatch.setattr(app_module, "_elevenlabs_stt", _fake)


def test_stt_multipart_file(client: TestClient) -> None:
    audio = b"\x00\x01\x02\x03RIFFfakewavdata"
    resp = client.post(
        "/stt",
        files={"file": ("rec.wav", audio, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == f"transcribed:{len(audio)}:rec.wav"


def test_stt_json_audio_b64(client: TestClient) -> None:
    audio = b"\x52\x49\x46\x46wav"
    resp = client.post(
        "/stt",
        json={"audio_b64": base64.b64encode(audio).decode("ascii"), "format": "wav"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == f"transcribed:{len(audio)}:audio.wav"


def test_stt_multipart_missing_file(client: TestClient) -> None:
    resp = client.post("/stt", files={})
    assert resp.status_code == 400


def test_stt_json_missing_audio_b64(client: TestClient) -> None:
    resp = client.post("/stt", json={"format": "wav"})
    assert resp.status_code == 400


def test_stt_unknown_content_type(client: TestClient) -> None:
    resp = client.post("/stt", content=b"garbage", headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 400
