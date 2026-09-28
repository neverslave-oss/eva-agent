"""
Tests for the ElevenLabs realtime receive path.

These are isolated and need NO live network / ElevenLabs credentials. They
exercise the buffer accumulation + GET /rt/receive semantics using a fake
RealtimeSession injected into app._SESSIONS, plus unit checks on the
RealtimeSession buffer fields.
"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from elevenlabs_rt import app as app_module
from elevenlabs_rt.realtime import RealtimeSession


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app_module.app)


def _fake_session(closed: bool = False) -> RealtimeSession:
    session = RealtimeSession(conversation_id="conv-fake", agent_id="agent-fake")
    session._closed = closed
    return session


def test_session_buffer_fields_default() -> None:
    session = RealtimeSession(conversation_id="c")
    assert session.audio_buffer == b""
    assert session.transcripts == []


def test_session_buffer_accumulates() -> None:
    session = RealtimeSession(conversation_id="c")
    session.audio_buffer += b"\x00\x01\x02"
    session.transcripts.append("hello")
    assert session.audio_buffer == b"\x00\x01\x02"
    assert session.transcripts == ["hello"]


def test_receive_returns_and_clears_audio_buffer(client: TestClient) -> None:
    sid = "sid-1"
    session = _fake_session()
    session.audio_buffer = b"\x01\x02\x03\x04"
    session.transcripts.append("assistant reply")
    app_module._SESSIONS[sid] = session

    resp = client.get(f"/rt/receive?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["audio_b64"] == base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")
    assert data["transcripts"] == ["assistant reply"]
    assert data["done"] is False

    # After the first receive the audio buffer is cleared (incremental polling).
    assert session.audio_buffer == b""


def test_receive_is_incremental(client: TestClient) -> None:
    sid = "sid-2"
    session = _fake_session()
    app_module._SESSIONS[sid] = session

    # First poll: nothing yet.
    first = client.get(f"/rt/receive?session_id={sid}").json()
    assert first["audio_b64"] == ""

    # Simulate the on_audio callback appending new PCM.
    session.audio_buffer += b"\xaa\xbb"
    second = client.get(f"/rt/receive?session_id={sid}").json()
    assert second["audio_b64"] == base64.b64encode(b"\xaa\xbb").decode("ascii")
    assert session.audio_buffer == b""


def test_receive_done_when_closed(client: TestClient) -> None:
    sid = "sid-3"
    session = _fake_session(closed=True)
    app_module._SESSIONS[sid] = session
    data = client.get(f"/rt/receive?session_id={sid}").json()
    assert data["done"] is True


def test_receive_404_for_unknown_session(client: TestClient) -> None:
    resp = client.get("/rt/receive?session_id=does-not-exist")
    assert resp.status_code == 404
