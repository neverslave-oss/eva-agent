"""Tests for ADR-015 AgentDiscovery."""
import time
import threading
from unittest.mock import patch, MagicMock
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.discovery import AgentDiscovery, DiscoveredPeer


def _minimal_config():
    return {
        "discovery": {
            "enabled": True,
            "interval_s": 60,
            "port_scan": True,
            "mdns": False,
            "coding_agents": False,
            "timeout_s": 0.5,
        },
        "delegation": {"enabled": False},
        "peers": [
            {"id": "static-peer", "url": "http://localhost:9999", "role": "production"}
        ],
        "api": {"port": 8779},
    }


def test_static_peers_loaded():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    peers = d.get_peers()
    ids = [p["id"] for p in peers]
    assert "static-peer" in ids


def test_self_port_not_discovered():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    # Patch _probe to return a valid response for all ports including self
    with patch.object(d, "_probe", return_value={"name": "kernel-evolving", "version": "1.0"}):
        d._scan_ports()
    peers = d.get_peers()
    # Self port (8779) should never be added via port-scan
    for p in peers:
        if p["source"] != "config":
            assert p["url"] != "http://localhost:8779", \
                f"Self port discovered as peer: {p}"


def test_probe_returns_none_on_error():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    # Port 1 is always closed / refused
    result = d._probe(1, "/health", timeout=0.2)
    assert result is None


def test_peer_registered_on_probe():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    with patch.object(d, "_probe", return_value={"name": "Kernel", "version": "1.0.3", "status": "ok"}):
        d._scan_ports()
    peers = d.get_peers()
    kernel_peers = [p for p in peers if p["agent_type"] == "kernel-base"]
    assert len(kernel_peers) >= 1


def test_stale_peers_expired():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    # Manually add a stale non-config peer
    stale = DiscoveredPeer(
        id="stale",
        url="http://localhost:9990",
        role="coding",
        agent_type="opencode",
        last_seen=time.time() - 1000,
        source="port-scan",
    )
    with d._lock:
        d._peers["stale"] = stale
    d._interval_s = 1
    d._expire_stale()
    assert "stale" not in d._peers


def test_get_peers_by_role():
    d = AgentDiscovery(_minimal_config(), self_port=8779)
    p = DiscoveredPeer(
        id="coder1",
        url="http://localhost:3000",
        role="coding",
        agent_type="opencode",
        source="port-scan",
    )
    with d._lock:
        d._peers["coder1"] = p
    coding = d.get_peers_by_role("coding")
    assert any(peer.id == "coder1" for peer in coding)
