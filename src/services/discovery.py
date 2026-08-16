"""
discovery.py — ADR-015 Agent auto-discovery.
Three-tier: port scan (Tier 1), mDNS (Tier 2, optional), coding agent detection (Tier 3).
"""
import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

WELL_KNOWN_AGENTS = [
    # (port, path, identity_key, agent_type)
    (8769,  "/",             "name",    "kernel-base"),
    (8779,  "/",             "name",    "kernel-evolving"),   # self — detected and skipped
    (8780,  "/",             "name",    "kernel-evolving"),   # shadow instance
    (18789, "/health",       "status",  "openclaw"),
    (3000,  "/api/info",     "version", "opencode"),
    (8888,  "/health",       "status",  "codex"),
    (40000, "/health",       "status",  "claude-code"),
    (3001,  "/health",       "status",  "aider"),
]

CODING_AGENT_FILES = [
    ("~/.opencode/session.json",    "opencode"),
    ("~/.claude/mcp_server_pid",    "claude-code"),
    ("~/.aider/server.json",        "aider"),
]


@dataclass
class DiscoveredPeer:
    id: str
    url: str
    role: str          # production | research | coding | orchestrator
    agent_type: str    # kernel | openclaw | opencode | claude-code | aider | codex
    version: str = ""
    capabilities: list = field(default_factory=list)
    mcp_endpoint: str = ""
    workspace: str = ""   # resolved workspace path (probed from peer status endpoint)
    last_seen: float = field(default_factory=time.time)
    source: str = "port-scan"   # port-scan | mdns | config | file
    healthy: bool = True


class AgentDiscovery:
    """
    ADR-015 three-tier agent auto-discovery.
    Tier 1: port scan of well-known local ports.
    Tier 2: mDNS/Zeroconf (optional, enabled via config).
    Tier 3: coding agent file detection.
    """

    def __init__(self, config: dict, self_port: int = 8779):
        self._config = config
        self._disc_cfg = config.get("discovery", {})
        self._self_port = self_port
        self._peers: dict[str, DiscoveredPeer] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._interval_s = float(self._disc_cfg.get("interval_s", 60))
        self._timeout_s = float(self._disc_cfg.get("timeout_s", 1.0))
        self._mdns_enabled = bool(self._disc_cfg.get("mdns", False))
        self._coding_agents_enabled = bool(self._disc_cfg.get("coding_agents", True))

        # Load static peers from config immediately
        for peer_cfg in config.get("peers", []):
            peer = DiscoveredPeer(
                id=peer_cfg.get("id", "unknown"),
                url=peer_cfg.get("url", ""),
                role=peer_cfg.get("role", "production"),
                agent_type=peer_cfg.get("agent_type", "kernel"),
                version=peer_cfg.get("version", ""),
                source="config",
            )
            with self._lock:
                self._peers[peer.id] = peer

    def start(self):
        """Start background discovery refresh loop."""
        t = threading.Thread(target=self._refresh_loop, daemon=True, name="agent-discovery")
        t.start()
        logger.info("[discovery] background refresh thread started (interval=%ss)", self._interval_s)

    def stop(self):
        """Signal the background thread to stop."""
        self._stop.set()

    def get_peers(self) -> list[dict]:
        """Return all known peers as list of dicts."""
        with self._lock:
            return [asdict(p) for p in self._peers.values()]

    def get_peers_by_role(self, role: str) -> list[DiscoveredPeer]:
        with self._lock:
            return [p for p in self._peers.values() if p.role == role]

    def get_peers_by_type(self, agent_type: str) -> list[DiscoveredPeer]:
        with self._lock:
            return [p for p in self._peers.values() if p.agent_type == agent_type]

    def _refresh_loop(self):
        """Main discovery loop — runs until stop() is called."""
        # Initial scan on startup
        self._scan_ports()
        if self._coding_agents_enabled:
            self._scan_coding_agent_files()

        while not self._stop.wait(self._interval_s):
            self._scan_ports()
            if self._coding_agents_enabled:
                self._scan_coding_agent_files()
            self._expire_stale()

    def _probe(self, port: int, path: str, timeout: float = 1.0) -> Optional[dict]:
        """HTTP GET localhost:{port}{path}. Returns parsed JSON or None on any error."""
        try:
            url = f"http://localhost:{port}{path}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                return json.loads(data)
        except Exception:
            return None

    def _classify_agent_type(self, response: dict, default_type: str) -> str:
        """Determine agent_type from response fields."""
        name = str(response.get("name", "")).lower()
        if "kernel-evolving" in name or "kernel evolving" in name:
            return "kernel-evolving"
        if "kernel" in name:
            return "kernel-base"
        if "opencode" in name:
            return "opencode"
        if "claude" in name:
            return "claude-code"
        if "codex" in name:
            return "codex"
        if "aider" in name:
            return "aider"
        if "openclaw" in name.lower() or "olly" in name.lower() or "hermes" in name.lower():
            return "openclaw"
        return default_type

    def _role_for_type(self, agent_type: str) -> str:
        if agent_type in ("opencode", "claude-code", "codex", "aider"):
            return "coding"
        if agent_type == "openclaw":
            return "orchestrator"
        return "production"

    def _scan_ports(self):
        """Tier 1: scan well-known ports for local agents."""
        for port, path, identity_key, default_type in WELL_KNOWN_AGENTS:
            if port == self._self_port:
                continue  # skip self

            response = self._probe(port, path, self._timeout_s)
            if response is None:
                # Mark existing non-config peer as unhealthy if we knew about it
                peer_id = f"{default_type}:{port}"
                with self._lock:
                    if peer_id in self._peers and self._peers[peer_id].source != "config":
                        if self._peers[peer_id].healthy:
                            logger.info("[discovery] peer %s at port %d went offline", peer_id, port)
                            self._peers[peer_id].healthy = False
                continue

            agent_type = self._classify_agent_type(response, default_type)
            # Skip if this response identifies as ourselves
            if agent_type == "kernel-evolving" and port == self._self_port:
                continue

            version = str(response.get("version", ""))
            peer_id = response.get("name") or f"{agent_type}:{port}"
            if isinstance(peer_id, str):
                peer_id = peer_id.lower().replace(" ", "-")

            role = self._role_for_type(agent_type)
            url = f"http://localhost:{port}"

            peer = DiscoveredPeer(
                id=peer_id,
                url=url,
                role=role,
                agent_type=agent_type,
                version=version,
                source="port-scan",
                last_seen=time.time(),
                healthy=True,
            )
            # For orchestrator peers (OpenClaw), probe workspace from status endpoint
            if agent_type == "openclaw":
                ws = self._probe_openclaw_workspace(port)
                if ws:
                    peer.workspace = ws
            self._register(peer)

    def _scan_coding_agent_files(self):
        """Tier 3: check filesystem for active coding agent sessions."""
        for file_pattern, agent_type in CODING_AGENT_FILES:
            fpath = os.path.expanduser(file_pattern)
            if not os.path.exists(fpath):
                continue

            url = ""
            try:
                with open(fpath) as f:
                    content = f.read().strip()
                    # Try JSON parsing for port/url info
                    if content.startswith("{"):
                        data = json.loads(content)
                        port = data.get("port") or data.get("serverPort")
                        if port:
                            url = f"http://localhost:{port}"
                    elif content.isdigit():
                        # PID file — agent is running but no URL known
                        url = f"file://{fpath}"
            except Exception:
                url = f"file://{fpath}"

            if not url:
                url = f"file://{fpath}"

            peer_id = f"{agent_type}-session"
            peer = DiscoveredPeer(
                id=peer_id,
                url=url,
                role="coding",
                agent_type=agent_type,
                source="file",
                last_seen=time.time(),
                healthy=True,
            )
            self._register(peer)

    def _probe_openclaw_workspace(self, port: int) -> str:
        """Probe OpenClaw's /status endpoint to extract its workspace path."""
        resp = self._probe(port, "/status", self._timeout_s)
        if resp:
            # OpenClaw /status returns {workspace: ...} or nested config.workspace
            ws = resp.get("workspace") or resp.get("config", {}).get("workspace", "")
            if ws:
                return os.path.expanduser(str(ws))
        # Fallback: derive from $HOME if probe fails
        home = os.path.expanduser("~")
        candidate = os.path.join(home, ".openclaw", "workspace")
        return candidate if os.path.isdir(candidate) else ""

    def _register(self, peer: DiscoveredPeer):
        """Register or update a discovered peer in the registry."""
        with self._lock:
            if peer.id in self._peers:
                existing = self._peers[peer.id]
                existing.last_seen = peer.last_seen
                existing.healthy = True
                if peer.version:
                    existing.version = peer.version
            else:
                self._peers[peer.id] = peer
                logger.info(
                    "[discovery] discovered new peer: %s (%s) at %s [source=%s]",
                    peer.id, peer.agent_type, peer.url, peer.source,
                )

    def _expire_stale(self):
        """Remove non-config peers not seen in 3x interval."""
        cutoff = time.time() - 3 * self._interval_s
        with self._lock:
            stale = [
                pid for pid, p in self._peers.items()
                if p.source != "config" and p.last_seen < cutoff
            ]
            for pid in stale:
                logger.info("[discovery] expiring stale peer: %s", pid)
                del self._peers[pid]

    def start_mdns(self):
        """Tier 2: announce self and browse for other kernel/openclaw instances via mDNS."""
        try:
            from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo
            import socket

            _version = self._config.get("self_identity", {}).get("version_tag", "unknown")
            _role = self._config.get("self_identity", {}).get("role", "research")
            _node_id = self._config.get("self_identity", {}).get("codename", "kernel-evolving")

            info = ServiceInfo(
                "_kernel-evolving._tcp.local.",
                f"{_node_id}._kernel-evolving._tcp.local.",
                addresses=[socket.inet_aton("127.0.0.1")],
                port=self._self_port,
                properties={
                    "version": _version,
                    "role": _role,
                    "api": "/",
                },
            )

            zc = Zeroconf()
            zc.register_service(info)
            logger.info("[discovery] mDNS: announced %s on port %d", _node_id, self._self_port)

            discovery_obj = self

            class _Listener:
                def add_service(self, zc_, type_, name):
                    svc_info = zc_.get_service_info(type_, name)
                    if svc_info:
                        addr = "127.0.0.1"  # local for now
                        port = svc_info.port
                        peer_id = name.split(".")[0]
                        peer = DiscoveredPeer(
                            id=peer_id,
                            url=f"http://{addr}:{port}",
                            role="research",
                            agent_type="kernel-evolving",
                            source="mdns",
                            last_seen=time.time(),
                        )
                        discovery_obj._register(peer)

                def remove_service(self, zc_, type_, name):
                    peer_id = name.split(".")[0]
                    with discovery_obj._lock:
                        if peer_id in discovery_obj._peers:
                            discovery_obj._peers[peer_id].healthy = False

                def update_service(self, zc_, type_, name):
                    pass

            _listener = _Listener()
            ServiceBrowser(zc, "_kernel-evolving._tcp.local.", _listener)
            ServiceBrowser(zc, "_openclaw._tcp.local.", _listener)
            logger.info("[discovery] mDNS: browsing for kernel-evolving and openclaw services")

        except ImportError:
            logger.warning(
                "[discovery] mDNS requested but 'zeroconf' package not installed. "
                "Falling back to Tier 1 port-scan only. Install with: pip install zeroconf"
            )
        except Exception as e:
            logger.warning("[discovery] mDNS startup failed (non-fatal): %s", e)
