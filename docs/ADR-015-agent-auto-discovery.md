# ADR-015: Agent Auto-Discovery — Peer Kernels, OpenClaw, and Coding Agents

**Status:** Proposed 📝  
**Date:** 2026-05-25  
**Author:** Olly + Fabio  
**Related:** ADR-004 (Self-Evolving), ADR-008 (Critic Replica), ADR-011 (Micro-Planner)

---

## Problem

kernel-evolving currently hardcodes its peer topology:

```yaml
peers:
  - id: kernel-base
    url: http://localhost:8769
    role: production
```

And has no awareness of:
- Other kernel-evolving instances on the same LAN
- OpenClaw (Olly/Hermes) gateway — the orchestration layer that can delegate tasks
- Coding agents: Codex CLI, opencode, Claude Code, Aider, Cursor — all expose MCP or HTTP interfaces when active
- Future agents in the same cluster

This creates three practical problems:

1. **Portability** — every deployment needs manual peer config. Moving to a new machine or Docker network requires editing `config.yaml`.
2. **Isolation** — kernel-evolving can't offload coding tasks to an active coding agent even if one is running locally. It either does everything itself (slow, imprecise) or fails.
3. **Cluster growth** — when a second kernel-evolving instance is spun up (different model, different specialisation), neither knows the other exists. Cooperative task routing is impossible.

---

## Decision

Implement a three-tier auto-discovery system that activates on boot and runs a lightweight background refresh loop every `discovery_interval_s` (default: 60s).

---

## Tier 1 — Local Port Scan (always on)

On startup, scan a well-known port list for local services:

```python
WELL_KNOWN_AGENTS = [
    # (port, identity_path, expected_field, agent_type)
    (8769,  "/",        "name",    "kernel-base"),
    (8779,  "/",        "name",    "kernel-evolving"),  # self — skip
    (8780,  "/",        "name",    "kernel-evolving"),  # shadow instance
    (18789, "/health",  "status",  "openclaw"),         # OpenClaw gateway
    (3000,  "/api/info","version", "opencode"),         # opencode
    (8888,  "/health",  "status",  "codex"),            # Codex CLI server
    (40000, "/health",  "status",  "claude-code"),      # Claude Code MCP
]
```

For each port:
1. `GET http://localhost:{port}{path}` with 1s timeout
2. Parse JSON — extract name/version/type
3. If the response contains a recognised identity field → register as a discovered peer
4. Cache result with TTL = `discovery_interval_s`

Self-detection: if a discovered peer returns the same `node_id` as `self_identity.codename`, skip it.

---

## Tier 2 — mDNS/Zeroconf (LAN, when `cluster.discovery: mdns`)

Announce self as a `_kernel-evolving._tcp.local.` mDNS service on startup:

```
Service name:  {node_id}._kernel-evolving._tcp.local.
Port:          8779 (or configured port)
TXT records:
  version=v1.9.0-evolving
  role=research-sandbox
  models=nemotron-3b
  adr=004,005,006,007,008,009,010,011,012,013,014
  api=/
```

Browse `_kernel-evolving._tcp.local.` to find other instances. Each discovered node is contacted via HTTP for capability exchange.

For OpenClaw, browse `_openclaw._tcp.local.` (if OpenClaw announces via mDNS — fallback to port scan at 18789).

Uses `zeroconf` Python package (already in requirements for future audio features). Graceful degradation: if `zeroconf` is not installed or mDNS is blocked, fall back to Tier 1 only.

---

## Tier 3 — Coding Agent Discovery

Coding agents expose MCP servers or HTTP APIs when active. Discovery approach per agent:

| Agent | Detection | Integration |
|-------|-----------|-------------|
| **opencode** | `GET http://localhost:3000/api/info` or `~/.opencode/session.json` | MCP client via `mcp` Python package |
| **Claude Code** | `~/.claude/mcp_server_pid` or `GET http://localhost:40000/health` | MCP client (stdio or HTTP) |
| **Codex CLI** | `ps aux \| grep "codex.*--server"` or `GET http://localhost:8888/health` | HTTP REST |
| **Aider** | `~/.aider/server.json` (port from JSON) | HTTP REST |
| **Cursor** | `~/.cursor/extensions/*/mcp_server.json` | MCP client |
| **GitHub Copilot** | OpenClaw configured provider — delegate via openclaw endpoint | HTTP via OpenClaw |

For all coding agents: register as a `coding` role peer. When a task is classified as `code_generation`, `refactor`, or `debug` by the micro-planner, the routing layer can delegate to the best available coding peer instead of synthesising a new skill locally.

---

## Peer Registry

Discovered peers are stored in memory (not persisted between restarts — re-discovered each boot):

```python
@dataclass
class DiscoveredPeer:
    id: str               # e.g. "kernel-base", "opencode-session-abc123"
    url: str              # http://localhost:3000
    role: str             # production | research | coding | orchestrator
    agent_type: str       # kernel | openclaw | opencode | claude-code | aider | codex
    version: str
    capabilities: list    # from /skills or /capabilities endpoint
    mcp_endpoint: str     # optional — MCP server URL/socket
    last_seen: float      # unix timestamp
    source: str           # port-scan | mdns | config
```

API endpoint: `GET /peers` — returns all currently known peers with their status and capabilities.

---

## Task Routing Integration

When a task enters `triage()`, after micro-planner decomposition, the routing layer checks:

```
task type classified → check peer registry
  coding task + coding peer available → delegate via MCP or HTTP
  evolution task + kernel-base peer available → delegate if configured
  orchestration hint + openclaw peer available → escalate
  → fallback: handle locally
```

Delegation is opt-in. The `delegation.enabled` config flag defaults to `false` — auto-discovery runs but routing only delegates when enabled:

```yaml
delegation:
  enabled: false         # set true to allow task delegation to discovered peers
  coding_agents: true    # delegate code tasks to coding agents when available
  openclaw: true         # escalate to OpenClaw for orchestration
  kernel_peers: true     # route to other kernel instances
```

---

## MCP Client Integration

For agents discovered via MCP (Claude Code, opencode, Cursor):

1. Connect to MCP server (HTTP or stdio) using the `mcp` Python package
2. Fetch `tools/list` → map to kernel-evolving tool schema
3. Register as a virtual skill: `coding/{agent_name}/{tool_name}`
4. When invoked, translate tool call → MCP `tools/call` → return result

This means coding agent capabilities appear natively in kernel-evolving's skill list and can be invoked transparently by the micro-planner.

---

## OpenClaw Integration

OpenClaw runs at `http://localhost:18789` (default gateway port). When discovered:

1. Fetch OpenClaw capabilities via `GET /api/discover` or health endpoint
2. Register as an `orchestrator` peer
3. Two-way sync:
   - kernel-evolving registers itself with OpenClaw via `POST /api/agents/register` (if supported)
   - OpenClaw can delegate tasks to kernel-evolving via `POST http://localhost:8779/message`
4. Collective memory skill (`context_provider: true`) already bridges shared knowledge — no duplication needed

---

## Config

New `config.yaml` section:

```yaml
discovery:
  enabled: true
  interval_s: 60                    # re-scan interval
  port_scan: true                   # Tier 1 — always safe
  mdns: false                       # Tier 2 — enable on LAN deployments
  coding_agents: true               # Tier 3 — check for active coding agents
  timeout_s: 1                      # per-probe HTTP timeout

delegation:
  enabled: false                    # master switch — discovery is passive until this is true
  coding_agents: true
  openclaw: true
  kernel_peers: true
  prefer_local: true                # prefer local peers over remote when both available
```

---

## Implementation Plan

1. **`src/discovery.py`** — `AgentDiscovery` class: port scan, mDNS browser, peer registry, `GET /peers` route
2. **`src/mcp_client.py`** — thin MCP client: connect (HTTP/stdio), list tools, call tool, map to skill schema
3. **`src/agent.py`** — routing hook: after micro-planner, check registry, delegate if `delegation.enabled`
4. **`config.yaml` + `config.container.yaml`** — add `discovery` and `delegation` sections
5. **`docs/`** — `PEER_PROTOCOL.md` — documents the `/peers` endpoint and mDNS service record schema for other agents to implement compatibility
6. **`start.sh`** — announce mDNS service when `discovery.mdns: true`

Estimated scope: ~400 lines new code, ~50 lines config, no breaking changes to existing behaviour.

---

## Consequences

**Positive:**
- kernel-evolving finds coding agents automatically — zero config on a developer machine
- Multiple kernel instances form a cooperative cluster without manual wiring
- OpenClaw and kernel-evolving become peers by default — OpenClaw can see kernel-evolving's skills; kernel-evolving can escalate to OpenClaw
- Docker deployments: Tier 2 (mDNS) works across containers on the same Docker network with `--network host` or a shared bridge

**Negative / trade-offs:**
- `zeroconf` adds a dependency (~200KB) — optional, graceful degradation
- `mcp` Python package adds a dependency — only imported when a coding agent is detected
- Discovery loop adds ~50ms to startup (port scan runs async)
- Delegation requires careful loop-prevention (a task delegated to peer A must not be re-delegated back)

**Not in scope (future):**
- Encrypted peer auth (trust-on-first-use model for now)
- Cross-machine WAN discovery (mDNS is LAN-only; WAN cluster needs explicit config)
- Automatic load balancing across kernel peers

---

## Related ADRs

- ADR-004: Two-tier evolution — delegation can trigger Tier 2 on a remote peer
- ADR-011: Micro-planner — task classification drives delegation routing decision
- ADR-013: Multi-provider — cloud providers remain separate from peer delegation
- ADR-014: Context pipeline — context providers remain local; peer context is a future extension
