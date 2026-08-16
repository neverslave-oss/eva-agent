# Peer Protocol — kernel-evolving ADR-015

kernel-evolving exposes a `/peers` endpoint and optionally announces via mDNS.
Any agent that implements this protocol can be auto-discovered.

## /peers endpoint

`GET http://localhost:8779/peers`

Response:
```json
{
  "peers": [
    {
      "id": "kernel-base",
      "url": "http://localhost:8769",
      "role": "production",
      "agent_type": "kernel",
      "version": "1.0.3",
      "capabilities": [],
      "mcp_endpoint": "",
      "last_seen": 1779700000.0,
      "source": "port-scan",
      "healthy": true
    }
  ],
  "discovery_enabled": true,
  "peer_count": 1
}
```

## mDNS Service Record (when `discovery.mdns: true`)

Service type: `_kernel-evolving._tcp.local.`
Service name: `{node_id}._kernel-evolving._tcp.local.`

TXT records:
- `version=v1.9.0-evolving`
- `role=research-sandbox`
- `api=/`

## Identity endpoint

For auto-discovery, any agent should respond to `GET /` with at minimum:
```json
{
  "name": "agent-name",
  "version": "x.y.z",
  "status": "ok"
}
```

kernel-evolving uses this to identify peer type from the `name` field.

## Agent types recognised

| name field contains | Classified as |
|---------------------|---------------|
| kernel-evolving     | kernel-evolving |
| Kernel              | kernel-base |
| opencode            | opencode |
| claude              | claude-code |
| codex               | codex |
| aider               | aider |
| openclaw / OpenClaw | openclaw |

## Well-known ports scanned

| Port  | Path        | Agent type    |
|-------|-------------|---------------|
| 8769  | /           | kernel-base   |
| 8779  | /           | kernel-evolving (self — skipped) |
| 8780  | /           | kernel-evolving (shadow) |
| 18789 | /health     | openclaw      |
| 3000  | /api/info   | opencode      |
| 3001  | /health     | aider         |
| 8888  | /health     | codex         |
| 40000 | /health     | claude-code   |

## Peer roles

| Role         | Description |
|--------------|-------------|
| production   | Stable, user-facing kernel instance |
| research     | Experimental kernel instance |
| coding       | Active coding agent (opencode, claude-code, aider, codex) |
| orchestrator | Orchestration layer (OpenClaw/Olly) |

## Delegation (future)

When `delegation.enabled: true` in config, kernel-evolving will route:
- `code_generation` / `refactor` / `debug` tasks → `coding` peers
- Orchestration escalations → `orchestrator` peers
- Evolution tasks → `production` peers

Delegation is passive (logging only) until explicitly enabled to prevent routing loops.
