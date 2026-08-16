---
date: 2026-08-06
agent: copilot
topic: Cross-platform onboarding and pairing — gap analysis across kernel-desktop-v1, kernel-mobile-v2, kernel-central, kernel-evolving
severity: high
tags: [cross-repo, audit, security, onboarding]
status: open
---

# Cross-platform onboarding & pairing — gap analysis

## Scope

Covers the full path from "user installs a client app" → "agent is responding on that device":

- **kernel-desktop-v1** — Laravel 13 + NativePHP desktop app; SetupWizard (7 steps); connects to kernel-central relay via WebSocket (TunnelService) and relays messages to kernel-evolving via HTTP
- **kernel-mobile-v2** — React Native/Expo app; dual-mode (direct LAN / proxy via kernel-central); no QR code or camera code exists
- **kernel-central** — Laravel 13 + Reverb relay at :8081; Device model with `pair_token`/`pair_secret`; QR pairing web page; broadcast events
- **kernel-evolving** — FastAPI agent at :8779; receives messages, runs triage, responds; multi-agent-collective-memory service at <collective-memory-host>:8010

## Findings

### A — Security (P0)

**A1. Broadcast events use public Channel** (`kernel-central`)
All three events (`MessageRelayRequested`, `DeviceConnected`, `DeviceDisconnected`) return `new Channel(...)` (unauthenticated). Any client that knows a device token can subscribe to another device's relay stream. The `Broadcast::channel()` auth closure in `routes/channels.php` is dead code. Fix: `PrivateChannel` in all three events + activate auth closure.
→ **XP1a** (prerequisite for XP5)

### B — Broken wiring (P1)

**B1. MessageForwarderService calls nonexistent endpoint** (`kernel-desktop-v1`)
`MessageForwarderService::forwardToKernelEvolving()` POSTs to `/chat/message`. kernel-evolving has no such route — the correct endpoint is `POST /message`. Every desktop chat message silently fails. Additionally, response parsing reads `response.message.text` but kernel-evolving returns `{"reply": str}`.
→ **XP1d**

**B2. TunnelService reconnect backoff never increments** (`kernel-desktop-v1`)
`$this->reconnectAttempts` is never incremented in the reconnect loop. WebSocket disconnects trigger unlimited immediate reconnects with no delay, flooding kernel-central's Reverb. The exponential backoff calculation is correct but never reached.
→ **XP1b**

**B3. Dead code appends :80/:443 to WebSocket URL unconditionally** (`kernel-desktop-v1`)
A dead block always appends the default port for the scheme to the WS URL, producing `ws://host:80/app/key:80` (double port). The port is already present in the base URL.
→ **XP1c**

### C — Missing features (P1)

**C1. No QR / camera code in mobile app** (`kernel-mobile-v2`)
`settingsStore` has mode=`direct`|`proxy` and `pairDevice()` is absent. No camera permission, no barcode scanner, no `PairScreen` component. The kernel-central QR pairing web page exists and issues `pair_token`/`pair_secret` but no mobile client flow consumes it.
→ **XP5** (depends on XP1a)

**C2. SetupWizard has no model-storage-path field** (`kernel-desktop-v1`)
`KernelEvolvingService::writeEnvFile()` has a `MODELS_PATH` placeholder that is never populated from user input. Users have no way to set `HF_HOME` from the wizard.
→ **XP2**

**C3. Settings::save() is a stub** (`kernel-desktop-v1`)
`app/Livewire/Settings.php::save()` emits a "saved" event with zero persistence. API key management is a no-op.
→ **XP3**

**C4. Telegram wizard has no bot token test** (`kernel-desktop-v1`)
The wizard collects a bot token but never validates it against the Telegram Bot API. Users who mis-type their token won't discover the error until first message.
→ **XP4**

**C5. Collective memory URL not configurable from desktop UI** (`kernel-desktop-v1`)
The collective-memory service URL (`config.yaml collective_memory.url`) is currently hardcoded to <collective-memory-host>:8010. No wizard step or settings field exposes it. XP6b (kernel-evolving HTTP client) is already complete — this is the desktop complement.
→ **XP6a**

### D — Kernel-evolving gaps already in tasks.md

**D1. Collective memory not queried for non-Telegram callers** → PD1 (FIXED 2026-08-06)
**D2. System prompt token budget hardcoded** → PD2 (FIXED 2026-08-06)
**D3. Collective memory HTTP wiring missing** → XP6b (FIXED 2026-08-06, service live at <collective-memory-host>:8010)

## Task index

| ID | Repo | Priority | Description |
|----|------|----------|-------------|
| XP1a | kernel-central | P0 | PrivateChannel in broadcast events |
| XP1b | kernel-desktop-v1 | P1 | Reconnect backoff fix |
| XP1c | kernel-desktop-v1 | P1 | Dead :80 port append fix |
| XP1d | kernel-desktop-v1 | P1 | Fix `/chat/message` endpoint + response parsing |
| XP2 | kernel-desktop-v1 | P1 | Model storage path in wizard |
| XP3 | kernel-desktop-v1 | P1 | Settings save() implementation |
| XP4 | kernel-desktop-v1 | P1 | Telegram bot token test in wizard |
| XP5 | kernel-mobile-v2 | P2 | QR pairing screen (requires XP1a) |
| XP6a | kernel-desktop-v1 | P1 | Collective memory URL in wizard/settings |
| XP6b | kernel-evolving | P1 | Collective memory HTTP client — **COMPLETE** |

## Implementation notes

- XP1a (kernel-central PrivateChannel) must be deployed before XP5 (mobile QR pairing) can work end-to-end. XP1b/c/d are independent and can be done in parallel.
- XP2/XP3/XP4/XP6a are all in the same Livewire files in kernel-desktop-v1 and can be batched in one PR.
- XP5 requires adding a new Expo dependency (expo-barcode-scanner or expo-camera v14+) and a new screen.
- XP6b is already live: `src/core/collective_memory_client.py` + `config.yaml collective_memory.url: http://<collective-memory-host>:8010`.
