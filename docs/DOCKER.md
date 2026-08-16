# Docker Guide — kernel-evolving

## Quick Start

```bash
cp .env.example .env
# Edit .env — fill in KERNEL_EVO_TELEGRAM_BOT_TOKEN + OPENAI_API_KEY at minimum

docker compose up -d
```

## Prerequisites

- Docker + NVIDIA Container Toolkit (for GPU mode)
- WSL2 / Linux host

### WSL2 Setup (once)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
sudo dockerd &
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Required | Description |
|---|---|---|
| `KERNEL_EVO_TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token |
| `KERNEL_EVO_TELEGRAM_CHAT_ID` | Yes | Your Telegram chat ID |
| `OPENAI_API_KEY` | Recommended | Cloud inference (Tier 2) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic fallback |
| `HF_TOKEN` | Optional | Private HF models |
| `MODELS_PATH` | Optional | Local HF cache dir (default: `~/.cache/huggingface`) |
| `KERNEL_WORKSPACE_PATH` | Optional | Host dir for the agent workspace (default: `~/.kernel-evolving/workspace`) |

## Model: Nemotron (default)

The container defaults to `nvidia/Nemotron-Labs-Diffusion-3B`. On first run with `task_inference: local`, the model is downloaded automatically via HF_HOME.

Set `MODELS_PATH` to reuse an existing cache:

```bash
MODELS_PATH=/path/to/huggingface/cache docker compose up -d
```

## Startup — entrypoint.sh

`entrypoint.sh` is the container entrypoint. It:

1. Reads `config.yaml` to detect `providers.task_inference`
2. If `local` → starts `model_server.py --lazy` in background, waits up to 60s for socket
3. If cloud provider → skips model server (saves VRAM)
4. Starts `uvicorn api:app` on port 8779

Force local mode even when config says cloud:
```bash
KERNEL_EVO_FORCE_LOCAL=1 docker compose up -d
```

## Profiles

### GPU (default)
```bash
docker compose up -d
```

### Sandbox / No-GPU (cloud-only mode)
```bash
docker compose --profile sandbox up -d
```
Uses `Dockerfile.sandbox-lite` — no GPU required, all inference via cloud providers.

### CLI shell
```bash
docker compose --profile cli run kernel-evolving-cli
```

## WSL2 Networking Notes

- `host.docker.internal` → resolves to host machine (for OpenClaw, kernel-base peer)
- If using `--network host`, set `OPENCLAW_ENDPOINT=http://localhost:18789`
- mDNS discovery (`mdns: true` in config) requires `--network host` or a shared bridge

## Health Check

```bash
curl http://localhost:8779/health
curl http://localhost:8779/version
```

## Think-at-Rest — Manual Trigger

```bash
curl -X POST http://localhost:8779/think/trigger
```

## Logs

```bash
docker compose logs -f kernel-evolving
# Model server log (inside container):
docker exec kernel-evolving cat /tmp/kernel_evolving_model_server.log
```

## Volumes

| Volume | Purpose |
|---|---|
| `kernel-memory` | Persistent agent memory (mounted at `/root`) |
| `kernel-ecosystem` | Ecosystem skills/routines cache |
| `MODELS_PATH` → `/models` | HF model cache |
| `KERNEL_WORKSPACE_PATH` → `/app/workspace` | Agent workspace — chat history, evolution DB, thought journal, trajectories. Bind-mounted so it's host-visible and survives container rebuilds (matches the bare-metal `kernel_workspace` path by default). |
