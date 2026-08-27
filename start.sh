#!/usr/bin/env bash
# kernel-evolving startup script
# Starts its OWN model server + API process.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_LOG="/tmp/kernel_evolving_api.log"
MODEL_LOG="/tmp/kernel_evolving_model_server.log"
SOCKET="/tmp/kernel_evolving_model.sock"
PORT=8779

# Optional alternate config: ./start.sh --config=config.reorganized.yaml
# Sets KERNEL_EVO_CONFIG so the API, telegram bot, and model server all use it.
CONFIG="$REPO/config.yaml"
for arg in "$@"; do
    case "$arg" in
        --config=*)
            CONFIG="${arg#--config=}"
            [[ "$CONFIG" != /* ]] && CONFIG="$REPO/$CONFIG"
            ;;
    esac
done
export KERNEL_EVO_CONFIG="$CONFIG"
echo "[kernel-evolving] Using config: $CONFIG"

# Prefer explicit KERNEL_EVO_PYTHON, then local venv, then system python3
if [[ -n "${KERNEL_EVO_PYTHON:-}" ]]; then
    PY="$KERNEL_EVO_PYTHON"
elif [[ -x "$REPO/.venv/bin/python3" ]]; then
    PY="$REPO/.venv/bin/python3"
elif [[ -x "$HOME/.miniconda/bin/python3" ]]; then
    PY="$HOME/.miniconda/bin/python3"
else
    PY="$(command -v python3)"
fi

echo "[kernel-evolving] Stopping API on port $PORT (if running)..."
fuser -k ${PORT}/tcp 2>/dev/null || true

# Always kill any existing model_server process so code changes take effect on restart
echo "[kernel-evolving] Stopping existing model server (if running)..."
pkill -f "src/core/inference/model_server.py" 2>/dev/null || true
rm -f "$SOCKET"
sleep 1

# Source .env
if [[ -f "$REPO/.env" ]]; then
    set -a; source "$REPO/.env"; set +a
    echo "[kernel-evolving] env loaded from .env"
else
    echo "[kernel-evolving] WARNING: no .env found — Telegram bot will be disabled"
fi

# EVOLUTION_ENABLED: default false — set to true in .env to enable autonomous skill evolution
export EVOLUTION_ENABLED=${EVOLUTION_ENABLED:-false}

GIT_VERSION=$(git -C "$REPO" describe --tags --abbrev=0 2>/dev/null || echo "")
PY_VERSION=$($PY -c "from src.infra.version import __version__; print(__version__)" 2>/dev/null || echo "unknown")
echo "[kernel-evolving] Version: ${GIT_VERSION:-$PY_VERSION}"

# --- Model server: only start when task_inference=local ---
# When providers.task_inference is set to a cloud provider (openai/anthropic/copilot/hf),
# the model server is not required for normal chat — skip startup to avoid VRAM conflicts.
# The model server is still started lazily if KERNEL_EVO_FORCE_LOCAL=1 is set,
# or when the socket already exists (reuse for synthesis/drafter roles).
_TASK_PROVIDER=$($PY -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    print(cfg.get('providers', {}).get('task_inference', 'local'))
except Exception:
    print('local')
" 2>/dev/null || echo 'local')
echo "[kernel-evolving] task_inference provider: $_TASK_PROVIDER"

_SOCK_ALIVE=0
if [[ -S "$SOCKET" ]]; then
    if $PY -c "import socket as _s,sys; s=_s.socket(_s.AF_UNIX,_s.SOCK_STREAM); s.settimeout(2); s.connect('$SOCKET'); s.close()" 2>/dev/null; then
        _SOCK_ALIVE=1
        echo "[kernel-evolving] Model server socket already up: $SOCKET (reusing)"
    else
        echo "[kernel-evolving] Stale socket detected — removing"
        rm -f "$SOCKET"
    fi
fi

# Model server startup policy:
#  - Local mode (task_inference=local OR KERNEL_EVO_FORCE_LOCAL=1): start the
#    model server EAGERLY (preloads Nemotron) so local inference is ready.
#  - Cloud mode: start the model server in LAZY mode (--lazy) so the local
#    thought slot (Gemma 4 E2B-it) is available for Think-at-Rest on demand,
#    without reserving VRAM for the cloud-only primary text model.
#  - Set KERNEL_EVO_SKIP_MODEL_SERVER=1 to disable the model server entirely
#    (zero footprint; Think-at-Rest local thoughts will be skipped).
if [[ "${KERNEL_EVO_SKIP_MODEL_SERVER:-0}" == "1" ]]; then
    echo "[kernel-evolving] KERNEL_EVO_SKIP_MODEL_SERVER=1 — skipping model server (voice/STT + local thoughts disabled)"
elif [[ $_SOCK_ALIVE -eq 0 ]]; then
    if [[ "$_TASK_PROVIDER" == "local" || "${KERNEL_EVO_FORCE_LOCAL:-0}" == "1" ]]; then
        echo "[kernel-evolving] Starting model server (eager — preloading Nemotron)..."
        _MODEL_ARGS=""
    else
        echo "[kernel-evolving] Cloud mode — starting model server LAZY (thought slot on demand, no eager VRAM)..."
        _MODEL_ARGS="--lazy"
    fi
    nohup $PY "$REPO/src/core/inference/model_server.py" \
        --config "$CONFIG" \
        $_MODEL_ARGS \
        >> "$MODEL_LOG" 2>&1 &
    MODEL_PID=$!
    echo "[kernel-evolving] Model server PID $MODEL_PID — log: $MODEL_LOG"
    echo "[kernel-evolving] Waiting for model server socket (up to 120s)..."
    _MODEL_READY=0
    for i in $(seq 1 120); do
        if [[ -S "$SOCKET" ]] && $PY -c "import socket as _s; s=_s.socket(_s.AF_UNIX,_s.SOCK_STREAM); s.settimeout(2); s.connect('$SOCKET'); s.close()" 2>/dev/null; then
            _MODEL_READY=1
            echo "[kernel-evolving] Model server ready after ${i}s"
            break
        fi
        if ! kill -0 "$MODEL_PID" 2>/dev/null; then
            echo "[kernel-evolving] ERROR: model server process died during startup"
            echo "---- model server log (last 40 lines) ----"
            tail -n 40 "$MODEL_LOG" || true
            exit 1
        fi
        sleep 1
    done
    if [[ $_MODEL_READY -eq 0 ]]; then
        echo "[kernel-evolving] ERROR: model server socket not ready after 120s"
        tail -n 40 "$MODEL_LOG" || true
        exit 1
    fi
fi

# --- Start API ---
cd "$REPO/src"
: > "$API_LOG"
nohup $PY -m uvicorn api:app --host 0.0.0.0 --port $PORT >> "$API_LOG" 2>&1 &
API_PID=$!
echo "[kernel-evolving] API started (PID $API_PID) — log: $API_LOG"

# Stream API startup logs in real time until health check settles.
tail -f "$API_LOG" &
TAIL_PID=$!

# Poll health endpoint for up to 30s (startup loads 60+ skills + seeds history — takes ~10-15s)
echo "[kernel-evolving] Waiting for API health (up to 30s)..."
_API_READY=0
for i in $(seq 1 30); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        _API_READY=1
        echo "[kernel-evolving] ✅ Ready on :$PORT (after ${i}s)"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[kernel-evolving] ERROR: API process died during startup"
        echo "---- API log (last 80 lines) ----"
        tail -n 80 "$API_LOG" || true
        kill "$TAIL_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
if [[ $_API_READY -eq 0 ]]; then
    echo "[kernel-evolving] ❌ Health check timed out after 30s"
    echo "---- API log (last 80 lines) ----"
    tail -n 80 "$API_LOG" || true
    exit 1
fi
