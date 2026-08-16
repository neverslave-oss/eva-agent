#!/usr/bin/env bash
# kernel-evolving Docker entrypoint
set -euo pipefail

CONFIG="/app/config.yaml"
SOCKET="/tmp/kernel_evolving_model.sock"

# Detect task_inference provider from config
TASK_PROVIDER=$(python3 -c "
import yaml, sys
try:
    c=yaml.safe_load(open('$CONFIG'))
    print(c.get('providers',{}).get('task_inference','local'))
except:
    print('local')
" 2>/dev/null || echo "local")

echo "[entrypoint] task_inference=${TASK_PROVIDER}"

if [[ "$TASK_PROVIDER" == "local" || "${KERNEL_EVO_FORCE_LOCAL:-0}" == "1" ]]; then
    echo "[entrypoint] Starting model server (lazy)..."
    python3 /app/src/model_server.py --config "$CONFIG" --lazy &
    MODEL_PID=$!
    echo "[entrypoint] Model server PID $MODEL_PID — waiting for socket..."
    for i in $(seq 1 60); do
        if [[ -S "$SOCKET" ]]; then
            echo "[entrypoint] Model server ready after ${i}s"
            break
        fi
        sleep 1
    done
    if [[ ! -S "$SOCKET" ]]; then
        echo "[entrypoint] WARNING: model server socket not ready — continuing anyway (lazy load)"
    fi
else
    echo "[entrypoint] Cloud provider ($TASK_PROVIDER) — skipping model server"
fi

cd /app/src
exec python3 -m uvicorn api:app --host 0.0.0.0 --port 8779
