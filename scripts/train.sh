#!/usr/bin/env bash
# train.sh — stop GPU services, run fine-tune, restart everything
# Usage: bash train.sh [--drafter] [extra finetune args...]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_DIR/scripts/finetune_from_trajectories.py"
PYTHON="${PYTHON:-$HOME/.miniconda/envs/opedDev_py311/bin/python}"
DATASET="${DATASET:-$HOME/.kernel-evolving/workspace/artifacts/trajectories/sft_ready_final.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.kernel-evolving/workspace/artifacts/finetune}"
LOG="/tmp/finetune_train.log"
# bitsandbytes needs CUDA 13 nvJitLink (Python 3.13 env)
export LD_LIBRARY_PATH="$HOME/.miniconda/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

GPU_THRESHOLD=85  # °C — pause above this
GPU_RESUME=75     # °C — resume below this

# ── helpers ─────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
gpu_temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null || echo 0; }

# ── 1. Stop GPU services ─────────────────────────────────────────────────────
log "=== Stopping GPU services ==="

# Base Kernel model_server (port 8769 uvicorn)
KERNEL_API_PID=$(pgrep -f "uvicorn api:app.*8769" || true)
KERNEL_MODEL_PID=$(pgrep -f "repositories/kernel/src/model_server.py\|kernel.*model_server" | grep -v "kernel-evolving" || true)
# kernel-evolving model_server + api
EVO_API_PID=$(pgrep -f "uvicorn api:app.*8779" || true)
EVO_MODEL_PID=$(pgrep -f "repositories/kernel-evolving/src/model_server.py" || true)

stop_pid() {
  local pid="$1" name="$2"
  if [ -n "$pid" ]; then
    log "  Stopping $name (PID $pid)"
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  else
    log "  $name not running, skipping"
  fi
}

stop_pid "$KERNEL_API_PID"   "kernel api (8769)"
stop_pid "$KERNEL_MODEL_PID" "kernel model_server"
stop_pid "$EVO_API_PID"      "kernel-evolving api (8779)"
stop_pid "$EVO_MODEL_PID"    "kernel-evolving model_server"

# Also kill Fantasia (SD-turbo) — holds ~2GB VRAM
FANTASIA_PID=$(pgrep -f "open-fantasia\|sd-turbo\|stabilityai" || true)
stop_pid "$FANTASIA_PID" "Fantasia SD-turbo"

# Also stop via systemd to prevent auto-restart
for svc in fantasia.service olly-voice.service olly-voice-openclaw.service; do
  systemctl --user stop "$svc" 2>/dev/null && log "  systemd: stopped $svc" || true
done

# Give GPU VRAM a moment to free
sleep 8

# Verify VRAM is actually clear before proceeding
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' MiB')
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB')
log "GPU VRAM — used: ${VRAM_USED} MiB, free: ${VRAM_FREE} MiB"

if [ "${VRAM_USED}" -gt 2000 ] 2>/dev/null; then
  log "⚠️  VRAM still occupied (${VRAM_USED} MiB) — force-killing remaining GPU processes"
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v pid | xargs -r kill -9 2>/dev/null || true
  sleep 5
  VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' MiB')
  log "GPU VRAM free after force-kill: ${VRAM_FREE} MiB"
fi
log ""

# ── 2. Temperature watcher (background) ─────────────────────────────────────
TRAIN_PID_FILE="/tmp/finetune_train.pid"
WATCHER_ACTIVE=true

gpu_watcher() {
  while $WATCHER_ACTIVE; do
    TEMP=$(gpu_temp)
    if [ "$TEMP" -ge "$GPU_THRESHOLD" ] 2>/dev/null; then
      TRAIN_PID=$(cat "$TRAIN_PID_FILE" 2>/dev/null || true)
      if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "⚠️  GPU ${TEMP}°C ≥ ${GPU_THRESHOLD}°C — pausing training (SIGSTOP)"
        kill -SIGSTOP "$TRAIN_PID"
        # Wait until cool enough
        while true; do
          sleep 15
          TEMP=$(gpu_temp)
          log "   Cooling... ${TEMP}°C (resume at ${GPU_RESUME}°C)"
          if [ "$TEMP" -le "$GPU_RESUME" ] 2>/dev/null; then
            log "✅ GPU ${TEMP}°C ≤ ${GPU_RESUME}°C — resuming training (SIGCONT)"
            kill -SIGCONT "$TRAIN_PID"
            break
          fi
        done
      fi
    fi
    sleep 30
  done
}

gpu_watcher &
WATCHER_PID=$!
log "GPU watcher started (PID $WATCHER_PID, pause ≥${GPU_THRESHOLD}°C, resume ≤${GPU_RESUME}°C)"

# ── 3. Run fine-tune ─────────────────────────────────────────────────────────
log "=== Starting fine-tune ==="
log "Dataset: $DATASET"
log "Output:  $OUTPUT_DIR"
log ""

set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_LAUNCH_BLOCKING=1 \
  "$PYTHON" "$SCRIPT" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --drafter \
    --local \
    "$@" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$TRAIN_PID_FILE"
log "Training PID: $TRAIN_PID"

# Stream log output
wait "$TRAIN_PID"
TRAIN_EXIT=$?
set -e

# Stop watcher
WATCHER_ACTIVE=false
kill "$WATCHER_PID" 2>/dev/null || true
rm -f "$TRAIN_PID_FILE"

log ""
if [ $TRAIN_EXIT -eq 0 ]; then
  log "✅ Fine-tune completed successfully"
else
  log "❌ Fine-tune exited with code $TRAIN_EXIT"
fi

# ── 4. Restart GPU services ──────────────────────────────────────────────────
log ""
log "=== Restarting GPU services ==="

# kernel-evolving
EVO_DIR="${EVO_DIR:-$HOME/.openclaw/workspace/repositories/kernel-evolving}"
log "  Restarting kernel-evolving..."
cd "$EVO_DIR"
nohup bash start.sh >> /tmp/kernel_evolving_start.log 2>&1 &
log "  kernel-evolving restarting (check /tmp/kernel_evolving_start.log)"

# Base kernel — NOT auto-restarted. Fabio must restart manually.
# Reason: base Kernel shares GPU VRAM and auto-restart after training
# was happening without explicit approval. Restart it yourself when ready:
#   cd repositories/kernel && bash start.sh
log "  ℹ️  Base kernel NOT auto-restarted (requires explicit approval — start manually when ready)"

log ""
log "=== Done. Training exit code: $TRAIN_EXIT ==="
exit $TRAIN_EXIT
