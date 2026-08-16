#!/usr/bin/env bash
# eval_adapter.sh — Full evaluation pipeline for fine-tuned LoRA adapter
#
# Steps:
#   1. Check adapter exists
#   2. Stop Fantasia + voice services
#   3. Start shadow instance (model_server + api) on shadow port/socket
#   4. Wait for shadow ready
#   5. Run sim eval against shadow (finetuned)
#   6. Run sim eval against live instance (baseline)
#   7. Compare pass rates
#   8. Stop shadow
#   9. Restart Fantasia + voice if they were running
#
# Usage: bash scripts/eval_adapter.sh [adapter_path]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_DIR/config.yaml"
PYTHON="python3"
LOG="/tmp/kernel_evo_shadow.log"

# ── Read config values ───────────────────────────────────────────────────────
_cfg() {
  python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
keys = '$1'.split('.')
val = cfg
for k in keys:
    val = val.get(k, None)
    if val is None:
        break
print(val if val is not None else '')
"
}

ADAPTER_OUTPUT_DIR=$(_cfg "evolution.finetune_gate.adapter_output_dir")
ADAPTER_OUTPUT_DIR="${ADAPTER_OUTPUT_DIR/#\~/$HOME}"
SHADOW_PORT=$(_cfg "evolution.finetune_gate.shadow_port")
SHADOW_SOCKET=$(_cfg "evolution.finetune_gate.shadow_socket")

# Fallback defaults
ADAPTER_OUTPUT_DIR="${ADAPTER_OUTPUT_DIR:-$HOME/.kernel-evolving/workspace/artifacts/finetune}"
SHADOW_PORT="${SHADOW_PORT:-8780}"
SHADOW_SOCKET="${SHADOW_SOCKET:-/tmp/kernel_evo_shadow.sock}"

# Allow override via arg
ADAPTER_PATH="${1:-$ADAPTER_OUTPUT_DIR/adapter_model.safetensors}"
ADAPTER_DIR="$(dirname "$ADAPTER_PATH")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== eval_adapter.sh ==="
log "Config:       $CONFIG"
log "Adapter dir:  $ADAPTER_DIR"
log "Shadow port:  $SHADOW_PORT"
log "Shadow sock:  $SHADOW_SOCKET"

# ── 1. Check adapter exists ───────────────────────────────────────────────────
if [ ! -f "$ADAPTER_PATH" ] && [ ! -d "$ADAPTER_DIR" ]; then
  log "❌ No adapter found at $ADAPTER_PATH — aborting"
  exit 2
fi
log "✅ Adapter found: $ADAPTER_PATH"

# ── 2. Track which GPU services are running ───────────────────────────────────
FANTASIA_WAS_RUNNING=false
VOICE_WAS_RUNNING=false

if systemctl --user is-active --quiet fantasia.service 2>/dev/null; then
  FANTASIA_WAS_RUNNING=true
fi
if systemctl --user is-active --quiet olly-voice.service 2>/dev/null || \
   systemctl --user is-active --quiet olly-voice-openclaw.service 2>/dev/null; then
  VOICE_WAS_RUNNING=true
fi

log "Fantasia was running: $FANTASIA_WAS_RUNNING"
log "Voice was running: $VOICE_WAS_RUNNING"

# Stop them to free VRAM for shadow instance
for svc in fantasia.service olly-voice.service olly-voice-openclaw.service; do
  systemctl --user stop "$svc" 2>/dev/null && log "  Stopped $svc" || true
done
sleep 3

# ── 3. Start shadow instance ──────────────────────────────────────────────────
# Remove stale shadow socket
rm -f "$SHADOW_SOCKET"

SHADOW_API_PID_FILE="/tmp/kernel_evo_shadow_api.pid"
SHADOW_MODEL_PID_FILE="/tmp/kernel_evo_shadow_model.pid"

log "Starting shadow model_server (socket: $SHADOW_SOCKET, adapter: $ADAPTER_DIR)..."
cd "$REPO_DIR"
nohup "$PYTHON" src/model_server.py \
  --config "$CONFIG" \
  --socket "$SHADOW_SOCKET" \
  --adapter "$ADAPTER_DIR" \
  >> "$LOG" 2>&1 &
SHADOW_MODEL_PID=$!
echo "$SHADOW_MODEL_PID" > "$SHADOW_MODEL_PID_FILE"
log "Shadow model_server PID: $SHADOW_MODEL_PID"

sleep 5

log "Starting shadow api.py on port $SHADOW_PORT..."
MODEL_SERVER_SOCKET="$SHADOW_SOCKET" \
nohup "$PYTHON" -m uvicorn api:app \
  --host 127.0.0.1 \
  --port "$SHADOW_PORT" \
  --log-level warning \
  >> "$LOG" 2>&1 &
SHADOW_API_PID=$!
echo "$SHADOW_API_PID" > "$SHADOW_API_PID_FILE"
log "Shadow api PID: $SHADOW_API_PID"

# ── 4. Wait for shadow to be ready ────────────────────────────────────────────
log "Waiting for shadow on http://localhost:$SHADOW_PORT/health ..."
READY=false
for i in $(seq 1 60); do
  if curl -sf "http://localhost:$SHADOW_PORT/health" > /dev/null 2>&1; then
    READY=true
    log "✅ Shadow ready (${i}s)"
    break
  fi
  sleep 5
done

if [ "$READY" = "false" ]; then
  log "❌ Shadow did not become ready in 300s — aborting"
  kill "$SHADOW_MODEL_PID" "$SHADOW_API_PID" 2>/dev/null || true
  exit 3
fi

# ── 5. Run finetuned eval ─────────────────────────────────────────────────────
log "Running finetuned eval on port $SHADOW_PORT..."
FINETUNED_EXIT=0
FINETUNED_RESULTS=$(python3 "$REPO_DIR/scripts/run_sim_eval.py" --port "$SHADOW_PORT" --label "finetuned" 2>&1) || FINETUNED_EXIT=$?
echo "$FINETUNED_RESULTS" | tee -a "$LOG"

FINETUNED_PASS=$(echo "$FINETUNED_RESULTS" | grep -oP '\d+(?=/\d+ tasks passed)' | tail -1 || echo "0")
FINETUNED_TOTAL=$(echo "$FINETUNED_RESULTS" | grep -oP '(?<=\d/)\d+(?= tasks passed)' | tail -1 || echo "8")

# ── 6. Run baseline eval ──────────────────────────────────────────────────────
BASELINE_LIVE_PORT=8779
BASELINE_EXIT=0
BASELINE_PASS=0
BASELINE_TOTAL=8

if curl -sf "http://localhost:$BASELINE_LIVE_PORT/health" > /dev/null 2>&1; then
  log "Running baseline eval on port $BASELINE_LIVE_PORT..."
  BASELINE_RESULTS=$(python3 "$REPO_DIR/scripts/run_sim_eval.py" --port "$BASELINE_LIVE_PORT" --label "baseline" 2>&1) || BASELINE_EXIT=$?
  echo "$BASELINE_RESULTS" | tee -a "$LOG"
  BASELINE_PASS=$(echo "$BASELINE_RESULTS" | grep -oP '\d+(?=/\d+ tasks passed)' | tail -1 || echo "0")
  BASELINE_TOTAL=$(echo "$BASELINE_RESULTS" | grep -oP '(?<=\d/)\d+(?= tasks passed)' | tail -1 || echo "8")
else
  log "⚠️  Live instance not running on port $BASELINE_LIVE_PORT — skipping baseline eval"
  BASELINE_PASS="N/A"
fi

# ── 7. Side-by-side comparison ────────────────────────────────────────────────
log ""
log "==============================="
log "  COMPARISON RESULTS"
log "==============================="
log "  Baseline (port $BASELINE_LIVE_PORT): $BASELINE_PASS/$BASELINE_TOTAL"
log "  Finetuned (port $SHADOW_PORT):       $FINETUNED_PASS/$FINETUNED_TOTAL"
log "==============================="

# ── 8. Stop shadow ────────────────────────────────────────────────────────────
log "Stopping shadow instance..."
kill "$SHADOW_API_PID" 2>/dev/null || true
kill "$SHADOW_MODEL_PID" 2>/dev/null || true
sleep 3
kill -9 "$SHADOW_API_PID" 2>/dev/null || true
kill -9 "$SHADOW_MODEL_PID" 2>/dev/null || true
rm -f "$SHADOW_SOCKET" "$SHADOW_API_PID_FILE" "$SHADOW_MODEL_PID_FILE"
log "Shadow stopped."

# ── 9. Restart Fantasia + voice if they were running ─────────────────────────
if [ "$FANTASIA_WAS_RUNNING" = "true" ]; then
  systemctl --user start fantasia.service 2>/dev/null && log "  Restarted fantasia.service" || log "  ⚠️  Could not restart fantasia.service"
fi
if [ "$VOICE_WAS_RUNNING" = "true" ]; then
  systemctl --user start olly-voice.service 2>/dev/null && log "  Restarted olly-voice.service" || \
  systemctl --user start olly-voice-openclaw.service 2>/dev/null && log "  Restarted olly-voice-openclaw.service" || \
  log "  ⚠️  Could not restart voice service"
fi

log "=== eval_adapter.sh done ==="
log "  Finetuned: $FINETUNED_PASS/$FINETUNED_TOTAL | Baseline: $BASELINE_PASS/$BASELINE_TOTAL"

# Export for callers
echo "FINETUNED_PASS=$FINETUNED_PASS"
echo "FINETUNED_TOTAL=$FINETUNED_TOTAL"
echo "BASELINE_PASS=$BASELINE_PASS"
echo "BASELINE_TOTAL=$BASELINE_TOTAL"

exit "$FINETUNED_EXIT"
