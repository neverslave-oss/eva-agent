#!/usr/bin/env bash
# eval_sequential.sh — Sequential single-GPU adapter evaluation
#
# Runs baseline (no adapter) then v2 adapter back-to-back.
# Only one model loaded at a time — no shadow replica.
#
# Usage: bash scripts/eval_sequential.sh [adapter_dir]
#   adapter_dir defaults to ~/.kernel-evolving/workspace/artifacts/finetune_v2

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8779
SOCKET="/tmp/kernel_evolving_model.sock"
API_LOG="/tmp/kernel_evo_seq_api.log"
MODEL_LOG="/tmp/kernel_evo_seq_model.log"
LOG="/tmp/kernel_evo_eval_sequential.log"
ADAPTER_DIR="${1:-$HOME/.kernel-evolving/workspace/artifacts/finetune_v2}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

: > "$LOG"
log "=== eval_sequential.sh ==="
log "Repo:        $REPO_DIR"
log "Port:        $PORT"
log "Adapter dir: $ADAPTER_DIR"

# ── helpers ──────────────────────────────────────────────────────────────────

stop_all_gpu() {
  log "Stopping all GPU consumers..."
  # kernel-evolving API + model server
  fuser -k ${PORT}/tcp 2>/dev/null || true
  pkill -f "model_server.py" 2>/dev/null || true
  # Fantasia
  systemctl --user stop fantasia.service 2>/dev/null || true
  # olly-voice
  systemctl --user stop olly-voice.service 2>/dev/null || true
  systemctl --user stop olly-voice-openclaw.service 2>/dev/null || true
  # stale shadow
  pkill -f "kernel_evo_shadow" 2>/dev/null || true
  fuser -k 8780/tcp 2>/dev/null || true
  rm -f "$SOCKET" /tmp/kernel_evo_shadow.sock
  sleep 4
  log "GPU consumers stopped."
}

start_model_server() {
  local adapter_flag="${1:-}"
  log "Starting model server${adapter_flag:+ with adapter $adapter_flag}..."
  rm -f "$SOCKET"
  if [[ -n "$adapter_flag" ]]; then
    nohup python3 "$REPO_DIR/src/model_server.py" \
      --config "$REPO_DIR/config.yaml" \
      --lazy \
      --adapter "$adapter_flag" \
      >> "$MODEL_LOG" 2>&1 &
  else
    nohup python3 "$REPO_DIR/src/model_server.py" \
      --config "$REPO_DIR/config.yaml" \
      --lazy \
      >> "$MODEL_LOG" 2>&1 &
  fi
  MODEL_PID=$!
  log "Model server PID: $MODEL_PID"

  # Wait for socket
  for i in $(seq 1 60); do
    if [[ -S "$SOCKET" ]] && python3 -c "import socket as _s; s=_s.socket(_s.AF_UNIX,_s.SOCK_STREAM); s.settimeout(2); s.connect('$SOCKET'); s.close()" 2>/dev/null; then
      log "Model server socket ready (${i}s)"
      return 0
    fi
    sleep 1
  done
  log "❌ Model server socket not ready in 60s"
  tail -20 "$MODEL_LOG" | tee -a "$LOG"
  exit 1
}

start_api() {
  log "Starting API on port $PORT..."
  : > "$API_LOG"
  cd "$REPO_DIR/src"
  nohup python3 -m uvicorn api:app --host 0.0.0.0 --port $PORT >> "$API_LOG" 2>&1 &
  API_PID=$!
  log "API PID: $API_PID"

  for i in $(seq 1 30); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
      log "API ready (${i}s)"
      return 0
    fi
    sleep 2
  done
  log "❌ API not ready in 60s"
  tail -20 "$API_LOG" | tee -a "$LOG"
  exit 1
}

run_eval() {
  local label="$1"
  log "Running eval: $label ..."
  python3 "$REPO_DIR/scripts/run_sim_eval.py" --port $PORT --label "$label" 2>&1 | tee -a "$LOG" || true
}

# ── 1. Kill everything GPU ────────────────────────────────────────────────────
stop_all_gpu

# ── 2. Baseline (no adapter) ──────────────────────────────────────────────────
log ""
log "=== ROUND 1: BASELINE (no adapter) ==="
start_model_server
start_api
run_eval "baseline"
BASELINE_RESULT=$(grep -oP '\d+/\d+ tasks passed' "$LOG" | tail -1 || echo "?/8")
log "Baseline result: $BASELINE_RESULT"

# ── 3. Stop everything again ──────────────────────────────────────────────────
stop_all_gpu

# ── 4. v2 adapter ─────────────────────────────────────────────────────────────
log ""
log "=== ROUND 2: v2 ADAPTER ($ADAPTER_DIR) ==="
start_model_server "$ADAPTER_DIR"
start_api
run_eval "finetuned_v2"
V2_RESULT=$(grep -oP '\d+/\d+ tasks passed' "$LOG" | tail -1 || echo "?/8")
log "v2 result: $V2_RESULT"

# ── 5. Stop everything ────────────────────────────────────────────────────────
stop_all_gpu

# ── 6. Summary ────────────────────────────────────────────────────────────────
log ""
log "==============================="
log "  EVAL RESULTS (Sim 14)"
log "==============================="
log "  Baseline (no adapter): $BASELINE_RESULT"
log "  v2 adapter:            $V2_RESULT"
log "==============================="
log "Full log: $LOG"
log "Results:  $HOME/.kernel-evolving/workspace/artifacts/eval_results/"

echo ""
echo "BASELINE=$BASELINE_RESULT"
echo "V2=$V2_RESULT"
