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
LOG="/tmp/kernel_evo_shadow.log"

# ── Python resolution (mirrors start.sh) ────────────────────────────────────
# The gate daemon runs under a systemd env where bare `python3` = /usr/bin/python3
# (no yaml module) — so never hardcode it. Prefer explicit KERNEL_EVO_PYTHON,
# then repo venv, then miniconda, then whatever python3 resolves to.
if [[ -n "${KERNEL_EVO_PYTHON:-}" ]]; then
  PYTHON="$KERNEL_EVO_PYTHON"
elif [[ -x "$REPO_DIR/.venv/bin/python3" ]]; then
  PYTHON="$REPO_DIR/.venv/bin/python3"
elif [[ -x "$HOME/.miniconda/bin/python3" ]]; then
  PYTHON="$HOME/.miniconda/bin/python3"
else
  PYTHON="$(command -v python3)"
fi

# ── Read config values ───────────────────────────────────────────────────────
_cfg() {
  "$PYTHON" -c "
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

# Allow override via arg — accept either a dir containing adapter_config.json/adapter weights,
# or a direct path to the safetensors file. When given a dir, point ADAPTER_PATH at the
# expected weights file inside it so adapter_config.json resolution lands on the right dir.
ADAPTER_ARG="${1:-}"
if [ -z "$ADAPTER_ARG" ]; then
  ADAPTER_PATH="$ADAPTER_OUTPUT_DIR/adapter_model.safetensors"
  ADAPTER_DIR="$ADAPTER_OUTPUT_DIR"
elif [ -d "$ADAPTER_ARG" ]; then
  ADAPTER_DIR="$ADAPTER_ARG"
  ADAPTER_PATH="$ADAPTER_ARG/adapter_model.safetensors"
elif [ -f "$ADAPTER_ARG" ]; then
  ADAPTER_PATH="$ADAPTER_ARG"
  ADAPTER_DIR="$(dirname "$ADAPTER_ARG")"
else
  ADAPTER_PATH="$ADAPTER_ARG"
  ADAPTER_DIR="$(dirname "$ADAPTER_ARG")"
fi

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

# ── 1b. Resolve the adapter's OWN base model dynamically ─────────────────────
# LoRA adapters are trained against a specific base model (Gemma, Qwen, Nemotron,
# etc). Loading them onto whatever model config.yaml currently points at (e.g. the
# live Nemotron inference model) silently mismatches and crashes. Read the base
# model straight from the adapter's own adapter_config.json — the ground truth
# peft wrote at training time — instead of assuming config.yaml's model.
ADAPTER_CFG_JSON="$ADAPTER_DIR/adapter_config.json"
if [ -f "$ADAPTER_CFG_JSON" ]; then
  ADAPTER_BASE_MODEL=$("$PYTHON" -c "
import json
try:
    d = json.load(open('$ADAPTER_CFG_JSON'))
    print(d.get('base_model_name_or_path') or '')
except Exception:
    print('')
")
else
  ADAPTER_BASE_MODEL=""
fi

if [ -n "$ADAPTER_BASE_MODEL" ]; then
  log "✅ Adapter's own base model (from adapter_config.json): $ADAPTER_BASE_MODEL"
else
  log "⚠️  No adapter_config.json/base_model_name_or_path found — falling back to config.yaml's model (may mismatch)"
fi

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

# bitsandbytes 4-bit quant needs libnvJitLink.so.13 (CUDA 13 runtime), which the
# nvidia-nvjitlink package ships to site-packages/nvidia/cu13/lib/. It is NOT on the
# default loader path, so without this bitsandbytes fails with:
#   🚨 CUDA SETUP ERROR: Missing dependency: libnvJitLink.so.13 🚨
# Prepend it so the shadow server can quantize its base model in 4-bit like production.
NVIDIA_CU13_LIB="$HOME/.miniconda/lib/python3.13/site-packages/nvidia/cu13/lib"
if [ -d "$NVIDIA_CU13_LIB" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  log "LD_LIBRARY_PATH prepended: $NVIDIA_CU13_LIB (libnvJitLink.so.13)"
else
  log "⚠️  nvidia/cu13 lib dir not found at $NVIDIA_CU13_LIB — 4-bit quant may fail"
fi

log "Starting shadow model_server (socket: $SHADOW_SOCKET, adapter: $ADAPTER_DIR)..."
cd "$REPO_DIR"
SHADOW_CMD=("$PYTHON" src/core/inference/model_server.py \
  --config "$CONFIG" \
  --socket "$SHADOW_SOCKET" \
  --adapter "$ADAPTER_DIR")
# Pass the adapter's own base model so the shadow loads the right base (Gemma/Qwen)
# instead of whatever config.yaml points at (Nemotron). Falls back gracefully if empty.
if [ -n "$ADAPTER_BASE_MODEL" ]; then
  SHADOW_CMD+=(--model "$ADAPTER_BASE_MODEL")
  log "  Base model override: $ADAPTER_BASE_MODEL"
fi
nohup "${SHADOW_CMD[@]}" >> "$LOG" 2>&1 &
SHADOW_MODEL_PID=$!
echo "$SHADOW_MODEL_PID" > "$SHADOW_MODEL_PID_FILE"
log "Shadow model_server PID: $SHADOW_MODEL_PID"

sleep 5

log "Starting shadow api.py on port $SHADOW_PORT..."
# api module lives in src/ — cd so uvicorn can import it (matches eval_sequential.sh's start_api)
(cd "$REPO_DIR/src" && MODEL_SERVER_SOCKET="$SHADOW_SOCKET" \
nohup "$PYTHON" -m uvicorn api:app \
  --host 127.0.0.1 \
  --port "$SHADOW_PORT" \
  --log-level warning \
  >> "$LOG" 2>&1 &)
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
FINETUNED_RESULTS=$("$PYTHON" "$REPO_DIR/scripts/run_sim_eval.py" --port "$SHADOW_PORT" --label "finetuned" 2>&1) || FINETUNED_EXIT=$?
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
  BASELINE_RESULTS=$("$PYTHON" "$REPO_DIR/scripts/run_sim_eval.py" --port "$BASELINE_LIVE_PORT" --label "baseline" 2>&1) || BASELINE_EXIT=$?
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
