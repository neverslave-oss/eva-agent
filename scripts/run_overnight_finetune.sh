#!/usr/bin/env bash
# run_overnight_finetune.sh — orchestrate overnight local fine-tuning for eva's
# three custom-model targets, sequentially, with per-target logs + a combined
# report. GPU services are stopped at the start and kernel-evolving is restarted
# at the end (base Kernel is left down — restart manually, as with train.sh).
#
# Targets:
#   1. eva:E2B-it  -> Gemma 4 E2B-it      (sft_ready_final.jsonl, messages render)
#   2. eva:1b      -> Qwen3.5-0.8B        (sft_ready_final.jsonl, messages render)
#   3. Nemotron FC -> Nemotron-Diffusion-3B (sft_nemotron_fc.jsonl, pre-rendered text)
#
# Usage:
#   bash scripts/run_overnight_finetune.sh [--dry-run] [--only <target>]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/.miniconda/envs/opedDev_py311/bin/python"
ARTS="$HOME/.kernel-evolving/workspace/artifacts"
DATA_DIR="$ARTS/trajectories"
OUT_DIR="$ARTS/finetune"
LOG_DIR="/tmp/eva_overnight"
mkdir -p "$LOG_DIR"

# bitsandbytes needs CUDA 13 nvJitLink (Python 3.13 env) — mirror of train.sh
export LD_LIBRARY_PATH="$HOME/.miniconda/lib/python3.13/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

# Model paths
GEMMA="$HOME/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"
QWEN="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
NEMOTRON="/mnt/e/models/huggingface/hub/models--nvidia--Nemotron-Labs-Diffusion-3B"

# Targets: name|model_path|dataset|output_subdir|extra_args
TARGETS=(
  "eva-e2b|$GEMMA|$DATA_DIR/sft_ready_final.jsonl|eva_e2b|--batch-size 1 --local --epochs 3"
  "eva-1b|$QWEN|$DATA_DIR/sft_ready_final.jsonl|eva_1b|--batch-size 1 --local --epochs 3"
  "nemotron-fc|$NEMOTRON|$DATA_DIR/sft_nemotron_fc.jsonl|nemotron_fc|--batch-size 1 --local --epochs 3"
)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/overnight.log"; }

ONLY="${2:-}"
DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY=1; fi
if [ "${2:-}" = "--dry-run" ]; then DRY=1; ONLY="${3:-}"; fi

log "=== eva overnight fine-tune ==="
log "Python: $PYTHON"
log "GPU (before): $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1) free"
log "DRY_RUN=${DRY:-0} ONLY=${ONLY:-all}"

if [ -z "${DRY:-}" ]; then
  log "Stopping GPU services (mirror train.sh)..."
  for pat in "uvicorn api:app.*8779" "repositories/kernel-evolving/src/model_server.py" "uvicorn api:app.*8769" "open-fantasia\|sd-turbo"; do
    pkill -f "$pat" 2>/dev/null && log "  stopped: $pat" || true
  done
  sleep 8
  log "GPU after stop: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1) free"
fi

results=()
for entry in "${TARGETS[@]}"; do
  IFS='|' read -r name model dataset subdir extra <<< "$entry"
  if [ -n "$ONLY" ] && [ "$name" != "$ONLY" ]; then
    log "  skipping $name (only=$ONLY)"
    continue
  fi
  if [ ! -f "$dataset" ]; then
    log "❌ $name: dataset missing $dataset"
    results+=("$name: MISSING_DATASET")
    continue
  fi
  out="$OUT_DIR/$subdir"
  log "── Target $name ──"
  log "  model: $model"
  log "  data:  $dataset"
  log "  out:   $out"
  if [ -n "${DRY:-}" ]; then
    log "  [dry-run] would run: $PYTHON scripts/finetune_from_trajectories.py $extra --dataset $dataset --model $model --output-dir $out"
    results+=("$name: DRY_OK")
    continue
  fi
  set +e
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 \
    "$PYTHON" "$REPO_DIR/scripts/finetune_from_trajectories.py" \
      --dataset "$dataset" --model "$model" --output-dir "$out" $extra \
      > "$LOG_DIR/finetune_$name.log" 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    log "✅ $name fine-tune OK (adapter: $out)"
    results+=("$name: OK")
  else
    log "❌ $name failed (rc=$rc) — see $LOG_DIR/finetune_$name.log"
    results+=("$name: FAIL")
  fi
done

if [ -z "${DRY:-}" ]; then
  log "Restarting kernel-evolving (base Kernel NOT auto-restarted)..."
  cd "$REPO_DIR" && nohup bash start.sh >> /tmp/kernel_evolving_start.log 2>&1 &
  log "kernel-evolving restarting — check /tmp/kernel_evolving_start.log"
fi

log "=== OVERNIGHT RESULTS ==="
for r in "${results[@]}"; do log "  $r"; done
log "Done."

exit 0
