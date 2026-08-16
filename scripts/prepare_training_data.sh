#!/usr/bin/env bash
# prepare_training_data.sh
# =========================
# Orchestrates the full data scaling pipeline:
#   1. Stitch PASS trajectories → stitched_trajectories.jsonl
#   2. Generate synthetic skill tasks → synthetic_skill_tasks.jsonl
#   3. Combine all sources + deduplicate → training_combined.jsonl
#
# Usage:
#   bash scripts/prepare_training_data.sh [--force]
#
# Options:
#   --force   Re-run stitching and synthesis even if outputs already exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TRAJ_DIR="${TRAJ_DIR:-$HOME/.kernel-evolving/workspace/artifacts/trajectories}"

SFT_READY="$TRAJ_DIR/sft_ready_final.jsonl"
STITCHED="$TRAJ_DIR/stitched_trajectories.jsonl"
SYNTHETIC="$TRAJ_DIR/synthetic_skill_tasks.jsonl"
COMBINED="$TRAJ_DIR/training_combined.jsonl"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Kernel-Evolving Training Data Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Stitch trajectories ───────────────────────────────────────────────
STITCHED_TODAY=0
if [[ -f "$STITCHED" ]]; then
    # Check if file was modified today
    FILE_DATE=$(date -r "$STITCHED" +%Y-%m-%d 2>/dev/null || stat -c %y "$STITCHED" 2>/dev/null | cut -d' ' -f1)
    TODAY=$(date +%Y-%m-%d)
    if [[ "$FILE_DATE" == "$TODAY" ]]; then
        STITCHED_TODAY=1
    fi
fi

if [[ $FORCE -eq 1 || $STITCHED_TODAY -eq 0 ]]; then
    echo ""
    echo "▶ Step 1: Stitching trajectories..."
    python3 "$SCRIPT_DIR/stitch_trajectories.py" \
        --input "$SFT_READY" \
        --output "$STITCHED" \
        --max-pairs 200 \
        --max-triples 100
else
    echo ""
    echo "✓ Step 1: Stitched trajectories already up-to-date (skip; use --force to regenerate)"
fi

STITCHED_COUNT=$(wc -l < "$STITCHED" | tr -d ' ')
echo "  → $STITCHED_COUNT stitched trajectories"

# ── Step 2: Generate synthetic tasks ─────────────────────────────────────────
SYNTHETIC_TODAY=0
if [[ -f "$SYNTHETIC" ]]; then
    FILE_DATE=$(date -r "$SYNTHETIC" +%Y-%m-%d 2>/dev/null || stat -c %y "$SYNTHETIC" 2>/dev/null | cut -d' ' -f1)
    TODAY=$(date +%Y-%m-%d)
    if [[ "$FILE_DATE" == "$TODAY" ]]; then
        SYNTHETIC_TODAY=1
    fi
fi

if [[ $FORCE -eq 1 || $SYNTHETIC_TODAY -eq 0 ]]; then
    echo ""
    echo "▶ Step 2: Generating synthetic skill tasks..."
    python3 "$SCRIPT_DIR/generate_skill_tasks.py" \
        --output "$SYNTHETIC"
else
    echo ""
    echo "✓ Step 2: Synthetic tasks already up-to-date (skip; use --force to regenerate)"
fi

SYNTHETIC_COUNT=$(wc -l < "$SYNTHETIC" | tr -d ' ')
echo "  → $SYNTHETIC_COUNT synthetic trajectories"

# ── Step 3: Combine + deduplicate ─────────────────────────────────────────────
echo ""
echo "▶ Step 3: Combining and deduplicating..."

ORIGINAL_COUNT=$(wc -l < "$SFT_READY" | tr -d ' ')

python3 - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

TRAJ_DIR = Path(os.environ.get("TRAJ_DIR", str(Path.home() / ".kernel-evolving/workspace/artifacts/trajectories")))
sources = [
    TRAJ_DIR / "sft_ready_final.jsonl",
    TRAJ_DIR / "stitched_trajectories.jsonl",
    TRAJ_DIR / "synthetic_skill_tasks.jsonl",
]
output_path = TRAJ_DIR / "training_combined.jsonl"

seen_ids = set()
all_trajectories = []
source_counts = {}

for src in sources:
    if not src.exists():
        print(f"  ⚠️  Missing: {src}", file=sys.stderr)
        continue
    count = 0
    dupes = 0
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            traj_id = traj.get("id", "")
            if traj_id in seen_ids:
                dupes += 1
                continue
            seen_ids.add(traj_id)
            all_trajectories.append(traj)
            count += 1
    source_counts[src.name] = count
    if dupes:
        print(f"  ⚠️  {src.name}: {dupes} duplicate(s) skipped")

with open(output_path, "w") as f:
    for traj in all_trajectories:
        f.write(json.dumps(traj, ensure_ascii=False) + "\n")

orig = source_counts.get("sft_ready_final.jsonl", 0)
stitched = source_counts.get("stitched_trajectories.jsonl", 0)
synthetic = source_counts.get("synthetic_skill_tasks.jsonl", 0)
total = len(all_trajectories)

print(f"\n  Original:   {orig:>6} trajectories (sft_ready_final.jsonl)")
print(f"  Stitched:   {stitched:>6} trajectories (stitched_trajectories.jsonl)")
print(f"  Synthetic:  {synthetic:>6} trajectories (synthetic_skill_tasks.jsonl)")
print(f"  ─────────────────────────────────────────")
print(f"  Total:      {total:>6} trajectories → training_combined.jsonl")
print(f"\n  Output: {output_path}")
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Pipeline complete → $COMBINED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
