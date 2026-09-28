#!/usr/bin/env python3
"""
prepare_sft_datasets.py — Generate SFT datasets for all eva custom-model targets.

Reads the clean trajectory export and produces a per-target SFT JSONL:
  - eva:E2B-it  -> Gemma  (rendered by finetune_from_trajectories from messages)
  - eva:1b      -> Qwen  (rendered by finetune_from_trajectories from messages)
  - Nemotron FC -> pre-rendered <tool_call> text (map_trajectories_to_nemotron)

Usage:
  python3 scripts/prepare_sft_datasets.py --export EXPORT.jsonl --outdir DIR
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Shared sft_ready (messages form) for Gemma/Qwen render path
    sft_ready = os.path.join(args.outdir, "sft_ready_final.jsonl")
    with open(args.export, encoding="utf-8") as f, open(sft_ready, "w", encoding="utf-8") as out:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} records -> {sft_ready} (shared messages for Gemma+Qwen)")

if __name__ == "__main__":
    main()
