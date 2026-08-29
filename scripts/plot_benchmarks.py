#!/usr/bin/env python3
"""
plot_benchmarks.py — generate a comparison chart from configs/*/results.json.

Reads each config's results.json and produces:
  1. configs/benchmark_avg_latency.png  — bar chart of average latency
  2. configs/benchmark_per_test.png     — grouped bar chart per test

Requires matplotlib. Run from the repo root:
    python3 scripts/plot_benchmarks.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO, "configs")

# Ordered list of (label_prefix, display_name)
CONFIG_ORDER = [
    ("01-baseline-cloud", "01 Cloud\n(Nemotron)"),
    ("02-local-gemma-primary", "02 Gemma\n(E2B 2.3B)"),
    ("03-local-qwen-primary", "03 Qwen\n(3.5-0.8B)"),
    ("04-local-omni-primary", "04 Omni\n(2.5-3B)"),
    ("05-local-janus-primary", "05 Janus\n(Pro-7B)"),
]

TESTS = ["text", "tool_calc", "tool_followup", "memory_context"]


def load_results(label_prefix: str):
    """Load results.json for a config label prefix."""
    for d in os.listdir(CONFIGS_DIR):
        if d.startswith(label_prefix) and os.path.isdir(os.path.join(CONFIGS_DIR, d)):
            rp = os.path.join(CONFIGS_DIR, d, "results.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    return json.load(f)
    # Fallback: <label_prefix>.json style (flat file)
    fp = os.path.join(CONFIGS_DIR, label_prefix + ".json")
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return None


def main():
    labels = []
    avgs = []
    per_test = {t: [] for t in TESTS}

    for prefix, display in CONFIG_ORDER:
        data = load_results(prefix)
        if data is None:
            print(f"[skip] no results for {prefix}")
            labels.append(display)
            avgs.append(0)
            for t in TESTS:
                per_test[t].append(0)
            continue
        by_label = {r["label"]: r["latency_s"] for r in data.get("results", [])}
        avg = data.get("avg_latency_s") or np.mean(list(by_label.values()))
        labels.append(display)
        avgs.append(avg)
        for t in TESTS:
            per_test[t].append(by_label.get(t, 0))

    x = np.arange(len(labels))

    # ── Chart 1: average latency ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(x, avgs, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average latency (s)")
    ax.set_title("kernel-evolving — average agent-path latency by config")
    for b, v in zip(bars, avgs):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(avgs) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(CONFIGS_DIR, "benchmark_avg_latency.png")
    fig.savefig(out1, dpi=120)
    print(f"saved {out1}")

    # ── Chart 2: per-test latency (log scale for the huge Gemma outlier) ──
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    width = 0.18
    for i, t in enumerate(TESTS):
        ax2.bar(x + (i - 1.5) * width, per_test[t], width * 0.9,
                label=t, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Latency (s, log scale)")
    ax2.set_title("kernel-evolving — per-test latency by config (log scale)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(CONFIGS_DIR, "benchmark_per_test.png")
    fig2.savefig(out2, dpi=120)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
