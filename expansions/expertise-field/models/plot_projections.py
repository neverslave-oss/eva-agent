#!/usr/bin/env python3
"""
plot_projections.py
===================
Render the REAL discovery pipeline output as a PNG plot image.

Reuses cluster_v0 on the real embedded seed data, projects the
intent_embedding vectors to 2D (PCA; UMAP if available), and saves a
matplotlib scatter plot colored by discovered cluster, highlighting the
anomaly-layer new-field seeds.

Output: models/datasets/fields_discovery_plot.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE.parent / "src" / "expertise_field"
sys.path.insert(0, str(SRC))
from cluster_v0 import select_k_by_silhouette, fit_clusters, detect_new_field_seeds  # noqa


def _load_rows(path):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    return [r for r in rows if r.get("intent_embedding")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(HERE / "datasets" / "embedded_seed.jsonl"))
    ap.add_argument("--out", default=str(HERE / "datasets" / "fields_discovery_plot.png"))
    ap.add_argument("--max-k", type=int, default=12)
    ap.add_argument("--umap", action="store_true", help="use UMAP if available")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_rows(args.inp)
    if not rows:
        print("No embedded rows found.", file=sys.stderr)
        sys.exit(1)

    X = np.array([r["intent_embedding"] for r in rows], dtype=float)
    n, d = X.shape
    print(f"Loaded {n} embedded rows, dim={d}")

    # same pipeline as visualize.py
    res = select_k_by_silhouette(X, k_min=2, k_max=min(args.max_k, n - 1))
    fit = fit_clusters(X, k=res.best_k)
    labels = fit.labels
    print(f"best k={res.best_k} (sil={res.best_score:.3f})")

    mask, _ = detect_new_field_seeds(X)
    outlier_idx = set(np.where(mask)[0].tolist())
    print(f"anomaly outliers: {len(outlier_idx)}")

    # 2D projection
    proj_name = "PCA"
    if args.umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            proj = reducer.fit_transform(X)
            proj_name = "UMAP"
        except Exception:
            proj_name = "PCA"
            from sklearn.decomposition import PCA
            proj = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=42).fit_transform(X)

    # ---- plot ----
    unique = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 8))

    for c in unique:
        idx = np.where(labels == c)[0]
        label_name = f"cluster {int(c)} (n={len(idx)})"
        ax.scatter(proj[idx, 0], proj[idx, 1], s=22, alpha=0.7,
                   color=cmap(int(c) % 20), label=label_name)

    # highlight anomaly new-field seeds with a star marker
    if outlier_idx:
        ids = np.array(sorted(outlier_idx))
        ax.scatter(proj[ids, 0], proj[ids, 1], s=90, facecolors="none",
                   edgecolors="red", linewidths=1.2, marker="*",
                   label=f"new-field seeds ({len(ids)})")

    ax.set_title(f"Expertise-Field Discovery — {proj_name} projection\n"
                 f"{n} activity events · k={res.best_k} (silhouette elbow={res.best_score:.3f})")
    ax.set_xlabel(f"{proj_name} 1")
    ax.set_ylabel(f"{proj_name} 2")
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.6)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")
    print(f"  absolute: {out.resolve()}")


if __name__ == "__main__":
    main()