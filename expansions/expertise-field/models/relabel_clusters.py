#!/usr/bin/env python3
"""
relabel_clusters.py
===================
Reduce + label the field-discovery clusters and render a review-friendly plot.

Motivation (user request):
  - Reduce cluster count: the earlier runs produced overlapping clusters (the
    big engineering 'dev/ops/infra/data' mass kept splitting into several
    weakly-separated blobs). We reduce to a small, defensible set.
  - Label the clusters: assign a human-readable discipline name per cluster
    based on its dominant source_workspace + skill_domains composition.
  - Use a DIFFERENT marker for anomaly/new-field seeds: the current red star
    overlaps cluster points and is hard to review. We switch to a distinct
    hollow marker + color.

Method:
  1. Down-project the 768-d embeddings via PCA -> 50-d (collapse noise).
  2. Select k by silhouette ELBOW (meaningful-improvement knee), not peak,
     on the reduced space, but capped low to avoid the fragmented engineering
     mass over-splitting.
  3. Label each cluster by dominant workspace + top skill_domain tags.
  4. Plot with:
       - clusters colored (tab10), each labelled "NAME (n=...)"
       - anomaly seeds drawn as hollow gold TRIANGLES (marker '^', face none),
         clearly distinct from cluster scatter
  5. Emit a labelled report.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except Exception:  # pragma: no cover
    HAVE_SK = False

HERE = Path(__file__).parent


def _load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8")]


def silhouette_knee(Xr, max_k=15, k_cap=None, min_gain=0.01):
    """Return k chosen by meaningful-improvement knee (not noisy peak)."""
    scores = {}
    k_models = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labs = km.fit_predict(Xr)
        if len(set(labs)) < 2:
            scores[k] = -1.0
            continue
        s = float(silhouette_score(Xr, labs))
        scores[k] = s
        k_models[k] = (km, labs)
    # knee: last k before improvement plateaus / declines
    chosen = 2
    best_s = -1.0
    for k in range(2, max_k + 1):
        if scores[k] > best_s:
            best_s = scores[k]
    prev = scores[2]
    for k in range(3, max_k + 1):
        if scores[k] < 0 or prev < 0:
            break
        gain = scores[k] - prev
        if gain > min_gain:
            chosen = k
        else:
            break  # plateau or decline -> stop (knee)
        prev = scores[k]
    if k_cap is not None:
        chosen = min(chosen, k_cap)
    return chosen, scores, k_models


def label_cluster(rows, idx):
    """Human discipline label from dominant workspace + skill_domains."""
    ws = Counter(rows[i].get("source_workspace", "?") for i in idx)
    dom = Counter()
    for i in idx:
        for t in rows[i].get("skill_domains", []):
            dom[t] += 1
    top_ws = ws.most_common(1)[0][0]
    top_dom = [t for t, _ in dom.most_common(3)]
    # interpret by the dominant tags
    doms = set(top_dom)
    # priority interpretation
    if "finance" in doms or top_ws == "invest":
        name = "Finance / Investment"
    elif top_ws == "marketing" or "media" in doms:
        name = "Media / Content / Branding"
    elif "security" in doms and dom["security"] / max(1, len(idx)) > 0.3:
        name = "Security"
    elif "plant" in doms:
        name = "Plant Science / Grow"
    elif "ops" in doms and dom["ops"] / max(1, len(idx)) > 0.5:
        name = "Systems / DevOps Ops"
    elif "data" in doms and dom["data"] / max(1, len(idx)) > 0.4:
        name = "Data / Analytics"
    else:
        name = "Engineering / Development"
    # workspace modifier for the engineering mass
    if name == "Engineering / Development" and top_ws == "main":
        name = "Software Engineering"
    return name, dict(ws.most_common()), dict(dom.most_common(5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(HERE / "datasets" / "embedded_seed.jsonl"))
    ap.add_argument("--plot", dest="plot",
                    default=str(HERE / "datasets" / "fields_discovery_labeled.png"))
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument("--max-k", type=int, default=15)
    ap.add_argument("--k-cap", type=int, default=6,
                    help="cap cluster count to avoid over-splitting eng mass")
    ap.add_argument("--k", type=int, default=None,
                    help="explicit cluster count (overrides knee)")
    ap.add_argument("--anomaly-q", type=float, default=0.9,
                    help="quantile threshold for anomaly/new-field seeds")
    args = ap.parse_args()

    if not HAVE_SK:
        print("scikit-learn unavailable; aborting.", file=sys.stderr)
        sys.exit(1)

    rows = [r for r in _load_rows(args.inp) if r.get("intent_embedding")]
    if not rows:
        print("No embedded rows.", file=sys.stderr)
        sys.exit(1)
    X = np.array([r["intent_embedding"] for r in rows], dtype=float)
    n, d = X.shape
    print(f"Loaded {n} rows, dim={d}")

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(args.pca_dims, n - 1), random_state=42)
    Xr = pca.fit_transform(Xs)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA -> {Xr.shape[1]} dims (explained var {evr:.3f})")

    # choose k: explicit override, else silhouette knee (capped)
    if args.k is not None:
        K = args.k
        km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(Xr)
        scores = {}
        for kk in range(2, args.max_k + 1):
            kkm = KMeans(n_clusters=kk, init="k-means++", n_init=10, random_state=42)
            ll = kkm.fit_predict(Xr)
            scores[kk] = float(silhouette_score(Xr, ll)) if len(set(ll)) > 1 else -1.0
        print(f"Chosen k={K} (explicit override)")
    else:
        K, scores, k_models = silhouette_knee(Xr, max_k=args.max_k, k_cap=args.k_cap)
        km, labels = k_models[K]
        print(f"Chosen k={K} (knee on reduced space, capped at {args.k_cap})")

    # per-cluster silhouette
    samples = silhouette_samples(Xr, labels)
    cluster_sil = {int(c): float(np.mean(samples[labels == c]))
                   for c in sorted(set(labels))}
    mean_s = float(np.mean(samples))

    # label each cluster
    labels_info = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        name, ws, dom = label_cluster(rows, idx)
        labels_info[int(c)] = {"name": name, "n": int(len(idx)), "ws": ws, "dom": dom,
                               "sil": round(cluster_sil[c], 3)}
        print(f"  c{c}: {name} (n={len(idx)}, sil={cluster_sil[c]:.3f}) "
              f"ws={list(ws.items())[:3]} dom={list(dom.keys())[:4]}")

    # --- post-hoc reduction: merge overlapping engineering clusters ---
    # Identify clusters whose label contains "Engineering" or "Systems" and merge
    # them into a single "Engineering & Systems" discipline to reduce the
    # fragmented dev/ops/infra/data mass the user flagged as overlapping.
    import re as _re
    eng_like = [c for c in sorted(set(labels))
                if "Engineering" in labels_info[int(c)]["name"]
                or "Systems" in labels_info[int(c)]["name"]]
    merged = None
    if len(eng_like) >= 2:
        # collapse eng-like clusters into new singleton label beyond current max
        merged = max(labels_info.keys()) + 1
        merged_name = "Engineering & Systems"
        merged_n = 0
        merged_ws = Counter()
        merged_dom = Counter()
        merged_sil = []
        eng_ids = set(int(c) for c in eng_like)   # cluster IDs to merge
        eng_labels = set()
        for c in eng_like:
            idx = np.where(labels == c)[0]
            eng_labels |= set(int(i) for i in idx)  # point indices (for plotting)
            merged_n += len(idx)
            for i in idx:
                merged_ws[rows[i].get("source_workspace", "?")] += 1
                for t in rows[i].get("skill_domains", []):
                    merged_dom[t] += 1
            merged_sil.append(labels_info[int(c)]["sil"])
        eng_idx = sorted(eng_labels)
        # re-mark labels on the raw clustering ids for plotting
        # mask based on CLUSTER IDs so only the eng clusters relabel to merged
        newlab = np.where(np.isin(labels, list(eng_ids)), merged, labels)
        # recompute merged perpendicular silhouette from samples directly
        label_map = {old: (merged if old in eng_labels else old) for old in labels_info.keys()}
        merged_sil_val = float(np.mean(samples[np.isin(labels, list(eng_ids))]))
        labels_info[merged] = {"name": merged_name, "n": merged_n,
                               "ws": dict(merged_ws), "dom": dict(merged_dom),
                               "sil": round(merged_sil_val, 3)}
        # drop the constituent labels from the display set
        for c in eng_like:
            labels_info.pop(int(c), None)
        labels = newlab
        print(f"\n[MERGE] collapsed engineering clusters {eng_like} -> "
              f"'{merged_name}' (n={merged_n}, sil={merged_sil_val:.3f})")

    # anomaly / new-field seeds (distance-based)
    nn = NearestNeighbors(n_neighbors=11)
    nn.fit(Xr)
    dist, _ = nn.kneighbors(Xr)
    d_self = dist[:, 1]
    thr = float(np.quantile(d_self, args.anomaly_q))
    outlier_idx = sorted(int(i) for i in np.where(d_self > thr)[0])
    print(f"Anomaly/new-field seeds: {len(outlier_idx)} "
          f"(kNN>q{args.anomaly_q:.2f}); threshold={thr:.3f}")

    # 2D projection for plotting
    proj = PCA(n_components=2, random_state=42).fit_transform(Xr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    uniq = sorted(set(labels))
    for cl in uniq:
        idx = np.where(labels == cl)[0]
        name = labels_info[int(cl)]["name"]
        ax.scatter(proj[idx, 0], proj[idx, 1],
                   color=cmap(int(cl) % 10), s=26, alpha=0.7,
                   label=f"{name} (n={len(idx)})")
    # NEW marker for anomaly seeds: hollow gold triangles (distinct from scatter)
    if outlier_idx:
        o = np.array(outlier_idx)
        ax.scatter(proj[o, 0], proj[o, 1],
                   marker="^", s=95, facecolors="none",
                   edgecolors="#F5B301", linewidths=1.6, zorder=5,
                   label=f"new-field seeds ({len(o)})  [hollow triangles]")
    ax.set_title(f"Reduced + Labeled: {K} disciplines (knee, PCA-{Xr.shape[1]}d)\n"
                 f"mean_sil={mean_s:.3f} · {len(outlier_idx)} new-field seeds "
                 f"= hollow triangles")
    ax.legend(fontsize=7, loc="best", ncol=2, framealpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)

    # right: silhouette vs k curve with chosen knee
    ax2 = axes[1]
    ks = sorted(k for k in scores if scores[k] >= 0)
    ax2.plot(ks, [scores[k] for k in ks], "o-", color="tab:blue")
    ax2.axvline(K, color="tab:green", linestyle="--", label=f"chosen k={K} (capped)")
    ax2.set_title(f"Silhouette vs k (reduced space)\n"
                  f"{ {k: round(scores[k],3) for k in ks} }")
    ax2.set_xlabel("k")
    ax2.set_ylabel("mean silhouette")
    ax2.legend()

    plt.suptitle("Expertise-Field Discovery — Reduced + Labeled (review-friendly)")
    plt.tight_layout()
    out = Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130)
    print(f"\nPlot -> {str(out)}")

    # report
    L = ["# Expertise-Field Discovery — Reduced + Labeled", ""]
    L.append(f"_Real run on {n} embedded ActivityEvent rows "
             f"(embeddinggemma-300m, {d}-d), @ {args.inp.split('/')[-1]}_")
    L.append("")
    L.append("## Method")
    L.append("- Down-projected embeddings to PCA **50-d** (explained var %.3f)" % evr)
    L.append(f"- **k = {K}** chosen by silhouette *knee* (meaningful-improvement), "
             f"capped at {args.k_cap} to avoid over-splitting the engineering mass.")
    L.append("- Each cluster labeled by dominant source_workspace + skill_domains.")
    L.append("")
    L.append("## Resulting disciplines (labeled)")
    L.append("")
    L.append("| cluster | label | n | sil | dominant ws | top domains |")
    L.append("|---|---|---|---|---|---|")
    for c in sorted(set(labels)):
        li = labels_info[int(c)]
        L.append(f"| c{c} | {li['name']} | {li['n']} | {li['sil']} | "
                 f"{', '.join(k for k,_ in list(li['ws'].items())[:3])} | "
                 f"{', '.join(li['dom'].keys())} |")
    L.append("")
    L.append(f"**Mean silhouette:** {mean_s:.3f}  (>~0.25 would be 'solid'; "
             "0.1–0.2 = weak-but-real structure on heterogeneous real logs)")
    L.append("")
    L.append("## How overlaps were reduced")
    L.append("- Earlier runs split the big `dev/ops/infra/data` engineering mass "
             "into 3–4 overlapping blobs. Reducing to a capped k merges those into "
             "cohesive, named disciplines rather than fragmented clusters.")
    L.append(f"- Anomaly/new-field seeds: **{len(outlier_idx)}** points beyond "
             f"kNN q{args.anomaly_q:.2f}; shown as **hollow gold triangles** (not "
             "red stars) for clear review separation.")
    L.append("")
    out_r = Path(args.plot.replace(".png", "_report.md"))
    out_r.write_text("\n".join(L), encoding="utf-8")
    print("Report ->", str(out_r))
    print("\n".join(L))


if __name__ == "__main__":
    main()