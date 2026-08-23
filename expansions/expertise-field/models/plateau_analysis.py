#!/usr/bin/env python3
"""
Plateau analysis: the honest field count is where the silhouette curve STOPS
climbing meaningfully.  On the audit run, silhouette went 0.077(2)->0.144(6)
then hovered 0.143-0.152 through k=13 — i.e. real discipline structure exists
in the 2..6 range and everything past ~6 is flat noise.  Pick k = the last k in
the rising run (the elbow), not the flat tail.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

HERE = Path(__file__).parent


def load_rows(path):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    return [r for r in rows if r.get("intent_embedding")]


def elbow_k(scores, min_rise=0.01):
    """Last k in the monotonic rising run that gains >= min_rise absolute."""
    chosen = 2
    prev = scores[2]
    for k in range(3, len(scores) + 1):
        if scores[k] < 0:
            break
        if scores[k] - prev >= min_rise:
            chosen = k
        prev = scores[k]
    return chosen


def main():
    rows = load_rows(HERE / "datasets" / "embedded_seed.jsonl")
    X = np.array([r["intent_embedding"] for r in rows], dtype=float)
    Xs = StandardScaler().fit_transform(X)
    Xr = PCA(n_components=min(50, X.shape[0] - 1), random_state=42).fit_transform(Xs)

    scores = {}
    for k in range(2, 16):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labs = km.fit_predict(Xr)
        scores[k] = float(silhouette_score(Xr, labs)) if len(set(labs)) > 1 else -1.0

    chosen = elbow_k(scores, min_rise=0.01)
    print("silhouette curve:")
    for k in sorted(scores):
        mark = " <-- elbow" if k == chosen else ""
        print(f"  k={k}: {scores[k]:.3f}{mark}")
    print(f"\nChosen elbow k = {chosen}")

    km = KMeans(n_clusters=chosen, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(Xr)
    samples = silhouette_samples(Xr, labels)
    mean_s = float(np.mean(samples))
    cluster_sil = {int(c): float(np.mean(samples[np.where(labels == c)[0]]))
                   for c in sorted(set(labels))}
    print(f"mean_sil={mean_s:.3f}")
    print("per-cluster silhouette:", {k: round(v, 2) for k, v in cluster_sil.items()})

    # Show the dominant topic/source for each cluster so we can interpret
    comp = {}
    for cl in sorted(set(labels)):
        idx = np.where(labels == cl)[0]
        toks = Counter()
        for i in idx:
            for t in rows[i].get("skill_domains", []):
                toks[t] += 1
            # also mine tools
            for t in rows[i].get("tools_used", []):
                toks["tool:" + t] += 1
            # filename date bucket
            src = rows[i].get("source_file", "")
            pref = src[:7]
            toks["month:" + pref] += 1 if pref else 0
        comp[cl] = toks.most_common(6)
        print(f"\ncluster {cl} (n={len(idx)}):")
        for t, c in comp[cl]:
            print(f"    {t}: {c}")

    # strong/weak clusters
    strong = [c for c, v in cluster_sil.items() if v >= 0.25]
    weak = [c for c, v in cluster_sil.items() if v < 0.1]
    print(f"\nSTRONG cohesive clusters (sil>=0.25): {strong}")
    print(f"WEAK clusters (sil<0.10): {weak}")


if __name__ == "__main__":
    main()