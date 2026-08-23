#!/usr/bin/env python3
"""
Revised audit plot — honest field discovery.
Uses the k=6 ELBOW (not the noisy 12/13 peak). Silhouette rises 0.077->0.144
from k=2->6 then goes flat, so 6 is where real (modest) discipline structure
stops improving. Flags weak clusters and anomaly new-field seeds.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).parent
rows = [r for r in (json.loads(l) for l in (HERE / "datasets" / "embedded_seed.jsonl").open(encoding="utf-8")) if r.get("intent_embedding")]
X = np.array([r["intent_embedding"] for r in rows], dtype=float)
Xs = StandardScaler().fit_transform(X)
Xr = PCA(n_components=min(50, X.shape[0] - 1), random_state=42).fit_transform(Xs)
X2 = PCA(n_components=2, random_state=42).fit_transform(Xr)

K = 6
km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(Xr)
samples = silhouette_samples(Xr, labels)
cluster_sil = {int(c): float(np.mean(samples[np.where(labels == c)[0]])) for c in sorted(set(labels))}
mean_s = float(np.mean(samples))

# anomaly seeds (distance-based)
nn = NearestNeighbors(n_neighbors=11)
nn.fit(Xr)
dist, _ = nn.kneighbors(Xr)
d_self = dist[:, 1]
thr = np.quantile(d_self, 0.9)
outlier_idx = [int(i) for i in np.where(d_self > thr)[0]]

# dominant tags per cluster for the legend
dom = {}
for cl in sorted(set(labels)):
    idx = np.where(labels == cl)[0]
    toks = Counter()
    for i in idx:
        for t in rows[i].get("skill_domains", []):
            toks[t] += 1
    dom[cl] = ", ".join(t for t, _ in toks.most_common(3)) or "untagged"

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
cmap = plt.get_cmap("tab10")
ax = axes[0]
for cl in sorted(set(labels)):
    idx = np.where(labels == cl)[0]
    ax.scatter(X2[idx, 0], X2[idx, 1], c=[cmap(cl % 10)], s=30, alpha=0.7,
               label=f"c{cl} (n={len(idx)}): {dom[cl]}")
for i in outlier_idx[:60]:
    ax.scatter(X2[i, 0], X2[i, 1], c="red", marker="*", s=190, zorder=5, edgecolors="black")
ax.set_title(f"Revision: {K} fields @ silhouette elbow (not peak)\nmean_sil={mean_s:.3f}; {len(outlier_idx)} red *=new-field seeds")
ax.legend(fontsize=6, loc="best")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# right: silhouette curve with elbow marked
ax2 = axes[1]
scores = {}
for k in range(2, 16):
    kkm = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labs = kkm.fit_predict(Xr)
    scores[k] = float(silhouette_score(Xr, labs)) if len(set(labs)) > 1 else -1.0
ks = sorted(k for k in scores if scores[k] >= 0)
ax2.plot(ks, [scores[k] for k in ks], "o-", color="tab:blue", label="mean silhouette")
ax2.axvline(K, color="red", linestyle="--", label=f"elbow k={K}")
ax2.set_title("Silhouette vs k (reduced 50-d)\nelbow at 6 = peak of meaningful rise; tail is noise")
ax2.set_xlabel("k")
ax2.set_ylabel("mean silhouette")
ax2.legend()

plt.suptitle("Expertise-Field Discovery — Revised Audit (honest; elbow not peak)")
plt.tight_layout()
out = HERE / "datasets" / "fields_discovery_revised.png"
plt.savefig(out, dpi=130)
print("saved", out)
print("mean_sil", round(mean_s, 3))
print("cluster_sil", {k: round(v, 2) for k, v in cluster_sil.items()})
print("outliers", len(outlier_idx))