"""
demo_silhouette_elbow.py — Sanity demo for the v0 clustering layer.

Generates synthetic activity-embedding blobs (the unit-normal style of real
intent embeddings), runs the silhouette-elbow k-selection, and prints the
scores so we can SEE the peak recover the true number of "fields".

Run: python src/expertise_field/demo_silhouette_elbow.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from expertise_field.cluster_v0 import select_k_by_silhouette, run_discovery_pipeline


def make_blobs(n_per=30, centers=4, dim=128, seed=21):
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for c in range(centers):
        mu = rng.normal(0, 1, dim) * 2.5 * (c + 1)
        pts = rng.normal(mu, 1.0, size=(n_per, dim))
        X.append(pts)
        labels.extend([c] * n_per)
    return np.vstack(X), np.array(labels)


def main():
    true_k = 4
    X, true_labels = make_blobs(centers=true_k)

    print(f"Dataset: {X.shape[0]} activity events, {X.shape[1]}-dim embeddings")
    print(f"TRUE number of natural fields injected: {true_k}\n")

    res = select_k_by_silhouette(X, k_min=2, k_max=9)
    print("Silhouette sweep (the 'peak' is our elbow):")
    for k, s in sorted(res.scores.items()):
        marker = "  <<< PEAK" if k == res.best_k else ""
        bar = "#" * int(s * 40)
        print(f"  k={k}: {s:.3f} {bar}{marker}")

    print(f"\n-> silhouette elbow picks k={res.best_k} "
          f"(mean silhouette {res.best_score:.3f})")
    print(f"-> recovered {'CORRECTLY' if res.best_k == true_k else 'expected-' + str(true_k)} "
          f"the {true_k} injected fields")

    print("\nFull v0 discovery pipeline over the same data:")
    out = run_discovery_pipeline(X)
    print(f"  estimated_fields      : {out['estimated_fields']}")
    print(f"  mean_silhouette       : {out['mean_silhouette']:.3f}")
    print(f"  per-cluster silhouettes: { {k: round(v,3) for k,v in out['cluster_silhouette'].items()} }")
    print(f"  weak_clusters         : {out['weak_clusters']}")
    print(f"  split_candidates      : {out['split_candidates']}")
    print(f"  new_field_seeds (anom): {out['n_new_field_seeds']}")


if __name__ == "__main__":
    main()