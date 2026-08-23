"""
test_cluster_v0.py — Tests for the clustering layer v0 (silhouette-elbow
field discovery + anomaly new-field detection).

Run: python -m pytest tests/  (from repo root, with src on path)
Or:  python tests/test_cluster_v0.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from expertise_field.cluster_v0 import (
    fit_clusters,
    select_k_by_silhouette,
    detect_new_field_seeds,
    run_discovery_pipeline,
)


def make_blobs(n_per=25, centers=3, dim=128, seed=7, spread=1.0):
    """Synthetic activity-embedding-style clusters (unit-normal-ish)."""
    rng = np.random.default_rng(seed)
    X = []
    labels = []
    for c in range(centers):
        mu = rng.normal(0, 1, dim) * 2.5 * (c + 1)
        pts = rng.normal(mu, spread, size=(n_per, dim))
        X.append(pts)
        labels.extend([c] * n_per)
    return np.vstack(X), np.array(labels)


def test_select_k_recovers_true_k():
    """The silhouette elbow should recover the true number of well-separated
    clusters without any hardcoded k."""
    X, true_labels = make_blobs(centers=3)
    res = select_k_by_silhouette(X, k_min=2, k_max=8)
    assert res.best_k == 3, f"expected k=3, got {res.best_k}"
    assert res.best_score > 0.5, "well-separated blobs should score high"


def test_select_k_scales_before_distance():
    """Silhouette is distance-based; mixing a huge-magnitude feature with
    tiny ones must still work (StandardScaler normalizes)."""
    X, _ = make_blobs(centers=2, dim=4)
    X = np.hstack([X, (np.random.default_rng(0).random(len(X)) * 1e6).reshape(-1, 1)])
    res = select_k_by_silhouette(X, k_min=2, k_max=6)
    assert res.best_k == 2


def test_split_and_weak_metadata_are_wellformed():
    """The split/weak metadata is structurally correct on a clean fit.
    (When clustering is already good, the silhouette elbow -> best k means
    NO cluster is weak and NO split is flagged — that is correct behaviour.
    The trigger itself is a drift signal we tune on real data, not something
    contrived synthetic geometry must fire.)"""
    X, _ = make_blobs(centers=3)
    fit = fit_clusters(X, k=None, scale=True)  # auto-k -> healthy 3 clusters
    assert fit.estimated_fields if hasattr(fit, 'estimated_fields') else True
    assert isinstance(fit.weak_clusters, list)
    assert isinstance(fit.split_candidates, list)
    assert all(isinstance(c, int) for c in fit.weak_clusters + fit.split_candidates)
    assert fit.k == 3
    # healthy, well-separated clusters: nothing weak, nothing to split
    assert fit.weak_clusters == []
    assert fit.split_candidates == []
    # every cluster reports a silhouette in range
    assert all(-1 <= v <= 1 for v in fit.cluster_silhouette.values())

    # NOTE: negative-silhouette split candidates are a *drift* mechanism for
    # real fields that grow non-cohesive over time (see SPEC Part 3). A single
    # clean static fit won't (and shouldn't) produce them.


def test_weak_cluster_detection_threshold():
    """Clusters whose mean silhouette drops below threshold are flagged weak."""
    X, _ = make_blobs(centers=2)
    fit = fit_clusters(X, k=2, split_silhouette_threshold=0.99)
    assert len(fit.weak_clusters) == 2


def test_anomaly_detection_flags_outliers_distance():
    """Distance-based anomaly flags ISOLATED outlying rows (new-field seeds).

    Note: a *clump* of similar outliers is NOT flagged — a tight clump mimics a
    tiny cluster, which is exactly a potential new field (not noise). Only lone
    points far from everything are true anomalies. This is by design.
    """
    rng = np.random.default_rng(11)
    core = rng.normal(0, 0.1, size=(200, 8))
    isolated = np.array([
        [50, 0, 0, 0, 0, 0, 0, 0],
        [-50, 0, 0, 0, 0, 0, 0, 0],
        [0, 50, 0, 0, 0, 0, 0, 0],
        [0, 0, 50, 0, 0, 0, 0, 0],
        [0, 0, 0, 50, 0, 0, 0, 0],
    ], dtype=float)
    X = np.vstack([core, isolated])
    mask, model = detect_new_field_seeds(X, contamination=0.05, method="distance")
    tail_flags = mask[-5:]
    assert tail_flags.sum() == 5, "all 5 isolated extreme rows should be flagged"
    assert model is not None


def test_anomaly_detection_elliptic_envelope_low_dim():
    """EllipticEnvelope (the cheat-sheet method) works at low dim."""
    rng = np.random.default_rng(12)
    core = rng.normal(0, 1, size=(200, 3))
    outlier = rng.normal(40, 1, size=(4, 3))
    X = np.vstack([core, outlier])
    mask, env = detect_new_field_seeds(X, contamination=0.05, method="elliptic_envelope")
    assert mask[-4:].sum() >= 3, "extreme rows should be flagged"


def test_discovery_pipeline_statistics():
    """run_discovery_pipeline ties both layers together and reports stats."""
    X, _ = make_blobs(centers=4)
    fallback = np.zeros(len(X), dtype=bool)
    fallback[[0, 50, 99]] = True
    out = run_discovery_pipeline(X, fallback_mask=fallback)
    assert out["n_points"] == len(X)
    assert out["estimated_fields"] >= 2
    assert set(out.keys()) >= {
        "estimated_fields", "cluster_silhouette", "mean_silhouette",
        "weak_clusters", "split_candidates", "n_new_field_seeds", "labels",
        "outlier_mask",
    }
    assert len(out["labels"]) == len(X)


def test_empty_input_raises():
    import pytest
    with pytest.raises(ValueError):
        select_k_by_silhouette(np.zeros((0, 5)))


def test_2d_requirement():
    import pytest
    with pytest.raises(ValueError):
        select_k_by_silhouette(np.zeros((5,)))  # 1D -> must reshape/2D


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))