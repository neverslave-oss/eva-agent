"""
cluster_v0.py — Clustering layer v0 for expertise_field.

Implements the silhouette-elbow method for data-driven field discovery
(SPEC Part 3 / Part 8). This is the "how many fields?" decision, made by
the data (not hardcoded), plus the per-cluster silhouette that powers the
split/drift trigger.

Two-layer design (cheat-sheet grounded):
    Layer 1 (discovery)  : KMeans over `intent_embedding` vectors, k chosen
                           by the silhouette elbow (peak, not a bend).
    Layer 2 (new-field)  : anomaly detection flags fallback/unassigned
                           activity as new-field seed. Uses a distance-based
                           outlier score (robust at high embedding dims)
                           with an optional EllipticEnvelope cross-check.
    (HDBSCAN remains the v1 candidate for arbitrary shapes + streaming.)

Design rules applied:
    * NO hardcoded cluster count — k is estimated from the data.
    * Feature scaling first (StandardScaler) — distance methods require it.
    * k selected by peak of mean silhouette, not an inertia "bend".
    * Per-cluster mean silhouette == split-trigger signal (a cluster whose
      cohesion degrades below threshold / has points nearer a neighbour
      cluster is a split candidate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

__all__ = [
    "select_k_by_silhouette",
    "fit_clusters",
    "detect_new_field_seeds",
    "run_discovery_pipeline",
    "SilhouetteResult",
    "ClusterFit",
]


@dataclass
class SilhouetteResult:
    """Outcome of a silhouette sweep across a range of k."""

    scores: dict[int, float]          # k -> mean silhouette width
    best_k: int                       # k maximizing mean silhouette
    best_score: float                 # mean silhouette at best_k
    per_point_best: np.ndarray        # silhouette_samples at best_k


@dataclass
class ClusterFit:
    """A fitted clustering run with field-discovery metadata."""

    kmeans: KMeans
    k: int
    labels: np.ndarray
    mean_silhouette: float
    per_point_silhouette: np.ndarray
    cluster_silhouette: dict[int, float]     # cluster_id -> mean silhouette
    weak_clusters: list[int]                 # cluster_ids below cohesion threshold
    split_candidates: list[int]              # clusters whose points lean to a neighbour
    fallback_seeds: list[int] = field(default_factory=list)  # global indices of anomaly rows


def _require_2d(X: np.ndarray, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError(f"{name} is empty; cannot cluster an empty dataset")
    return X


def select_k_by_silhouette(
    X: np.ndarray,
    k_min: int = 2,
    k_max: Optional[int] = None,
    scale: bool = True,
    n_init: int = 10,
    random_state: int = 42,
) -> SilhouetteResult:
    """
    Sweep k in [k_min, k_max] and pick the k maximizing mean silhouette
    width (the silhouette *peak*, not an inertia bend).

    Parameters
    ----------
    X : 2D array of dense feature vectors (e.g. intent_embedding).
    k_min : smallest k to test (default 2; 1-cluster = "no fields yet").
    k_max : largest k to test. Defaults to min(sqrt(n), n-1) — a reasonable
            cap so we don't over-search beyond distinct regions.
    scale : StandardScaler normalize X before clustering (recommended).
    n_init / random_state : passed to KMeans (k-means++ init).
    """
    X = _require_2d(X)
    n = X.shape[0]
    if scale:
        X = StandardScaler().fit_transform(X)
    if k_max is None:
        k_max = max(k_min, min(int(np.sqrt(n)) + 1, n - 1))
    k_max = min(k_max, n - 1)
    if k_min > k_max:
        raise ValueError(f"k_min ({k_min}) > available k_max ({k_max}); not enough rows")

    scores: dict[int, float] = {}
    best_k, best_score = k_min, -1.0
    best_samples: Optional[np.ndarray] = None
    best_model: Optional[KMeans] = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            scores[k] = -1.0  # degenerate: one cluster won; not meaningful
            continue
        s = float(silhouette_score(X, labels))
        scores[k] = s
        if s > best_score:
            best_score, best_k = s, k
            best_samples = silhouette_samples(X, labels)
            best_model = km

    if best_model is None or best_samples is None:
        raise ValueError("No valid k found; dataset may have no meaningful clusters")

    return SilhouetteResult(
        scores=scores,
        best_k=best_k,
        best_score=best_score,
        per_point_best=best_samples,
    )


def fit_clusters(
    X: np.ndarray,
    k: Optional[int] = None,
    scale: bool = True,
    split_silhouette_threshold: float = 0.25,
    random_state: int = 42,
    n_init: int = 10,
) -> ClusterFit:
    """
    Fit KMeans at a given k (or auto-select k via silhouette peak) and compute
    cluster-level cohesion metadata that drives the split/drift trigger.
    """
    X = _require_2d(X)
    Xw = StandardScaler().fit_transform(X) if scale else X

    if k is None:
        res = select_k_by_silhouette(Xw, scale=False)
        k = res.best_k
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xw)
        samples = res.per_point_best
    else:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Xw)
        samples = silhouette_samples(Xw, labels)

    mean_s = float(np.mean(samples))

    cluster_sil = {}
    weak: list[int] = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        cluster_sil[int(c)] = float(np.mean(samples[idx]))
        if cluster_sil[int(c)] < split_silhouette_threshold:
            weak.append(int(c))

    # Split candidates: a cluster whose points have poor placement. K-Means
    # assigns every point to its NEAREST center, so comparing center distances
    # can never signal a split. The correct signal is per-point SILHOUETTE:
    # a negative silhouette means the point is nearer to a neighbouring cluster
    # (in the cohesion sense) than to its own -> poor assignment. A cluster
    # with >= `split_neg_ratio` of its points below zero is a split candidate.
    split_neg_ratio = 0.25
    split_candidates: list[int] = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        neg_frac = float(np.mean(samples[idx] < 0))
        if neg_frac >= split_neg_ratio:
            split_candidates.append(c)

    return ClusterFit(
        kmeans=km,
        k=k,
        labels=labels,
        mean_silhouette=mean_s,
        per_point_silhouette=samples,
        cluster_silhouette=cluster_sil,
        weak_clusters=weak,
        split_candidates=split_candidates,
    )


def detect_new_field_seeds(
    X: np.ndarray,
    contamination: float = 0.1,
    scale: bool = True,
    random_state: int = 42,
    method: str = "distance",
) -> tuple[np.ndarray, Optional[object]]:
    """
    Layer 2 (anomaly): flag points statistically unlike the rest. These are
    the fallback / unassigned activity rows -> seeds of NEW fields.

    method:
        "distance" (default) : standardized k-NN distance to nearest neighbour.
            Robust and cheap in high-dimensional embedding spaces. A point is
            an outlier if its distance to its kth-nearest-inlier exceeds the
            (1-contamination) quantile of that distance distribution.
        "elliptic_envelope"  : Gaussian-covariance EllipticEnvelope (from the
            cheat sheet). Note: fails to "full rank" on high-dim, few-sample
            data; use mainly for low-dim cross-checks.

    Returns (boolean outlier mask, fitted model-or-None).
    """
    X = _require_2d(X, name="X (anomaly)")
    Xw = StandardScaler().fit_transform(X) if scale else X

    if method == "elliptic_envelope":
        from sklearn.covariance import EllipticEnvelope

        env = EllipticEnvelope(contamination=contamination, random_state=random_state)
        pred = env.fit_predict(Xw)
        return (pred == -1), env

    # distance-based: k-NN to the (1:contamination) quantile threshold
    n = Xw.shape[0]
    k_nn = max(1, int(contamination * n))
    nn = NearestNeighbors(n_neighbors=min(k_nn + 1, n))
    nn.fit(Xw)
    dist, _ = nn.kneighbors(Xw)
    d_self = dist[:, 1] if dist.shape[1] > 1 else dist[:, 0]
    # contamination quantile of the nn-distance distribution
    thr = np.quantile(d_self, 1 - contamination)
    mask = d_self > thr
    return mask, nn


def run_discovery_pipeline(
    X: np.ndarray,
    fallback_mask: Optional[np.ndarray] = None,
    scale: bool = True,
    anomaly_method: str = "distance",
) -> dict:
    """
    End-to-end v0 discovery over a batch of activity embeddings.

    - Layer 1: KMeans + silhouette elbow -> discovered fields (clusters).
    - Layer 2: anomaly detection -> new-field seed rows.
    """
    fit = fit_clusters(X, k=None, scale=scale)
    outliers, model = detect_new_field_seeds(X, scale=scale, method=anomaly_method)
    if fallback_mask is not None:
        fallback_mask = np.asarray(fallback_mask, dtype=bool)
        seeds = np.where(outliers & fallback_mask)[0].tolist()
    else:
        seeds = np.where(outliers)[0].tolist()

    return {
        "n_points": len(X),
        "estimated_fields": fit.k,
        "cluster_silhouette": fit.cluster_silhouette,
        "mean_silhouette": fit.mean_silhouette,
        "weak_clusters": fit.weak_clusters,
        "split_candidates": fit.split_candidates,
        "new_field_seeds": seeds,
        "n_new_field_seeds": len(seeds),
        "labels": fit.labels,
        "outlier_mask": outliers,
    }