# models/ — Clustering layer (v0)

Experimental, isolated clustering models for the `expertise_field` module.
THIS CODE IS ISOLATED (kernel source is read-only); it lives in the module
repo until integration, then perches in the kernel-evolving workspace.

## Layout

```
models/
├── data_schema.py      # ActivityEvent dataclass + field definitions
├── schema.json         # machine-readable activity schema
├── collector.py        # collector stub: log activity -> JSONL dataset
├── datasets/           # collected training rows (activity_YYYYMMDD.jsonl)
└── README.md           # this file
```

The clustering *estimators* live in `src/expertise_field/cluster_v0.py`
(not under `models/`) so they're importable as part of the package. The
`models/` dir is the *data + schema + model-artifact* home.

## Design intent

We discover expertise fields by **unsupervised learning over activity
history** — we do NOT declare fields (no hardcoded count; consistency with
the "fields are discovered, not locked to a fixed number" decision).

- **Facts** (cheat sheet): ML-Specialization-Cheat-Sheet.md, Course 3
  (Unsupervised). Tools: K-Means (k via silhouette elbow), anomaly detection,
  `StandardScaler` required for distance methods.
- **Layer 1 — clustering**: K-Means over `intent_embedding` vectors; k chosen
  by the **silhouette peak** (not an inertia bend) → emergent fields.
- **Layer 2 — anomaly**: distance-based k-NN outlier score → flags
  fallback/unassigned activity as **new-field seeds**.
- **Upgrade path**: HDBSCAN (v1) for arbitrary cluster shapes + streaming,
  no fixed k, built-in noise→new-field detection.

### Key behavioural notes (learned from tests)

- **An isolated outlier** = genuine anomaly → new-field seed.
- **A tight clump of similar outliers is NOT an anomaly** — it mimics a tiny
  cluster, i.e. a *potential new field*, not noise. Correct by design.
- **Negative-silhouette split candidates are a DRIFT mechanism**: a field
  grows non-cohesive over time. A single clean static K-Means fit won't
  produce them (K-Means optimizes placement) — the silhouette elbow always
  picks the healthiest k. Splits are detected by re-running clustering over
  time and watching for cohesion loss, not by inspecting one snapshot.

## Current status (v0)

- `select_k_by_silhouette` — sweep k, pick peak silhouette (data-driven k)
- `fit_clusters` — KMeans + per-cluster silhouette + weak/split metadata
- `detect_new_field_seeds` — distance-based outlier → new-field seeds
  (optionally `elliptic_envelope` for low-dim cross-check)
- `run_discovery_pipeline` — end-to-end over a batch of embeddings
- `demo_silhouette_elbow.py` — synthetic sanity check (recovers true k)
- **17 tests green** (`tests/test_cluster_v0.py` + Step 1/2 tests)

## Next

1. Collect real `ActivityEvent` rows (wire `collector.py` into sessions).
2. Build the embedding contract (pin model + dimension + version so vectors
   stay comparable across runs).
3. Fit v0 on real data; validate discovered clusters against real intents.
4. Evaluate HDBSCAN as v1 (arbitrary shapes + streaming).