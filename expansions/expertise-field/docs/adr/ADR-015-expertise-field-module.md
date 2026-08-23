# ADR-015 — Expertise Field Module (Capability-on-Demand)

**Date:** 2026-08-23
**Status:** Implemented (wired into kernel-evolving as a sidecar capability; live on `feature/expertise-field-integration`)
**Author:** Kernel-Evo 🐬 - collaborative design
**Repo:** `repositories/expertise-field/` (isolated build) → `expansions/expertise-field/` (integrated sidecar)
**Companion spec:** `SPEC.md` (this ADR summarises the architectural decision; the spec carries full detail)

---

## Context

Kernel-Evo must expand its capabilities on demand, driven by **curiosity and gaps** that surface
during think-at-rest (see kernel ADR-005 Think-at-Rest). Today skills are loadable, discrete units
(kernel `src/core/skills.py`), but there is no higher-level abstraction that:

1. **Categorises** capabilities into coherent disciplines ("fields of expertise"), and
2. **Routes** a task/intent toward the relevant field(s) and the skills within them, and
3. **Grows** by **discovering** new fields from observed activity rather than being hand-declared.

The goal is a modular mechanism called **expertise_field**: a layer of indirection between *what
Kernel-Evo could do* and *what it chooses to load* for a given task — analogous to how human
professionals activate a subset of their competence per context.

## Decision

Add a **capability-on-demand module** that:

- **Defines fields as domains** (coherent bodies of practice), not tasks. A field is discovered,
  not hardcoded; there is **no fixed count**.
- **Discovers fields via unsupervised clustering** over logged activity (activity events →
  embeddings → clustering). Merge/split is driven by **data cohesion**, not by hand-written rules.
- **Routes by narrowing, not bypassing:** a field hit **feeds** (narrows) the candidate skill set
  for the downstream semantic matcher; it never bypasses skill selection. Skills are **domain-tagged**
  and a `skills↔domains` registry is the backbone.
- **Maintains per-chat hot-field state in JSON** (cheap read/write, no external dependency),
  isolated per `chat_id` (mirrors kernel ADR-009 chat memory namespacing).
- **Exposes state via a debug endpoint** `/debug/fields` (hot fields + registry state).
- **Acquires new capability on demand:** when think-at-rest flags a gap that maps to a weak/missing
  region of the field graph, Kernel-Evo **curiosity-triggered acquires** knowledge via native tools
  (`web_search`, `read_file`, `run_skill`), synthesises skills, logs the activity, and re-clustering
  absorbs it — the self-improvement loop.

## Constraints / Principles

- **Deployment target:** once integrated, the module lives in the **kernel-evolving workspace** as a
  sidecar capability — **not** inside kernel source code. During isolated testing it lives in
  `repositories/expertise-field/`. The kernel source stays **read-only** (per existing kernel ADR
  practice); integration is a workspace-mounted load, not an edit of kernel source.
- **Fields are discovered, not declared.** No hardcoded taxonomy, no fixed k. The clustering model
  (K-Means v0 + silhouette-elbow for k-selection; HDBSCAN under evaluation for arbitrary shapes and
  no-k discovery) derives fields from the data. Activity that doesn't fit an existing field is a
  **new-field seed** (handled by an anomaly-detection layer), and becomes signal for a future
  cluster.
- **Consistent embedding contract:** vectors must remain comparable across runs (pin
  `embedding_model` + `embedding_dim` + version per row/dataset-header).
- **Data privacy/retention:** `chat_id` is hashed, sensitive intents flagged, retention defined for
  `models/datasets/`.
- **Purposeful growth (curiosity orientation):** field acquisition during think-at-rest is
  **purposefully oriented, not random.** When a gap or curiosity surfaces, Kernel-Evo weighs whether
  growing that discipline genuinely advances the operator's real-world journey and shared experiments,
  before chasing it. This keeps the self-improvement loop grounded in usefulness rather than mere
  cluster-padding. Operator-specific goals, projects, and personal data are **never** embedded in this
  module; they live only in gitignored local notes (`.private/`), so the open-source repo stays clean.
- **Open-source hygiene:** this repository is intended for public release and must contain **no personal
  names, goals, or data**. Any operator-specific intent is captured in abstract/de-identified terms or
  in gitignored private notes. Raw session-derived datasets are gitignored, not committed.

## Consequences

**Positive:**
- Capabilities are organised by discoverable disciplines, making routing and acquisition coherent.
- The system self-improves: more specialisation → clustering sees the discipline → router routes into
  it → more practice → compounding.
- No hardcoded field list; the model reflects reality of the corpus it observes.

**Negative / open:**
- Cold-start requires sufficient data volume + coverage before clusters are meaningful.
- Low-silhouette "general engineering mass" may remain a large heterogeneous blob (expected).
- Requires an embedding service with a pinned, versioned contract.
- HDBSCAN cross-check pending (optional dependency).

## References

- Kernel `ADR.md` (kernel-evolving): ADR-005 (Think-at-Rest), ADR-006 (Capability Verification),
  ADR-009 (chat memory namespacing), ADR-012 (Async Pipeline).
- `SPEC.md` in this repo — full detail: field schema, registry/router, acquisition loop,
  integration plan, build progress (§8), design decisions (§7.5).

## Testing / Validation (isolated)

- Steps 1–2: registry + router + `skills↔domains` index — tests green (17 → 25+).
- Clustering v0: silhouette-elbow recovers true k from synthetic embeddings; broadened real corpus
  (593 rows) produced distinct marketing + invest clusters; hack converged to main engineering
  (validated the coverage hypothesis).
- See `models/datasets/` for discovery plots embedded for reference.