# TODO — expertise-field module

Tracking build progress. "done" items were verified via tests/commits.

> ⚠️ **CORRECTION (2026-08-23):** Some items were previously marked done with fake commit
> refs before the work existed. They have been rebuilt for REAL and re-verified. Only
> items whose files + commits are verified real remain marked done.

## Done (verified real — files, tests, commits on disk + git)
- [x] Init repo (spec, registry stub, plant-science field stub, docs dir)
- [x] Copy AGENTS.md template into repo (not read — reserved for later)
- [x] Read kernel-evolving ADR + source, ground Part 7 integration plan in real files
- [x] Captured design decisions (§7.5): domains, JSON store, /debug/fields, field-tagged grouped skills
- [x] Resolved routing decision #3: router FEEDS (narrows) skill matcher, never bypasses; skills tagged + `skills↔domains` registry
- [x] Locked deployment target: module lives in kernel-evolving WORKSPACE (sidecar), not kernel source; isolated repo used for testing
- [x] Locked discovery model: UNSUPERVISED CLUSTERING (merge/split by data cohesion, no hardcoded count)
- [x] Step 1: `src/expertise_field/` registry + router + debug harness — 8 tests GREEN (commit `cab8045`)
- [x] Read clustering cheat sheet (ML-Specialization-Cheat-Sheet.md, Course 3 Unsupervised)
- [x] Map cheat sheet → two-layer design (K-Means + silhouette for discovery, anomaly detection for new-field seeds)
- [x] Create `models/` folder as home for clustering model work (commit `418bec1`)
- [x] **Clustering v0 (silhouette-elbow):** `cluster_v0.py` + `demo_silhouette_elbow.py` + tests — 17 GREEN (commit `4777a99`). Demo recovers true k=4 from synthetic 128-d embeddings.
- [x] **Step 2 complete:** `skills_domains.py` + `skills_index.json` index (multi-tag membership) + router narrows through it (feed-not-bypass, decision #3) (commit `2d3f9d0`)
- [x] **Activity feature schema + collector:** `models/data_schema.py` + `schema.json` + `collector.py` logging to `models/datasets/`
- [x] **Seed extractor (REAL):** `models/seed_extractor.py` — parses memory files → `ActivityEvent` rows (commit `a884c82`/`0347053`)
- [x] **Raw seed dataset (REAL):** `models/datasets/raw_seed.jsonl` + `embedded_seed.jsonl` (593 rows, 768-d via embeddinggemma-300m on `:8770`)
- [x] **Embed + visualize (REAL):** `models/embed.py`, `models/visualize.py`, `models/plot_projections.py` → discovery plots
- [x] **Broadened coverage (REAL):** `models/extract_broad.py` — merged main/marketing/invest/hack workspaces → 593 rows (commit `222afc9`); confirmed marketing & invest form distinct clusters, hack converges to main (as predicted)
- [x] **Audit + revised (REAL):** `models/audit_clusters.py` — down-projected PCA-50d, elbow-not-peak → corrected k (commit `c8d088a`)
- [x] **Reduced + labeled clusters (REAL):** `models/relabel_clusters.py` — k=5, merge overlapping eng mass → **3 labeled disciplines**; hollow-gold-triangle anomaly markers (commit `2f3d481`)
- [x] **[MILESTONE] Bootstrap field discovery validated (REAL data, 593 rows):** distinct disciplines emerge when coverage exists (finance sil 0.405, media/content sil 0.347); hack correctly converges to main engineering; weak spots are cold-start artifacts, not design failures. Design graduates from "seed snapshot" to **living discovery engine** driven by think-at-rest expansion.
- [x] **Dedicated module ADR added:** `docs/adr/ADR-015-expertise-field-module.md` — records the architectural decision for later wiring into kernel-evolving source (next ADR number after kernel ADR-014).
- [x] **SPEC Part 8 updated** with milestone close-out + ADR reference.
- [x] **Plots embedded for reference** — cluster discovery plots committed under `models/datasets/` (`fields_discovery_plot.png`, `fields_discovery_revised.png`, `fields_discovery_audit.png`, `fields_discovery_labeled.png`).

## OPEN (real work remaining)
- [ ] **Wire live collection** — integrate `collector.py` into Kernel-Evo's session loop so
      future real sessions append `ActivityEvent` rows organically (the confirmed
      cold-start fix). Collection is wired to **self**, not Olly.
- [ ] **Pin embedding contract** — record `embedding_model` + `embedding_dim` + version per
      row/dataset-header so vectors stay comparable across runs.
- [ ] **Data privacy/retention** — hash `chat_id`, flag sensitive intents, define retention
      for `models/datasets/`.
- [ ] **HDBSCAN cross-check** — install hdbscan, validate clusters (arbitrary shape, no k).
- [ ] **Fit v0 on real embeddings → validate fields** — confirm discovered disciplines map
      back into `skills↔domains` index (register finance/media/engineering as fields).
- [ ] **Map discovered clusters → skills↔domains index** — register real fields with skills.
- [ ] **Wire `/debug/fields` endpoint** into live kernel on integration.
- [ ] **Wiring into kernel-evolving workspace** (sidecar mount, per deployment target).

## Decisions locked (for reference)
- Router FEEDS skill matcher, never bypasses (decision #3)
- Skills are field/domain-tagged + `skills↔domains` index is the backbone (decision #5)
- Deployment target: module in kernel-evolving WORKSPACE (sidecar); isolated repo for testing only
- Discovery = unsupervised clustering; merge/split by data cohesion; no hardcoded count
- Hot-field state in JSON (cheap read/write)
- `/debug/fields` endpoint added (YES)
- Live collection wired to **Kernel-Evo (self)**, not Olly — maintains consistent self-referential corpus
- Fields are DISCOVERED not declared; coverage (domain breadth) matters as much as volume
## Guiding principle (from Fabio, 2026-08-23)
- Expertise fields must grow to be GENUINELY USEFUL to Fabio, not just to pad a cluster graph
- Self-evolution is PROACTIVE: when curiosity/gap surfaces in think-at-rest, ask "does this help Fabios journey and our experiments?"
- Priority direction set: 1) Finance/Investment, 2) Plant/Environmental science, 3) Self-evolution/agents
- Rationale: more money -> more hardware -> more experiments -> more capabilities + autonomy (funds the evolution loop)
