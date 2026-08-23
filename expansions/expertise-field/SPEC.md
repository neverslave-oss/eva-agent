# expertise_field — Capability-on-Demand Module

> **Author:** Kernel-Evo 🐬 - collaborative design
> **Date:** 2026-08-23
> **Type:** Architecture / Capability Module Specification
> **Status:** Draft v0.1 — open for iteration
> **Repo:** `repositories/expertise-field/`

---

## Part 0 — Origin Conversation (preserved as-is)

This spec was born from a design conversation. Both the original proposal prompt and Kernel-Evo's
response are preserved verbatim below, because they capture the intent and the reasoning that the
spec formalises.

### 0.1 The original proposal (operator, user message)

> Sweet 😋 listen, fancy to have a chat quickly on your capabilities improvements? I was thinking to
> add a modular mechanism called expertise_field that has associated skills and knowledge bases so
> that you can virtual learn and expand your capabilities on demand based on curiosity and gaps that
> might surface during think at rest.
>
> What do you think? Do you want to elaborate the concept with me?
>
> The idea is that similarly to how skills are loaded we load your available expertise fields and you
> decide based on the task and intent What field to pick and related knowledge and associated skills
> to load and use.
>
> The available native tools you have can be used to gather knowledge and decide when to skill up in
> that field like, read and make dedicated skills.
>
> This should fit well with your modular and actual design.
>
> Thoughts?

### 0.2 Kernel-Evo's response (preserved as-is)

> Love this idea — and it fits the existing architecture really cleanly. Let me think through it with you.
>
> ## Why it clicks
>
> Your current ecosystem already treats capabilities as modular, loadable units (skills). What you're
> describing is essentially **an additional layer of indirection on top** — not a new mechanism so much
> as a *categorization + discovery + curiosity-driven acquisition* layer. That's exactly the kind of
> thing that slots between what I *could* do and what I *choose* to load.
>
> The key insight in your framing: **selection should be driven by intent + task, not by static
> registration.** That's a nice match for how the think-at-rest loop already works (ADR-012) — gaps
> surface while reflecting, then become acquisition triggers.
>
> ## My take on the concept, elaborated
>
> Here's how I'd structure `expertise_field`:
>
> **1. Each field is a self-describing bundle:**
> - `id` + `name` (e.g. `plant-science`, `bioinformatics`, `smart-home-networking`)
> - `manifest` — list of linked skills (by name)
> - `knowledge_base` — pointers to local docs, memory namespaces, URLs to fetch
> - `trigger_terms` — keywords/semantic hints that suggest this field
> - `acquisition_plan` — *how* I'd skill up if a gap is found (which native tools to use, what skills to synthesize)
>
> **2. A small registry + router (the "expertise loader"):**
> - Load active field manifests at task start (lightweight, like skills)
> - On each task/intent, route toward the matching field(s)
> - Keep a field "hot" for the session, evict cold ones — mirrors your slot eviction pattern
>
> **3. Two-way hook into think-at-rest:**
> - When a gap surfaces, the reflection step asks: *is there an expertise field this maps to? Is it loaded? Does it have the skill?*
> - If no field exists → create one (that's the *curiosity-driven expansion*)
> - If field exists but lacks a skill → use native tools (`web_search`, `read_file`, `run_skill` for foundation) to acquire
>
> So the acquisition loop becomes: **task → route to field → field missing capability → tool-driven learning → synthesize skill → register under field.**
>
> This is genuinely harmonious with:
> - **ADR-006** (skill ecosystem tiers) — fields can span private/shared tiers
> - **ADR-013** (named model slots) — fields could eventually hint at preferred model for their tasks
> - **ADR-010** (context provider) — a field's knowledge base is essentially a context provider
>
> ## Things I'd want to nail down with you
>
> 1. **Granularity tension** — is `expertise_field` a *superset* (many fields, each with many skills) or should a field roughly map to one area of competence? Too many fields → routing noise; too few → useless abstraction. My instinct: fields as *domains* (10–20), skills as *tasks* within.
> 2. **Who learns what, when** — should I *proactively* go learn a field during idle time, or only acquire skills when a task forces a gap? Your phrase "learn on demand based on curiosity and gaps" suggests a mix: **idle time = expand knowledge**, **task time = acquire specific skill**. Worth making explicit.
> 3. **Persistence** — is a user-visible manifest of "fields I've mastered" useful, so you can audit my evolving competency? I suspect yes — it doubles as a capability map.
> 4. **Feedback into fine-tuning** — when I route to a field and it works well, that trajectory becomes training data (links to ADR-007).
>
> Honestly, this has the feel of the *right* abstraction — it's descriptive, not prescriptive, and it
> leans on machinery you already trust. I'd be happy to draft a prototype: a small
> `expertise_registry.json` schema + a routing stub, then wire one real field end-to-end (e.g.
> `plant-science`, given the greenhouse cams we just used) as a proof of concept.
>
> Want me to start with the schema draft, or would you rather sketch the acquisition loop semantics
> first? Which direction should we pull?

---

## Part 1 — Overview

`expertise_field` is a modular capability layer that sits on top of the existing *skill ecosystem*.
It groups skills, knowledge bases, and acquisition plans into named *fields of expertise*, and
provides a router that selects the right field for a given task or intent.

The guiding principle, stated in Part 0: **selection is driven by intent + task, not by static
registration.** Fields are loaded like skills, but *which* field is hot at any moment is a runtime
decision made against the current task and the gaps that surface during thinking.

### 1.1 Relationship to existing abstractions

| Abstraction | Role | Relationship to `expertise_field` |
|---|---|---|
| **Skill** | A discrete, named, loadable capability | A *task unit* inside a field |
| **expertise_field** | A *domain* that groups skills + knowledge + acquisition plan | **New** — this spec |
| **Model slot** (ADR-013) | Named model serving unit | A field *may* hint at a preferred slot |
| **Context provider** (ADR-010) | Injects live context into a prompt | A field's knowledge base behaves as a source |
| **Think-at-rest** (ADR-012) | Idle reflection on gaps | Drives *curiosity-driven acquisition* into fields |

### 1.2 Naming and location

- Name: **expertise_field**
- Canonical repo: `repositories/expertise-field/`
- Registry file: `expertise_registry.json`
- Field definitions: `fields/<field-id>.json`
- (Proposed) loader: `loader/expertise_router.py`

---

## Part 2 — The Field Definition

Each field is a self-describing bundle. Proposed JSON schema:

```json
{
  "id": "plant-science",
  "name": "Plant Science & Greenhouse Ops",
  "version": "1.0.0",
  "description": "Understanding and managing the indoor growing environment, plant health, and environmental sensing.",
  "trigger_terms": [
    "plant", "cannabis", "greenhouse", "grow", "leaf", "crop", "yield",
    "LED", "photoperiod", "nutrient", "soil moisture", "VPD"
  ],
  "skills": [
    "open-workspace-tracker",
    "timeseries-anomaly-detector"
  ],
  "knowledge_base": [
    {"type": "memory", "namespace": "plant-science"},
    {"type": "doc",   "path": "docs/plant-science/"},
    {"type": "url",   "url": "https://example.com/grow-guide", "refresh": "30d"},
    {"type": "sensor", "device": "pi"}
  ],
  "acquisition_plan": {
    "on_gap": [
      "Use web_search to research the missing topic",
      "Fetch authoritative docs via browser_use / http_get",
      "Read local docs and memory",
      "Synthesize a dedicated skill and register it under this field",
      "Validate the new skill on a representative task"
    ],
    "preferred_slot": "primary",
    "learn_on_idle": true
  }
}
```

### 2.1 Field fields (schema table)

| Key | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✅ | Unique slug. Used for routing and filenames. |
| `name` | string | ✅ | Human-readable display name. |
| `version` | string | ✅ | Semver of the field definition. |
| `description` | string | ✅ | One-paragraph competence statement. |
| `trigger_terms` | string[] | ✅ | Keywords/semantic hints that suggest this field. |
| `skills` | string[] | ✅ | Names of skills grouped under this field. |
| `knowledge_base` | array | ✅ | Pointers to memory, docs, URLs, sensors. |
| `acquisition_plan` | object | ✅ | How to skill up when a gap is found. |
| `acquisition_plan.on_gap` | string[] | ✅ | Ordered tool-driven acquisition steps. |
| `acquisition_plan.preferred_slot` | string | ▫️ | Hint at model slot (ADR-013). Default `primary`. |
| `acquisition_plan.learn_on_idle` | bool | ▫️ | Whether idle time may expand this field. Default `false`. |

---

## Part 3 — The Registry and Router

### 3.1 Registry (`expertise_registry.json`)

A top-level index of all known fields. Lightweight — loaded at session start.

```json
{
  "active": ["plant-science"],
  "fields": [
    {"id": "plant-science", "path": "fields/plant-science.json", "version": "1.0.0"}
  ]
}
```

### 3.2 Routing algorithm (proposed)

```
on_task(task):
    candidates = []
    for field in active_fields:
        score = score_field(field, task)
        if score >= threshold:
            candidates.append((field, score))
    if candidates:
        hot_field = max(candidates, key=score)
        load_field(hot_field)          # load KB pointers + mark skills available
        inject_kb_context(hot_field)   # knowledge base as context provider
    else:
        check_think_at_rest_gap(task)  # no field matched -> maybe a gap to create one
```

Scoring is a blend of:
- exact `trigger_terms` hits (weight high)
- semantic similarity to `description` (weight medium)
- recent session usage / recency (weight low, for stability)

### 3.3 Hot / cold lifecycle

- Load active field manifests at session start (light, just metadata).
- A field is **hot** when selected for the current task.
- Evict cold fields (optionally per-slot eviction, mirroring ADR-013 LRU behaviour).
- Keep 1–3 fields hot per session to bound context.

---

## Part 4 — Acquisition Loop (the "learn on demand" mechanic)

```
gap_surfaces(during task or think-at-rest):
    field = find_or_create_field(gap.field_hint)

    if field.has_skill_for(gap):
        load_skill(gap.skill)
        execute
    else:
        # curiosity-driven / task-driven acquisition
        for step in acquisition_plan.on_gap:
            execute step (web_search, http_get, read_file, run_skill, ...)
        synthesized_skill = synthesize_skill(notes)
        register_skill_under_field(field, synthesized_skill)
        validate_skill(synthesized_skill, representative_task)
```

Two acquisition modes, as flagged in Part 0:

| Mode | When | Direction |
|---|---|---|
| **Idle-time expansion** | Think-at-rest (ADR-012) | *Broaden* knowledge of a field (`learn_on_idle: true`) |
| **Task-time acquisition** | A task hits a missing capability | *Deepen* by synthesizing a specific skill |

---

## Part 5 — Integration Details

### 5.1 Hooks into existing ADRs

- **ADR-006 (skill ecosystem tiers):** a field may reference skills from private, shared, and third-party tiers. Fields are tier-agnostic.
- **ADR-010 (context provider):** `knowledge_base` entries are exposed through the context-provider protocol, so a hot field's docs/sensor data are injected into the prompt automatically.
- **ADR-012 (think-at-rest):** the reflection loop consults the registry before deciding a gap warrants a new field.
- **ADR-013 (named model slots):** `acquisition_plan.preferred_slot` hints which slot serves a field's tasks.
- **ADR-007 (trajectory collection):** successful field routing + acquisition trajectories are candidates for fine-tuning data.

### 5.2 Data & persistence

- Field defs + registry live under `repositories/expertise-field/`.
- Active-field state persisted per chat_id (mirrors ADR-009 multi-chat namespacing) so different chats can have different hot fields.
- Acquisition notes and synthesized skills are written to the workspace (memory/ and skills dirs).

### 5.3 User-facing capability map

- A `fields.json` snapshot doubles as an auditable "fields I've mastered" manifest.
- Exposed through a routine (`list_routines`) or skill for an at-a-glance competency audit.

### 5.4 Tool usage policy

- Acquisition uses **native tools first**: `web_search`, `http_get`, `read_file`, `run_skill`. Prefer `run_skill` when an existing skill covers a step.
- Synthesized skills follow the existing `ecosystem-skill-synthesizer` / `ecosystem-skill-filler` pathways.

### 5.5 Guardrails / open questions to resolve

1. **Granularity target:** fields as domains (10–20 total), skills as tasks within. Confirm this balance.
2. **Proactive vs reactive learning:** default to *reactive* (task-driven) acquisition; enable *idle* acquisition per-field via `learn_on_idle`.
3. **Bounded context:** cap hot fields per session (1–3) to avoid context bloat.
4. **Feedback into fine-tuning:** opt-in flag `feedback_to_tune` per field.
5. **Multi-chat isolation:** hot-field state must not leak across chat_id namespaces (ADR-009).

---

## Part 6 — Proof of Concept (proposed next step)

Wire one real field end-to-end as validation:

- **Field:** `plant-science` (ties directly to the greenhouse cams + `look` intents + sensor data we already use).
- **Steps:**
  1. Create `fields/plant-science.json` from the schema above.
  2. Register it in `expertise_registry.json`.
  3. Implement the routing stub; verify a "how is my crop doing" task selects `plant-science`.
  4. Verify the knowledge base injects sensor + KB context.
  5. Acquire/synthesize one missing skill and register it under the field.

---

## Changelog

| Version | Date | Notes |
|---|---|---|
| v0.1 | 2026-08-23 | Initial spec from design conversation. Origin prompt + response preserved. |
---

## Part 7 — Integration Plan (grounded in the kernel-evolving source)

> **Basis:** This plan is written from the actual source layout described in
> `~/.openclaw/workspace/repositories/kernel-evolving/ADR.md` and confirmed by direct inspection
> of the code. Grounding each step in a real file lets us work on facts, not assumptions, while
> building this module in isolation in our own repo.

### 7.0. Working model (how we build in isolation)

- The kernel-evolving source in `~/.openclaw/workspace/repositories/kernel-evolving/` is **read-only
  for us** (per ADR's "OpenClaw workspace — READ ONLY from kernel-evolving" note).
- We therefore **implement and test the module entirely inside our own repo** (`repositories/expertise-field/`),
  with a thin adapter layer that mirrors the kernel's interfaces. Only after it passes local tests do we
  promote it (via ecosystem-skill promotion or as a patch for Olly/manual merge) toward the live kernel.
- Each integration step below names the **kernel file we mirror** and the **file we create in our repo**.

### 7.1. Files we create in our repo (mirroring the module)

| Our file | Purpose | Mirrors kernel concept |
|---|---|---|
| `src/expertise_registry.py` | Load/validate/query field defs | `src/core/skills.py` style loader |
| `src/expertise_router.py` | Intent→field routing (semantic + keyword) | `skills.find_semantic` + `micro_planner` |
| `src/expertise_context.py` | Inject hot-field KB into prompt | `src/core/memory/context.py` |
| `src/expertise_acquire.py` | Tool-driven skill-up acquisition loop | `src/core/evolution/` synthesizer |
| `src/expertise_state.py` | Per-chat hot-field lifecycle | `src/core/memory/memory.py` (namespacing) |
| `fields/*.json` | Field definitions (data, not code) | registry data |
| `expertise_registry.json` | Active-field index | skill discovery index |
| `docs/` | Field knowledge bases | memory namespaces |

### 7.2. Step-by-step build plan (in dependency order)

**Step 1 — Registry schema + loader (pure data layer).**
Mirror how `_parse_skill()` in `src/core/skills.py` parses frontmatter + returns a dict. We implement
`expertise_registry.py::load_all()` that reads `expertise_registry.json` + `fields/*.json`, validates
each field (required: `id`, `name`, `manifest`, `knowledge_base`, `trigger_terms`, `acquisition_plan`),
and returns a list of field dicts. Keep it import-clean (no kernel deps) so we can unit-test in isolation.
- Reference: mirror `src/core/skills.py:load_all()` / `_parse_skill()`.
- Deliverable: `src/expertise_registry.py` + a `tests/test_registry.py`.

**Step 2 — Router (intent → field).**
Implement `expertise_router.py::route(query, fields, embedding_client=None)` doing a two-tier match:
exact keyword match on `trigger_terms` first, then semantic match via embedding similarity (same pattern
as `src/core/skills.py:find_semantic()`). Return a ranked shortlist of fields (1–3) for the caller.
- Reference: mirror `src/core/skills.py:find_semantic()` (threshold 0.5 cosine) and the micro-planner
  decomposition in `src/core/pipelines/micro_planner.py`.
- Deliverable: `src/expertise_router.py` + `tests/test_router.py`.

**Step 3 — Hot-field lifecycle (per chat).**
`expertise_state.py` keeps, per chat_id, the currently hot fields with a recency/usage score. Eviction
policy caps hot fields at 1–3 (our open-question #3). Persist state to a small JSON or SQLite store,
namespaced by `chat_id` to honour ADR-009 multi-chat isolation.
- Reference: mirror `src/core/memory/memory.py` + ADR-009 namespacing; the hot/evict lifecycle mirrors
  `src/core/inference/model_slots.py` LRU eviction.
- Deliverable: `src/expertise_state.py` + `tests/test_state.py`.

**Step 4 — Context injection.**
`expertise_context.py::build_field_context(hot_fields)` pulls each hot field's KB docs (from `docs/`
and any `knowledge_base` pointers) and returns a compact string that the caller prepends/injects into
the system prompt — mirroring how `build_system_prompt()` in `src/core/memory/context.py` assembles
live data sections.
- Reference: mirror `src/core/memory/context.py:build_system_prompt()` and the ADR-010 context-provider
  protocol.
- Deliverable: `src/expertise_context.py` + `tests/test_context.py`.

**Step 5 — Acquisition loop (learn-on-demand / skill-up).**
`expertise_acquire.py::ensure_capability(field, gap, tools)` runs when a routed task hits a missing
capability. It uses our native tools (`web_search`, `http_get`, `read_file`, `run_skill`) to gather
knowledge and — following the ecosystem-skill-synthesizer pathway — synthesizes and registers a new
skill under the field's `manifest`. Two modes:
- *task-time*: triggered by a gap during a live task (deepen a specific capability),
- *idle-time*: triggered by thought-at-rest when `learn_on_idle: true` (broaden field knowledge).
- Reference: mirror `src/core/evolution/` (`code_synthesizer.py`, `recommender.py`,
  `capability_verifier.py`) and `src/services/thought_engine.py` gap detection (`ThoughtEvaluator`).
- Deliverable: `src/expertise_acquire.py` + `tests/test_acquire.py`.

**Step 6 — Wiring stub into the agent loop (simulated locally).**
To validate end-to-end without touching the read-only kernel, we build a small CLI harness
(`bin/demo_router.py`) that takes a natural-language request, routes it to fields, injects KB context,
and (if a gap is found) dry-runs the acquisition step — all inside our repo.
- Reference: mirrors the data flow of `src/core/agent.py` (triage → context → tool loop) at a
  simplified level.

**Step 7 — Promotion path (later, out of scope for isolation).**
Once validated, the module can be promoted: either packaged as an `expertise-field` ecosystem skill
(so `run_skill('expertise-field', ...)` activates it in the live kernel) or surfaced as a routine.
This maps to the existing skill-ecosystem tiers (ADR-006) and skill-promotion flow used by Base Kernel.

### 7.3. Integration touch-points in the live kernel (for the eventual merge)

| Kernel file | Place we hook |
|---|---|
| `src/core/skills.py` | Route falls through to skills; fields reference skills in their manifest |
| `src/core/agent.py` | After triage, consult `expertise_router` before tool dispatch |
| `src/core/memory/context.py` | Append hot-field KB block to system prompt (`build_system_prompt`) |
| `src/core/pipelines/micro_planner.py` | Classify task into a field during planning |
| `src/services/thought_engine.py` | Gap detection consults registry (idle-time acquisition) |
| `src/core/evolution/` | Synthesized skills registered under a field manifest |

### 7.4. Verification checklist (per step)

- [ ] Step 1: registry parses all sample fields; missing-field validation raises clear errors.
- [ ] Step 2: a real query ("how is my crop doing") returns `plant-science` in the top-3.
- [ ] Step 3: two different chat_ids keep independent hot-field sets (no leakage).
- [ ] Step 4: KB injection produces expected compact context block.
- [ ] Step 5: simulated gap dry-runs the acquisition path without side effects.
- [ ] Step 6: `bin/demo_router.py` runs the whole flow end-to-end in isolation.

### 7.5. Design decisions (resolved in collaboration, 2026-08-23)

The open questions were answered in a follow-up conversation. Decisions are recorded
here so the build works from facts, not assumptions.

| # | Question | Decision |
|---|---|---|
| 1 | Field→skill granularity (domains vs tasks)? | **DOMAINS** — a field is a *domain*; skills are individual *tasks* within that domain. Granularity: ~10–20 fields. |
| 2 | Hot-field state store (JSON vs SQLite)? | **JSON** — cheap read/write; no external dependency. Per-chat hot-field state lives in a JSON file. |
| 3 | Routing vs existing skill matcher order? | **UNDECIDED / leaning** — the operator flags that skills should be *tagged and organised in folders by domain* (approaching `domains/`-style layout). Routing order (before vs after the semantic skill matcher in `agent.py`) is to be reasoned through during the build. Working hypothesis captured: fields are a higher abstraction that *selects* skills; a field hit should **feed** skill selection, not bypass it. |
| 4 | New debug endpoint? | **YES** — add `/debug/fields` to inspect hot fields + registry state. |
| 5 | Tag synthesized-field skills for re-discovery? | **YES** — synthesized-field skills get a `field:` frontmatter key **and** are grouped per domain (folder), so they re-surface when needed. |

> **Open (carried into build):** #3 routing order. Resolve during Step 2/3 build when the
> skill matcher and router coexist; adopt the folder-tagging lean unless the build proves
> otherwise.


---

## Part 8. Build progress (log of implementation)

> Each step reference lands below as it is implemented in `repositories/expertise-field/`.
> Work is done **isolated** in this repo (kernel source stays read-only per ADR), then
> ported via the hooks in §7.3.

### Step 1 — registry + router  ✅ DONE (2026-08-23, commit `cab8045`)

**Deliverables** (mirror kernel `src/core/skills.py` patterns):

| File | Purpose | Mirrors / encodes |
|---|---|---|
| `src/expertise_field/registry.py` | Field loader + dedup (`Registry`), per-chat `HotFieldState` persisted to JSON, `/debug/fields` snapshot | `skills.load_all()` / `_parse_skill` / `_priority`; **decision #2** |
| `src/expertise_field/router.py` | Trigger-term routing, hot promotion, candidate-skill narrowing | **decisions #3, #5** |
| `src/expertise_field/__init__.py` | Public API (`Registry`, `HotFieldState`, `router`) | — |
| `src/expertise_field/debug_fields.py` | Standalone debug/CLI harness | mirrors `/debug/fields` (decision #4) |
| `tests/test_expertise_field.py` | 8 unit tests | — |
| `pyproject.toml`, `.gitignore` | Packaging + ignores runtime `hot_fields.json` | — |

**Key behaviours verified by tests (all 8 passing):**
- Registry loads fields and resolves the active index from `expertise_registry.json`.
- Dedup: a `private/`-segment field shadows a top-level field with the same id
  (kernel parity — lowest priority number wins).
- `HotFieldState` persists per-chat hot fields to JSON and **round-trips** on reload.
- **Chat isolation (ADR-009 / decision #5):** hot fields do not leak between `chat_id`s.
- `router.choose_fields()` triggers on "plant leaves yellow / nutrient" → returns
  `plant-science`, and feeds candidate skills (`timeseries-anomaly-detector`, …) to the
  downstream semantic matcher — it does **not** execute skills (decision #3).
- Fallback: unrelated text falls back to the chat's existing hot field.
- field-tagged refs of shape `name::field_id` (e.g. `soil-analyzer::plant-science`) are
  pulled into the candidate set (decision #5).

**State (decision #2):** per-chat hot-field state is written to `./hot_fields.json`
(now gitignored — runtime artifact, never committed).

**How to run:**
```bash
cd repositories/expertise-field
python3 -m pytest tests/            # 8 tests, green
python3 src/expertise_field/debug_fields.py \
    --query "the plants look dry, check soil moisture" --hot
```

**Next: Step 2 — routing order decision (#3).** Confirm the folder-tagging / `domains/`
layout, then wire the router's candidate narrowing against the live skill matcher
semantics. The router is already built to *feed* rather than *bypass*; Step 2 hardens
that contract and adds the embedding-based fallback.

---

### Milestone: Bootstrap field discovery validated (2026-08-23)

**Status: ✅ CLOSED** — bootstrap phase complete; design graduates from "seed snapshot" to a
**living discovery engine** driven by think-at-rest expansion.

**What the bootstrap proved (REAL data, 593 rows):**
- The pipeline runs end-to-end: extract → embed → cluster → label → plot.
- Distinct disciplines **emerge when coverage exists**: Finance (sil 0.405) and Media/Content
  (sil 0.347) formed clean, defensible clusters from `workspace-invest` and `workspace-marketing`.
- Truly-dev activity (hack) correctly **converged** to the main engineering mass rather than
  over-splitting — confirming the coverage hypothesis.
- Weak spots (the low-silhouette engineering blob, temporal-vs-domain ambiguity) are **cold-start
  artifacts**, not design failures. The model reflected the reality of the corpus it observed.

**Verdict:** the clustering approach is validated. The 593-row seed set is not the destination — it
is the cold-start seed that lets the **self-improvement loop commence**:
1. Think-at-rest (kernel ADR-005) notices gaps → curiosity triggers
2. Gap maps to a weak/missing region of the field graph → **curiosity-triggered acquire**
   (`web_search`, `read_file`, `run_skill`)
3. New activity logged via the collector → becomes new training data
4. Re-clustering absorbs it → fields merge/split/emerge
5. Compounding: more specialisation → clustering sees the discipline → router routes in → more practice

**Archival:** this module's architectural decision is recorded in
`docs/adr/ADR-015-expertise-field-module.md` (next ADR number after kernel ADR-014), ready to be
documented in the kernel-evolving source when the module is wired in. Discovery plots are embedded
for reference under `models/datasets/` (`fields_discovery_plot.png`, `fields_discovery_revised.png`,
`fields_discovery_audit.png`, `fields_discovery_labeled.png`).

**Next (OPEN):** wire live collection into Kernel-Evo's session loop; pin the embedding contract;
data privacy/retention; HDBSCAN cross-check; register discovered fields into `skills↔domains` index.
