# expertise_field

A modular capability layer for Kernel-Evo. Groups skills + knowledge bases + acquisition plans into
named *fields of expertise*, with an intent-driven router.

**Design rationale, origin conversation, and full spec:** see [`SPEC.md`](SPEC.md).
**Integration plan, grounded in kernel source:** SPEC §7 (integration) + §8 (build progress).

## Layout

```
expertise-field/
├── README.md
├── SPEC.md                  # complete specification (incl. origin + build progress)
├── AGENTS.md                # template, for later use
├── expertise_registry.json  # active-field index
├── fields/
│   └── plant-science.json   # field definition stub
├── src/expertise_field/     # the module (step 1 implemented)
│   ├── __init__.py
│   ├── registry.py          # Registry + per-chat HotFieldState (JSON)
│   ├── router.py            # trigger routing + candidate-skill narrowing
│   └── debug_fields.py      # standalone debug/CLI harness
├── tests/
│   └── test_expertise_field.py   # 8 tests, green
└── docs/
```

## Status

- **v0.1** — design captured; integration + PoC proposed.
- **v0.3** — design decisions resolved (§7.5): DOMAINS, JSON store, `/debug/fields`,
  field-tagged grouped skills. Routing order (#3) carried into build.
- **v0.4 — Step 1 implemented (commit `cab8045`)** — registry + router live, 8 tests green.

## Run

```bash
python3 -m pytest tests/                       # 8 tests, green
python3 src/expertise_field/debug_fields.py \
    --query "the plants look dry, check soil moisture" --hot
```