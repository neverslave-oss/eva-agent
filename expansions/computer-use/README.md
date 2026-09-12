# computer_use expansion

Modular computer-use sidecar for Kernel-Evolving.

This expansion follows the same sidecar + bridge pattern used by `expertise-field`.

Current defaults (decision locked):
- **Default target:** desktop-first (Kernel already has native browser-use tooling)
- **State retention:** per-chat, with per-run checkpoints under each chat

Planners (all behind the same orchestrator interface):
- **Deterministic `Planner`** — baseline step loop
- **`LLMPlanner`** — vision-brain perceive→decide→act loop (streams screenshots to
  Telegram via `watch_callback`)
- **`RLPlanner`** — tabular Q-learning perceive→decide→act loop (see
  `.specs/plans/2026-09-12-computer-use-rl-planner.md`). Discrete state from
  screen-hash / driver / outcome / progress / step / url buckets; ε-greedy over a
  Q-table; verified rewards; fail-closed dead-state abort. Streams screenshots to
  Telegram via `watch_callback` (mirrors the LLM path).

Dependency policy:
- Avoid legacy pins.
- Keep baseline dependencies modern and reviewed against official docs.
- Version snapshot is tracked in `docs/dependencies.md`.

See `SPEC.md` for architecture and milestones.
