# computer_use expansion

Modular computer-use sidecar for Kernel-Evolving.

This expansion follows the same sidecar + bridge pattern used by `expertise-field`.

Current defaults (decision locked):
- **Default target:** desktop-first (Kernel already has native browser-use tooling)
- **State retention:** per-chat, with per-run checkpoints under each chat

Dependency policy:
- Avoid legacy pins.
- Keep baseline dependencies modern and reviewed against official docs.
- Version snapshot is tracked in `docs/dependencies.md`.

See `SPEC.md` for architecture and milestones.
