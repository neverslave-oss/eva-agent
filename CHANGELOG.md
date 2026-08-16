# Changelog

All notable changes to **kernel-evolving** are documented here.

## [1.19.0] - 2026-05-25

### Added
- `/voice-clone` bare command: inline sample picker with usage hint and one-tap voice switching
- `voice-clone` skill: `exec:` field for direct script dispatch (bypasses LLM), `intents:` for natural language routing
- `clone_voice.sh` v2: smart arg parsing — auto-detects sample from text (ricky/italian/english), no path required
- Core skills dispatch test suite: 31 tests covering presence, commands, intents, exec paths for voice-clone, collective-memory, security-scanner etc.
- New test files: `test_core_skills_dispatch.py`, `test_memory.py`, `test_provider_infer_with_tools.py`, `test_model_client_unit.py`

### Fixed
- `skills.py`: `{skill_dir}` token was not substituted in exec templates → `No such file or directory`
- `skills.py`: `skill_dir` assignment was after its use → `UnboundLocalError`
- `memory.py`: `_sanitise()` strips error/VRAM-guard assistant turns and orphan consecutive user turns that caused stuck/looping responses
- `/new` command: now scopes clear to `chat_id` only — previously cleared all sessions via `clear_all()`
- `/status` command: model name now read live from `model_client.health()` instead of hardcoded `Gemma 4 E2B-it`
- `telegram_bot.py`: double memory persistence for plain chat turns removed
- Ecosystem cleanup: removed 37 synthesised noise/duplicate skills (80 → 43)
- Telegram command picker: dynamic `/skill_<slug>` + `/run_<slug>` for all installed skills/routines (100-command budget, correct 32-char slug limits)


## [1.18.6] - 2026-05-25

### Changed
- Added and structured this changelog for release tracking.
- Normalized release documentation after the Docker + ADR-015 rollout.

## [1.18.5] - 2026-05-25

### Changed
- Version alignment across runtime and docs:
  - `src/version.py` → `1.18.5`
  - `config.yaml` and `config.container.yaml` `self_identity.version_tag` → `v1.18.5-evolving`
  - README version badge updated accordingly.

## [1.10.1-evolving] - 2026-05-25

### Documentation
- README aligned to the shipped architecture:
  - Docker startup via `entrypoint.sh`
  - writable `/models` cache for first-run Nemotron download
  - sandbox profile usage
  - discovery/network notes for containerized deployments
- Sim14 section updated with current snapshot and caveats.

## [1.10.0-evolving] - 2026-05-25

### Added
- ADR-015 auto-discovery implementation:
  - `src/discovery.py` (port-scan + optional mDNS + coding agent probes)
  - `src/mcp_client.py`
  - `/peers` endpoint in API
  - passive delegation hook in agent routing
  - `docs/PEER_PROTOCOL.md`
  - unit tests: `tests/test_discovery.py`

### Changed
- Docker portability pass:
  - new `entrypoint.sh`
  - compose/env wiring cleanup
  - container config portability updates
  - full dependency alignment for container runtime
- Think-at-Rest observability:
  - logging setup improvements
  - manual `/think/trigger` endpoint
- Eval hygiene:
  - provider-switch unload behavior improvements in eval flow.
- Local default model set to **Nemotron-Labs-Diffusion-3B**.

## [1.18.4] - 2026-05-25

### Fixed
- Telegram chat-id propagation to avoid cross-session memory contamination.

---

> Note: historical pre-1.18.4 entries are available in git history and release notes.