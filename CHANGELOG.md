# Changelog

All notable changes to **kernel-evolving (EVA)** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-05

### Added
- HF Router request tuning in `provider.py`: `_hf_post()` helper with configurable
  per-attempt timeout (`HF_TIMEOUT`, default 45s) and retries (`HF_RETRIES`, default 2)
  with brief backoff, used by `_call_hf` and `_hf_tool_loop`. The HF Router
  (`router.huggingface.co`) intermittently hangs; a fixed 120s timeout previously
  blocked the tool loop for minutes before falling back to local.

### Fixed
- Restored the known-working pre-auth-gate `auth_gate.py` (in-process Allow/Deny
  approval) on `main`, reverting the `feat/exec-shell-approve-all` regression that
  caused approval buttons to not appear and conversation turns to be lost.
- Provider tests no longer depend on the live shell environment: added a
  `_clear_provider_env` autouse fixture to `tests/test_provider.py` and
  `tests/test_provider_infer_with_tools.py` so tests assert against their own
  controlled config (the `PROVIDER_TASK_INFERENCE` env var was overriding test config).
- `tests/integration/test_telegram_integration.py::test_my_recent_thoughts_in_system_prompt`
  checked the wrong journal path; now looks in `~/.kernel-evolving/workspace/thoughts/`
  where journals are actually written.

### Tests
- Full suite green: 719 passed, 16 skipped, 1 xfailed, 0 failures.

## [1.0.0] - 2026-08-16

### Added
- Fresh open-source launch of the **EVA** agent (Kernel-Evolving): a self-evolving,
  local-first AI agent with autonomous skill acquisition, Think-at-Rest idle
  reflection, multi-provider inference, and a Telegram-native control plane.
- Machine-specific paths, credentials, and internal working notes removed and
  replaced with environment variables (see `.env.example`).
- Native any-to-any support (Qwen2.5-Omni) and unified vision-language
  (Janus-Pro-7B) alongside the classic Nemotron-Diffusion and Gemma 4 E2B-it.
- ADR-driven architecture docs, Mermaid architecture diagram, and Hippocratic
  License HL3-LAW-MIL-SV.


> Note: historical pre-1.18.4 entries are available in git history and release notes.