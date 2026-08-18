# Fix Spec — AGENTS.md Core Context Overwritten by Session Learnings

**Date:** 2026-08-18
**Status:** in-progress (audit complete, fix not yet implemented)
**Branch:** `fix/agent-identity-override` (from `dev`)
**Related:** `IDENTITY-INJECTION-UTILITY-PROMPT-CONFLICT-2026-08-06.md`, `ADR-022`

## Problem

The core identity file `~/.kernel-evolving/workspace/AGENTS.md` was **overwritten and lost
all its template content**. It now contains **only** accumulated "Session learnings" sections
(40 lines, zero template markers), so the agent no longer receives its core persona,
architecture, tool reference, and workflow rules on every inference.

The repository template (`AGENTS.md` at repo root, identical copy in
`src/assets/agent-templates/AGENTS.md`) is intact — only the **workspace runtime copy** is
corrupted.

## Evidence

- `~/.kernel-evolving/workspace/AGENTS.md` → 40 lines, **0** matches for template markers
  (`Goals and personality`, `How a request flows`, `Who you are`, `Architecture reference`).
- Repo root `AGENTS.md` ≡ `src/assets/agent-templates/AGENTS.md` (diff shows IDENTICAL).
- Workspace file mtime: **2026-08-06 17:35** — corruption predates the recent log window.
- `/tmp/kernel_evolving_api.log` (2026-08-17 → 2026-08-18): no `[IdentityConsolidator]`
  append lines, no `write_file` to AGENTS.md in the window — the overwrite happened earlier,
  outside this log window.
- `docs/ADR-022` documents a **manual rewrite** of AGENTS.md on 2026-07-03 with "corrective
  learnings" — the first known removal of template content.

## Root Cause (three compounding vulnerabilities)

1. **`write_file` can overwrite the identity file — no protection.**
   `src/core/tools.py` `write_file` allows writing anywhere under `~/.kernel-evolving`
   (the workspace guard only redirects paths *outside* that prefix). The model can therefore
   call `write_file('~/.kernel-evolving/workspace/AGENTS.md', ...)` and replace the whole
   file with arbitrary content (e.g. a "learnings" write, an identity edit, or a corrupted
   dump). There is no protected-files list for identity files.

2. **`refresh_identity_files()` cannot repair a corrupted file.**
   `src/infra/setup.py` `_copy_file_if_absent()` copies the template to the workspace only
   when the destination is **absent or ≤ 20 chars**. Once the runtime AGENTS.md has *any*
   content (even corrupted session-learnings-only content), the template is never restored
   on boot — the corruption is permanent until manually fixed.

3. **The consolidator appends (correctly) but to whatever is there.**
   `src/services/thought_engine.py` `_run_identity_consolidator()` correctly **appends**
   `## Session learnings (date)` sections (verified by `test_think_at_rest_memory.py`). But
   it appends to the existing file, so once the template is gone the file just keeps growing
   with learnings and the identity context stays lost.

**Net effect:** identity template lost → `_load_agents_md()` in `context.py` injects only the
session-learnings persona → agent loses core context → degraded behaviour (as seen in the
recent live tests).

## Goal

Ensure the core identity template is **always present and never overwritten by session
learnings**, while still allowing the agent to accumulate session learnings in an
**append-only** fashion.

## Proposed Fix

### 1. Templates live in `src/assets/agent-templates/` (reference copies)

- Keep `AGENTS.md` + `SOUL.md` as the canonical reference templates under
  `src/assets/agent-templates/`.
- **`setup.py` must copy from `src/assets/agent-templates/`**, NOT from the repo root. The
  repo-root `AGENTS.md` is the agent file used to work on the repository itself and must not
  be treated as the Kernel-Evolving runtime identity.
- Update `_REPO_ROOT`-based source resolution in `setup.py` to point at
  `src/assets/agent-templates/AGENTS.md` / `SOUL.md`.

### 2. Self-healing `refresh_identity_files()` — restore template, preserve learnings

- Add a **repair step** that detects when the runtime AGENTS.md has lost its identity
  template (e.g. missing a sentinel marker like `## How a request flows` or `## Who you are`).
- On detection: **merge** — write the full template, then re-append any existing
  `## Session learnings` sections that were present, so no learnings are lost.
- This makes the fix idempotent and safe on every boot (never overwrites valid content).

### 3. Protected identity files in `write_file`

- Add `~/.kernel-evolving/workspace/AGENTS.md` (and `SOUL.md`, `IDENTITY.md`) to a
  **protected-files list** in the `write_file` tool. A model write to these paths is rejected
  (or redirected to a safe location) so the identity file can never be clobbered by the model.

### 4. Append-only session learnings (defense in depth)

- Keep `_run_identity_consolidator()` appending to a **dedicated `## Session learnings`
  section** at the end of AGENTS.md, but make the append robust:
  - Always anchor the append to the end of the file.
  - Ensure the template portion (before the learnings anchor) is never replaced by the
    consolidator.
  - Optionally cap the number of learnings sections to bound growth.

## Files Affected

- `src/infra/setup.py` — copy template from `src/assets/agent-templates/`; add repair/merge
  logic in `refresh_identity_files()`.
- `src/core/tools.py` — `write_file` protected-files list for identity files.
- `src/services/thought_engine.py` — harden the append (anchor + template-preservation).
- `src/core/memory/context.py` — (verify) `_load_agents_md()` still injects the restored
  template persona.
- `tests/test_setup.py`, `tests/test_think_at_rest_memory.py`, `tests/test_all_native_tools.py`
  — add/adjust tests for repair + protected write.
- This spec: `.specs/fixes/FIX-AGENT-OVERRIDE-ISSUE-2026-08-18.md`.

## Success Criteria

- [ ] `refresh_identity_files()` on a corrupted workspace AGENTS.md restores the full template
      **and** preserves existing session learnings (merge, no data loss).
- [ ] `setup.py` copies from `src/assets/agent-templates/` (repo-root AGENTS.md no longer used
      as the runtime source).
- [ ] `write_file` refuses to write to AGENTS.md / SOUL.md / IDENTITY.md (returns a clear
      protected-path message).
- [ ] `_run_identity_consolidator()` appends learnings without ever removing the template.
- [ ] Existing tests pass; new tests cover the repair-merge and protected-write paths.

## Non-goals

- Not changing the general `write_file` workspace guard (paths outside `~/.kernel-evolving`
  stay redirected to tmp).
- Not moving session learnings out of AGENTS.md into a separate file (out of scope unless
  requested) — the merge approach keeps them in place but append-only.
- Not re-architecting the identity injection pipeline (covered by the separate
  `IDENTITY-INJECTION-UTILITY-PROMPT-CONFLICT` fix).

## Status

[ ] Not started
