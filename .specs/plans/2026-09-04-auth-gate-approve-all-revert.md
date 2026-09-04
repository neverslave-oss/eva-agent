# Plan — Auth-gate approve-all feature: regression, revert, and revisit

**Date:** 2026-09-04
**Status:** Feature reverted from `main`; working agent restored. Feature kept on `feat/exec-shell-approve-all` for future revisit.
**Branches:**
- `main` — working baseline + non-feature work, approve-all feature **stripped**.
- `feat/exec-shell-approve-all` — contains the approve-all feature + stop-on-3-denials (kept for future work).
- `debug/pre-auth-gate` — known-working baseline (`a60747a`).

---

## 1. Issue report (user DEBUG NOTES)

During the tools loop, `exec_shell` authorization requests were **not delivered to the user for approval**, and the **agent lost the conversation** (user messages not persisted). Reported as introduced by the `feat/exec-shell-approve-all` branch after merge.

### Symptoms observed
- Approval buttons did not appear in Telegram (or appeared intermittently).
- `exec_shell` returned `(authorization timed out — command blocked)`.
- Conversation turns were lost (user messages missing from `chat_history_evolving.db`; e.g. the DEBUG NOTES message had a gap at message id 1338).

---

## 2. Root cause analysis

### Primary trigger — cross-process approval timeout
- `request_auth()` runs in the **model_server** process (the tool loop); `resolve_auth()` runs in the **API/bot** process (Telegram callback handler).
- Each process has its **own in-memory `_pending` dict**, so a button tap in the bot process could not resolve the blocking request in the model_server process → every approval timed out.

### Secondary trigger — stop-on-3-denials (`f33cac8`)
- After `MAX_CONSECUTIVE_DENIES` (3) non-approvals, `request_stop()` sets a stop marker.
- With the cross-process timeout counting each timeout as a denial, after 3 timeouts the tool loop saw `is_stop_requested()` and aborted mid-turn → the conversation turn was not persisted → "the agent lost the conversation".

### Why unit tests did not catch it
- The test suite passes on both the working and broken branches (738 vs 707 passed) because the regression is a **runtime/integration** issue (cross-process auth + Telegram bot + conversation save) not covered by the unit tests.

---

## 3. Attempted fixes

### Attempt 1 — Cross-process file handoff (rejected)
Persisted each pending auth request to a tmp file (`_AUTH_DIR`) so `resolve_auth` (bot process) could reach `request_auth` (model_server process); `request_auth` polled the file.

**Result:** **Intermittent timeouts persisted.** Logs showed `resolve_auth` writing `"allow"` to the file, but `request_auth` not reliably consuming it (stale files remained, and `(authorization timed out — command blocked)` still occurred). The handoff was not reliable in practice.

**Lesson:** A file-based cross-process handoff for a blocking approval is fragile (race conditions, process mismatches, cleanup). Not acceptable for production.

### Attempt 2 — Restore working pre-auth-gate `auth_gate.py` (adopted)
Restored `src/core/auth_gate.py` to the **exact known-working version** (`a60747a`), which uses the **in-process `_pending` dict** with `event.wait()` + `resolve_auth` setting the event. This is the version where approval buttons appear and resolve reliably.

**Result:** Working agent restored. `auth_gate.py` is byte-identical to `a60747a`. 56 auth+tools tests pass. Stack restarted on this version.

---

## 4. Current state of `main`

- `auth_gate.py` — pre-auth-gate working version (in-process approval, Allow/Deny buttons, no approve-all, no stop-on-3-denials, no cross-process handoff).
- All other non-feature work kept: sensors control, AGENTS.md docs, dynamic native-tool injection, contextual self-docs, gitignore cleanup.
- Feature-specific code (approve-all button, `_auto_allow`, `_deny_count`, `request_stop`/`is_stop_requested`, `/stop` wiring) **removed** from `main`, retained on `feat/exec-shell-approve-all`.

---

## 5. Considerations for revisiting the feature

If we re-introduce approve-all / stop-on-3-denials in the future, we must address the root cause first:

1. **Process model:** Determine definitively whether `request_auth` and `resolve_auth` run in the same process in the deployed topology. If separate, a reliable cross-process handoff is REQUIRED before adding blocking features.
   - Options: (a) run the bot callback handler in the same process as the tool loop; (b) use a robust shared store (SQLite/Redis) with proper locking instead of tmp files; (c) make approval resolution non-blocking (queue the decision and let the tool loop check it).
2. **Stop-on-3-denials:** The rule must not abort the tool loop in a way that loses the conversation. Any stop path must still persist the user message + assistant response as a complete turn.
3. **Test coverage:** Add an **integration test** that exercises the full flow (tool loop → approval → button tap → resolution → conversation persisted), not just unit tests of `auth_gate`.
4. **Markdown/HTML delivery:** Approval buttons must be delivered reliably (HTML parse mode with escaped command text) so they always appear.

---

## 6. Next steps

- [x] Restore working `auth_gate.py` on `main`.
- [x] Restart kernel-evolving on the working version.
- [x] User confirms approval buttons appear and resolve in Telegram.
- [x] Apply HF Router timeout/retry fixes to `provider.py` (`_HF_TIMEOUT`=45s, `_HF_RETRIES`=2, `_hf_post()` helper).
- [x] Fix 7 provider test failures (root cause: `PROVIDER_TASK_INFERENCE=hf` env override; added `_clear_provider_env` autouse fixture to isolate tests).
- [x] Fix pre-existing test bug in `test_my_recent_thoughts_in_system_prompt` (wrong journal path — real journals live in `~/.kernel-evolving/workspace/thoughts/YYYY-MM-DD.md`).
- [x] All tests green (final full-suite run: 719 passed, 0 failed).
- [x] Tag `main` as `v1.1.0` (first minor release after the `v1.0.0` launch tag; `v1.0.0`/`v1.10.0`/`v1.30.x` tags in the local namespace belong to the separate `gitea-backup`/`kernel-evolving-backup` repo, not this one).
- [x] Push `main` + `v1.1.0` to remotes (origin, gitea, neverslave-oss) — force-pushed the fixed `main` (67d3060) over the broken `b866a82` that contained the approve-all feature commits (78d43ae, f33cac8, 36c2718).
- [ ] Clean up `fix/working-rebuild` and `debug/pre-auth-gate` branches once done.
- [ ] When revisiting the feature: implement a reliable cross-process handoff + conversation-safe stop + integration test per section 5.
