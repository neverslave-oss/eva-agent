# Architecture

Pipeline: observe -> plan -> act -> verify -> checkpoint.

- Orchestrator coordinates stages.
- Router selects driver strategy.
- Safety policy gate validates every action.

## Planners

Three planners share the same orchestrator interface (`plan()` → `ActionBatch`):

- **`Planner`** — deterministic baseline.
- **`LLMPlanner`** — vision-brain perceive→decide→act loop; streams each step's
  screenshot to Telegram via `watch_callback`.
- **`RLPlanner`** — tabular Q-learning. Perceive → discretize state (screen-hash /
  driver / outcome / progress / step / url buckets) → ε-greedy action over a
  Q-table → execute → verify → reward → Q-update. Fail-closed dead-state abort.
  Streams screenshots to Telegram via `watch_callback` (mirrors the LLM path).

## Progress & goal signals (RL)

- **`goal_met`** — injectable verifier (modality-agnostic); default never claims done.
- **`progress_met`** — general intrinsic signal: the observation signature (URL +
  text + screen hash) differs from the previous step. Works for browser AND desktop.

See `.specs/plans/2026-09-12-computer-use-rl-planner.md` for the RL design and
remaining-work list.
