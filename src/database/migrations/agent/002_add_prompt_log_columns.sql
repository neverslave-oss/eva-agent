-- migrations/agent/002_add_prompt_log_columns.sql
-- Additional columns added to prompt_log by prompt_logger.py ALTER TABLE pattern
-- (already included in 001 but kept as an idempotent guard for older DBs)

-- Nothing to add: history and user_message were included in 001_initial_schema.sql.
-- This file intentionally empty — placeholder for future prompt_log schema additions.
SELECT 1;
