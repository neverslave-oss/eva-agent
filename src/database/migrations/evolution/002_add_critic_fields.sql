-- migrations/evolution/002_add_critic_fields.sql
-- ADR-006/007/010: add columns to evolution_events and synthesis_trajectories
-- SQLite does not support IF NOT EXISTS on ALTER TABLE, so each migration
-- file is applied only once (tracked by _schema_version table).

ALTER TABLE evolution_events ADD COLUMN verification_result    TEXT;
ALTER TABLE evolution_events ADD COLUMN verification_reasoning TEXT;
ALTER TABLE evolution_events ADD COLUMN recommendations        TEXT;
ALTER TABLE evolution_events ADD COLUMN event_type             TEXT;
ALTER TABLE evolution_events ADD COLUMN entity_name            TEXT;
ALTER TABLE evolution_events ADD COLUMN meta                   TEXT;

ALTER TABLE synthesis_trajectories ADD COLUMN critic_score   REAL;
ALTER TABLE synthesis_trajectories ADD COLUMN critic_issues  TEXT;
ALTER TABLE synthesis_trajectories ADD COLUMN suggested_name TEXT;
