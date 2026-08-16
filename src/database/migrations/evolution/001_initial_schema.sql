-- migrations/evolution/001_initial_schema.sql
-- Core evolution tables

CREATE TABLE IF NOT EXISTS evolution_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    task          TEXT NOT NULL,
    found         INTEGER NOT NULL,
    installed     TEXT,
    confidence    TEXT,
    retry         INTEGER,
    escalated     INTEGER,
    provider_used TEXT,
    gap           TEXT,
    resolved      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS synthesis_trajectories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    task          TEXT NOT NULL,
    gap           TEXT,
    prompt        TEXT,
    output_skill  TEXT,
    output_script TEXT,
    validation    TEXT,
    provider      TEXT,
    skill_name    TEXT
);
