-- migrations/agent/001_initial_schema.sql
-- Agent-domain tables: failed_requests, promoted_signals, probes, prompt_log

CREATE TABLE IF NOT EXISTS failed_requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id          TEXT    NOT NULL,
    ts               TEXT    NOT NULL,
    user_message     TEXT    NOT NULL,
    agent_response   TEXT    NOT NULL,
    failure_type     TEXT    NOT NULL,
    resolved         INTEGER NOT NULL DEFAULT 0,
    anchor_task      TEXT,
    request_hash     TEXT
);
CREATE INDEX IF NOT EXISTS idx_failed_requests_chat ON failed_requests(chat_id);
CREATE INDEX IF NOT EXISTS idx_failed_requests_ts   ON failed_requests(ts);

CREATE TABLE IF NOT EXISTS promoted_signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern      TEXT    NOT NULL,
    count        INTEGER NOT NULL DEFAULT 1,
    first_seen   TEXT    NOT NULL,
    last_seen    TEXT    NOT NULL,
    promoted     INTEGER NOT NULL DEFAULT 0,
    promoted_at  TEXT
);

CREATE TABLE IF NOT EXISTS probes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subject      TEXT    NOT NULL,
    thought_text TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    score        REAL    NOT NULL,
    created_at   TEXT    NOT NULL,
    expires_at   TEXT    NOT NULL,
    triggered_at TEXT,
    triggered_by TEXT,
    resolved     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prompt_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    chat_id      TEXT    NOT NULL DEFAULT '',
    prompt_len   INTEGER NOT NULL,
    prompt       TEXT    NOT NULL,
    history      TEXT    NOT NULL DEFAULT '[]',
    user_message TEXT    NOT NULL DEFAULT '',
    provider     TEXT    NOT NULL DEFAULT '',
    model        TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_prompt_log_ts      ON prompt_log (ts);
CREATE INDEX IF NOT EXISTS idx_prompt_log_chat_id ON prompt_log (chat_id);
