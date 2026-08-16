-- migrations/memory/001_initial_schema.sql
-- Chat history: sessions, messages

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    chat_id       TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bot        TEXT    NOT NULL DEFAULT 'kernel-evolving',
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bot_session ON messages(bot, session_id, id);
CREATE INDEX IF NOT EXISTS idx_bot_created ON messages(bot, created_at);
