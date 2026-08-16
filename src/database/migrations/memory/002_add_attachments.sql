-- migrations/memory/002_add_attachments.sql
-- Attachments table for chat history

CREATE TABLE IF NOT EXISTS attachments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bot           TEXT    NOT NULL DEFAULT 'kernel-evolving',
    session_id    TEXT    NOT NULL,
    chat_id       TEXT    NOT NULL DEFAULT '',
    kind          TEXT    NOT NULL CHECK(kind IN ('document','photo','voice','audio','video','other')),
    local_path    TEXT    NOT NULL,
    original_name TEXT    NOT NULL DEFAULT '',
    mime_type     TEXT    NOT NULL DEFAULT '',
    caption       TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attachments_session ON attachments(bot, session_id);
CREATE INDEX IF NOT EXISTS idx_attachments_chat    ON attachments(bot, chat_id);
