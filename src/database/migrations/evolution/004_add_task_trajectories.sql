-- migrations/evolution/004_add_task_trajectories.sql
-- ADR-013: task_trajectories for fine-tuning dataset collection

CREATE TABLE IF NOT EXISTS task_trajectories (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    task           TEXT NOT NULL,
    provider       TEXT NOT NULL,
    model_name     TEXT,
    call_type      TEXT NOT NULL,
    tool_calls     TEXT NOT NULL,
    final_reply    TEXT NOT NULL,
    artifacts      TEXT,
    critic_score   REAL,
    critic_verdict TEXT,
    token_count    INTEGER,
    elapsed_s      REAL
);
