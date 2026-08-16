-- migrations/evolution/003_add_recommendation_hits.sql
-- RecommendationStore table (ADR-007)

CREATE TABLE IF NOT EXISTS recommendation_hits (
    skill_name TEXT PRIMARY KEY,
    hit_count  INTEGER NOT NULL DEFAULT 0,
    last_seen  TEXT
);
