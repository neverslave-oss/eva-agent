"""
database/base.py — BaseRepository ABC with migration support.

All repository classes inherit from BaseRepository. They get:
- A `connection()` factory with WAL mode + foreign keys enabled.
- An abstract `_ensure_schema()` that must be implemented.
- A `migrate()` helper that runs numbered .sql files from a directory,
  tracking applied versions in a `_schema_version` table.
"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class BaseRepository(ABC):
    """Abstract base for all SQLite repository classes."""

    # Subclasses must set this (or pass it to __init__)
    db_path: Path

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connection(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection with recommended pragmas."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @abstractmethod
    def _ensure_schema(self) -> None:
        """Create tables / indexes for this repository.
        Called once during __init__ by concrete subclasses."""

    def migrate(self, migration_dir: Path) -> None:
        """Run numbered .sql migration files from *migration_dir*.

        Files must be named ``NNN_description.sql`` (e.g. ``001_initial_schema.sql``).
        Applied migrations are tracked in the ``_schema_version`` table so that
        re-running migrate() is idempotent.

        Args:
            migration_dir: Directory containing numbered ``.sql`` files.
        """
        conn = self.connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _schema_version (
                    migration   TEXT PRIMARY KEY,
                    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

            already_applied: set[str] = {
                row[0]
                for row in conn.execute("SELECT migration FROM _schema_version").fetchall()
            }

            migration_files = sorted(
                f for f in Path(migration_dir).glob("*.sql")
            )

            for sql_file in migration_files:
                name = sql_file.name
                if name in already_applied:
                    continue
                sql_text = sql_file.read_text(encoding="utf-8")
                # Execute statements one by one so ALTER TABLE errors are ignorable
                statements = [s.strip() for s in sql_text.split(";") if s.strip()]
                for stmt in statements:
                    try:
                        conn.execute(stmt)
                    except Exception as e:
                        err = str(e).lower()
                        # Tolerate "already exists" errors — idempotent DDL
                        if any(kw in err for kw in ("already exists", "duplicate column")):
                            pass
                        else:
                            raise
                conn.execute(
                    "INSERT INTO _schema_version (migration) VALUES (?)", (name,)
                )
                conn.commit()
        finally:
            conn.close()


class MigrationRunner:
    """Standalone helper to run migrations against an arbitrary db_path.

    Useful for CLI scripts or tests that don't want a full repository instance.

    Example::

        runner = MigrationRunner(db_path, migration_dir)
        runner.run()
    """

    def __init__(self, db_path: Path | str, migration_dir: Path | str) -> None:
        self.db_path = Path(db_path)
        self.migration_dir = Path(migration_dir)

    def run(self) -> None:
        """Apply all pending migrations."""

        class _Repo(BaseRepository):
            def _ensure_schema(self) -> None:
                pass

        repo = _Repo(self.db_path)
        repo.migrate(self.migration_dir)
