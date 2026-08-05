"""
Minimal forward-only migration runner.

    python -m migrate           # apply against config.DATABASE_URL
    python migrate.py

Applies migrations/*.sql in filename order, once each, recording every applied
filename in schema_migrations. Idempotent: a second run is a no-op. Each
migration runs in its own transaction, so a failing migration rolls back
cleanly and leaves earlier ones applied.
"""

import psycopg

from library_rag import config

# Repo root, not package data: migrations are applied by an operator against a
# database, and keeping them visible at the top level is how anyone looking for
# "what is the schema" finds it.
MIGRATIONS_DIR = config.PROJECT_ROOT / "migrations"

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migration_files() -> list:
    return sorted(MIGRATIONS_DIR.glob("*.sql"), key=lambda p: p.name)


def applied_filenames(conn) -> set:
    conn.execute(ENSURE_TABLE_SQL)
    conn.commit()
    rows = conn.execute("SELECT filename FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def run(database_url: str = None) -> list:
    """Apply all pending migrations. Returns the list of filenames applied this
    run (empty if the schema was already up to date)."""
    database_url = database_url or config.DATABASE_URL
    newly_applied = []
    with psycopg.connect(database_url, autocommit=False) as conn:
        done = applied_filenames(conn)
        for path in migration_files():
            if path.name in done:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
            newly_applied.append(path.name)
            print(f"applied {path.name}")
    return newly_applied
