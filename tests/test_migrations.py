"""The migration runner applies each file once and is a no-op on re-run."""
import psycopg

import migrate


def _table_exists(url, name):
    with psycopg.connect(url) as c:
        row = c.execute("SELECT to_regclass(%s)", (name,)).fetchone()
        return row[0] is not None


def test_migrations_apply_once_then_noop(fresh_database):
    # First run applies 0001 (and any later migrations); second run applies none.
    first = migrate.run(fresh_database)
    assert "0001_init.sql" in first

    second = migrate.run(fresh_database)
    assert second == []

    # Schema is really there, and each filename is recorded exactly once.
    assert _table_exists(fresh_database, "books")
    assert _table_exists(fresh_database, "chunks")
    with psycopg.connect(fresh_database) as c:
        rows = c.execute(
            "SELECT filename, COUNT(*) FROM schema_migrations GROUP BY filename"
        ).fetchall()
    counts = dict(rows)
    assert counts["0001_init.sql"] == 1
