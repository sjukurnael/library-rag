"""claim_next_book(): SKIP LOCKED gives distinct claims; the reaper reclaims
stale books; attempts past the cap fail instead of looping forever."""
from concurrent.futures import ThreadPoolExecutor

import db


def _seed(conn, n, status="discovered"):
    for i in range(n):
        conn.execute(
            "INSERT INTO books (drive_file_id, title, status) VALUES (%s, %s, %s)",
            (f"file-{i}", f"Book {i}", status),
        )
    conn.commit()


def test_concurrent_claims_are_distinct(conn, test_database_url):
    n = 8
    _seed(conn, n)

    def claim_one(_):
        # Each thread needs its own connection for SKIP LOCKED to apply.
        with db.get_conn(test_database_url) as c:
            row = db.claim_next_book(c)
            return row["id"] if row else None

    with ThreadPoolExecutor(max_workers=n) as pool:
        claimed = list(pool.map(claim_one, range(n)))

    got = [c for c in claimed if c is not None]
    assert len(got) == n
    assert len(set(got)) == n  # no book claimed twice


def test_fresh_and_stale_claims(conn):
    # Fresh, never-claimed book is claimable.
    _seed(conn, 1)
    row = db.claim_next_book(conn)
    assert row is not None
    assert row["attempts"] == 1

    # Immediately re-claiming finds nothing: it's freshly claimed, not stale.
    assert db.claim_next_book(conn) is None

    # Backdate the claim past the stale window -> reclaimable, attempts bumps.
    conn.execute(
        "UPDATE books SET claimed_at = now() - interval '31 minutes' WHERE id = %s",
        (row["id"],),
    )
    conn.commit()
    reclaimed = db.claim_next_book(conn)
    assert reclaimed is not None
    assert reclaimed["id"] == row["id"]
    assert reclaimed["attempts"] == 2


def test_heartbeat_keeps_a_slow_book_from_being_stolen(conn):
    """claimed_at is a lease, not a lock. Without a heartbeat it expires on a
    fixed timer, so a book slower than the window is taken from a live worker
    and processed twice."""
    _seed(conn, 1)
    mine = db.claim_next_book(conn)

    # Simulate a long stage: the claim ages past the stale window.
    conn.execute(
        "UPDATE books SET claimed_at = now() - interval '6 minutes' WHERE id = %s",
        (mine["id"],),
    )
    conn.commit()
    assert db.claim_next_book(conn) is not None, "stale claim should be reclaimable"

    # Now do it again, but heartbeat before the window elapses.
    conn.execute(
        "UPDATE books SET claimed_at = now() - interval '4 minutes', attempts = 1 "
        "WHERE id = %s",
        (mine["id"],),
    )
    conn.commit()
    db.touch_claim(conn, mine["id"])
    assert db.claim_next_book(conn) is None, "heartbeat did not protect the claim"


def test_attempts_over_cap_are_failed(conn):
    _seed(conn, 1)
    # Already at the cap, and stale so it would otherwise be claimed.
    conn.execute(
        "UPDATE books SET attempts = %s, "
        "claimed_at = now() - interval '31 minutes'",
        (db.config.MAX_ATTEMPTS,),
    )
    conn.commit()

    row = db.claim_next_book(conn)
    assert row is not None
    assert row["status"] == "failed"
    assert row["error"] is not None

    # Nothing processable remains.
    assert db.claim_next_book(conn) is None
