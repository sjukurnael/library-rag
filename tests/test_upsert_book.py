"""upsert_book(): idempotent on source_id, and the rules for what a re-upsert
is allowed to disturb.

These assertions used to live in test_discover.py and reach upsert_book through
ingest.discover(). discover() is gone -- books now reach the queue from the
mirror-backed producers (queue_drive_folder / queue_drive_files) and from
register_upload -- but every rule below still governs those callers, so the
coverage moved here rather than leaving with it.
"""
from library_rag import db


def _upsert(conn, *, md5="aaa", title="Romans.pdf", size=1048576):
    db.upsert_book(conn, "a", title, md5, size, source="drive")


def test_insert_lands_claimable(conn):
    _upsert(conn)
    source_id, title, md5, size_bytes, status = conn.execute(
        "SELECT source_id, title, md5, size_bytes, status FROM books"
    ).fetchone()
    assert (source_id, title, md5, size_bytes) == ("a", "Romans.pdf", "aaa", 1048576)
    # 'discovered' is the column default, and it is the whole handoff to the
    # worker: it is what claim_next_book will hand out.
    assert status == "discovered"


def test_is_idempotent_and_does_not_touch_updated_at(conn):
    _upsert(conn)
    before = conn.execute(
        "SELECT updated_at FROM books WHERE source_id = 'a'"
    ).fetchone()[0]

    _upsert(conn)  # identical -> no change
    after = conn.execute(
        "SELECT updated_at FROM books WHERE source_id = 'a'"
    ).fetchone()[0]
    assert before == after
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1


def test_updates_on_md5_change(conn):
    _upsert(conn)
    _upsert(conn, md5="zzz", size=999)
    row = conn.execute(
        "SELECT md5, size_bytes FROM books WHERE source_id = 'a'"
    ).fetchone()
    assert row[0] == "zzz"
    assert row[1] == 999


def test_a_done_book_whose_bytes_changed_is_requeued(conn):
    """A new md5 means the indexed chunks describe a document that no longer
    exists. Leaving the book 'done' guarantees the index disagrees with Drive,
    and nothing in any status count reveals it."""
    _upsert(conn)
    conn.execute(
        "UPDATE books SET status='done', attempts=2, claimed_at=now() "
        "WHERE source_id='a'"
    )
    conn.commit()

    _upsert(conn, md5="zzz")

    status, attempts, claimed = conn.execute(
        "SELECT status, attempts, claimed_at FROM books WHERE source_id='a'"
    ).fetchone()
    assert status == "discovered", f"edited book left as {status!r}"
    assert attempts == 0, "retry budget not reset for a genuinely new document"
    assert claimed is None
    assert db.claim_next_book(conn) is not None, "requeued book is not claimable"


def test_a_rename_alone_does_not_reprocess(conn):
    """A title change is metadata. Re-ingesting a 400-page book because someone
    renamed it in Drive would be a costly false positive."""
    _upsert(conn)
    conn.execute("UPDATE books SET status='done' WHERE source_id='a'")
    conn.commit()

    _upsert(conn, title="Romans (2nd ed).pdf")

    status, title = conn.execute(
        "SELECT status, title FROM books WHERE source_id='a'"
    ).fetchone()
    assert status == "done", "a rename should not requeue the book"
    assert title == "Romans (2nd ed).pdf", "the new title should still be recorded"
