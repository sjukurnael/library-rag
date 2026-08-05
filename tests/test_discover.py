"""discover(): idempotent upsert, updates only on real change, PDF-only."""
from library_rag import db, ingest


def _fake_listing(files):
    """Return a list_children callable that ignores folder_id and yields
    `files`."""
    return lambda folder_id: list(files)


PDF_A = {"id": "a", "name": "Romans.pdf", "mimeType": "application/pdf",
         "size": "1048576", "md5Checksum": "aaa"}
PDF_B = {"id": "b", "name": "Genesis.pdf", "mimeType": "application/pdf",
         "size": "2097152", "md5Checksum": "bbb"}
FOLDER = {"id": "f", "name": "sub", "mimeType": "application/vnd.google-apps.folder"}
JUNK = {"id": "j", "name": "index.html", "mimeType": "text/html", "size": "10"}


def _rows(conn):
    return conn.execute(
        "SELECT source_id, title, md5, size_bytes, status FROM books ORDER BY id"
    ).fetchall()


def test_discover_inserts_only_pdfs(conn):
    n = ingest.discover(conn, "folder", _fake_listing([PDF_A, PDF_B, FOLDER, JUNK]))
    assert n == 2
    rows = _rows(conn)
    ids = {r[0] for r in rows}
    assert ids == {"a", "b"}
    assert all(r[4] == "discovered" for r in rows)
    assert dict((r[0], r[3]) for r in rows)["a"] == 1048576


def test_discover_is_idempotent_and_does_not_touch_updated_at(conn):
    ingest.discover(conn, "folder", _fake_listing([PDF_A]))
    before = conn.execute(
        "SELECT updated_at FROM books WHERE source_id = 'a'"
    ).fetchone()[0]

    ingest.discover(conn, "folder", _fake_listing([PDF_A]))  # identical -> no change
    after = conn.execute(
        "SELECT updated_at FROM books WHERE source_id = 'a'"
    ).fetchone()[0]
    assert before == after
    assert conn.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 1


def test_discover_updates_on_md5_change(conn):
    ingest.discover(conn, "folder", _fake_listing([PDF_A]))
    changed = {**PDF_A, "md5Checksum": "zzz", "size": "999"}
    ingest.discover(conn, "folder", _fake_listing([changed]))
    row = conn.execute(
        "SELECT md5, size_bytes FROM books WHERE source_id = 'a'"
    ).fetchone()
    assert row[0] == "zzz"
    assert row[1] == 999


def test_a_done_book_whose_bytes_changed_is_requeued(conn):
    """A new md5 means the indexed chunks describe a document that no longer
    exists. Leaving the book 'done' guarantees the index disagrees with Drive,
    and nothing in any status count reveals it."""
    ingest.discover(conn, "folder", _fake_listing([PDF_A]))
    conn.execute(
        "UPDATE books SET status='done', attempts=2, claimed_at=now() "
        "WHERE source_id='a'"
    )
    conn.commit()

    ingest.discover(conn, "folder", _fake_listing([{**PDF_A, "md5Checksum": "zzz"}]))

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
    ingest.discover(conn, "folder", _fake_listing([PDF_A]))
    conn.execute("UPDATE books SET status='done' WHERE source_id='a'")
    conn.commit()

    ingest.discover(conn, "folder", _fake_listing([{**PDF_A, "name": "Romans (2nd ed).pdf"}]))

    status, title = conn.execute(
        "SELECT status, title FROM books WHERE source_id='a'"
    ).fetchone()
    assert status == "done", "a rename should not requeue the book"
    assert title == "Romans (2nd ed).pdf", "the new title should still be recorded"
