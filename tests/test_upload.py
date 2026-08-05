"""
The upload source: registering a PDF, and the HTTP endpoint in front of it.

A book can now arrive from Drive or from a user's disk. What these tests pin is
that the two sources differ in exactly ONE place -- how the bytes arrive --
and that everything downstream (the queue, the claim, the pipeline) does not
know or care which it got.

The endpoint tests never let ingestion actually run: _drain_queue is replaced
with a recorder, so what is asserted is that the upload was ACCEPTED and
QUEUED. Whether the pipeline then works is what the rest of the suite is for.
"""
import io

import pytest
from fastapi.testclient import TestClient

import api
import config
import db
import ingest

MINIMAL_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def _pdf(body: bytes = b"some book text") -> bytes:
    return MINIMAL_PDF + body


@pytest.fixture
def client(conn, test_database_url, monkeypatch):
    """A TestClient whose requests hit the throwaway test database.

    api.py calls db.get_conn() with no argument, which reads
    config.DATABASE_URL at call time -- so pointing that at the test database
    is enough to keep the endpoint off the real one.
    """
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    monkeypatch.setattr(config, "VOYAGE_API_KEY", "test-key-not-used")
    drained = []
    monkeypatch.setattr(api, "_drain_queue", lambda: drained.append(True))
    c = TestClient(api.app)
    c.drained = drained
    return c


# ------------------------------------------------------- register_upload --

def test_an_uploaded_book_is_stored_content_addressed_and_queued(conn, tmp_path):
    src = tmp_path / "Some Book.pdf"
    src.write_bytes(_pdf())

    book = ingest.register_upload(conn, src)

    assert book["source"] == "upload"
    assert book["source_id"].startswith("upload:")
    assert book["title"] == "Some Book.pdf"
    # 'discovered' is what claim_next_book hands out -- an upload joins the same
    # queue a Drive book does rather than getting its own scheduling path.
    assert book["status"] == "discovered"
    assert db.claim_next_book(conn) is not None, "upload did not enter the queue"

    stored = ingest.upload_path(book["source_id"])
    assert stored.exists(), "the original was not kept"
    assert stored.read_bytes() == src.read_bytes()
    assert stored.parent == config.UPLOAD_DIR


def test_reuploading_identical_bytes_is_the_same_book(conn, tmp_path):
    """Content-addressed identity. The same document under a different filename
    is the same document, and must not be embedded a second time."""
    a = tmp_path / "first.pdf"
    b = tmp_path / "renamed-copy.pdf"
    a.write_bytes(_pdf())
    b.write_bytes(_pdf())

    first = ingest.register_upload(conn, a)
    second = ingest.register_upload(conn, b)

    assert first["id"] == second["id"]
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 1


def test_a_finished_book_is_not_requeued_by_an_identical_reupload(conn, tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(_pdf())
    book = ingest.register_upload(conn, src)
    conn.execute("UPDATE books SET status = 'done' WHERE id = %s", (book["id"],))
    conn.commit()

    again = ingest.register_upload(conn, src)
    assert again["status"] == "done", "identical bytes should not reopen a done book"


def test_different_bytes_are_a_different_book_even_with_the_same_name(conn, tmp_path):
    """The inverse guarantee: one upload can never silently replace another
    just because a user happened to name two files the same thing."""
    src = tmp_path / "book.pdf"
    src.write_bytes(_pdf(b"first edition"))
    first = ingest.register_upload(conn, src)

    src.write_bytes(_pdf(b"second edition, revised"))
    second = ingest.register_upload(conn, src)

    assert first["id"] != second["id"]
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 2


def test_upload_path_is_derived_from_the_hash_not_the_title(tmp_path):
    """The filename a client sends never reaches the filesystem, so there is
    nothing to traverse out of."""
    path = ingest.upload_path("upload:abc123def")
    assert path == config.UPLOAD_DIR / "abc123def.pdf"


# ---------------------------------------------------------- source split --

def test_downloader_dispatches_on_source(conn, tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(_pdf())
    uploaded = ingest.register_upload(conn, src)

    calls = []
    drive_download = lambda sid, dest: calls.append((sid, dest))  # noqa: E731

    # An upload copies from UPLOAD_DIR and never touches Drive.
    dest = tmp_path / "copied.pdf"
    ingest.downloader_for(uploaded, drive_download)(uploaded["source_id"], dest)
    assert dest.read_bytes() == src.read_bytes()
    assert calls == [], "an upload reached the Drive downloader"

    # A Drive book goes the other way.
    ingest.downloader_for({"source": "drive"}, drive_download)("file-id", "/tmp/x.pdf")
    assert calls == [("file-id", "/tmp/x.pdf")]


def test_a_missing_original_fails_loudly(conn, tmp_path):
    """The book row can outlive its file. That must be an error naming the
    path, not a zero-byte PDF that fails three stages later as 'no chunks'."""
    src = tmp_path / "book.pdf"
    src.write_bytes(_pdf())
    book = ingest.register_upload(conn, src)
    ingest.upload_path(book["source_id"]).unlink()

    download = ingest.downloader_for(book, lambda *a: None)
    with pytest.raises(RuntimeError, match="uploaded original is missing"):
        download(book["source_id"], str(tmp_path / "out.pdf"))


# ---------------------------------------------------------------- HTTP --

def _post(client, name, content):
    return client.post(
        "/api/books/upload",
        files={"file": (name, io.BytesIO(content), "application/pdf")},
    )


def test_a_valid_pdf_is_accepted_and_processing_is_scheduled(client):
    r = _post(client, "Doctrine.pdf", _pdf())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"] == "Doctrine.pdf"
    assert body["status"] == "discovered"
    assert body["already_indexed"] is False
    assert client.drained == [True], "ingestion was never kicked off"


def test_a_file_that_is_not_a_pdf_is_rejected_by_its_bytes(client):
    """Named .pdf and declared application/pdf, but it is not one. Both of those
    are chosen by the client; the magic bytes are not."""
    r = _post(client, "malware.pdf", b"MZ\x90\x00 this is a windows executable")
    assert r.status_code == 400
    assert "not a PDF" in r.json()["detail"]
    assert client.drained == []


def test_a_non_pdf_extension_is_rejected(client):
    r = _post(client, "notes.txt", _pdf())
    assert r.status_code == 400
    assert "Only PDF" in r.json()["detail"]


def test_an_empty_file_is_rejected(client):
    r = _post(client, "empty.pdf", b"")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_an_oversize_upload_is_rejected(client, monkeypatch):
    """The cap is enforced while streaming, so this never has to allocate a
    real 64 MB to prove it."""
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    r = _post(client, "huge.pdf", _pdf(b"x" * 4096))
    assert r.status_code == 413
    assert client.drained == []


def test_reuploading_an_indexed_book_reports_it_instead_of_reprocessing(client, conn):
    first = _post(client, "book.pdf", _pdf())
    book_id = first.json()["id"]
    conn.execute("UPDATE books SET status = 'done' WHERE id = %s", (book_id,))
    conn.commit()
    client.drained.clear()

    again = _post(client, "book.pdf", _pdf())
    assert again.json()["already_indexed"] is True
    assert client.drained == [], "an already-indexed book was queued for reprocessing"


def test_books_endpoint_separates_searchable_from_in_flight(client, conn):
    _post(client, "queued.pdf", _pdf(b"still working"))
    d = client.get("/api/books").json()

    assert d["books"] == [], "a book with no chunks must not look searchable"
    assert [b["title"] for b in d["pending"]] == ["queued.pdf"]
    assert d["pending"][0]["source"] == "upload"


def test_a_failed_book_stays_visible_with_its_error(client, conn):
    r = _post(client, "broken.pdf", _pdf())
    conn.execute(
        "UPDATE books SET status = 'failed', error = 'extraction produced no chunks' "
        "WHERE id = %s",
        (r.json()["id"],),
    )
    conn.commit()

    pending = client.get("/api/books").json()["pending"]
    assert pending[0]["status"] == "failed"
    assert "no chunks" in pending[0]["error"], (
        "a failed upload that disappears tells the user it succeeded"
    )


# -------------------------------------------------------------- delete --

def test_purge_removes_the_row_its_chunks_and_its_files(conn, tmp_path):
    src = tmp_path / "book.pdf"
    src.write_bytes(_pdf())
    book = ingest.register_upload(conn, src)
    book_id = book["id"]

    # Stand in for what a real run leaves behind.
    original = ingest.upload_path(book["source_id"])
    (config.PDF_DIR / f"{book_id}.pdf").write_bytes(_pdf())
    (config.MARKDOWN_DIR / f"{book_id}.md").write_text("# extracted")
    (config.MARKDOWN_DIR / f"{book_id}.manifest.json").write_text("{}")
    db.insert_chunks_and_finish(conn, book_id, [{
        "ordinal": 0, "heading_trail": None, "page_start": 1, "page_end": 2,
        "content": "body", "token_count": 2, "embedding": [0.0] * config.EMBED_DIM,
    }])
    assert conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0] == 1

    result = ingest.purge_book(conn, book_id)

    assert result["book"]["id"] == book_id
    assert conn.execute("SELECT count(*) FROM books WHERE id=%s", (book_id,)).fetchone()[0] == 0
    assert conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0] == 0, "chunks outlived their book and still answer searches"
    assert not original.exists(), "the uploaded original was left on disk"
    assert not (config.PDF_DIR / f"{book_id}.pdf").exists()
    assert not (config.MARKDOWN_DIR / f"{book_id}.md").exists()
    assert not (config.MARKDOWN_DIR / f"{book_id}.manifest.json").exists()
    assert len(result["removed"]) == 4


def test_purging_a_drive_book_never_touches_an_upload_path(conn):
    """A Drive source_id has no "upload:" prefix, so treating it as one would
    build a nonsense path -- and Drive still holds the bytes either way."""
    conn.execute(
        "INSERT INTO books (source_id, title, source, status) "
        "VALUES ('drive-abc', 'From Drive.pdf', 'drive', 'done')"
    )
    conn.commit()
    book_id = conn.execute(
        "SELECT id FROM books WHERE source_id = 'drive-abc'"
    ).fetchone()[0]

    result = ingest.purge_book(conn, book_id)
    assert result["book"]["source"] == "drive"
    assert result["removed"] == []


def test_purging_an_unknown_book_is_not_an_error(conn):
    assert ingest.purge_book(conn, 999999) is None


def test_delete_endpoint_removes_the_book(client, conn):
    r = _post(client, "unwanted.pdf", _pdf())
    book_id = r.json()["id"]

    d = client.delete(f"/api/books/{book_id}")
    assert d.status_code == 200, d.text
    assert d.json()["title"] == "unwanted.pdf"
    assert conn.execute(
        "SELECT count(*) FROM books WHERE id = %s", (book_id,)
    ).fetchone()[0] == 0


def test_deleting_an_unknown_book_is_a_404(client):
    assert client.delete("/api/books/999999").status_code == 404


def test_a_deleted_book_can_be_uploaded_again(client, conn):
    """Content-addressing must not make a deletion permanent: the same bytes
    were the same book, so re-uploading has to create a fresh one rather than
    matching a row that no longer exists."""
    first = _post(client, "book.pdf", _pdf()).json()["id"]
    client.delete(f"/api/books/{first}")

    second = _post(client, "book.pdf", _pdf())
    assert second.status_code == 200
    assert second.json()["id"] != first
    assert second.json()["already_indexed"] is False
