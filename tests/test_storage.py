"""
The storage mirror and the PDF-viewer route.

Supabase itself is never contacted: conftest blanks the SUPABASE_* config, so
storage.enabled() is False by default, and every test that wants storage
behaviour monkeypatches the module functions -- the assertions here are about
OUR wiring (when things are mirrored, when the shared object may be deleted,
which source the route serves), not about Supabase's API.
"""
import pytest
from fastapi.testclient import TestClient

from library_rag import config, db, ingest, storage
from library_rag.web import api

# ------------------------------------------------------------- unit level --

def test_storage_is_a_noop_when_unconfigured(tmp_path):
    """With no SUPABASE_* config (the test default), every call is a cheap
    False/None -- no client is built, nothing raises."""
    assert storage.enabled() is False
    assert storage.put_original("abc", tmp_path / "x.pdf") is False
    assert storage.delete_original("abc") is False
    assert storage.signed_original_url("abc") is None


def test_signed_url_relative_path_is_absolutised(monkeypatch):
    monkeypatch.setattr(config, "SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setattr(config, "SUPABASE_SERVICE_KEY", "sb_secret_x")

    class FakeBucket:
        def create_signed_url(self, key, expires):
            return {"signedURL": f"/object/sign/library-bucket/{key}?token=t"}

    monkeypatch.setattr(storage, "_bucket", lambda: FakeBucket())
    url = storage.signed_original_url("abc")
    assert url == (
        "https://proj.supabase.co/storage/v1"
        "/object/sign/library-bucket/originals/abc.pdf?token=t"
    )


# -------------------------------------------------------- purge semantics --

def _register(conn, tmp_path, content: bytes, name: str):
    src = tmp_path / name
    src.write_bytes(content)
    return ingest.register_upload(conn, src, title=name)


def test_purge_deletes_the_object_with_the_last_reference(conn, tmp_path, monkeypatch):
    deleted = []
    monkeypatch.setattr(storage, "delete_original", lambda md5: deleted.append(md5) or True)

    book = _register(conn, tmp_path, b"%PDF-1.4 unique-bytes", "a.pdf")
    ingest.purge_book(conn, book["id"])
    assert deleted == [book["md5"]]


def test_purge_keeps_the_object_while_a_twin_row_remains(conn, tmp_path, monkeypatch):
    """Same bytes known under two rows (e.g. once from Drive, once uploaded):
    purging one row must NOT delete the shared bucket object."""
    deleted = []
    monkeypatch.setattr(storage, "delete_original", lambda md5: deleted.append(md5) or True)

    book = _register(conn, tmp_path, b"%PDF-1.4 twin-bytes", "a.pdf")
    # A second row with the same md5, as Drive discovery would create it.
    db.upsert_book(conn, "drivefile123", "a (drive).pdf", book["md5"], 21, source="drive")

    ingest.purge_book(conn, book["id"])
    assert deleted == [], "object deleted while another book still references it"


# ---------------------------------------------------------- viewer route --

@pytest.fixture
def client(conn, test_database_url, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    return TestClient(api.app)


def test_pdf_route_redirects_to_a_signed_url_when_storage_has_it(
    client, conn, tmp_path, monkeypatch
):
    book = _register(conn, tmp_path, b"%PDF-1.4 stored", "s.pdf")
    monkeypatch.setattr(
        storage, "signed_original_url",
        lambda md5, **kw: f"https://bucket.example/{md5}.pdf",
    )
    r = client.get(f"/api/books/{book['id']}/pdf", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"https://bucket.example/{book['md5']}.pdf"


def test_pdf_route_serves_the_local_original_inline(client, conn, tmp_path):
    """No storage (the default): an upload's original streams from disk, as a
    browser-viewable PDF rather than a download."""
    book = _register(conn, tmp_path, b"%PDF-1.4 local", "l.pdf")
    r = client.get(f"/api/books/{book['id']}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "inline" in r.headers.get("content-disposition", "")
    assert r.content == b"%PDF-1.4 local"


def test_pdf_route_404s_helpfully_when_the_bytes_are_gone(client, conn, tmp_path):
    book = _register(conn, tmp_path, b"%PDF-1.4 doomed", "d.pdf")
    ingest.upload_path(book["source_id"]).unlink()
    r = client.get(f"/api/books/{book['id']}/pdf")
    assert r.status_code == 404
    assert "search still works" in r.json()["detail"]


def test_pdf_route_404s_for_an_unknown_book(client):
    assert client.get("/api/books/999999/pdf").status_code == 404


# ------------------------------------------------------- ingest mirroring --

def test_process_book_mirrors_the_verified_bytes(conn, tmp_path, monkeypatch):
    """The mirror call happens after download, keyed by the book's md5."""
    put = []
    monkeypatch.setattr(storage, "put_original", lambda md5, path: put.append(md5) or True)

    book = _register(conn, tmp_path, b"%PDF-1.4 mirror-me", "m.pdf")
    claimed = db.claim_next_book(conn, "upload")

    class FailFast(Exception):
        pass

    # Stop the pipeline right after the download stage: extraction and
    # embedding are other tests' business.
    monkeypatch.setattr(
        ingest.extract_mod, "extract",
        lambda *a, **k: (_ for _ in ()).throw(FailFast()),
    )
    with pytest.raises(FailFast):
        ingest.process_book(
            conn, claimed,
            download_file=ingest.downloader_for(claimed, None),
            ocr_client=None, voyage_client=None,
        )
    assert put == [book["md5"]]
