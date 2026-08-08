"""
The storage mirror and the PDF-viewer route.

Supabase itself is never contacted: conftest blanks the SUPABASE_* config, so
storage.enabled() is False by default, and every test that wants storage
behaviour monkeypatches the module functions -- the assertions here are about
OUR wiring (when things are mirrored, when the shared object may be deleted,
which source the route serves), not about Supabase's API.
"""
import json
import types

import pytest
from fastapi.testclient import TestClient

from library_rag import config, db, ingest, storage
from library_rag.pipeline import extract
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


# ------------------------------------------- the local copy, after mirroring --
#
# process_book now DELETES local files once the bucket confirms them, which is
# the only place in this project that removes a user's data. What follows pins
# the rule down: a file goes only after its own bytes are known to be up, and
# reads fall back to the bucket so a machine that never extracted a book can
# still rechunk it.


@pytest.fixture
def fake_bucket(monkeypatch):
    """An in-memory stand-in for the bucket.

    Storage is OFF by default here (conftest blanks SUPABASE_*), so anything
    exercising the mirror has to switch it on deliberately -- which is also what
    guarantees a test can never reach the real Supabase.
    """
    objects: dict = {}
    failing: set = set()

    def put_text(key, text, content_type):
        if key in failing:
            return False
        objects[key] = text
        return True

    monkeypatch.setattr(storage, "enabled", lambda: True)
    monkeypatch.setattr(storage, "put_text", put_text)
    monkeypatch.setattr(storage, "get_text", lambda key: objects.get(key))
    monkeypatch.setattr(
        storage, "delete_keys",
        lambda keys: all(objects.pop(k, None) or True for k in keys),
    )

    # SimpleNamespace, not a class body: `objects = objects` inside one is a
    # local assignment that cannot see the enclosing function's name.
    return types.SimpleNamespace(objects=objects, fail=failing.add)


def _write_outputs(book_id: int, md: str = "# Book\n\ntext", **manifest):
    md_path = config.MARKDOWN_DIR / f"{book_id}.md"
    mf_path = config.MARKDOWN_DIR / f"{book_id}.manifest.json"
    md_path.write_text(md, encoding="utf-8")
    mf_path.write_text(json.dumps(manifest or {"page_count": 3}), encoding="utf-8")
    return md_path, mf_path


def test_sync_outputs_uploads_both_and_clears_the_local_pair(fake_bucket):
    md_path, mf_path = _write_outputs(7)

    assert extract.sync_outputs(7) is True

    assert fake_bucket.objects[storage.markdown_key(7)] == "# Book\n\ntext"
    assert storage.manifest_key(7) in fake_bucket.objects
    # The point of the change: nothing is left behind locally.
    assert not md_path.exists() and not mf_path.exists()


def test_a_file_whose_own_upload_fails_is_kept(fake_bucket):
    """Per-file, not all-or-nothing: the manifest goes up and is cleared, the
    markdown fails and stays. No partial failure can drop unstored bytes."""
    md_path, mf_path = _write_outputs(8)
    fake_bucket.fail(storage.markdown_key(8))

    assert extract.sync_outputs(8) is False

    assert md_path.exists(), "markdown must survive its own failed upload"
    assert not mf_path.exists()
    assert storage.markdown_key(8) not in fake_bucket.objects


def test_a_lone_manifest_still_syncs(fake_bucket):
    """The rechunk case: load_markdown served the .md from the bucket, so no
    local one exists. An all-or-nothing rule would see it missing, refuse, and
    strand the refreshed manifest on one machine."""
    mf_path = config.MARKDOWN_DIR / "9.manifest.json"
    mf_path.write_text(json.dumps({"chunk_count": 12}), encoding="utf-8")

    assert extract.sync_outputs(9) is True
    assert fake_bucket.objects[storage.manifest_key(9)]
    assert not mf_path.exists()


def test_sync_outputs_leaves_everything_alone_while_storage_is_off():
    """A purely local install must behave exactly as it did before the mirror
    existed -- nothing uploaded, and nothing deleted."""
    md_path, mf_path = _write_outputs(10)

    assert storage.enabled() is False
    assert extract.sync_outputs(10) is False
    assert md_path.exists() and mf_path.exists()


def test_load_markdown_prefers_local_then_the_bucket(fake_bucket):
    _write_outputs(11, md="local copy")
    assert extract.load_markdown(11) == "local copy"

    (config.MARKDOWN_DIR / "11.md").unlink()
    fake_bucket.objects[storage.markdown_key(11)] = "stored copy"
    assert extract.load_markdown(11) == "stored copy"


def test_load_markdown_is_none_when_a_book_has_neither(fake_bucket):
    """None, not an exception: run_rechunk skips on it rather than dying."""
    assert extract.load_markdown(12) is None


def test_read_manifest_falls_back_to_the_bucket(fake_bucket):
    fake_bucket.objects[storage.manifest_key(13)] = json.dumps({"page_count": 42})
    assert extract.read_manifest(13)["page_count"] == 42


def test_update_manifest_merges_into_the_stored_copy(fake_bucket):
    """The silent-drop bug: with the local file already synced away, a naive
    update would write a manifest holding ONLY the new fields, losing the
    extractor version and page count that report.py reads."""
    fake_bucket.objects[storage.manifest_key(14)] = json.dumps(
        {"page_count": 42, "extractor": "pymupdf4llm-1.28.0"}
    )

    extract.update_manifest(14, chunk_count=9)

    merged = extract.read_manifest(14)
    assert merged == {
        "page_count": 42, "extractor": "pymupdf4llm-1.28.0", "chunk_count": 9,
    }


def test_purge_removes_the_extraction_objects(conn, tmp_path, monkeypatch):
    """Keyed by book_id, so nothing else can reference them and the book they
    name is gone -- they go unconditionally, unlike the shared original."""
    deleted = []
    monkeypatch.setattr(storage, "delete_keys", lambda keys: deleted.extend(keys) or True)
    monkeypatch.setattr(storage, "delete_original", lambda md5: False)

    book = _register(conn, tmp_path, b"%PDF-1.4 purge-me", "p.pdf")
    ingest.purge_book(conn, book["id"])

    assert deleted == [
        storage.markdown_key(book["id"]), storage.manifest_key(book["id"]),
    ]
