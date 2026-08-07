"""
The Drive metadata mirror: sync, the recursive path CTE, resumable title
embedding, and hybrid search over drive_files.

Drive is a fake service; embeddings come from conftest's lexical_vector, which
has the one property this needs -- texts sharing words are close -- and no
network. A real embedder would put model variance inside the thing being
measured (see conftest.lexical_vector's own docstring).
"""
import types

import pytest
from fastapi.testclient import TestClient
from pgvector import HalfVector

from library_rag import config, db
from library_rag.drive import mirror
from library_rag.web import api

from .conftest import lexical_vector

FOLDER = "application/vnd.google-apps.folder"
PDF = "application/pdf"


# ------------------------------------------------------------------ fakes --

def _f(fid, name, parent=None, mime=FOLDER):
    return {"id": fid, "name": name, "mimeType": mime,
            "parents": [parent] if parent else [],
            "webViewLink": f"https://drive.google.com/file/d/{fid}/view"}


def _p(fid, name, parent, size_mb=2.0):
    d = _f(fid, name, parent, PDF)
    d["size"] = str(int(size_mb * 1024 * 1024))
    d["md5Checksum"] = f"md5-{fid}"
    return d


class FakeFiles:
    def __init__(self, items):
        self.items = items
        self.queries = []

    def list(self, **kw):
        q = kw["q"]
        self.queries.append(q)
        mime = q.split("mimeType = '", 1)[1].split("'", 1)[0]
        hits = [i for i in self.items if i["mimeType"] == mime]
        return types.SimpleNamespace(
            execute=lambda: {"files": hits, "nextPageToken": None}
        )


class FakeService:
    def __init__(self, items):
        self._files = FakeFiles(items)

    def files(self):
        return self._files


class FakeVoyage:
    """Mirrors the slice of voyageai.Client that pipeline/embed.py uses."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, model, input_type):
        self.calls += 1
        return types.SimpleNamespace(
            embeddings=[lexical_vector(t) for t in texts],
            total_tokens=sum(len(t) for t in texts),
        )


# A small library: root -> Books -> Commentaries -> a PDF, five levels deep.
TREE = [
    _f("root", "Exclusive"),
    _f("books", "Books", "root"),
    _f("comm", "Commentaries", "books"),
    _f("rom", "Romans", "comm"),
    _p("pdf1", "Jensen_Romans Self Study.pdf", "rom", 3.0),
    _p("pdf2", "Snodgrass_Stories With Intent Parables.pdf", "books", 7.1),
    _p("pdf3", "Dodd_The Parables of the Kingdom.pdf", "books", 7.7),
]


@pytest.fixture
def synced(conn):
    mirror.sync(conn, FakeService(TREE))
    return conn


# ------------------------------------------------------------------- sync --

def test_sync_mirrors_folders_and_pdfs_separately(conn):
    counts = mirror.sync(conn, FakeService(TREE))
    assert counts["folders"] == 4
    assert counts["files"] == 3
    assert db.drive_sync_status(conn)["files"] == 3


def test_sync_is_idempotent(synced):
    before = synced.execute("SELECT count(*) FROM drive_files").fetchone()[0]
    mirror.sync(synced, FakeService(TREE))
    assert synced.execute("SELECT count(*) FROM drive_files").fetchone()[0] == before


def test_sync_removes_rows_drive_no_longer_returns(synced):
    """A stale row is worse than a missing one: it renders as a browsable book
    and Index on it fails at download with a 404 the user cannot act on."""
    counts = mirror.sync(synced, FakeService([i for i in TREE if i["id"] != "pdf3"]))
    assert counts["removed"] == 1
    assert synced.execute(
        "SELECT count(*) FROM drive_files WHERE file_id = 'pdf3'").fetchone()[0] == 0


def test_a_renamed_file_loses_its_stale_embedding(synced):
    synced.execute("UPDATE drive_files SET embedding = %s WHERE file_id = 'pdf1'",
                   (HalfVector(lexical_vector("old title")),))
    synced.commit()

    renamed = [dict(i, name="Something Entirely Different.pdf")
               if i["id"] == "pdf1" else i for i in TREE]
    mirror.sync(synced, FakeService(renamed))

    assert synced.execute(
        "SELECT embedding FROM drive_files WHERE file_id='pdf1'").fetchone()[0] is None, (
        "a vector describing the old title would keep matching a name that is gone"
    )


def test_an_unchanged_file_keeps_its_embedding(synced):
    synced.execute("UPDATE drive_files SET embedding = %s WHERE file_id = 'pdf1'",
                   (HalfVector(lexical_vector("x")),))
    synced.commit()
    mirror.sync(synced, FakeService(TREE))
    assert synced.execute(
        "SELECT embedding FROM drive_files WHERE file_id='pdf1'").fetchone()[0] is not None


def test_sync_prunes_everything_outside_the_library_root(conn, monkeypatch):
    """Drive has no "within this subtree" filter, so a sync pulls the whole
    account. The first real run mirrored 126 files of coursework, sheet music
    and a resume alongside the library -- a browse UI that surfaces someone's
    personal documents is one nobody can show anyone."""
    outside = TREE + [
        _f("personal", "Family Pictures"),
        _p("tax", "2025 Tax Return.pdf", "personal"),
    ]
    monkeypatch.setattr(config, "DRIVE_ROOT_FOLDER_ID", "books")
    counts = mirror.sync(conn, FakeService(outside))

    kept = {r[0] for r in conn.execute("SELECT file_id FROM drive_files").fetchall()}
    assert kept == {"books", "comm", "rom", "pdf1", "pdf2", "pdf3"}
    assert "tax" not in kept and "root" not in kept
    assert counts["out_of_scope"] == 3


def test_no_configured_root_mirrors_everything(conn, monkeypatch):
    monkeypatch.setattr(config, "DRIVE_ROOT_FOLDER_ID", None)
    counts = mirror.sync(conn, FakeService(TREE))
    assert counts["out_of_scope"] == 0
    assert conn.execute("SELECT count(*) FROM drive_files").fetchone()[0] == 7


# ------------------------------------------------------------- the paths --

def test_paths_are_materialised_root_first(synced):
    path = synced.execute(
        "SELECT path FROM drive_files WHERE file_id = 'pdf1'").fetchone()[0]
    assert path == "Exclusive / Books / Commentaries / Romans / Jensen_Romans Self Study.pdf"


def test_every_row_gets_a_path(synced):
    assert synced.execute(
        "SELECT count(*) FROM drive_files WHERE path IS NULL").fetchone()[0] == 0


def test_a_row_whose_parent_is_outside_the_mirror_is_treated_as_a_root(conn):
    """Not an edge case: a shared drive's top folder has a parent we were never
    granted access to. Treating only NULL as a root leaves every path NULL."""
    orphaned = [dict(i, parents=["never-shared-with-us"]) if i["id"] == "root" else i
                for i in TREE]
    mirror.sync(conn, FakeService(orphaned))

    assert conn.execute(
        "SELECT count(*) FROM drive_files WHERE path IS NULL").fetchone()[0] == 0
    assert conn.execute(
        "SELECT path FROM drive_files WHERE file_id='root'").fetchone()[0] == "Exclusive"


# -------------------------------------------------------------- embedding --

def test_embed_titles_only_touches_rows_without_one(synced):
    voyage = FakeVoyage()
    assert mirror.embed_titles(synced, voyage) == 3
    assert mirror.pending_count(synced) == 0

    voyage.calls = 0
    assert mirror.embed_titles(synced, voyage) == 0
    assert voyage.calls == 0, "re-running must not re-pay for work already done"


def test_embedding_is_resumable(synced):
    """At ~450 batches over the real drive, an all-or-nothing pass is a bad bet
    on the network. Interrupting must keep what was already paid for."""
    mirror.embed_titles(synced, FakeVoyage(), limit=1)
    assert mirror.pending_count(synced) == 2
    mirror.embed_titles(synced, FakeVoyage())
    assert mirror.pending_count(synced) == 0


def test_folders_are_never_embedded(synced):
    mirror.embed_titles(synced, FakeVoyage())
    assert synced.execute(
        "SELECT count(*) FROM drive_files WHERE mime_type = %s AND embedding IS NOT NULL",
        (FOLDER,)).fetchone()[0] == 0


def test_search_text_strips_filename_noise_and_keeps_the_folders(conn):
    noisy = TREE + [_p("noise", "Levine_Short Stories ( PDFDrive ).pdf", "books")]
    mirror.sync(conn, FakeService(noisy))

    got = conn.execute(
        "SELECT search_text FROM drive_files WHERE file_id='noise'").fetchone()[0]
    assert "PDFDrive" not in got
    assert ".pdf" not in got
    assert "_" not in got, "underscores are separators here, not content"
    assert "Books" in got, "the folder often carries more signal than the filename"


def test_the_last_word_of_a_title_is_searchable(conn):
    """Postgres's parser reads "Parables.pdf" as a FILE token, not a word, so
    without stripping the suffix a search for "parables" misses the final --
    usually most specific -- word of every title in the drive."""
    mirror.sync(conn, FakeService(TREE))
    lexemes = conn.execute(
        "SELECT tsv::text FROM drive_files WHERE file_id='pdf2'").fetchone()[0]
    assert "parabl" in lexemes, lexemes
    assert "parables.pdf" not in lexemes


def test_tsv_and_the_embedding_read_the_same_text(conn):
    """One cleaning, one column. Two implementations of "what is this book
    called" is how the dense and lexical legs come to disagree."""
    mirror.sync(conn, FakeService(TREE))
    row = conn.execute(
        "SELECT search_text, tsv::text FROM drive_files WHERE file_id='pdf1'"
    ).fetchone()
    assert "Commentaries" in row[0]
    assert "commentari" in row[1], "tsv must be derived from search_text"


# ---------------------------------------------------------------- browse --

def test_children_separates_folders_from_files(synced):
    d = db.drive_children(synced, "books")
    assert [f["title"] for f in d["folders"]] == ["Commentaries"]
    assert {f["title"] for f in d["files"]} == {
        "Snodgrass_Stories With Intent Parables.pdf",
        "Dodd_The Parables of the Kingdom.pdf",
    }


def test_children_with_no_parent_returns_the_roots(synced):
    d = db.drive_children(synced, None)
    assert [f["title"] for f in d["folders"]] == ["Exclusive"]


def test_an_indexed_book_is_flagged_with_its_status(synced):
    db.upsert_book(synced, "pdf2", "Snodgrass.pdf", "m", 100, source="drive")
    synced.commit()

    files = {f["file_id"]: f for f in db.drive_children(synced, "books")["files"]}
    assert files["pdf2"]["indexed"] is True
    assert files["pdf2"]["status"] == "discovered"
    assert files["pdf3"]["indexed"] is False


def test_an_upload_does_not_leak_into_the_indexed_flag(synced):
    """indexed is a join on source_id, which only means a Drive file id when
    source='drive'."""
    db.upsert_book(synced, "pdf2", "Some Upload.pdf", "m", 100, source="upload")
    synced.commit()
    files = {f["file_id"]: f for f in db.drive_children(synced, "books")["files"]}
    assert files["pdf2"]["indexed"] is False


def test_breadcrumb_is_root_first(synced):
    trail = [c["name"] for c in db.drive_breadcrumb(synced, "rom")]
    assert trail == ["Exclusive", "Books", "Commentaries", "Romans"]


# ---------------------------------------------------------------- search --

def _vec(text):
    return lexical_vector(text)


@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_every_mode_ranks_the_matching_title_first(synced, mode):
    mirror.embed_titles(synced, FakeVoyage())
    rows = db.search_drive_files(synced, _vec("parables kingdom"), "parables kingdom",
                                5, mode=mode)
    assert rows, f"{mode} returned nothing"
    assert rows[0]["file_id"] in {"pdf2", "pdf3"}


def test_search_never_returns_a_folder(synced):
    mirror.embed_titles(synced, FakeVoyage())
    rows = db.search_drive_files(synced, _vec("Romans"), "Romans", 10)
    assert all(r["mime_type"] == PDF for r in rows)


def test_a_title_with_no_embedding_is_still_findable_by_word(synced):
    """The lexical leg must not require the dense one. A title synced five
    seconds ago has no vector yet, and vanishing from search until the embed
    pass catches up would look like the sync had lost it."""
    rows = db.search_drive_files(synced, _vec("parables"), "parables", 5)
    assert any(r["file_id"] == "pdf2" for r in rows)


def test_search_carries_the_indexed_flag(synced):
    mirror.embed_titles(synced, FakeVoyage())
    db.upsert_book(synced, "pdf2", "x", "m", 1, source="drive")
    synced.commit()
    rows = db.search_drive_files(synced, _vec("parables"), "parables", 5)
    assert {r["file_id"]: r["indexed"] for r in rows}.get("pdf2") is True


def test_an_unknown_mode_is_rejected(synced):
    with pytest.raises(ValueError, match="unknown search mode"):
        db.search_drive_files(synced, _vec("x"), "x", 5, mode="magic")


# ----------------------------------------------------------------- routes --

@pytest.fixture
def web(synced, test_database_url, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    monkeypatch.setattr(config, "VOYAGE_API_KEY", "test-key-not-used")
    return TestClient(api.app)


def test_children_route_returns_folders_files_and_a_breadcrumb(web):
    d = web.get("/api/drive/children", params={"parent": "comm"}).json()
    assert [f["title"] for f in d["folders"]] == ["Romans"]
    assert [c["name"] for c in d["breadcrumb"]] == ["Exclusive", "Books", "Commentaries"]


def test_children_route_with_no_parent_returns_roots(web):
    d = web.get("/api/drive/children").json()
    assert [f["title"] for f in d["folders"]] == ["Exclusive"]
    assert d["breadcrumb"] == []


def test_sync_status_route_reports_pending_titles(web, synced):
    s = web.get("/api/drive/sync").json()
    assert s == {"folders": 4, "files": 3, "embedded": 0, "pending": 3,
                 "synced_at": s["synced_at"]}
    assert s["synced_at"] is not None


def test_search_route_rejects_an_empty_query(web):
    assert web.get("/api/drive/search", params={"q": "   "}).status_code == 400


def test_static_assets_are_whitelisted_not_served_by_path(web):
    """This process also holds credentials.json and token.json; a path parameter
    that reaches the filesystem is the wrong thing to be casual about."""
    assert web.get("/static/app.css").status_code == 200
    assert web.get("/static/../../../etc/passwd").status_code == 404
    assert web.get("/static/token.json").status_code == 404
