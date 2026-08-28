"""embed + store: the same model embeds queries and documents; chunks+done
commit atomically; a crash before commit leaves no orphan chunks; reprocessing
replaces chunks rather than duplicating them."""
import pytest

from library_rag import config, db
from library_rag.pipeline import chunking, embed
from tests.conftest import deterministic_vector, lexical_vector, make_book


def _seed_book(conn, status="chunked"):
    row = conn.execute(
        "INSERT INTO books (source_id, title, status) VALUES (%s, %s, %s) "
        "RETURNING id",
        ("x", "X", status),
    ).fetchone()
    conn.commit()
    return row[0]


def _valid_chunks(n):
    return [
        {
            "ordinal": i,
            "heading_trail": "H",
            "page_start": 1,
            "page_end": 1,
            "content": f"chunk {i}",
            "token_count": 2,
            "embedding": [0.1] * config.EMBED_DIM,
        }
        for i in range(n)
    ]


def test_query_and_documents_share_model(fake_voyage):
    embed.embed_query("a question", fake_voyage)
    embed.embed_documents(["a passage", "another"], fake_voyage)
    models = {call[1] for call in fake_voyage.calls}
    input_types = {call[2] for call in fake_voyage.calls}
    assert models == {config.EMBED_MODEL}  # one model, both sides
    assert input_types == {"query", "document"}


def test_insert_chunks_and_finish_is_atomic(conn, fake_voyage):
    book_id = _seed_book(conn)
    md = "<!-- page: 1 -->\n\n# Heading\n\nSome body text about covenant theology."
    chunks = chunking.chunk_markdown(md)
    embeddings, _ = embed.embed_documents([c["content"] for c in chunks], fake_voyage)
    for c, e in zip(chunks, embeddings):
        c["embedding"] = e
        c["token_count"] = len(c["content"])

    db.insert_chunks_and_finish(conn, book_id, chunks)

    status = conn.execute(
        "SELECT status FROM books WHERE id = %s", (book_id,)
    ).fetchone()[0]
    n = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    assert status == "done"
    assert n == len(chunks) > 0


def test_crash_before_commit_leaves_no_orphans(conn):
    book_id = _seed_book(conn, status="chunked")
    chunks = _valid_chunks(1)
    # Second chunk has a wrong-dimension vector -> the INSERT fails inside the
    # single transaction, before the commit, simulating a crash mid-write.
    chunks.append({**_valid_chunks(1)[0], "ordinal": 1, "embedding": [0.0] * 5})

    with pytest.raises(Exception):
        db.insert_chunks_and_finish(conn, book_id, chunks)
    conn.rollback()

    n = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    status = conn.execute(
        "SELECT status FROM books WHERE id = %s", (book_id,)
    ).fetchone()[0]
    assert n == 0  # no orphan chunks
    assert status == "chunked"  # book untouched -> still reclaimable


def test_reprocess_replaces_without_duplicates(conn):
    book_id = _seed_book(conn)
    db.insert_chunks_and_finish(conn, book_id, _valid_chunks(3))
    # Reprocess the same book: delete-then-insert keeps it at 3, and the
    # UNIQUE(book_id, ordinal) constraint would fail if it were duplicating.
    db.insert_chunks_and_finish(conn, book_id, _valid_chunks(3))
    n = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    assert n == 3


def test_a_book_that_stops_extracting_drops_its_old_chunks(conn):
    """A failed re-index must not leave the previous index answering searches.

    The zero-chunk path returns before insert_chunks_and_finish, and that DELETE
    is the only chunk cleanup on the --rechunk route -- process_book's wipe
    never runs there. The book was left 'failed' with its old chunks still in
    the table, and db.search has no status filter, so they kept being retrieved
    for a book the library reported as broken.
    """
    from library_rag import ingest

    db.upsert_book(conn, "upload:stale", "Stale.pdf", "m", 1, source="upload")
    book_id = conn.execute(
        "SELECT id FROM books WHERE source_id = 'upload:stale'"
    ).fetchone()[0]
    db.insert_chunks_and_finish(conn, book_id, [{
        "ordinal": 0, "heading_trail": None, "page_start": 1, "page_end": 1,
        "content": "text from the previous successful run", "token_count": 6,
        "embedding": deterministic_vector("previous"),
    }])
    assert conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0] == 1

    # Re-chunk, but the markdown now yields nothing.
    n_chunks, _ = ingest.chunk_embed_and_finish(conn, book_id, "   ", None)

    assert n_chunks == 0
    assert conn.execute(
        "SELECT status FROM books WHERE id = %s", (book_id,)
    ).fetchone()[0] == "failed"
    assert conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s", (book_id,)
    ).fetchone()[0] == 0, "stale chunks still answer searches for a failed book"


def test_finishing_a_book_also_builds_its_topic_profile(conn, monkeypatch):
    """A book with chunks but no topic vectors is searchable by the tutor and
    INVISIBLE to the librarian: find_books reads book_topic_vectors, so the book
    simply never comes up and nothing anywhere reports an error. Three books sat
    like that until a coverage count caught it.

    So the profile is part of finishing, not a separate backfill."""
    from library_rag import ingest
    from library_rag.pipeline import embed as embed_mod

    monkeypatch.setattr(embed_mod, "embed_documents",
                        lambda texts, client: ([lexical_vector(t) for t in texts], 0))
    book = make_book(conn, "profiled", "A Book About The Covenant", status="discovered")
    md = "\n\n".join(
        f"## Chapter {i}\n\nThe covenant with Abraham and its later readers, part {i}."
        for i in range(1, 9))

    n, _ = ingest.chunk_embed_and_finish(conn, book, md, object())
    assert n > 0

    vectors = conn.execute(
        "SELECT count(*) FROM book_topic_vectors WHERE book_id = %s", (book,)
    ).fetchone()[0]
    assert vectors > 0, "the book finished ingestion invisible to the librarian"
    assert conn.execute("SELECT status::text FROM books WHERE id = %s",
                        (book,)).fetchone()[0] == "done"


def test_a_failed_profile_still_leaves_a_searchable_book(conn, monkeypatch):
    """The chunks are committed before the profile is attempted. A profile that
    cannot be built must leave a working book behind rather than roll one back:
    searchable-but-undiscoverable is a smaller problem than not indexed at all."""
    from library_rag import ingest
    from library_rag.pipeline import embed as embed_mod
    from library_rag.pipeline import profile as profile_mod

    monkeypatch.setattr(embed_mod, "embed_documents",
                        lambda texts, client: ([lexical_vector(t) for t in texts], 0))
    monkeypatch.setattr(profile_mod, "build",
                        lambda conn_, bid: (_ for _ in ()).throw(RuntimeError("boom")))

    book = make_book(conn, "halfway", "Half Profiled", status="discovered")
    n, _ = ingest.chunk_embed_and_finish(
        conn, book, "## One\n\nSome text about the covenant.", object())

    assert n > 0
    assert conn.execute("SELECT status::text FROM books WHERE id = %s",
                        (book,)).fetchone()[0] == "done", "a profile failure must not fail the book"
    assert conn.execute("SELECT count(*) FROM chunks WHERE book_id = %s",
                        (book,)).fetchone()[0] == n
