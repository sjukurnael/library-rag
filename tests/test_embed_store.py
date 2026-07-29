"""embed + store: the same model embeds queries and documents; chunks+done
commit atomically; a crash before commit leaves no orphan chunks; reprocessing
replaces chunks rather than duplicating them."""
import pytest

import config
import db
from pipeline import chunking, embed


def _seed_book(conn, status="chunked"):
    row = conn.execute(
        "INSERT INTO books (drive_file_id, title, status) VALUES (%s, %s, %s) "
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
