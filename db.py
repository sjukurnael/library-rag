"""
Postgres access layer: connection helper, the books work-queue, and CRUD for
books/chunks. The schema itself lives in migrations/ (applied by migrate.py) --
nothing here creates tables.

Every function owns its own commit. No transaction spans a network call (Drive
download, OCR, embedding); only a book's status row marks it claimed. Crash
recovery is the reaper clause of claim_next_book, not a held lock, so a worker
killed mid-book always leaves the database in a clean, resumable state -- with
one exception by design: insert_chunks_and_finish writes all of a book's chunks
and flips it to 'done' in a SINGLE transaction, so a book is never left
half-indexed.
"""
import contextlib

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

import config

# Statuses a book can be claimed from (anything not terminal-or-skipped).
TERMINAL_STATUSES = ("done", "failed", "needs_ocr")


@contextlib.contextmanager
def get_conn(database_url: str | None = None):
    conn = psycopg.connect(database_url or config.DATABASE_URL, autocommit=False)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


# ------------------------------------------------------------------ queue --

def upsert_book(
    conn,
    drive_file_id: str,
    title: str,
    md5: str | None,
    size_bytes: int | None,
) -> None:
    """Idempotent on drive_file_id. On conflict, refresh title/md5/size ONLY
    when one of them actually changed -- so re-running --discover never bumps
    updated_at or disturbs the status of a book already in flight or done."""
    conn.execute(
        """
        INSERT INTO books (drive_file_id, title, md5, size_bytes)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (drive_file_id) DO UPDATE
            SET title = EXCLUDED.title,
                md5 = EXCLUDED.md5,
                size_bytes = EXCLUDED.size_bytes,
                updated_at = now()
            WHERE books.title IS DISTINCT FROM EXCLUDED.title
               OR books.md5 IS DISTINCT FROM EXCLUDED.md5
               OR books.size_bytes IS DISTINCT FROM EXCLUDED.size_bytes
        """,
        (drive_file_id, title, md5, size_bytes),
    )
    conn.commit()


def claim_next_book(conn):
    """Atomically claim one processable book. Returns its row as a dict, or None
    if the queue is empty.

    Processable = status not in (done, failed, needs_ocr) AND either never
    claimed or claimed longer than CLAIM_STALE_MINUTES ago (the reaper: a book
    left 'processing' by a dead worker becomes claimable again). ORDER BY status
    DESC prioritises the most-advanced not-yet-done books (chunked > extracted >
    downloaded > discovered) so near-finished work completes first.

    A book claimed for the (MAX_ATTEMPTS+1)th time is marked 'failed' here rather
    than handed out again; the caller sees status == 'failed' and skips it.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE books
            SET claimed_at = now(),
                attempts = attempts + 1,
                updated_at = now()
            WHERE id = (
                SELECT id FROM books
                WHERE status NOT IN ('done','failed','needs_ocr')
                  AND (claimed_at IS NULL
                       OR claimed_at < now() - make_interval(mins => %(stale)s))
                ORDER BY status DESC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            {"stale": config.CLAIM_STALE_MINUTES},
        )
        row = cur.fetchone()
        if row is not None and row["attempts"] > config.MAX_ATTEMPTS:
            cur.execute(
                """
                UPDATE books
                SET status = 'failed',
                    error = %s,
                    claimed_at = NULL,
                    updated_at = now()
                WHERE id = %s
                RETURNING *
                """,
                (f"exceeded MAX_ATTEMPTS ({config.MAX_ATTEMPTS})", row["id"]),
            )
            row = cur.fetchone()
    conn.commit()
    return row


def fetch_book_by_drive_id(conn, drive_file_id: str):
    """One book row as a dict, or None. Same shape claim_next_book returns, so
    callers can hand it straight to process_book. Used by the --local path,
    which addresses a book directly instead of taking it off the queue."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM books WHERE drive_file_id = %s", (drive_file_id,))
        return cur.fetchone()


# ------------------------------------------------------------ book state --

def set_status(
    conn, book_id: int, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE books SET status = %s, error = %s, updated_at = now() WHERE id = %s",
        (status, error, book_id),
    )
    conn.commit()


def mark_downloaded(conn, book_id: int) -> None:
    conn.execute(
        "UPDATE books SET status = 'downloaded', updated_at = now() WHERE id = %s",
        (book_id,),
    )
    conn.commit()


def mark_extracted(
    conn, book_id: int, page_count: int, has_text_layer: bool
) -> None:
    conn.execute(
        """
        UPDATE books
        SET status = 'extracted', page_count = %s, has_text_layer = %s,
            updated_at = now()
        WHERE id = %s
        """,
        (page_count, has_text_layer, book_id),
    )
    conn.commit()


def mark_chunked(conn, book_id: int) -> None:
    conn.execute(
        "UPDATE books SET status = 'chunked', updated_at = now() WHERE id = %s",
        (book_id,),
    )
    conn.commit()


def delete_chunks_for_book(conn, book_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE book_id = %s", (book_id,))
    conn.commit()


def reset_done_to_extracted(conn) -> int:
    """For --rechunk: move every 'done' book back to 'extracted' so its chunks
    can be rebuilt from local markdown. Returns the count reset."""
    cur = conn.execute(
        "UPDATE books SET status = 'extracted', updated_at = now() "
        "WHERE status = 'done' RETURNING id"
    )
    n = len(cur.fetchall())
    conn.commit()
    return n


def fetch_rechunkable_books(conn):
    """Books whose markdown has been extracted already (status 'extracted' or
    'chunked') -- candidates for a chunk+embed rebuild from local markdown."""
    cur = conn.execute(
        "SELECT id, title FROM books WHERE status IN ('extracted','chunked') "
        "ORDER BY id"
    )
    return cur.fetchall()


# ---------------------------------------------------------------- chunks --

def insert_chunks_and_finish(conn, book_id: int, chunks: list) -> None:
    """Insert all of a book's chunks AND flip it to 'done' in ONE transaction.
    Either the whole book lands or none of it does -- a crash before commit
    leaves zero chunks and the book still claimable, never half-indexed."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE book_id = %s", (book_id,))
        cur.executemany(
            """
            INSERT INTO chunks
                (book_id, ordinal, heading_trail, page_start, page_end,
                 content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    book_id,
                    c["ordinal"],
                    c["heading_trail"],
                    c["page_start"],
                    c["page_end"],
                    c["content"],
                    c["token_count"],
                    HalfVector(c["embedding"]),
                )
                for c in chunks
            ],
        )
        cur.execute(
            "UPDATE books SET status = 'done', error = NULL, updated_at = now() "
            "WHERE id = %s",
            (book_id,),
        )
    conn.commit()


def build_hnsw_index(conn) -> None:
    conn.execute(
        "CREATE INDEX IF NOT EXISTS chunks_hnsw "
        "ON chunks USING hnsw (embedding halfvec_cosine_ops)"
    )
    conn.commit()


def search(conn, query_embedding, k: int, book_id: int | None = None) -> list:
    """Nearest chunks as dicts, closest first.

    dict rows rather than tuples: every caller wants a different subset of the
    columns, and positional unpacking means adding one column here silently
    breaks all of them at once. `ordinal` and `total_chunks` are carried so a
    citation can say where in the book a passage sits, not just which page.
    """
    params = [HalfVector(query_embedding)]
    where = ""
    if book_id is not None:
        where = "WHERE c.book_id = %s"
        params.append(book_id)
    params.append(k)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT c.id AS chunk_id, c.book_id, c.ordinal, c.heading_trail,
                   c.page_start, c.page_end, c.content, c.token_count,
                   b.title, b.page_count,
                   (SELECT count(*) FROM chunks x WHERE x.book_id = c.book_id)
                       AS total_chunks,
                   c.embedding <=> %s::halfvec AS distance
            FROM chunks c JOIN books b ON b.id = c.book_id
            {where}
            ORDER BY distance
            LIMIT %s
            """,
            params,
        )
        return cur.fetchall()


# ---------------------------------------------------------------- report --

def status_counts(conn):
    """List of (status, count), ordered by the enum's natural order."""
    cur = conn.execute(
        "SELECT status, COUNT(*) FROM books GROUP BY status ORDER BY status"
    )
    return cur.fetchall()


def fetch_report_data(conn):
    """Per-book: id, title, size_bytes, page_count, has_text_layer, status,
    chunk_count, total_tokens."""
    cur = conn.execute(
        """
        SELECT b.id, b.title, b.size_bytes, b.page_count, b.has_text_layer,
               b.status, COUNT(c.id) AS chunk_count,
               COALESCE(SUM(c.token_count), 0) AS total_tokens
        FROM books b LEFT JOIN chunks c ON c.book_id = b.id
        GROUP BY b.id
        ORDER BY b.id
        """
    )
    return cur.fetchall()
