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
    updated_at or disturbs a book already in flight or done.

    A CHANGED md5 is different from a changed title: it means the file's bytes
    are not the ones we ingested, so whatever is in `chunks` describes a
    document that no longer exists. Those books are reset to 'discovered' and
    re-enter the queue. Previously the new md5 was recorded and the book left
    'done', which quietly guaranteed the index disagreed with Drive -- the
    failure is invisible because every status count still reads clean.

    A title-only change (a rename in Drive) is metadata and does not reprocess.
    """
    conn.execute(
        """
        INSERT INTO books (drive_file_id, title, md5, size_bytes)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (drive_file_id) DO UPDATE
            SET title = EXCLUDED.title,
                md5 = EXCLUDED.md5,
                size_bytes = EXCLUDED.size_bytes,
                updated_at = now(),
                status = CASE
                    WHEN books.md5 IS DISTINCT FROM EXCLUDED.md5
                         AND EXCLUDED.md5 IS NOT NULL
                         AND books.md5 IS NOT NULL
                    THEN 'discovered'::doc_status
                    ELSE books.status
                END,
                claimed_at = CASE
                    WHEN books.md5 IS DISTINCT FROM EXCLUDED.md5
                         AND EXCLUDED.md5 IS NOT NULL
                         AND books.md5 IS NOT NULL
                    THEN NULL ELSE books.claimed_at
                END,
                attempts = CASE
                    WHEN books.md5 IS DISTINCT FROM EXCLUDED.md5
                         AND EXCLUDED.md5 IS NOT NULL
                         AND books.md5 IS NOT NULL
                    THEN 0 ELSE books.attempts
                END
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


def touch_claim(conn, book_id: int) -> None:
    """Refresh claimed_at -- the worker's heartbeat, called between pipeline
    stages.

    claimed_at is a lease, not a lock: nothing is held during the minutes a book
    takes to download, extract and embed, and other workers stay off it only
    because the claim looks recent. Without a heartbeat that lease expires on a
    fixed timer, so it cannot tell "the worker died" from "this book is just
    slow" -- a long OCR job gets stolen mid-flight and processed twice. Touching
    it as each stage completes makes "still alive" and "still holding it" the
    same signal, which is what lets CLAIM_STALE_MINUTES stay short enough to
    recover a genuinely dead worker quickly.
    """
    conn.execute("UPDATE books SET claimed_at = now() WHERE id = %s", (book_id,))
    conn.commit()


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


SEARCH_MODES = ("hybrid", "dense", "lexical")

# Columns every mode returns, so callers never branch on how a row was found.
# `ordinal` and `total_chunks` are carried so a citation can say where in the
# book a passage sits, not just which page.
_CHUNK_COLUMNS = """
    c.id AS chunk_id, c.book_id, c.ordinal, c.heading_trail,
    c.page_start, c.page_end, c.content, c.token_count,
    b.title, b.page_count,
    (SELECT count(*) FROM chunks x WHERE x.book_id = c.book_id) AS total_chunks
"""


def search(
    conn,
    query_embedding,
    k: int,
    book_id: int | None = None,
    *,
    query_text: str | None = None,
    mode: str | None = None,
    tsquery_mode: str | None = None,
    lexical_weight: float | None = None,
) -> list:
    """Retrieve the k best chunks as dicts, best first.

    mode:
      "hybrid"  -- RRF over the dense and lexical legs (see config.RRF_K).
      "dense"   -- cosine over the embedding only.
      "lexical" -- Postgres full-text over the `tsv` column only.

    Defaults to "hybrid" when query_text is supplied and "dense" when it is not,
    so a caller that only has a vector keeps working unchanged. "dense" and
    "lexical" exist mainly so evaluate.py can score the legs against the fusion
    -- a hybrid that is not measurably better than its own dense leg is just a
    slower dense search, and the only way to know is to be able to run both.

    dict rows rather than tuples: every caller wants a different subset of the
    columns, and positional unpacking means adding one column here silently
    breaks all of them at once.
    """
    # config.SEARCH_MODE, not a literal: which mode ships is a measured
    # decision recorded there, and it should be changeable in one place.
    mode = mode or (config.SEARCH_MODE if query_text else "dense")
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
    if mode in ("hybrid", "lexical") and not query_text:
        raise ValueError(f"mode={mode!r} needs query_text")
    if mode == "dense":
        return _search_dense(conn, query_embedding, k, book_id)
    return _search_fused(
        conn, query_embedding, query_text, k, book_id, mode, tsquery_mode,
        lexical_weight,
    )


def _search_dense(conn, query_embedding, k: int, book_id: int | None) -> list:
    params = [HalfVector(query_embedding)]
    where = ""
    if book_id is not None:
        where = "WHERE c.book_id = %s"
        params.append(book_id)
    params.append(k)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {_CHUNK_COLUMNS},
                   c.embedding <=> %s::halfvec AS distance
            FROM chunks c JOIN books b ON b.id = c.book_id
            {where}
            ORDER BY distance
            LIMIT %s
            """,
            params,
        )
        return cur.fetchall()


def _search_fused(
    conn, query_embedding, query_text, k, book_id, mode, tsquery_mode=None,
    lexical_weight=None,
) -> list:
    """Both legs, fused by Reciprocal Rank Fusion.

    One SQL statement for both fused modes, with the unwanted leg's candidate
    pool set to 0 rather than a second near-identical query. Two queries that
    must stay in lockstep on filtering, page columns and book scoping are two
    queries that will eventually disagree about one of them.

    row_number() is taken OUTSIDE each leg's LIMIT so the ranks are the ranks
    within the candidate pool, which is what RRF is defined over.

    `distance` is recomputed in the final SELECT for every fused row, including
    rows only the lexical leg found. It costs one vector op per surviving
    candidate (at most 2 * HYBRID_CANDIDATES) and it means `distance` is never
    NULL -- agent/research.py's weak-match cutoff reads it on every row, and a
    lexical-only hit with no distance would silently read as a strong match.
    """
    dense_pool = 0 if mode == "lexical" else config.HYBRID_CANDIDATES
    lexical_pool = config.HYBRID_CANDIDATES
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            WITH dense AS (
                SELECT d.id, row_number() OVER (ORDER BY d.distance, d.id) AS rank
                FROM (
                    SELECT c.id, c.embedding <=> %(vec)s::halfvec AS distance
                    FROM chunks c
                    WHERE (%(book_id)s::bigint IS NULL
                           OR c.book_id = %(book_id)s::bigint)
                    ORDER BY 2
                    LIMIT %(dense_pool)s
                ) d
            ),
            q AS (
                SELECT CASE WHEN %(tsquery_mode)s = 'or' THEN
                    -- Stem and drop stopwords with Postgres's own english
                    -- config (to_tsvector), then OR the surviving lexemes.
                    -- quote_literal each one: raw lexemes can contain '/', '@'
                    -- or ':' (URLs, emails, "3:16") and would fail the cast.
                    -- NULLIF('') covers an all-stopword query -- `tsv @@ NULL`
                    -- is NULL, so the leg returns nothing instead of erroring.
                    NULLIF(array_to_string(ARRAY(
                        SELECT quote_literal(l)
                        FROM unnest(tsvector_to_array(
                            to_tsvector('english', %(text)s))) l
                    ), ' | '), '')::tsquery
                ELSE
                    websearch_to_tsquery('english', %(text)s)
                END AS query
            ),
            lexical AS (
                SELECT l.id, row_number() OVER (ORDER BY l.score DESC, l.id) AS rank
                FROM (
                    SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
                    FROM chunks c, q
                    WHERE c.tsv @@ q.query
                      AND (%(book_id)s::bigint IS NULL
                           OR c.book_id = %(book_id)s::bigint)
                    ORDER BY 2 DESC
                    LIMIT %(lexical_pool)s
                ) l
            ),
            fused AS (
                SELECT COALESCE(d.id, l.id) AS id,
                       COALESCE(%(w_dense)s / (%(rrf_k)s + d.rank), 0)
                         + COALESCE(%(w_lexical)s / (%(rrf_k)s + l.rank), 0)
                         AS rrf_score,
                       d.rank AS dense_rank,
                       l.rank AS lexical_rank
                FROM dense d FULL OUTER JOIN lexical l ON l.id = d.id
            )
            SELECT {_CHUNK_COLUMNS},
                   c.embedding <=> %(vec)s::halfvec AS distance,
                   f.rrf_score, f.dense_rank, f.lexical_rank
            FROM fused f
            JOIN chunks c ON c.id = f.id
            JOIN books b ON b.id = c.book_id
            ORDER BY f.rrf_score DESC, distance
            LIMIT %(k)s
            """,
            {
                "vec": HalfVector(query_embedding),
                "text": query_text,
                "book_id": book_id,
                "dense_pool": dense_pool,
                "lexical_pool": lexical_pool,
                "tsquery_mode": tsquery_mode or config.LEXICAL_TSQUERY,
                "rrf_k": config.RRF_K,
                "w_dense": 1.0,
                "w_lexical": (
                    config.RRF_LEXICAL_WEIGHT if lexical_weight is None
                    else lexical_weight
                ),
                "k": k,
            },
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
