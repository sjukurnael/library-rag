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

from library_rag import config

# Statuses a book can be claimed from (anything not terminal-or-skipped).
TERMINAL_STATUSES = ("done", "failed", "needs_ocr")

# Drive's marker for a folder. Folders are ordinary file rows with this
# mime_type -- there is no separate folder table, in Drive or in the mirror.
FOLDER_MIME = "application/vnd.google-apps.folder"


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
    source_id: str,
    title: str,
    md5: str | None,
    size_bytes: int | None,
    source: str = "drive",
) -> None:
    """Idempotent on source_id. On conflict, refresh title/md5/size ONLY
    when one of them actually changed -- so re-running --discover never bumps
    updated_at or disturbs a book already in flight or done.

    A CHANGED md5 is different from a changed title: it means the file's bytes
    are not the ones we ingested, so whatever is in `chunks` describes a
    document that no longer exists. Those books are reset to 'discovered' and
    re-enter the queue. Previously the new md5 was recorded and the book left
    'done', which quietly guaranteed the index disagreed with Drive -- the
    failure is invisible because every status count still reads clean.

    A title-only change (a rename in Drive) is metadata and does not reprocess.

    `source` is deliberately NOT in the DO UPDATE list. A book's source is fixed
    at creation, and a conflicting insert claiming a different one means two
    sources have produced the same source_id -- which cannot happen while Drive
    ids and "upload:<md5>" live in disjoint namespaces. Silently rewriting it
    would move the book's bytes somewhere the downloader will not look.
    """
    conn.execute(
        """
        INSERT INTO books (source_id, source, title, md5, size_bytes)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source_id) DO UPDATE
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
        (source_id, source, title, md5, size_bytes),
    )
    conn.commit()


def claim_next_book(conn, source: str | None = None):
    """Atomically claim one processable book. Returns its row as a dict, or None
    if the queue is empty.

    `source` restricts the claim to one kind of book. The web upload path uses
    it: a user adding a PDF has asked for that PDF to be indexed, not for a
    backlog of Drive books to start downloading behind it -- and a Drive book
    claimed from inside the API process can block for minutes on OAuth that has
    nobody to prompt. Left None (the CLI worker) it claims from every source.

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
                  AND (%(source)s::text IS NULL OR source = %(source)s::text)
                  AND (claimed_at IS NULL
                       OR claimed_at < now() - make_interval(mins => %(stale)s))
                ORDER BY status DESC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            {"stale": config.CLAIM_STALE_MINUTES, "source": source},
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


def fetch_book_by_source_id(conn, source_id: str):
    """One book row as a dict, or None. Same shape claim_next_book returns, so
    callers can hand it straight to process_book. Used by the upload path, which
    addresses a book directly instead of taking it off the queue."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM books WHERE source_id = %s", (source_id,))
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


def delete_book(conn, book_id: int):
    """Delete a book and its chunks. Returns the deleted row as a dict, or None
    if there was no such book.

    Chunks go with it through ON DELETE CASCADE rather than a second statement,
    so there is no window in which the book is gone and its chunks are not --
    orphaned chunks would keep answering searches for a book the library no
    longer lists.

    Returns the row because the caller needs source and source_id to know which
    files on disk belonged to it, and after the DELETE there is nowhere else to
    read them from.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("DELETE FROM books WHERE id = %s RETURNING *", (book_id,))
        row = cur.fetchone()
    conn.commit()
    return row


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
    NULL -- retrieval/tools.py's weak-match cutoff reads it on every row, and a
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


# ---------------------------------------------------------- drive mirror --

# Every row a Drive-browser view returns, in one place so the tree, the search
# and the agent all render identically. `indexed`/`book_id`/`status` come from a
# LEFT JOIN rather than a column: a Drive file exists whether or not we hold it,
# and most never will be held.
_DRIVE_COLUMNS = """
    d.file_id, d.name AS title, d.mime_type, d.parent_id, d.path,
    d.web_view_link AS url,
    round(d.size_bytes / 1048576.0, 1)::float8 AS size_mb,
    (b.id IS NOT NULL) AS indexed, b.id AS book_id, b.status::text AS status
"""

_DRIVE_JOIN = """
    LEFT JOIN books b
        ON b.source_id = d.file_id AND b.source = 'drive'
"""


def drive_children(conn, parent_id: str | None) -> dict:
    """One folder's contents: subfolders and PDFs, separately.

    parent_id=None means the roots -- rows with no parent, or whose parent is
    outside the mirror (a shared drive's top folder has a parent we cannot see,
    so treating only NULL as a root would show an empty library).

    Folders and files are returned as two lists rather than one sorted mix
    because the UI renders them differently, and separating them here means the
    frontend never has to branch on mime_type.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT {_DRIVE_COLUMNS}
            FROM drive_files d
            {_DRIVE_JOIN}
            LEFT JOIN drive_files p ON p.file_id = d.parent_id
            WHERE CASE
                    WHEN %(parent)s::text IS NULL
                        THEN d.parent_id IS NULL OR p.file_id IS NULL
                    ELSE d.parent_id = %(parent)s::text
                  END
            ORDER BY lower(d.name)
            """,
            {"parent": parent_id},
        )
        rows = cur.fetchall()
    folders = [r for r in rows if r["mime_type"] == FOLDER_MIME]
    return {
        "folders": folders,
        "files": [r for r in rows if r["mime_type"] != FOLDER_MIME],
        "folder_count": len(folders),
    }


def drive_breadcrumb(conn, file_id: str) -> list:
    """Root-first ancestry of one folder, for the header trail.

    Walked in SQL rather than by repeated round trips: the same recursive-CTE
    shape mirror.materialise_paths uses, but upward and for one node.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH RECURSIVE up AS (
                SELECT file_id, name, parent_id, 0 AS depth
                FROM drive_files WHERE file_id = %s
              UNION ALL
                SELECT d.file_id, d.name, d.parent_id, up.depth + 1
                FROM drive_files d JOIN up ON d.file_id = up.parent_id
            )
            SELECT file_id, name FROM up ORDER BY depth DESC
            """,
            (file_id,),
        )
        return cur.fetchall()


def drive_sync_status(conn) -> dict:
    """What the mirror currently holds, and whether it is finished."""
    row = conn.execute(
        """
        SELECT count(*) FILTER (WHERE mime_type = %s) AS folders,
               count(*) FILTER (WHERE mime_type <> %s) AS files,
               count(*) FILTER (WHERE mime_type <> %s AND embedding IS NOT NULL)
                   AS embedded,
               max(synced_at) AS synced_at
        FROM drive_files
        """,
        (FOLDER_MIME, FOLDER_MIME, FOLDER_MIME),
    ).fetchone()
    folders, files, embedded, synced_at = row
    return {
        "folders": folders,
        "files": files,
        "embedded": embedded,
        "pending": files - embedded,
        "synced_at": synced_at.isoformat() if synced_at else None,
    }


def search_drive_files(conn, query_embedding, query_text: str, k: int, mode=None,
                       lexical_weight=None) -> list:
    """Rank the mirror by meaning and by words, fused with RRF.

    Same algorithm as search() over chunks -- two rankers, rank-based fusion,
    1/(RRF_K + rank) -- against drive_files instead. The measured result that
    hybrid ties dense over CHUNKS does not carry here, and was re-measured
    rather than assumed: over 57,527 titles, fusing beat both legs alone
    (8/8 hit-rate@5 vs dense 7/8 and lexical 5/8) -- but only at a REDUCED
    lexical weight. See config.DRIVE_RRF_LEXICAL_WEIGHT for the sweep and for
    what an equal weight does to "the end times".

    Rows that have never been embedded simply lose the dense leg rather than
    being excluded: a title synced five seconds ago is still findable by word
    while the embed pass catches up.
    """
    mode = mode or "hybrid"
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
    dense_pool = 0 if mode == "lexical" else config.HYBRID_CANDIDATES
    lexical_pool = 0 if mode == "dense" else config.HYBRID_CANDIDATES

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            WITH dense AS (
                SELECT x.file_id, row_number() OVER (ORDER BY x.distance, x.file_id) AS rank
                FROM (
                    SELECT d.file_id, d.embedding <=> %(vec)s::halfvec AS distance
                    FROM drive_files d
                    WHERE d.mime_type <> %(folder)s AND d.embedding IS NOT NULL
                    ORDER BY 2
                    LIMIT %(dense_pool)s
                ) x
            ),
            q AS (
                SELECT NULLIF(array_to_string(ARRAY(
                    SELECT quote_literal(l)
                    FROM unnest(tsvector_to_array(
                        to_tsvector('english', %(text)s))) l
                ), ' | '), '')::tsquery AS query
            ),
            lexical AS (
                SELECT y.file_id, row_number() OVER (ORDER BY y.score DESC, y.file_id) AS rank
                FROM (
                    SELECT d.file_id, ts_rank_cd(d.tsv, q.query) AS score
                    FROM drive_files d, q
                    WHERE d.tsv @@ q.query AND d.mime_type <> %(folder)s
                    ORDER BY 2 DESC
                    LIMIT %(lexical_pool)s
                ) y
            ),
            fused AS (
                SELECT COALESCE(dn.file_id, lx.file_id) AS file_id,
                       COALESCE(1.0 / (%(rrf_k)s + dn.rank), 0)
                         + COALESCE(%(w_lex)s / (%(rrf_k)s + lx.rank), 0) AS rrf_score,
                       dn.rank AS dense_rank, lx.rank AS lexical_rank
                FROM dense dn FULL OUTER JOIN lexical lx ON lx.file_id = dn.file_id
            )
            SELECT {_DRIVE_COLUMNS}, f.rrf_score, f.dense_rank, f.lexical_rank
            FROM fused f
            JOIN drive_files d ON d.file_id = f.file_id
            {_DRIVE_JOIN}
            ORDER BY f.rrf_score DESC, lower(d.name)
            LIMIT %(k)s
            """,
            {
                "vec": HalfVector(query_embedding),
                "text": query_text,
                "folder": FOLDER_MIME,
                "dense_pool": dense_pool,
                "lexical_pool": lexical_pool,
                "rrf_k": config.RRF_K,
                "w_lex": (config.DRIVE_RRF_LEXICAL_WEIGHT
                          if lexical_weight is None else lexical_weight),
                "k": k,
            },
        )
        return cur.fetchall()
