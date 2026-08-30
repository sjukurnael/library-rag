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
from psycopg.types.json import Jsonb

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
    when one of them actually changed -- so re-queuing a folder never bumps
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


def fetch_book(conn, book_id: int):
    """One book row as a dict by primary key, or None. The PDF-viewer route
    needs source, source_id and md5 to decide where the bytes live."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM books WHERE id = %s", (book_id,))
        return cur.fetchone()


def md5_in_use(conn, md5: str) -> bool:
    """Does ANY book still reference these bytes? Two rows can share an md5
    (the same content indexed once from Drive and once as an upload), and the
    bucket holds ONE object per md5 -- so a purge may only delete the object
    when the last referencing row is gone."""
    return conn.execute(
        "SELECT EXISTS(SELECT 1 FROM books WHERE md5 = %s)", (md5,)
    ).fetchone()[0]


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


# The statuses the pipeline never leaves on its own. A book here is not "in
# progress", it is stopped -- and 353 of them accumulated on the Processing page
# with no way to clear them but 353 clicks.
STUCK_STATUSES = ("failed", "needs_ocr")


def stuck_books(conn, older_than_hours: int | None = None) -> list:
    """Books stopped in a terminal state, oldest first.

    `older_than_hours=None` means all of them. The cutoff is on updated_at,
    which is when the book last changed state -- so "older than 24 hours" means
    "gave up more than a day ago", not "was uploaded more than a day ago".
    """
    sql = """
        SELECT id, title, status::text AS status, source, source_id,
               EXTRACT(EPOCH FROM now() - updated_at) AS age_s
        FROM books
        WHERE status = ANY(%(stuck)s)
    """
    params = {"stuck": list(STUCK_STATUSES), "hours": older_than_hours}
    if older_than_hours is not None:
        sql += " AND updated_at < now() - make_interval(hours => %(hours)s)"
    sql += " ORDER BY updated_at"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def drive_link_for_book(conn, book_id: int) -> str | None:
    """The Google Drive viewer URL for a book, or None if it did not come from
    Drive.

    Kept out of the books table on purpose: the link belongs to the mirrored
    file, and drive_files is what a re-sync refreshes. Joining on demand means
    a moved or re-shared file is right the next time it is asked for, rather
    than stale in a column nobody thought to update.
    """
    row = conn.execute(
        """
        SELECT d.web_view_link
        FROM books b
        JOIN drive_files d ON d.file_id = b.source_id
        WHERE b.id = %s AND b.source = 'drive'
        """,
        (book_id,),
    ).fetchone()
    return row[0] if row and row[0] else None


def delete_books(conn, book_ids: list) -> list:
    """Delete many books in ONE statement, returning the rows.

    The per-book path costs two round trips each, which is fine for the one
    book someone clicks and is not fine for 353: against a hosted Postgres that
    alone is most of a minute before any file work starts.
    """
    if not book_ids:
        return []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("DELETE FROM books WHERE id = ANY(%s) RETURNING *", (list(book_ids),))
        rows = cur.fetchall()
    conn.commit()
    return rows


def md5s_still_used(conn, md5s: list) -> set:
    """Which of these md5s some surviving book still references.

    Same question as md5_in_use, asked once for a whole batch. The bucket holds
    one object per md5 and two rows can share it, so an object may only go when
    the last row referencing it has.
    """
    md5s = [m for m in set(md5s) if m]
    if not md5s:
        return set()
    rows = conn.execute(
        "SELECT DISTINCT md5 FROM books WHERE md5 = ANY(%s)", (md5s,)
    ).fetchall()
    return {r[0] for r in rows}


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
    book_ids: list | None = None,
    query_text: str | None = None,
    mode: str | None = None,
    tsquery_mode: str | None = None,
    lexical_weight: float | None = None,
) -> list:
    """Retrieve the k best chunks as dicts, best first.

    `book_ids` is a classroom: the set of books this search may see at all.
    `book_id` narrows to one book. They COMPOSE rather than override, and that
    is deliberate -- the tutor's Session already refuses a book_id outside its
    classroom before calling, so composing here is the second lock on the same
    door. A containment property worth having is worth having twice; if the
    caller's check is ever wrong, this one still cannot leak a chunk.

    Passing NEITHER searches every chunk, and on a full corpus that is now a
    trap rather than a feature. Migration 0010 dropped both global chunk indexes
    -- hnsw and the tsv GIN -- because retrieval was becoming scoped, so an
    unscoped search has nothing to jump with: over 1.76M chunks the dense leg is
    a sequential scan and the lexical leg computes ts_rank_cd on every row.
    Measured, it does not finish inside two minutes.

    Nothing on the reader-facing path does this. The tutor's Session requires a
    classroom and cannot express "everything"; the two remaining unscoped
    callers are cli/search.py (a developer's spot-check) and evaluation/
    harness.py (a measuring instrument that runs against a small test corpus and
    genuinely means the whole of it). The default stays permissive for those two
    rather than forcing a sentinel on them -- but if a third caller ever wants
    it, that is the moment to make scope mandatory.

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
        return _search_dense(conn, query_embedding, k, book_id, book_ids)
    return _search_fused(
        conn, query_embedding, query_text, k, book_id, mode, tsquery_mode,
        lexical_weight, book_ids,
    )


def _search_dense(
    conn, query_embedding, k: int, book_id: int | None, book_ids: list | None = None,
) -> list:
    params = [HalfVector(query_embedding)]
    clauses = []
    # Scope first, then narrow. An empty list is NOT the same as None: a
    # classroom with no books must match nothing, where None means "everything".
    # `= ANY('{}')` is false for every row, which is exactly right -- and is why
    # this tests `is not None` rather than truthiness.
    if book_ids is not None:
        clauses.append("c.book_id = ANY(%s)")
        params.append(list(book_ids))
    if book_id is not None:
        clauses.append("c.book_id = %s")
        params.append(book_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
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
    lexical_weight=None, book_ids=None,
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
                    WHERE (%(book_ids)s::bigint[] IS NULL
                           OR c.book_id = ANY(%(book_ids)s::bigint[]))
                      AND (%(book_id)s::bigint IS NULL
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
                      AND (%(book_ids)s::bigint[] IS NULL
                           OR c.book_id = ANY(%(book_ids)s::bigint[]))
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
                "book_ids": None if book_ids is None else list(book_ids),
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
    round(coalesce(d.size_bytes, d.subtree_bytes) / 1048576.0, 1)::float8 AS size_mb,
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


def drive_file(conn, file_id: str) -> dict | None:
    """One mirror row, shaped like the browse rows (size_mb covers folders via
    subtree_bytes) plus the raw subtree_bytes the folder-index gate compares."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT file_id, name AS title, mime_type, parent_id, path,
                   round(coalesce(size_bytes, subtree_bytes) / 1048576.0, 1)::float8
                       AS size_mb,
                   subtree_bytes
            FROM drive_files WHERE file_id = %s
            """,
            (file_id,),
        )
        return cur.fetchone()


def queue_drive_folder(conn, folder_id: str) -> dict:
    """Queue every PDF in a folder's subtree for ingestion, in one statement.

    Metadata comes from the mirror, not from per-file Drive calls: the single-
    file route's one get_file is fine for hand-picked books, but a folder of a
    hundred would spend minutes inside the request re-fetching what sync
    already wrote. The mirror's md5 can be stale if a file changed in Drive
    since -- that book fails its checksum during ingest, visibly, which is the
    right degradation.

    Rows already in books are left completely alone (anti-join, plus ON
    CONFLICT DO NOTHING for races), so re-clicking never resets a book that is
    done or mid-flight. New rows land at the column-default 'discovered', which
    is what makes them claimable.

    Deduplicated on md5, not only on source_id. A shared drive files the same
    book in several folders -- 967 of the Books subtree's 9,921 PDFs are byte
    identical to another one -- and a source_id anti-join alone would download,
    extract, chunk and embed every copy. The DISTINCT ON collapses duplicates
    inside this batch; the second anti-join catches the ones already ingested
    under a different file_id. A NULL md5 dedupes on its own file_id, i.e. not
    at all: collapsing every unchecksummed file into one row would be far worse
    than processing them.

    The cost is that the losing copy gets no books row, so browse shows it as
    unindexed. That is the right trade -- the bytes ARE indexed, under the other
    file_id -- and it is why total_pdfs still counts the whole subtree rather
    than the deduped set: the reader asked about a folder, not about distinct
    checksums.

    Returns {"queued": inserted, "total_pdfs": pdfs in the subtree}.
    """
    row = conn.execute(
        """
        WITH RECURSIVE sub AS (
            SELECT file_id FROM drive_files WHERE file_id = %(folder)s
          UNION ALL
            SELECT d.file_id FROM drive_files d JOIN sub s ON d.parent_id = s.file_id
        ),
        all_pdfs AS (
            SELECT d.file_id, d.name, d.md5, d.size_bytes
            FROM drive_files d JOIN sub USING (file_id)
            WHERE d.mime_type <> %(folder_mime)s
        ),
        pdfs AS (
            SELECT DISTINCT ON (coalesce(md5, file_id))
                   file_id, name, md5, size_bytes
            FROM all_pdfs
            ORDER BY coalesce(md5, file_id), file_id
        ),
        ins AS (
            INSERT INTO books (source_id, title, md5, size_bytes, source)
            SELECT p.file_id, p.name, p.md5, p.size_bytes, 'drive'
            FROM pdfs p
            WHERE NOT EXISTS (SELECT 1 FROM books b WHERE b.source_id = p.file_id)
              AND NOT EXISTS (SELECT 1 FROM books b
                              WHERE p.md5 IS NOT NULL AND b.md5 = p.md5)
            ON CONFLICT (source_id) DO NOTHING
            RETURNING 1
        )
        SELECT (SELECT count(*) FROM all_pdfs) AS total_pdfs,
               (SELECT count(*) FROM ins) AS queued
        """,
        {"folder": folder_id, "folder_mime": FOLDER_MIME},
    ).fetchone()
    conn.commit()
    return {"total_pdfs": row[0], "queued": row[1]}


def queue_drive_files(conn, file_ids: list, limit_bytes: int | None = None) -> dict:
    """Queue an explicit list of mirror PDFs (e.g. one search's results).

    Same contract as queue_drive_folder -- mirror metadata, rows already in
    books untouched, md5 duplicates collapsed -- but gated on the bytes actually
    PENDING rather than a stored total: a result list is arbitrary, so its
    "size" is whatever is not yet queued. Over the limit nothing is inserted and
    over_limit says so; non-PDF or unknown ids are simply not matched.

    pending_bytes is measured BEFORE the md5 collapse, so it over-states what
    will actually be fetched. Deliberate: the size gate should err towards
    refusing, and a caller told "1.2 GB" that then transfers 1.0 GB has been
    warned in the safe direction.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE b.id IS NULL),
                   coalesce(sum(d.size_bytes) FILTER (WHERE b.id IS NULL), 0),
                   count(*)
            FROM drive_files d
            LEFT JOIN books b ON b.source_id = d.file_id
            WHERE d.file_id = ANY(%s) AND d.mime_type <> %s
            """,
            (list(file_ids), FOLDER_MIME),
        )
        pending, pending_bytes, matched = cur.fetchone()
        if limit_bytes is not None and pending_bytes > limit_bytes:
            return {"queued": 0, "matched": matched,
                    "pending_bytes": pending_bytes, "over_limit": True}
        cur.execute(
            """
            INSERT INTO books (source_id, title, md5, size_bytes, source)
            SELECT p.file_id, p.name, p.md5, p.size_bytes, 'drive'
            FROM (
                SELECT DISTINCT ON (coalesce(d.md5, d.file_id))
                       d.file_id, d.name, d.md5, d.size_bytes
                FROM drive_files d
                WHERE d.file_id = ANY(%s) AND d.mime_type <> %s
                ORDER BY coalesce(d.md5, d.file_id), d.file_id
            ) p
            WHERE NOT EXISTS (SELECT 1 FROM books b WHERE b.source_id = p.file_id)
              AND NOT EXISTS (SELECT 1 FROM books b
                              WHERE p.md5 IS NOT NULL AND b.md5 = p.md5)
            ON CONFLICT (source_id) DO NOTHING
            """,
            (list(file_ids), FOLDER_MIME),
        )
        queued = cur.rowcount
    conn.commit()
    return {"queued": queued, "matched": matched,
            "pending_bytes": pending_bytes, "over_limit": False}


def drive_sync_status(conn) -> dict:
    """What the mirror holds, how much of it is actually indexed, and whether
    the sync is finished.

    Coverage rides along on THIS statement rather than getting one of its own.
    The seq scan over drive_files is already being paid for by the counts, and
    `books` is small enough to hash in a few kilobytes -- EXPLAIN ANALYZE puts
    the joined statement at the same ~46 ms as the counts alone. A second
    endpoint on its own poll timer would have cost an entire extra scan of
    57,000 rows to report something this query already has in hand.

    Two denominators, because they answer different questions. A thousand
    tracts and one 400 MB scanned folio are the same fraction of the shelf and
    wildly different fractions of the work, so a single figure would always
    flatter whichever of the two is further along.

    `indexed` means `status = 'done'` and nothing else. A book that failed, or
    is waiting on OCR, has a row in `books` and is not searchable; counting it
    would overstate coverage in the one number whose entire job is not to.
    Everything with a row but no chunks is `working` instead, so the two
    buckets never sum to more than the truth.

    Uploads are invisible here on purpose. They are not in Drive, so they
    cannot be a fraction of it -- the sidebar's book count already covers them.
    """
    row = conn.execute(
        """
        SELECT count(*) FILTER (WHERE d.mime_type = %(folder)s) AS folders,
               count(*) FILTER (WHERE d.mime_type <> %(folder)s) AS files,
               count(*) FILTER (WHERE d.mime_type <> %(folder)s
                                  AND d.embedding IS NOT NULL) AS embedded,
               -- ::bigint because sum(bigint) is numeric, and a byte count
               -- crossing the wire as a Decimal serialises as a float that
               -- cannot represent 153,330,392,318 exactly.
               coalesce(sum(d.size_bytes)
                        FILTER (WHERE d.mime_type <> %(folder)s), 0)::bigint
                   AS total_bytes,
               count(*) FILTER (WHERE d.mime_type <> %(folder)s
                                  AND b.status = 'done') AS indexed_files,
               coalesce(sum(d.size_bytes) FILTER (WHERE d.mime_type <> %(folder)s
                                  AND b.status = 'done'), 0)::bigint
                   AS indexed_bytes,
               count(*) FILTER (WHERE d.mime_type <> %(folder)s
                                  AND b.id IS NOT NULL
                                  AND b.status <> 'done') AS working_files,
               coalesce(sum(d.size_bytes) FILTER (WHERE d.mime_type <> %(folder)s
                                  AND b.id IS NOT NULL
                                  AND b.status <> 'done'), 0)::bigint
                   AS working_bytes,
               max(d.synced_at) AS synced_at
        FROM drive_files d
        -- The same LEFT JOIN the browse views use, aggregated instead of
        -- listed. No foreign key backs it: a Drive file exists whether or not
        -- we hold it, and by these numbers most never will.
        LEFT JOIN books b
            ON b.source_id = d.file_id AND b.source = 'drive'
        """,
        {"folder": FOLDER_MIME},
    ).fetchone()
    (folders, files, embedded, total_bytes, indexed_files, indexed_bytes,
     working_files, working_bytes, synced_at) = row
    return {
        "folders": folders,
        "files": files,
        "embedded": embedded,
        "pending": files - embedded,
        "total_bytes": total_bytes,
        "indexed_files": indexed_files,
        "indexed_bytes": indexed_bytes,
        "working_files": working_files,
        "working_bytes": working_bytes,
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


# ---------------------------------------------------------- research runs --
#
# An agent run's state, kept here rather than in the web process's memory. See
# migrations/0009_research_runs.sql for why. The division of labour mirrors the
# books queue: the producer owns the row and appends to it, any reader can
# follow along, and nothing about a run depends on which container is serving.

# What a run's `error` says when its worker stopped reporting. Named because
# both the row and the SSE frame use it, and because it is the one failure whose
# message should not guess: the run did not "fail", and the server did not
# necessarily "restart" -- its worker went quiet, and the honest report is that.
RESEARCH_INTERRUPTED = (
    "the worker stopped reporting -- its container was probably replaced or "
    "shut down mid-run"
)


def create_research_run(conn, run_id: str, question: str,
                        classroom_id: int | None = None,
                        agent: str = "tutor", model: str | None = None) -> None:
    """The row, written before any work starts.

    Ordering matters: the request that mints a run_id must not return until the
    row exists, or a client fast enough to come straight back with it would get
    a 404 for a run that is about to be perfectly fine.

    classroom_id is nullable for the rows written before classrooms existed;
    every new run has one. See 0012 for why the table is still called
    research_runs when what runs against a classroom is the tutor.
    """
    conn.execute(
        "INSERT INTO research_runs (run_id, question, classroom_id, agent, model) "
        "VALUES (%s, %s, %s, %s, %s)",
        (run_id, question, classroom_id, agent, model),
    )
    conn.commit()


def append_research_event(conn, run_id: str, payload: dict) -> int:
    """Append one event and bump the heartbeat. Returns its seq.

    The sequence number is computed IN the insert rather than counted by the
    caller, so the producer holds no state that a crash could desynchronise --
    the next seq is always a fact about the table. Safe without locking because
    a run has exactly one writer; the primary key makes that assumption
    enforced rather than assumed.

    The heartbeat rides along in the same commit. Two statements, one
    transaction: a visible event whose run looks dead would be worse than
    either.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_events (run_id, seq, payload)
            SELECT %s, COALESCE(MAX(seq) + 1, 0), %s
            FROM research_events WHERE run_id = %s
            RETURNING seq
            """,
            (run_id, Jsonb(payload), run_id),
        )
        seq = cur.fetchone()[0]
    conn.execute(
        "UPDATE research_runs SET heartbeat_at = now() WHERE run_id = %s", (run_id,)
    )
    conn.commit()
    return seq


def finish_research_run(
    conn,
    run_id: str,
    *,
    status: str,
    answer: str | None = None,
    stop_reason: str | None = None,
    iterations: int | None = None,
    searches: int | None = None,
    usage: dict | None = None,
    error: str | None = None,
) -> None:
    """Close a run out. Called AFTER its last event is appended, so a reader
    that sees a terminal status can trust one more drain to be complete."""
    u = usage or {}
    conn.execute(
        """
        UPDATE research_runs
        SET status = %s, answer = %s, stop_reason = %s, iterations = %s,
            searches = %s, error = %s,
            input_tokens = %s, output_tokens = %s,
            cache_read_tokens = %s, cache_write_tokens = %s,
            finished_at = now(), heartbeat_at = now()
        WHERE run_id = %s
        """,
        (
            status, answer, stop_reason, iterations, searches, error,
            # Anthropic's names on the left of the mapping, ours on the right;
            # .get with a default because a scripted client in the tests has no
            # usage at all and an absent counter is zero, not a crash.
            u.get("input_tokens", 0),
            u.get("output_tokens", 0),
            u.get("cache_read_input_tokens", 0),
            u.get("cache_creation_input_tokens", 0),
            run_id,
        ),
    )
    conn.commit()


def research_run_state(conn, run_id: str, stale_seconds: int):
    """{"status", "stale"} for one run, or None if there is no such run.

    `stale` is computed by Postgres rather than by comparing timestamps in
    Python: the heartbeat is written by one container and read by another, and
    now() on the database is the only clock both of them share.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT status,
                   heartbeat_at < now() - make_interval(secs => %s) AS stale
            FROM research_runs
            WHERE run_id = %s
            """,
            (stale_seconds, run_id),
        )
        row = cur.fetchone()
    # This runs inside a poll loop. Ending the read transaction each time keeps
    # the connection from sitting idle-in-transaction for the length of a run.
    conn.commit()
    return row


def fetch_research_events(conn, run_id: str, after: int) -> list:
    """Events from `after` onwards, in order. Same cursor semantics the route
    has always had: after=N means "I have seen 0..N-1"."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT seq, payload FROM research_events
            WHERE run_id = %s AND seq >= %s
            ORDER BY seq
            """,
            (run_id, after),
        )
        rows = cur.fetchall()
    conn.commit()
    return rows


def mark_research_interrupted(conn, run_id: str) -> None:
    """Close out a run whose worker went quiet.

    Guarded on status = 'running' so this can never overwrite a real outcome:
    if the worker finished between the staleness check and this update, the
    WHERE matches nothing and the run keeps the result it earned.
    """
    conn.execute(
        """
        UPDATE research_runs
        SET status = 'interrupted', error = %s, finished_at = now()
        WHERE run_id = %s AND status = 'running'
        """,
        (RESEARCH_INTERRUPTED, run_id),
    )
    conn.commit()


def fetch_research_run(conn, run_id: str):
    """The whole row, for anyone asking what a run did after the fact."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM research_runs WHERE run_id = %s", (run_id,))
        return cur.fetchone()


# --------------------------------------------------------------- profiles --
# The librarian's tier. These read chunks that already exist and write a small
# per-book representation; nothing here calls an API or costs money, so a
# profile can be rebuilt at any time from what is already stored.


def books_needing_profile(conn, limit: int | None = None) -> list:
    """Ids of finished books that have chunks but no topic vectors yet.

    Driven off the absence of vectors rather than a flag on books, so a book
    whose profile was deleted (or whose chunks were rebuilt by --rechunk) comes
    back into scope without anything having to remember to mark it. The join
    against chunks is what keeps a `done` book with zero chunks -- which the
    extractor can produce -- out of a pass that would divide by zero on it.
    """
    sql = """
        SELECT b.id
        FROM books b
        WHERE b.status = 'done'
          AND EXISTS (SELECT 1 FROM chunks c WHERE c.book_id = b.id)
          AND NOT EXISTS (SELECT 1 FROM book_topic_vectors v WHERE v.book_id = b.id)
        ORDER BY b.id
    """
    if limit is not None:
        sql += " LIMIT %s"
        return [r[0] for r in conn.execute(sql, (limit,)).fetchall()]
    return [r[0] for r in conn.execute(sql).fetchall()]


def chunks_for_profile(conn, book_id: int) -> list:
    """(embedding, heading_trail, content) for one book, in reading order.

    Content comes back whole rather than truncated in SQL: the caller needs only
    its first few words for a fallback label, but slicing here would mean
    left(content, N) on every row of a table whose content column is TOASTed,
    and the caller already has the row in memory.
    """
    return conn.execute(
        """
        SELECT embedding, heading_trail, content
        FROM chunks WHERE book_id = %s ORDER BY ordinal
        """,
        (book_id,),
    ).fetchall()


def store_book_profile(conn, book_id: int, toc: str | None, source: str,
                       vectors: list) -> None:
    """Replace a book's profile and topic vectors, atomically.

    Delete-then-insert rather than upsert: k is derived from the chunk count, so
    a rebuild after --rechunk can produce FEWER vectors than last time, and an
    upsert keyed on (book_id, ordinal) would leave the surplus ordinals behind
    as orphans that still answer searches. The delete is the only way the row
    count can shrink.

    `vectors` is [(ordinal, label, members, embedding_list), ...]; ordinal 0 is
    the whole-book mean by convention (see the migration).

    Commits, like every other writer here. get_conn() does not commit on exit --
    it closes, which rolls back -- and psycopg opens an implicit transaction on
    the first statement, so the conn.transaction() below is a SAVEPOINT inside
    it rather than a transaction of its own. Releasing a savepoint persists
    nothing; without this commit a whole backfill runs, reports success, and
    leaves an empty table.
    """
    with conn.transaction():
        conn.execute("DELETE FROM book_topic_vectors WHERE book_id = %s", (book_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO book_topic_vectors (book_id, ordinal, label, members, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [(book_id, o, lab, n, HalfVector(vec)) for o, lab, n, vec in vectors],
            )
        conn.execute(
            """
            INSERT INTO book_profiles (book_id, toc, source, built_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (book_id) DO UPDATE
                SET toc = EXCLUDED.toc,
                    source = EXCLUDED.source,
                    built_at = now()
            """,
            (book_id, toc, source),
        )
    conn.commit()


def profile_counts(conn) -> dict:
    """How much of the searchable corpus the librarian can actually see."""
    row = conn.execute(
        """
        SELECT (SELECT count(*) FROM books WHERE status = 'done'),
               (SELECT count(*) FROM book_profiles),
               (SELECT count(*) FROM book_topic_vectors),
               (SELECT count(*) FROM book_profiles WHERE source = 'headings'),
               (SELECT count(*) FROM book_profiles WHERE source = 'text')
        """
    ).fetchone()
    return {"done": row[0], "profiled": row[1], "vectors": row[2],
            "from_headings": row[3], "from_text": row[4]}


def _enable_ef_search(conn) -> None:
    """Raise hnsw.ef_search for this session.

    Touching a vector operator first is not superstition: pgvector registers its
    GUCs when the module loads, and the module loads lazily on first use. A bare
    SET on a fresh connection raises 'unrecognized configuration parameter'.
    """
    conn.execute("SELECT '[1]'::halfvec(1) <=> '[1]'::halfvec(1)")
    conn.execute(f"SET hnsw.ef_search = {int(config.PROFILE_EF_SEARCH)}")


def search_book_profiles(conn, query_embedding, query_text: str, k: int = 30) -> list:
    """Rank BOOKS for a study topic. The librarian's recall tool.

    Two legs, fused by RRF, the same shape as search() over chunks:

      contents -- HNSW over binary-quantized topic centroids, then an exact
                  rerank of the shortlist against the full-precision vectors.
                  Binary costs nothing here once ef_search is raised (measured:
                  identical to an exact scan) and keeps the index at 41 MB.
      titles   -- the filename and folder path, which carry what no embedding of
                  the text can: author names, series, and the curated folder
                  taxonomy a human filed the book under.

    A book scores on its BEST-matching centroid, not its average -- that is the
    whole reason multi-vector profiles exist. Averaging a book's topics back
    together reproduces the single whole-book vector that measured 68% against
    the 96% of separate centroids.

    Returns `matched_topics` alongside the score because they are different
    recommendations. A book matching 6 of its 14 centroids covers the subject;
    one matching a single centroid mentions it in one chapter. Both belong in a
    session, and collapsing them into one number throws that distinction away.
    """
    _enable_ef_search(conn)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            WITH shortlist AS (
                SELECT v.book_id, v.label, v.members,
                       v.embedding <=> %(vec)s::halfvec AS distance
                FROM book_topic_vectors v
                ORDER BY v.bits <~> binary_quantize(%(vec)s::halfvec)::bit(1024)
                LIMIT %(shortlist)s
            ),
            scored AS (
                SELECT book_id, label, members, distance,
                       min(distance) OVER (PARTITION BY book_id) AS best
                FROM shortlist
            ),
            contents AS (
                SELECT book_id, min(best) AS best,
                       -- Counted against the book's OWN best match, not against
                       -- shortlist membership: reaching an 800-row shortlist out
                       -- of 101,385 centroids means something, reaching it out of
                       -- six does not, and a signal that only works at one corpus
                       -- size is not a signal.
                       count(*) FILTER (WHERE distance <= best + %(margin)s)
                           AS matched_topics,
                       (array_agg(label ORDER BY distance))[1] AS best_label,
                       (array_agg(members ORDER BY distance))[1] AS best_members
                FROM scored GROUP BY book_id
            ),
            content_ranked AS (
                SELECT c.*, row_number() OVER (ORDER BY best, book_id) AS rank
                FROM contents c
            ),
            title_hits AS (
                SELECT b.id AS book_id, d.embedding <=> %(vec)s::halfvec AS distance
                FROM drive_files d
                JOIN books b ON b.source_id = d.file_id
                WHERE d.embedding IS NOT NULL AND b.status = 'done'
                ORDER BY 2
                LIMIT %(shortlist)s
            ),
            titles AS (
                SELECT book_id, row_number() OVER (ORDER BY distance, book_id) AS rank
                FROM title_hits
            ),
            -- Drive the final join from the CANDIDATES, never from books. Starting
            -- FROM books and filtering afterwards made the planner walk all 9,032
            -- rows against two materialised CTEs: 5.6 s, against 30 ms for either
            -- leg on its own.
            cand AS (
                SELECT book_id FROM content_ranked
                UNION
                SELECT book_id FROM titles
            )
            SELECT b.id AS book_id, b.title, b.page_count,
                   cr.best AS distance, cr.matched_topics,
                   cr.best_label, cr.best_members,
                   coalesce(1.0 / (%(rrf_k)s + cr.rank), 0)
                     + coalesce(%(tw)s / (%(rrf_k)s + t.rank), 0) AS score
            FROM cand
            JOIN books b ON b.id = cand.book_id
            LEFT JOIN content_ranked cr ON cr.book_id = cand.book_id
            LEFT JOIN titles t ON t.book_id = cand.book_id
            ORDER BY score DESC, b.id
            LIMIT %(k)s
            """,
            {
                "vec": HalfVector(query_embedding),
                "shortlist": config.PROFILE_SHORTLIST,
                "rrf_k": config.RRF_K,
                "tw": config.PROFILE_TITLE_WEIGHT,
                "margin": config.PROFILE_TOPIC_MARGIN,
                "k": k,
            },
        )
        return cur.fetchall()


def look_inside(conn, book_id: int, query_embedding, k: int = 3) -> list:
    """The best passages for a topic INSIDE one book. The librarian's verifier.

    Exact, not approximate, and it needs no index: a btree jump to one book's
    ~200 chunks is a small enough population to compare exhaustively, which is
    both cheaper and more accurate than a graph walk over all 1.76M. This is the
    same property that let migration 0010 drop chunks_hnsw.

    ~12 ms per book, so an agent can verify thirty candidates inside a second
    and quote real text back rather than guessing from a title.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, ordinal, heading_trail, page_start, page_end, content,
                   embedding <=> %s::halfvec AS distance
            FROM chunks
            WHERE book_id = %s
            ORDER BY distance
            LIMIT %s
            """,
            (HalfVector(query_embedding), book_id, k),
        )
        return cur.fetchall()


# ------------------------------------------------------------- classrooms --
# A classroom is the shelf a reader assembles and the tutor may not look outside
# of. See migrations/0012_classrooms.sql for why the membership cascade only
# ever runs one way, and config.CLASSROOM_MAX_BOOKS for why there is a cap.


class ClassroomFull(Exception):
    """Raised instead of silently truncating an add that would exceed the cap.

    Truncating would be the worse failure: the reader sees "added" and gets a
    shelf missing whichever books the ordering happened to drop, and only finds
    out later when the tutor cannot answer from a book they believe is there.
    """

    def __init__(self, have: int, adding: int, cap: int):
        self.have, self.adding, self.cap = have, adding, cap
        super().__init__(
            f"This classroom holds {have} of {cap} books; adding {adding} more "
            f"would exceed it. Remove some first, or start another classroom."
        )


def create_classroom(conn, owner_email: str | None, name: str,
                     brief: str | None = None) -> int:
    row = conn.execute(
        """
        INSERT INTO classrooms (owner_email, name, brief)
        VALUES (%s, %s, %s) RETURNING id
        """,
        (owner_email, name, brief),
    ).fetchone()
    conn.commit()
    return row[0]


def list_classrooms(conn, owner_email: str | None = None) -> list:
    """Every classroom, newest-used first, with its book count.

    The count is derived, not stored. It is one index scan over
    classroom_books, and a denormalised copy is a copy that drifts the first
    time a book is deleted out from under a shelf.

    owner_email=None returns every classroom rather than none: auth can be
    disabled entirely (config.auth_enabled()), and in that mode there is no
    owner to filter by.
    """
    where = "" if owner_email is None else "WHERE c.owner_email = %(owner)s"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT c.id, c.name, c.brief, c.owner_email,
                   c.created_at, c.last_used_at,
                   count(cb.book_id) AS book_count,
                   count(cb.book_id) FILTER (WHERE b.status <> 'done') AS arriving,
                   -- How heavy the shelf is. books is already joined for
                   -- `arriving`, so both sums are free; coalesce because an
                   -- empty shelf sums to NULL, and a shelf whose books predate
                   -- size_bytes should read 0 rather than blank.
                   coalesce(sum(b.size_bytes), 0) AS size_bytes,
                   coalesce(sum(b.page_count), 0) AS pages
            FROM classrooms c
            LEFT JOIN classroom_books cb ON cb.classroom_id = c.id
            LEFT JOIN books b ON b.id = cb.book_id
            {where}
            GROUP BY c.id
            ORDER BY c.last_used_at DESC
            """,
            {"owner": owner_email},
        )
        return cur.fetchall()


def fetch_classroom(conn, classroom_id: int):
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM classrooms WHERE id = %s", (classroom_id,))
        return cur.fetchone()


def classroom_books(conn, classroom_id: int) -> list:
    """The shelf, for display: title, state, and why each book is here."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT b.id AS book_id, b.title, b.page_count, b.status::text AS status,
                   b.size_bytes,
                   cb.added_by, cb.rationale, cb.added_at,
                   count(c.id) AS chunks
            FROM classroom_books cb
            JOIN books b ON b.id = cb.book_id
            LEFT JOIN chunks c ON c.book_id = b.id
            WHERE cb.classroom_id = %s
            GROUP BY b.id, b.title, b.page_count, b.status, b.size_bytes,
                     cb.added_by, cb.rationale, cb.added_at
            ORDER BY cb.added_at
            """,
            (classroom_id,),
        )
        return cur.fetchall()


def classroom_book_ids(conn, classroom_id: int, *, ready_only: bool = True) -> list:
    """The tutor's scope.

    ready_only because a book still being ingested has no chunks: including it
    would widen the scope by a book that cannot answer, and the tutor would
    report "nothing found in X" for a book that is merely still arriving. The
    shelf shows it as arriving; the search does not see it yet.
    """
    sql = "SELECT book_id FROM classroom_books cb"
    if ready_only:
        sql += " JOIN books b ON b.id = cb.book_id AND b.status = 'done'"
    sql += " WHERE cb.classroom_id = %s ORDER BY book_id"
    return [r[0] for r in conn.execute(sql, (classroom_id,)).fetchall()]


def add_classroom_books(conn, classroom_id: int, picks: list) -> int:
    """Add books to a shelf. `picks` is [(book_id, added_by, rationale), ...].

    Raises ClassroomFull rather than adding a subset. Counted against the books
    that are actually NEW, so re-adding a book already on the shelf is a no-op
    that can never push it over the cap.
    """
    have = conn.execute(
        "SELECT count(*) FROM classroom_books WHERE classroom_id = %s",
        (classroom_id,),
    ).fetchone()[0]
    wanted = {int(p[0]) for p in picks}
    existing = set(
        r[0] for r in conn.execute(
            "SELECT book_id FROM classroom_books "
            "WHERE classroom_id = %s AND book_id = ANY(%s)",
            (classroom_id, list(wanted)),
        ).fetchall()
    ) if wanted else set()
    new = wanted - existing
    if have + len(new) > config.CLASSROOM_MAX_BOOKS:
        raise ClassroomFull(have, len(new), config.CLASSROOM_MAX_BOOKS)

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO classroom_books (classroom_id, book_id, added_by, rationale)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (classroom_id, book_id) DO NOTHING
            """,
            [(classroom_id, int(b), by, why) for b, by, why in picks],
        )
        added = cur.rowcount
    touch_classroom(conn, classroom_id)
    conn.commit()
    return added


def remove_classroom_book(conn, classroom_id: int, book_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM classroom_books WHERE classroom_id = %s AND book_id = %s",
            (classroom_id, book_id),
        )
        gone = cur.rowcount > 0
    touch_classroom(conn, classroom_id)
    conn.commit()
    return gone


def rename_classroom(conn, classroom_id: int, name: str, brief: str | None = None) -> None:
    conn.execute(
        """
        UPDATE classrooms
        SET name = %s, brief = coalesce(%s, brief), last_used_at = now()
        WHERE id = %s
        """,
        (name, brief, classroom_id),
    )
    conn.commit()


def delete_classroom(conn, classroom_id: int) -> bool:
    """Delete a shelf. Its membership and its tutor runs cascade; the BOOKS do
    not -- emptying a shelf must never touch the library."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM classrooms WHERE id = %s", (classroom_id,))
        gone = cur.rowcount > 0
    conn.commit()
    return gone


def touch_classroom(conn, classroom_id: int) -> None:
    """Float a classroom to the top of the list. Does NOT commit -- callers are
    already inside a write, and a commit here would split their transaction."""
    conn.execute(
        "UPDATE classrooms SET last_used_at = now() WHERE id = %s", (classroom_id,)
    )


def classrooms_holding(conn, book_id: int) -> list:
    """Which shelves hold this book. Read before deleting one, so the confirm
    can say "and removes it from 3 classrooms" instead of surprising someone."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id, c.name FROM classroom_books cb
            JOIN classrooms c ON c.id = cb.classroom_id
            WHERE cb.book_id = %s ORDER BY c.name
            """,
            (book_id,),
        )
        return cur.fetchall()


def _passage_needles(passage: str):
    """Search strings to try for a quoted passage, most specific first.

    The model does not quote cleanly, and measuring a real run is how that
    became visible: of twelve recommendations, five could not be located by
    matching the opening words. The quotes showed why.

    It stitches fragments with an ellipsis -- "characterized in various ways...
    Grigor Tat'ewac'i says" -- so a needle taken off the front runs straight
    into the "..." and matches nothing. Each fragment is therefore tried on its
    own, longest first, since the longest is the most specific.

    And it trims. A quote whose first ten words are right but whose eleventh was
    dropped still fails a ten-word needle, so each fragment is retried shorter.
    Five words is the floor: scoped to one book that is still specific, and
    below it a needle starts matching common phrasing.
    """
    import re as _re
    fragments = _re.split(r"\.\.\.|\u2026|\[\s*\.\.\.\s*\]", passage)
    cleaned = []
    for fragment in fragments:
        fragment = " ".join(fragment.split()).strip(" \"'\u201c\u201d\u2018\u2019")
        if len(fragment.split()) >= 5:
            cleaned.append(fragment)
    cleaned.sort(key=lambda f: -len(f.split()))

    for fragment in cleaned:
        words = fragment.split()
        for n in (10, 7, 5):
            if len(words) >= n:
                yield " ".join(words[:n])


def page_of_passage(conn, book_id: int, passage: str) -> tuple | None:
    """(page_start, page_end) for a quoted passage, or None if it is not found.

    Matched against the text rather than reported by the model. The librarian
    quotes a passage look_inside handed it, but the page that passage sat on is
    not part of what `recommend` asks for -- and a page number retyped by a
    model is one that can be wrong, silently, in a citation whose entire job is
    to be checkable. So the words are located instead.

    Whitespace is normalised on both sides because look_inside collapses runs of
    it before the model ever sees the text, so a quote will not match the stored
    chunk byte for byte. LIKE wildcards are escaped for the same reason a search
    box escapes them: an underscore in the passage is an underscore.

    The heading trail is searched as well as the body. The librarian sometimes
    quotes a section title rather than prose, and those live in their own column
    -- a chunk-only search returns nothing for a quote that is genuinely there.

    Returns None rather than guessing. A passage the model paraphrased outright
    will not be found, and no page number is a better answer than a plausible
    one nobody can check.
    """
    if not passage:
        return None

    for needle in _passage_needles(passage):
        escaped = needle
        for ch in ("\\", "%", "_"):
            escaped = escaped.replace(ch, "\\" + ch)

        row = conn.execute(
            """
            SELECT page_start, page_end
            FROM chunks
            WHERE book_id = %s
              AND (regexp_replace(content, '\\s+', ' ', 'g') ILIKE '%%' || %s || '%%'
                   OR regexp_replace(coalesce(heading_trail, ''), '\\s+', ' ', 'g')
                      ILIKE '%%' || %s || '%%')
            ORDER BY page_start
            LIMIT 1
            """,
            (book_id, escaped, escaped),
        ).fetchone()
        if row:
            return (row[0], row[1])
    return None


RUNS_PAGE = 30


def list_runs(conn, *, agent: str | None = None, limit: int = RUNS_PAGE,
              offset: int = 0) -> dict:
    """Runs newest-first, with enough per row to render a list without opening
    any of them.

    The classroom is joined rather than stored: a run outlives the shelf it was
    taken off (0014 made classroom_id SET NULL on delete), so the name has to
    come from the classrooms table when it is still there and be absent, not
    wrong, when it is not.

    `total` is a second query on purpose. Counting in the same statement as a
    LIMIT means either a window function over every matching row or a lateral,
    and this table is small enough that two cheap queries beat one clever one.
    """
    where = "WHERE r.agent = %s" if agent else ""
    params = [agent] if agent else []

    total = conn.execute(
        f"SELECT count(*) FROM research_runs r {where}", params
    ).fetchone()[0]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
        SELECT r.run_id, r.agent, r.question, r.status::text, r.model,
               r.iterations, r.searches, r.error,
               r.input_tokens, r.output_tokens,
               r.cache_read_tokens, r.cache_write_tokens,
               r.started_at, r.finished_at,
               r.classroom_id, c.name AS classroom_name,
               EXTRACT(EPOCH FROM (coalesce(r.finished_at, now()) - r.started_at))
                   AS seconds
        FROM research_runs r
        LEFT JOIN classrooms c ON c.id = r.classroom_id
        {where}
        ORDER BY r.started_at DESC
        LIMIT %s OFFSET %s
            """,
            params + [limit, offset],
        )
        rows = cur.fetchall()

    return {"runs": rows, "total": total, "limit": limit, "offset": offset}
