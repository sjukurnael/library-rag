"""
Mirror Drive's metadata into Postgres so the library can be browsed and searched
locally.

Three passes, deliberately separate because they have different costs and
different failure modes:

    sync(conn, service)          ~66 Drive calls, ~2 min   -- rows
    materialise_paths(conn)      one recursive CTE         -- the tree
    embed_titles(conn, voyage)   ~450 embed batches        -- semantic search

Nothing here downloads a file. A row in `drive_files` means "this exists in
Drive", never "we have it"; that is `books`, joined on file_id.
"""
from pgvector import HalfVector

from library_rag import config
from library_rag.drive import client as drive_client
from library_rag.pipeline import embed as embed_mod

FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"

EMBED_CHUNK = 512  # rows fetched per embed round; embed_documents batches within


# ------------------------------------------------------------------- sync --

def _row(item: dict) -> tuple:
    parents = item.get("parents") or []
    size = item.get("size")
    return (
        item["id"],
        item.get("name", ""),
        item.get("mimeType", ""),
        parents[0] if parents else None,
        int(size) if size is not None else None,
        item.get("md5Checksum"),
        item.get("webViewLink"),
    )


def sync(conn, service, *, progress=None) -> dict:
    """Replace the mirror with what Drive currently holds. Returns counts.

    Folders and PDFs are fetched as two queries rather than one unfiltered
    listing: the drive holds plenty of files we will never browse (.DS_Store,
    index.html, epubs), and pulling them would triple the row count to show
    nothing.

    Rows Drive no longer returns are DELETED, not left behind. A stale row is
    worse than a missing one -- it renders as a browsable book, and clicking
    Index on it fails at download with a 404 the user cannot act on.
    """
    counts = {}
    seen = set()
    for label, mime in (("folders", FOLDER_MIME), ("files", PDF_MIME)):
        items = drive_client._list(service, f"mimeType = '{mime}' and trashed = false")
        counts[label] = len(items)
        seen.update(i["id"] for i in items)
        if progress:
            progress(f"{label}: {len(items)}")
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO drive_files
                    (file_id, name, mime_type, parent_id, size_bytes, md5,
                     web_view_link, synced_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (file_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        mime_type = EXCLUDED.mime_type,
                        parent_id = EXCLUDED.parent_id,
                        size_bytes = EXCLUDED.size_bytes,
                        md5 = EXCLUDED.md5,
                        web_view_link = EXCLUDED.web_view_link,
                        synced_at = now(),
                        -- A renamed file must be re-embedded: its vector
                        -- describes the old title. Leaving the stale vector in
                        -- place is worse than having none, because search would
                        -- keep matching a name that no longer exists.
                        embedding = CASE WHEN drive_files.name IS DISTINCT FROM EXCLUDED.name
                                         THEN NULL ELSE drive_files.embedding END
                """,
                [_row(i) for i in items],
            )
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM drive_files WHERE NOT (file_id = ANY(%s))", (list(seen),)
        )
        counts["removed"] = cur.rowcount
    conn.commit()
    counts["paths"] = materialise_paths(conn)
    counts["out_of_scope"] = prune_outside_root(conn)
    return counts


def prune_outside_root(conn, root_id=None) -> int:
    """Drop everything not descended from the configured library root.

    Drive's files.list has no "within this subtree" filter, so a sync pulls
    every file the account can see. On a personal account that is coursework,
    sheet music and tax documents alongside the library -- 126 such files on the
    first real run. They are worse than noise in a browse UI, so they are removed
    after paths exist rather than filtered during the listing, which is the only
    point at which descent is knowable.

    root_id=None with no configured root mirrors the whole drive, unpruned.
    """
    root_id = root_id or config.DRIVE_ROOT_FOLDER_ID
    if not root_id:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE keep AS (
                SELECT file_id FROM drive_files WHERE file_id = %s
              UNION ALL
                SELECT d.file_id FROM drive_files d JOIN keep k ON d.parent_id = k.file_id
            )
            DELETE FROM drive_files
            WHERE file_id NOT IN (SELECT file_id FROM keep)
            """,
            (root_id,),
        )
        n = cur.rowcount
    conn.commit()
    return n


# -------------------------------------------------------------- the tree --

def materialise_paths(conn) -> int:
    """Fold the flat rows into display paths, and derive the search text.

    A recursive CTE, run once after the rows land, rather than computed per row
    during sync -- the syncer sees files in arbitrary order, so a child often
    arrives before its parent has a path to extend.

    Roots are rows whose parent_id is NULL *or* points outside the mirror. That
    second case is not an edge case: the shared drive's own top folder has a
    parent we were never granted access to, so without it the recursion starts
    nowhere and every path comes back NULL.

    search_text is set in the same statement because it is derived from `path`
    and is therefore uncomputable until this point.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE roots AS (
                SELECT d.file_id, d.name AS path
                FROM drive_files d
                LEFT JOIN drive_files p ON p.file_id = d.parent_id
                WHERE d.parent_id IS NULL OR p.file_id IS NULL
            ),
            tree AS (
                SELECT file_id, path FROM roots
              UNION ALL
                SELECT d.file_id, t.path || ' / ' || d.name
                FROM drive_files d
                JOIN tree t ON d.parent_id = t.file_id
            )
            UPDATE drive_files d
            SET path = t.path,
                search_text = regexp_replace(
                    regexp_replace(t.path, '\(\s*pdfdrive\s*\)|\.pdf|_+', ' ', 'gi'),
                    '\s+', ' ', 'g')
            FROM tree t
            WHERE d.file_id = t.file_id
              AND d.path IS DISTINCT FROM t.path
            """
        )
        n = cur.rowcount
    conn.commit()
    return n


# ------------------------------------------------------------ embeddings --

# What gets embedded is drive_files.search_text, written by materialise_paths
# above. `tsv` is generated from that same column, so the dense and lexical legs
# cannot disagree about what a book is called -- cleaning the text twice, in two
# languages, is a drift waiting to happen.
#
# It is the full PATH, cleaned, not the filename: the folders frequently carry
# more signal than the file does. "Books / Commentaries / Romans / jbsg_rom.pdf"
# is only findable through them.

def pending_count(conn) -> int:
    return conn.execute(
        "SELECT count(*) FROM drive_files "
        "WHERE mime_type = %s AND embedding IS NULL", (PDF_MIME,)
    ).fetchone()[0]


def embed_titles(conn, voyage, *, limit=None, progress=None) -> int:
    """Embed titles that have none. Returns how many were embedded.

    Resumable by construction: the selection is `embedding IS NULL`, so killing
    this halfway and re-running continues rather than restarting. That matters
    at ~450 batches -- an all-or-nothing pass would be a bad bet on the network.

    Only PDFs. Folders are navigated, not searched.
    """
    done = 0
    while limit is None or done < limit:
        take = EMBED_CHUNK if limit is None else min(EMBED_CHUNK, limit - done)
        rows = conn.execute(
            "SELECT file_id, search_text FROM drive_files "
            "WHERE mime_type = %s AND embedding IS NULL "
            "ORDER BY file_id LIMIT %s",
            (PDF_MIME, take),
        ).fetchall()
        if not rows:
            break
        texts = [r[1] for r in rows]
        vectors, _ = embed_mod.embed_documents(texts, voyage)
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE drive_files SET embedding = %s WHERE file_id = %s",
                [(HalfVector(v), r[0]) for v, r in zip(vectors, rows)],
            )
        # Commit per chunk, not at the end: an interrupted run must keep the
        # work it already paid Voyage for.
        conn.commit()
        done += len(rows)
        if progress:
            progress(f"embedded {done} (remaining {pending_count(conn)})")
    return done


def build_index(conn) -> None:
    """HNSW over the title embeddings. Built after the embed pass, never before
    -- indexing an empty table then filling it is strictly more work than the
    reverse (same reasoning as chunks_hnsw)."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS drive_files_hnsw ON drive_files "
        "USING hnsw (embedding halfvec_cosine_ops)"
    )
    conn.commit()
