-- A local mirror of the Drive metadata, so the library can be browsed and
-- searched without talking to Drive.
--
-- Why mirror at all: the drive holds ~57,500 PDFs across ~8,200 folders, and a
-- full listing is 58 paginated API calls taking ~110 seconds. Nothing
-- interactive can afford a Drive round trip per click, and Drive's own `name
-- contains` matches a WORD PREFIX only -- 'parab' finds "Parables", 'arable'
-- finds nothing -- so finding a book means already knowing a word in its title.
-- Mirrored, the tree is an indexed SQL lookup and the search can be semantic.
--
-- This is metadata ONLY. No file contents, no downloads. A row here means "this
-- exists in Drive", not "we have it" -- that is `books`, joined on file_id.

CREATE TABLE drive_files (
    file_id       TEXT PRIMARY KEY,     -- Drive's own id, stable across renames
    name          TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    -- Drive stores containment as an array on the CHILD, and a file may have
    -- more than one parent. We keep the first: this is a browsable view, not a
    -- faithful reproduction of Drive's graph, and a file appearing under two
    -- folders would otherwise need a join table to say something the UI has no
    -- way to show.
    parent_id     TEXT,
    size_bytes    BIGINT,
    md5           TEXT,
    web_view_link TEXT,
    -- "Books / Commentaries / Romans". Materialised after each sync by a
    -- recursive CTE rather than written per row: the syncer sees files in
    -- arbitrary order and a parent's own path may not exist yet when its child
    -- arrives. Computed once, at the end, when every edge is present.
    path          TEXT,
    embedding     HALFVEC(1024),
    -- Name AND path, because the folder often carries more signal than the
    -- filename ("Books / Commentaries / Romans" vs "jbsg_rom.pdf").
    tsv           tsvector GENERATED ALWAYS AS (
                      to_tsvector('english',
                          coalesce(name, '') || ' ' || coalesce(path, ''))
                  ) STORED,
    synced_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The tree lookup: children of one folder. Every expand in the browser is this.
CREATE INDEX ON drive_files (parent_id);
CREATE INDEX ON drive_files USING GIN (tsv);

-- Deliberately NO foreign key to books. A Drive file exists whether or not we
-- hold it, and most never will be held; "is this indexed" is a LEFT JOIN on
-- books.source_id, not a constraint.
--
-- HNSW is deliberately NOT created here, for the same reason as chunks_hnsw in
-- 0001: building it against an empty table is wasted work. It is built after
-- the title-embedding pass by `drive_sync --index`.
