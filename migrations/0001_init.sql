-- Phase 1 initial schema. Applied by `python -m migrate` (see migrate.py).
-- The HNSW index is deliberately NOT created here -- it is built once, after
-- bulk load, by `python ingest.py --index` (building it against an empty or
-- partial table is wasted work).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE doc_status AS ENUM
  ('discovered','downloaded','extracted','chunked','done','failed','needs_ocr');

CREATE TABLE books (
    id             BIGSERIAL PRIMARY KEY,
    drive_file_id  TEXT UNIQUE NOT NULL,
    title          TEXT,
    md5            TEXT,
    size_bytes     BIGINT,
    page_count     INT,                 -- from PyMuPDF
    has_text_layer BOOLEAN,
    status         doc_status NOT NULL DEFAULT 'discovered',
    attempts       INT NOT NULL DEFAULT 0,
    claimed_at     TIMESTAMPTZ,
    error          TEXT,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id            BIGSERIAL PRIMARY KEY,
    book_id       BIGINT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    ordinal       INT NOT NULL,
    heading_trail TEXT,                 -- "Ch 3 > The Exile > Return"
    page_start    INT,
    page_end      INT,
    content       TEXT NOT NULL,
    token_count   INT,
    embedding     HALFVEC(1024),        -- voyage-3 dimension; halfvec = half the RAM
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (book_id, ordinal)
);
CREATE INDEX ON chunks (book_id);
CREATE INDEX ON chunks USING GIN (tsv);
-- HNSW deliberately NOT here; built by `ingest.py --index` AFTER bulk load:
--   CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding halfvec_cosine_ops);
