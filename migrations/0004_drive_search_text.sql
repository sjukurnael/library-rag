-- Fix: the lexical leg could not match the last word of any title.
--
-- Postgres's default text parser classifies "Parables.pdf" as a *file* token,
-- not a word, so to_tsvector('english', 'Snodgrass_Stories With Intent
-- Parables.pdf') yields:
--
--     'intent':4  'parables.pdf':5  'snodgrass':1  'stori':2
--
-- and a search for "parables" matched nothing. Every filename in this corpus
-- ends in .pdf, so this silently broke the final -- usually most specific --
-- word of all 57,527 titles. Invisible with short synthetic test names, obvious
-- the moment a real one is indexed.
--
-- The same noise hurt the dense leg, where Python was stripping it separately
-- before embedding. Two cleaning implementations of the same text is a drift
-- waiting to happen, so the cleaning becomes ONE column that both legs read:
-- `tsv` is generated from it, and the embed pass selects it rather than
-- recomputing.

ALTER TABLE drive_files DROP COLUMN tsv;

-- Plain, not generated: a generated column cannot reference another generated
-- column, and `tsv` must be derived from this. Written by
-- mirror.materialise_paths, in the same statement as `path` -- it depends on
-- `path`, which is itself only computable once every edge has landed.
ALTER TABLE drive_files ADD COLUMN search_text TEXT;

ALTER TABLE drive_files ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(search_text, name))) STORED;

CREATE INDEX ON drive_files USING GIN (tsv);

-- Existing rows: recompute in place rather than forcing a full Drive re-sync.
-- Same expression mirror.materialise_paths uses -- strip the ".pdf" suffix, the
-- "( PDFDrive )" watermark this corpus is full of, and underscore separators,
-- each of which otherwise indexes as a term and dilutes the real words.
UPDATE drive_files
SET search_text = regexp_replace(
        regexp_replace(coalesce(path, name), '\(\s*pdfdrive\s*\)|\.pdf|_+', ' ', 'gi'),
        '\s+', ' ', 'g');

-- Every title was embedded from the Python-cleaned text, which differed from
-- this. Clear the vectors so the next embed pass rebuilds them from the column
-- both legs now share. embed_titles is resumable, so this is a re-run rather
-- than something that has to succeed atomically.
UPDATE drive_files SET embedding = NULL;
