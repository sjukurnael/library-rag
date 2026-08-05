-- A book can now come from two places: a Drive folder, or a file the user
-- uploaded. Before this, `drive_file_id` was already carrying both -- run_local
-- stuffed "local:<filename>" into a column whose name promises a Drive id --
-- and a column name that lies is a bug waiting for someone to trust it.
--
-- source_id is the book's identity WITHIN its source, and stays UNIQUE:
--   drive  -> the Drive file id
--   upload -> "upload:<md5 of the bytes>"
--
-- Content-addressing uploads (rather than keying on filename) means re-uploading
-- the identical file is a no-op instead of a duplicate, and two different books
-- that happen to share a filename can never overwrite each other.

ALTER TABLE books RENAME COLUMN drive_file_id TO source_id;

ALTER TABLE books ADD COLUMN source TEXT NOT NULL DEFAULT 'drive';

-- Books ingested through the old --local path are uploads that predate the
-- column, so label them as such.
--
-- Re-keying their source_id to "upload:<md5>" is deliberately NOT done here.
-- It needs the file's md5 and a copy into data/uploads/, and a SQL migration
-- can do neither -- it would rewrite the id to point at a file that was never
-- put there. That half of the move belongs to a one-off script with filesystem
-- access, run once alongside this migration.
UPDATE books SET source = 'upload' WHERE source_id LIKE 'local:%';

ALTER TABLE books ADD CONSTRAINT books_source_check
    CHECK (source IN ('drive', 'upload'));

CREATE INDEX ON books (source);
