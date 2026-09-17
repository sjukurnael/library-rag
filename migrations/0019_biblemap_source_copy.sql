-- A verbatim copy of the Bible map's source files, for the "Show in original
-- data" viewer.
--
-- 0018's tables hold what the loader MADE of the spreadsheets: parsed IDs,
-- NULL for "37?", a set where the cell repeated an ID. When the map looks
-- wrong, the question is whether the parse or the spreadsheet is at fault, and
-- that needs the spreadsheet as it was -- every column, including the 36 the
-- map never reads, as the text a person would see in the cell.
--
-- Loaded in the same transaction as 0018's tables (biblemap.load), so the copy
-- always matches the map built from it. Rows are JSONB arrays aligned with
-- source_files.columns rather than one SQL column per spreadsheet column: the
-- four files have 5 to 43 columns between them, and the viewer only ever shows
-- a row whole.

CREATE TABLE biblemap.source_files (
    file_key  TEXT PRIMARY KEY,          -- events | places | new_places | people
    filename  TEXT NOT NULL,             -- the file it was read from, for the header
    columns   JSONB NOT NULL,            -- header row, in file order
    row_count INTEGER NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE biblemap.source_rows (
    file_key  TEXT NOT NULL REFERENCES biblemap.source_files ON DELETE CASCADE,
    -- Numbered as a spreadsheet program numbers it, header = row 1, so a row
    -- found here is the same row number when the file is opened in Excel.
    row_num   INTEGER NOT NULL,
    -- The event, place or person this row became; NULL for a row that became
    -- nothing (a blank row, a place with no ID). Not a foreign key: places come
    -- from two files, and this is a pointer for the viewer, not a relationship.
    record_id INTEGER,
    cells     JSONB NOT NULL,            -- array of strings, aligned with columns
    PRIMARY KEY (file_key, row_num)
);

-- "Show in original data" on an event, place or person looks up by record.
CREATE INDEX source_rows_record_idx ON biblemap.source_rows (file_key, record_id);
