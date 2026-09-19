-- Editing the Bible map's data in the app.
--
-- 0019 stored the source files verbatim so the viewer could SHOW them. This
-- makes that copy the thing you edit, and the map a derivation of it: a cell
-- change re-runs the same parser the loader runs and replaces places/people/
-- events. The spreadsheets in biblemap_project become an import and export
-- format rather than the source of truth.

-- Who touched a row, and when. `origin` separates rows that came from a file
-- from rows the app created, which is what tells the loader that a re-import
-- would destroy work (cli/biblemap.py refuses without --force) and what the
-- viewer marks with a dot.
ALTER TABLE biblemap.source_rows
    ADD COLUMN origin    TEXT NOT NULL DEFAULT 'file',   -- 'file' | 'app'
    ADD COLUMN edited_at TIMESTAMPTZ,
    ADD COLUMN editor    TEXT;

-- Inserting a row "directly after row 7" renumbers every row below it, and a
-- single UPDATE that shifts row_num by one collides with itself row by row
-- while the statement is still running. Deferring the check to COMMIT lets the
-- renumber be one statement; INITIALLY IMMEDIATE keeps every other write
-- checked as it happens.
ALTER TABLE biblemap.source_rows DROP CONSTRAINT source_rows_pkey;
ALTER TABLE biblemap.source_rows
    ADD CONSTRAINT source_rows_pkey PRIMARY KEY (file_key, row_num)
    DEFERRABLE INITIALLY IMMEDIATE;

-- Position and identity, split.
--
-- event_id used to BE the position: the first event was 1, the second 2. That
-- made inserting a row in the middle renumber every event after it, so
-- /biblemap#event=40 would silently start meaning a different event and every
-- link anyone had kept would point somewhere plausible and wrong.
--
-- So `seq` carries the order (the timeline sorts by it) and event_id becomes a
-- stable name: an existing event keeps its id forever, and a new one takes the
-- next free number regardless of where it sits in the story. They are equal
-- today and will drift apart with the first insert, which is the point.
ALTER TABLE biblemap.events ADD COLUMN seq INTEGER;
UPDATE biblemap.events SET seq = event_id;
ALTER TABLE biblemap.events ALTER COLUMN seq SET NOT NULL;
CREATE UNIQUE INDEX events_seq_idx ON biblemap.events (seq);

-- Every change, append-only. Not an audit trail for its own sake: when the map
-- says something surprising, "was this in the export, or did someone type it
-- last Tuesday" is the first question, and `origin` alone cannot answer it.
-- It is also everything an undo button would need.
CREATE TABLE biblemap.source_edits (
    id          BIGSERIAL PRIMARY KEY,
    at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- NULL when the app runs with sign-in off, as it does locally. Same call
    -- 0017 made for research_runs.owner_email.
    editor      TEXT,
    file_key    TEXT NOT NULL,
    row_num     INTEGER NOT NULL,
    action      TEXT NOT NULL,          -- 'insert' | 'update' | 'delete'
    column_name TEXT,                   -- NULL for insert/delete
    old_value   TEXT,
    new_value   TEXT,
    -- The whole row as it was, so a delete can be described (and one day undone)
    -- after the row itself is gone.
    row_snapshot JSONB
);

CREATE INDEX source_edits_at_idx ON biblemap.source_edits (at DESC);
