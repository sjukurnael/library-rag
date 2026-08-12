-- The Bible, one row per verse. English only.
--
-- The Berean Standard Bible, dedicated to the public domain in April 2023 and
-- downloaded from bereanbible.com. 31,102 verses across 66 books, ~7 MB. For
-- scale, one `chunks` row already carries 2 KB of embedding, so the whole
-- Bible costs less disk than about 3,500 chunks.
--
-- In `public`, not its own schema. `bible_verses` cannot collide with anything
-- here, and not having to remember a schema qualifier in every query is worth
-- more than namespace tidiness for a single table.

CREATE TABLE bible_verses (
    book       SMALLINT NOT NULL,   -- 1..66 in canonical order, Genesis..Revelation
    -- Denormalised on purpose. 66 names repeated across 31k rows costs a few
    -- hundred KB and removes a join from every query the app makes. It also
    -- means the table reads correctly on its own in the Supabase editor, with
    -- no lookup table to cross-reference.
    book_name  TEXT     NOT NULL,   -- 'Genesis'. NB the source says 'Psalm', singular.
    chapter    SMALLINT NOT NULL,
    verse      SMALLINT NOT NULL,
    -- '' for the 16 verses the BSB carries as an empty placeholder: Matthew
    -- 17:21, Mark 9:44, John 5:4, Acts 8:37 and the rest. They are in the KJV
    -- and absent from modern critical editions, and the number is kept so that
    -- Matthew 17 does not jump from verse 20 to 22 and read as a loader bug.
    text       TEXT     NOT NULL,

    -- The natural key IS the identity. A surrogate id would be a second way to
    -- name the same verse, and this one is the way anybody would actually cite
    -- it. The index it creates is also the only index the table needs: reading
    -- a chapter is `WHERE book = ? AND chapter = ?`, a leftmost-prefix lookup.
    PRIMARY KEY (book, chapter, verse)
);

-- Deliberately no index for search. It is `text ILIKE '%...%'`, which no btree
-- can serve, over 31k rows of ~7 MB -- a sequential scan measured in tens of
-- milliseconds. The fix if that ever stops being true is one pg_trgm GIN index,
-- and adding it before it is needed is an unexplained thing in the schema.
