-- Classrooms: the shelf a reader assembles before asking anything.
--
-- Retrieval is scoped now (see 0010). Nothing searches all 1.76M chunks, so
-- something has to say WHICH books a question is asked against, and that is a
-- classroom: a named set of books, filled by the librarian or by hand, that the
-- tutor may not look outside of.
--
-- Scoping is not only a performance property. Measured on this corpus, a
-- 20-book scope answers in ~15 ms through chunks_book_id_idx and does it
-- EXACTLY, where the global HNSW it replaced was both approximate and, on a
-- filtered query, silently wrong. But the reason to build it this way is that
-- scope is signal: a reader who says "I am studying the atonement" and picks
-- fifteen books has told us something no ranking function could infer from the
-- question alone.

CREATE TABLE classrooms (
    id           BIGSERIAL PRIMARY KEY,

    -- Who may open it. TEXT rather than a FK to allowed_users(email): access is
    -- granted and revoked by editing that table, and a hard reference would
    -- mean revoking someone's access deletes their work rather than hiding it.
    owner_email  TEXT NOT NULL,

    name         TEXT NOT NULL,
    -- What the reader actually typed. Kept separate from `name` because the
    -- brief is what the librarian searched on -- worth showing next to the
    -- shelf so a reader can see why these books and not others, and worth
    -- re-running verbatim when they come back to widen it.
    brief        TEXT,

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Ordering for the classroom list. Touched on every tutor turn, so the
    -- shelf someone is actually working in floats to the top.
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX classrooms_owner_idx ON classrooms (owner_email, last_used_at DESC);


CREATE TABLE classroom_books (
    classroom_id BIGINT NOT NULL REFERENCES classrooms(id) ON DELETE CASCADE,
    -- ON DELETE CASCADE deletes the MEMBERSHIP, never the book. A book removed
    -- from the library is gone from every shelf that held it, which is right;
    -- emptying a shelf must not touch the library, which is why the cascade
    -- only ever runs in this direction.
    book_id      BIGINT NOT NULL REFERENCES books(id) ON DELETE CASCADE,

    -- 'librarian' or 'reader'. Worth recording because the two are answerable
    -- to different things: a librarian pick can be judged against the passage
    -- it quoted, a reader's pick is simply what they wanted.
    added_by     TEXT NOT NULL,
    -- The passage look_inside returned when the book was recommended -- the
    -- evidence for this pick, not a summary of the book. Nullable: a book added
    -- by hand from the Drive view has no such evidence, and inventing one would
    -- be worse than leaving it blank.
    rationale    TEXT,

    added_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One row per book per classroom. Adding a book twice is a no-op, not an
    -- error and not a duplicate on the shelf.
    PRIMARY KEY (classroom_id, book_id)
);

-- "Which classrooms hold this book?" -- asked when a book is deleted, and when
-- the Drive view marks a row as already on the current shelf.
CREATE INDEX classroom_books_book_idx ON classroom_books (book_id);


-- Tutor turns belong to a classroom.
--
-- The table is still called research_runs, and the name is now wrong: what runs
-- against a classroom is the tutor. Renaming it would touch every db.*_research_*
-- helper and both api.py routes to rename a string, so the column goes here and
-- the mismatch is recorded rather than chased. Every OTHER column still says
-- what it means -- question, searches, stop_reason and the four token counters
-- describe a tutor turn exactly as well as they described a research one.
--
-- Nullable, because rows written before this migration have no classroom and
-- because the un-scoped research chat still exists. A NULL classroom_id means
-- "asked against the whole library", which is a real answer, not missing data.
ALTER TABLE research_runs ADD COLUMN classroom_id BIGINT REFERENCES classrooms(id);

CREATE INDEX research_runs_classroom_idx
    ON research_runs (classroom_id, started_at DESC);
