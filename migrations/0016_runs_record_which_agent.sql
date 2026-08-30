-- One table for both agents' runs.
--
-- The tutor's runs have been durable since 0009; the librarian's have not
-- existed at all after they finished -- no run id, no row, nothing to look at
-- the next morning. That was deliberate while its output was a shortlist you
-- acted on within the minute. It stops being right the moment anyone asks
-- "which books did it consider, and why did it drop that one" -- a question
-- about a run that no longer exists.
--
-- research_events is already generic: (run_id, seq, payload jsonb), with no
-- opinion about which agent produced the payload. So only the runs table needs
-- to learn the difference.
--
-- The name stays research_runs. Renaming it would touch every db.*_research_*
-- helper and api.py for no functional gain -- the same call 0012 made when it
-- hung classroom_id off a table whose name had already stopped being accurate.
-- What the columns hold, per agent:
--
--     question  -- the tutor's question, or the librarian's brief
--     answer    -- the tutor's answer, or the librarian's closing note
--     searches  -- searches run: search_library for one, find_books for the other
--
-- 'tutor' as the default backfills every existing row correctly: until now, a
-- run in this table could only have been a tutor run.

ALTER TABLE research_runs
    ADD COLUMN agent TEXT NOT NULL DEFAULT 'tutor'
        CHECK (agent IN ('tutor', 'librarian')),
    -- Which model actually ran it. Both loops report this on their 'done' event
    -- now that a reader can pick per run, and "what did this cost" is not
    -- answerable without it.
    ADD COLUMN model TEXT;

-- The runs page lists newest-first, filtered by agent.
CREATE INDEX ON research_runs (agent, started_at DESC);
