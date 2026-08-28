-- Research runs, moved out of process memory.
--
-- An agent run is long-running work that outlives the request which started it
-- -- the same shape as a book ingest, and nothing like an ordinary endpoint. It
-- was previously a dict in the web process, which fails on two axes:
--
--   across TIME  -- a restart, a redeploy, or a scale-to-zero loses every
--                   in-flight run, and Cloud Run does all three routinely;
--   across SPACE -- the POST that mints a run_id and the GET that streams it
--                   are separate requests, so with more than one container the
--                   GET can land on an instance that has never heard of the run
--                   and answers 404 blaming a restart that never happened.
--
-- Both axes are one fix, and it is the fix `books` already uses: the state is a
-- row, so any process that can reach Postgres can serve it.
--
-- The secondary payoff is the reason to want this even on a single instance.
-- Every failure above used to destroy the only record that the run happened, so
-- "how often does this break" was unanswerable -- the evidence was deleted by
-- the same event that caused the problem. A failed run is now a row you can
-- count, exactly like a book sitting at status 'failed'.

CREATE TYPE research_status AS ENUM ('running', 'done', 'failed', 'interrupted');

CREATE TABLE research_runs (
    -- The id handed to the browser. TEXT rather than BIGSERIAL because it is
    -- minted by secrets.token_hex before any row exists, and because a run id
    -- travels to the client: a sequential integer would let anyone enumerate
    -- other people's questions by counting.
    run_id       TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    status       research_status NOT NULL DEFAULT 'running',

    -- The outcome, once there is one.
    answer       TEXT,
    -- Anthropic's stop_reason for the final turn. 'end_turn' is a real answer,
    -- 'max_tokens' is a truncated one, 'refusal' is an empty one -- and the loop
    -- currently treats all three identically, presenting the last two as
    -- finished work. Recording it makes that countable before it is fixed;
    -- changing the loop's behaviour is a separate decision from measuring it.
    stop_reason  TEXT,
    iterations   INT,
    searches     INT,
    error        TEXT,

    -- Cost. Four counters rather than one total because the interesting number
    -- is cache_read against input: that ratio is the whole answer to "is prompt
    -- caching working". Nothing else in this system costs money per call, and
    -- the SDK's usage object was being dropped on the floor.
    input_tokens       BIGINT NOT NULL DEFAULT 0,
    output_tokens      BIGINT NOT NULL DEFAULT 0,
    cache_read_tokens  BIGINT NOT NULL DEFAULT 0,
    cache_write_tokens BIGINT NOT NULL DEFAULT 0,

    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    -- Bumped on every event. A row left at 'running' with an old heartbeat is a
    -- run whose worker died: the same reaper idea as books.claimed_at, and the
    -- only way a reader on a DIFFERENT instance can tell "still thinking" from
    -- "the container doing this is gone". Without it, moving the state to
    -- Postgres would trade a 404 for a stream that hangs forever.
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- For "what has this agent been doing lately", which is the query the whole
-- table exists to make possible.
CREATE INDEX ON research_runs (started_at DESC);

CREATE TABLE research_events (
    run_id  TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    -- 0-based and dense. This IS the `after` cursor the SSE route already
    -- takes, so a browser holding a saved offset keeps working unchanged:
    -- `after=N` still means "I have seen 0..N-1, send me N onwards".
    seq     INT NOT NULL,
    -- The event dict exactly as the loop yielded it. JSONB rather than columns
    -- because the shape is the loop's to define and it varies by type -- and
    -- because `payload->>'query'` is enough to answer the retrieval questions
    -- this table was added for.
    payload JSONB NOT NULL,
    -- Doubles as the index for the range scan the stream does every poll.
    PRIMARY KEY (run_id, seq)
);
