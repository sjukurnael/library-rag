-- The librarian's tier: a small, always-cached representation of every book.
--
-- Retrieval is scoped to a study session, so nothing searches all 2.9M chunks
-- any more (see 0010). But something still has to search all 43,000 BOOKS, and
-- that is what these two tables are for. The trick is granularity: one vector
-- per file is too coarse and 61 per book is too many, so a book gets ~6.
--
-- Measured on 203 books / 12,320 chunks, 22 eval questions with known
-- ground-truth books, ranking all 203 by best-matching vector:
--
--                            median rank   hit@1   hit@5   hit@10   hit@20
--     one whole-book vector       6        36.4%   50.0%    68.2%    81.8%
--     ~6 topic centroids          1        63.6%   77.3%    95.5%   100.0%
--
-- The single-vector row is why book_topic_vectors is multi-row. Averaging a
-- 900-page systematic theology into one point buries every chapter in it: a
-- question about the atonement matches "generic theology" weakly instead of
-- matching the atonement chapter strongly. Splitting into clusters keeps each
-- vector sharp, and is what makes a book findable by a topic its TITLE never
-- mentions -- the case the whole session design depends on.
--
-- Centroids are computed, not selected. k-means over the book's own chunk
-- embeddings; each centroid is the MEAN of its cluster, so it corresponds to no
-- actual chunk and is deliberately more general than any single passage. No
-- LLM, no API calls -- the vectors were already paid for at ingest.

CREATE TABLE book_profiles (
    book_id  BIGINT PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,

    -- Structure as text, for display and for the lexical leg. Preference
    -- order is recorded in `source` because it predicts profile quality:
    --   'outline'  -- the PDF's embedded bookmarks (doc.get_toc()), the
    --                author's own chapter titles. Best, and free: no text
    --                extraction needed.
    --   'headings' -- distinct heading_trail values from the chunks, i.e. a
    --                TOC recovered from pymupdf4llm's font-size mapping.
    --   'text'     -- neither was available; falls back to opening words.
    -- Measured on the first 203 books: 28 had no usable headings at all, but
    -- they averaged 4.6 chunks each -- fragments and front matter, which do not
    -- need structure because they only cover one thing. Structure-poor and
    -- short are the same population.
    toc      TEXT,
    source   TEXT NOT NULL,

    -- Deliberately unused for now. An LLM summary is the one thing centroids
    -- cannot capture -- averaging loses stance, so a chapter ARGUING FOR penal
    -- substitution and one REFUTING it produce similar centroids. Costed at
    -- ~$8 with Haiku if targeted only at the structure-poor books, which is why
    -- the column exists before the feature does: adding it to the tsv
    -- expression later would rewrite the table.
    summary  TEXT,

    tsv      tsvector GENERATED ALWAYS AS (
                 to_tsvector('english',
                     coalesce(toc, '') || ' ' || coalesce(summary, ''))
             ) STORED,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Worth indexing here, unlike on chunks: 43,000 rows, and the librarian's
-- lexical leg genuinely searches all of them. config.py already records that
-- over TITLES hybrid beat dense 8/8 vs 7/8 (MRR 0.812 vs 0.719); profiles are
-- the same kind of corpus -- short, proper-noun dense, full of exact terms
-- like "supralapsarian" that lexical catches and dense blurs.
CREATE INDEX book_profiles_tsv_idx ON book_profiles USING GIN (tsv);


CREATE TABLE book_topic_vectors (
    book_id   BIGINT NOT NULL REFERENCES books(id) ON DELETE CASCADE,

    -- 0 is the whole-book mean, 1..k are the topic clusters. Keeping both
    -- lets the librarian work at either altitude: "I'm studying systematic
    -- theology" matches no single chapter well but matches ordinal 0 strongly,
    -- while "penal substitution" matches one cluster. The ordinal tells the
    -- caller which kind of match it got.
    ordinal   INT NOT NULL,

    -- What a human is shown. A vector cannot be displayed, so a recommendation
    -- needs text to justify itself -- "matched SS4, The Self-Substitution of
    -- God" is a rationale a reader can check; "matched centroid 3" is not.
    -- Derived from the cluster's MEMBERS after clustering, never chosen before.
    label     TEXT,
    -- How many chunks this centroid averages. A book that matched on a cluster
    -- of 40 chunks covers the topic; one that matched a cluster of 2 mentions
    -- it. Different recommendations, and collapsing them into a single score
    -- would throw that away.
    members   INT NOT NULL,

    -- Full precision, for the exact rerank over the few hundred books that
    -- survive the binary pass. TOASTed, so it costs disk and not cache.
    embedding HALFVEC(1024) NOT NULL,

    -- What actually gets indexed. binary_quantize is IMMUTABLE, so this is a
    -- generated column and cannot drift from the vector it compresses.
    --
    -- Binary is free at THIS tier specifically. It loses ranking precision,
    -- which at chunk level would mean stacking approximation on approximation
    -- and needing an exact rerank -- but here a human reads the librarian's 30
    -- candidates and picks. The reader IS the rerank pass.
    --
    -- 136 bytes against halfvec's 2,052. At ~254,000 vectors (43,147 books,
    -- k scaled to length) that is ~94 MB rather than ~671 MB, which is the
    -- difference between fitting in a 1 GB instance's cache and not.
    bits      BIT(1024) GENERATED ALWAYS AS
                  (binary_quantize(embedding)::bit(1024)) STORED,

    PRIMARY KEY (book_id, ordinal)
);

CREATE INDEX book_topic_bits_hnsw
    ON book_topic_vectors USING hnsw (bits bit_hamming_ops);
