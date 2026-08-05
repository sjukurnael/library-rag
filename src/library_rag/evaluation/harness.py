"""
Retrieval quality harness: does search actually put the right passage in front
of the model?

    python -m evaluate                 # score the default (hybrid) config
    python -m evaluate --compare       # every mode x tsquery mode, side by side
    python -m evaluate --show-misses   # what came back instead
    python -m evaluate --min-hit-rate 0.80   # non-zero exit if it regressed

Why this exists: every other test in this repo asserts that a function does what
it says. None of them can tell you whether retrieval is any GOOD -- chunking,
the embedding model, the fusion, RRF_K, the candidate pool and the tsquery
semantics all silently trade quality against each other, and the only visible
symptom of getting one wrong is that answers quietly get worse. "The tests pass
and the demo looks fine" is not a measurement.

The metrics:
  hit-rate@k -- fraction of questions where an expected passage appears anywhere
                in the top k. This is what actually matters, because everything
                retrieved is handed to the model at once; rank 1 and rank 8 are
                both "the model can see it".
  MRR        -- mean of 1/rank of the first expected passage. Rank does still
                matter at the margin (a passage at rank 1 survives any later
                truncation), so MRR is reported as a tiebreaker between configs
                whose hit-rate is identical.

Ground truth is (book, page span), not chunk_id -- see eval/questions.json.

This module holds the scoring logic ONLY; the corpus it scores against is
whatever you point it at. tests/test_eval.py runs these same functions over a
small synthetic corpus with a deterministic embedder, which is how the harness
itself gets exercised in CI with no Postgres corpus and no Voyage key.
"""
import json
from pathlib import Path

from library_rag import db

# Package data, resolved relative to this module rather than the repo root --
# the question sets belong to the harness and travel with it.
QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
QUESTIONS_PATH = QUESTIONS_DIR / "questions.json"


# ------------------------------------------------------------- scoring --

def load_questions(path=None) -> dict:
    data = json.loads(Path(path or QUESTIONS_PATH).read_text())
    if not data.get("questions"):
        raise ValueError(f"no questions in {path or QUESTIONS_PATH}")
    return data


def matches(row, target) -> bool:
    """Is this retrieved chunk one of the passages we were hoping for?

    Book by case-insensitive substring, pages by OVERLAP rather than equality:
    chunk boundaries move whenever chunking changes, so a chunk covering pp.8-10
    must still count against a target of pp.9-10. Requiring containment would
    turn every re-chunk into a fake regression.
    """
    title = (row.get("title") or "").lower()
    if target["book"].lower() not in title:
        return False
    want_start, want_end = target["pages"]
    got_start, got_end = row.get("page_start"), row.get("page_end")
    if got_start is None or got_end is None:
        return False
    return got_start <= want_end and want_start <= got_end


def first_hit_rank(rows, expect) -> int | None:
    """1-based rank of the first retrieved chunk matching any expected target,
    or None if the top k contains none of them."""
    for rank, row in enumerate(rows, start=1):
        if any(matches(row, t) for t in expect):
            return rank
    return None


def score(per_question: list) -> dict:
    """Aggregate a list of {"rank": int|None} into hit-rate and MRR."""
    n = len(per_question)
    if not n:
        return {"n": 0, "hit_rate": 0.0, "mrr": 0.0, "hits": 0}
    ranks = [q["rank"] for q in per_question]
    hits = sum(1 for r in ranks if r is not None)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mrr": sum(1.0 / r for r in ranks if r is not None) / n,
    }


def run_questions(
    conn, questions, vectors, k, *, mode, tsquery_mode=None, lexical_weight=None
) -> list:
    """Score every question under one retrieval config.

    `vectors` maps question id -> embedding, passed in rather than computed here
    so a --compare run embeds each question once and every config is scored
    against the identical vector. Re-embedding per config would put Voyage's
    own variance inside the thing being measured.
    """
    out = []
    for q in questions:
        rows = db.search(
            conn,
            vectors[q["id"]],
            k,
            query_text=q["question"],
            mode=mode,
            tsquery_mode=tsquery_mode,
            lexical_weight=lexical_weight,
        )
        out.append(
            {
                "id": q["id"],
                "question": q["question"],
                "rank": first_hit_rank(rows, q["expect"]),
                "rows": rows,
            }
        )
    return out
