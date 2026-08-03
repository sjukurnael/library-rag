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
import argparse
import json
import sys
from pathlib import Path

import config
import db

QUESTIONS_PATH = config.BASE_DIR / "eval" / "questions.json"


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


# ------------------------------------------------------------------ cli --

def _embed_all(questions) -> dict:
    from pipeline import embed as embed_mod

    client = embed_mod.build_client()
    print(f"Embedding {len(questions)} questions with {config.EMBED_MODEL} ...")
    return {q["id"]: embed_mod.embed_query(q["question"], client) for q in questions}


def _print_run(label, results, k):
    s = score(results)
    print(
        f"{label:<22} hit-rate@{k} {s['hit_rate']:6.1%}  "
        f"({s['hits']}/{s['n']})   MRR {s['mrr']:.3f}"
    )
    return s


def _print_misses(results):
    missed = [r for r in results if r["rank"] is None]
    if not missed:
        print("\nNo misses.")
        return
    print(f"\n{len(missed)} miss(es):")
    for r in missed:
        print(f"\n  {r['id']}: {r['question']}")
        for row in r["rows"][:3]:
            pages = f"pp.{row['page_start']}-{row['page_end']}"
            print(f"    got  {row['distance']:.4f}  {pages:<12} {row['title'][:52]}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", default=None, help="Path to a question set.")
    parser.add_argument("-k", type=int, default=None, help="Override the set's k.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Score every mode x tsquery mode instead of just the default.",
    )
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Exit 1 if the default config scores below this. For gating.",
    )
    args = parser.parse_args()

    data = load_questions(args.questions)
    questions = data["questions"]
    k = args.k or data.get("k", 8)
    vectors = _embed_all(questions)

    configs = (
        [
            ("dense", None),
            ("lexical", "and"),
            ("lexical", "or"),
            ("hybrid", "and"),
            ("hybrid", "or"),
        ]
        if args.compare
        else [(None, None)]
    )

    default = None
    with db.get_conn() as conn:
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        books = conn.execute(
            "SELECT count(*) FROM books WHERE status = 'done'"
        ).fetchone()[0]
        print(f"Corpus: {total} chunks across {books} books\n")
        for mode, tsq in configs:
            results = run_questions(
                conn, questions, vectors, k, mode=mode, tsquery_mode=tsq
            )
            label = (
                f"{mode}/{tsq}" if tsq else (mode or f"default ({config.LEXICAL_TSQUERY})")
            )
            s = _print_run(label, results, k)
            if mode is None or (mode == "hybrid" and tsq == config.LEXICAL_TSQUERY):
                default, default_results = s, results

    if args.show_misses and default is not None:
        _print_misses(default_results)

    if args.min_hit_rate is not None:
        if default is None:
            print("\nNo default config was scored; cannot gate.", file=sys.stderr)
            return 1
        if default["hit_rate"] < args.min_hit_rate:
            print(
                f"\nFAIL: hit-rate {default['hit_rate']:.1%} is below the "
                f"{args.min_hit_rate:.1%} floor.",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: hit-rate {default['hit_rate']:.1%} meets the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
