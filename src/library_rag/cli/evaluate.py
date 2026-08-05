"""
Score retrieval quality against the curated question sets.

    python -m library_rag.cli.evaluate
    python -m library_rag.cli.evaluate --compare
    python -m library_rag.cli.evaluate --show-misses
    python -m library_rag.cli.evaluate --min-hit-rate 0.80

The scoring itself lives in library_rag.evaluation.harness, which has no CLI so
tests/test_eval.py can run the same functions over a synthetic corpus.
"""
import argparse
import sys

from library_rag import config, db
from library_rag.evaluation.harness import (
    QUESTIONS_DIR,  # noqa: F401  -- re-exported for --questions help text
    load_questions,
    run_questions,
    score,
)


def _embed_all(questions) -> dict:
    from library_rag.pipeline import embed as embed_mod

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
