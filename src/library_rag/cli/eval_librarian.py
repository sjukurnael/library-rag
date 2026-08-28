"""
Does the librarian recommend books that are actually about the topic?

    python -m library_rag.cli.eval_librarian
    python -m library_rag.cli.eval_librarian --k 20 --detail

Scored against ABSENT topics: not against curated ground truth, and not against
random books.

Random books were the first attempt and it was a tautology. Nearest-neighbour
search returns nearer-than-random results by construction, so every brief passed
20/20 -- including "quantum chromodynamics" and "optimising the PostgreSQL query
planner", put to a library of theology. A test that cannot fail measures nothing.

What does discriminate is absolute distance. Real briefs reach a best passage at
0.26-0.36 cosine; briefs on subjects this corpus does not hold bottom out around
0.51-0.58, because the nearest thing to a nonsense query is still not near. So
the absent briefs set the bar: whatever they can reach is what "no real match"
looks like in this embedding space, and a genuine recommendation must beat it.
Recomputed every run, so it follows the corpus and survives a change of model.

This exists because questions.json cannot answer the question any more. It pins
each question to one book -- true at 203 books, false at 8,680, where "John"
resolves to 424 titles. Measured against it the librarian scored 32% hit@10
while returning, for a providence question, three books about providence ranked
above the one the question happened to be transcribed from. That is not a
retrieval failure; it is a ground truth that expired.
"""
import argparse
import json
import statistics
import sys

from library_rag import config, db
from library_rag.pipeline import embed as embed_mod

BRIEFS_FILE = config.PROJECT_ROOT / "src/library_rag/evaluation/briefs.json"
# Percentile of the absent briefs' distances to place the bar at. 10, not 50:
# the floor is the BEST a nonsense query manages, not its typical result.
ABSENT_PERCENTILE = 10


def _best_distance(conn, book_id, vec):
    """How near this book's closest passage gets to the brief. None if empty."""
    rows = db.look_inside(conn, book_id, vec, k=1)
    return float(rows[0]["distance"]) if rows else None


def _percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    return s[max(0, min(len(s) - 1, round(pct / 100 * (len(s) - 1))))]


def _distances(conn, brief, k, voyage):
    """(best-passage distance, recommendation row) for the top-k books."""
    vec = embed_mod.embed_query(brief, voyage)
    out = []
    for r in db.search_book_profiles(conn, vec, brief, k=k):
        d = _best_distance(conn, r["book_id"], vec)
        if d is not None:
            out.append((d, r))
    return out


def run(k: int, detail: bool) -> int:
    spec = json.loads(BRIEFS_FILE.read_text())
    voyage = embed_mod.build_client()
    rows = []

    with db.get_conn() as conn:
        total = conn.execute(
            "SELECT count(*) FROM books WHERE status = 'done'"
        ).fetchone()[0]
        print("corpus: {:,} searchable books   |   {} briefs, {} absent controls, "
              "top-{}\n".format(total, len(spec["briefs"]), len(spec["absent"]), k))

        # Calibrate: how near can a query on a subject the corpus lacks get?
        floor = []
        print("{:<52}{:>8}{:>9}".format("ABSENT control", "best", "median"))
        print("-" * 70)
        for brief in spec["absent"]:
            ds = [d for d, _ in _distances(conn, brief, k, voyage)]
            floor.extend(ds)
            print("{:<52}{:>8.3f}{:>9.3f}".format(
                brief[:50], min(ds), statistics.median(ds)))
        bar = _percentile(floor, ABSENT_PERCENTILE)
        print("-" * 70)
        print("bar = {}th percentile of what absent topics reach: {:.3f}\n".format(
            ABSENT_PERCENTILE, bar))

        for brief in spec["briefs"]:
            ds = _distances(conn, brief, k, voyage)
            rows.append({
                "brief": brief, "n": len(ds),
                "justified": sum(1 for d, _ in ds if d <= bar),
                "best": min(d for d, _ in ds),
                "median": statistics.median([d for d, _ in ds]),
            })
            if detail:
                print("  " + brief)
                for d, r in ds[:5]:
                    print("    {} {:.3f}  topics={:<3} {}".format(
                        "OK " if d <= bar else "   ", d,
                        r["matched_topics"] or 0, r["title"][:52]))
                print()

    print("{:<52}{:>8}{:>9}{:>12}".format("brief", "best", "median", "justified"))
    print("-" * 82)
    for r in rows:
        print("{:<52}{:>8.3f}{:>9.3f}{:>8}/{:<3}".format(
            r["brief"][:50], r["best"], r["median"], r["justified"], r["n"]))
    print("-" * 82)
    tj = sum(r["justified"] for r in rows)
    tn = sum(r["n"] for r in rows)
    print("{:<52}{:>8}{:>9}{:>8}/{:<3}  ({:.0f}%)".format(
        "OVERALL", "", "", tj, tn, 100 * tj / tn))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=20, help="Recommendations per brief.")
    p.add_argument("--detail", action="store_true", help="Show the top 5 per brief.")
    a = p.parse_args()
    return run(a.k, a.detail)


if __name__ == "__main__":
    sys.exit(main())
