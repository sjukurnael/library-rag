"""
Build the librarian's book profiles.

    python -m library_rag.cli.profile              # backfill what is missing
    python -m library_rag.cli.profile --limit 500
    python -m library_rag.cli.profile --rebuild    # every done book, from scratch
    python -m library_rag.cli.profile --status

Safe to run while ingestion is running, and safe to interrupt. It reads chunk
embeddings that are already stored and writes one small row set per book, so
there is no download, no API call and nothing to pay for -- which is why the
default is a resumable backfill rather than a one-shot pass: a book that
finishes ingesting after this starts is simply picked up by the next run.
"""
import argparse
import sys
import time

from library_rag import db
from library_rag.pipeline import profile


def run_status() -> None:
    with db.get_conn() as conn:
        c = db.profile_counts(conn)
    pct = 100.0 * c["profiled"] / c["done"] if c["done"] else 0.0
    print(f"{'searchable books':<22}{c['done']:>8}")
    print(f"{'profiled':<22}{c['profiled']:>8}   ({pct:.1f}%)")
    print(f"{'topic vectors':<22}{c['vectors']:>8}")
    print(f"{'  from headings':<22}{c['from_headings']:>8}")
    print(f"{'  from text only':<22}{c['from_text']:>8}")


def run_build(limit: int | None, rebuild: bool) -> None:
    with db.get_conn() as conn:
        if rebuild:
            sql = "SELECT id FROM books WHERE status = 'done' ORDER BY id"
            ids = [r[0] for r in conn.execute(sql).fetchall()]
            if limit:
                ids = ids[:limit]
        else:
            ids = db.books_needing_profile(conn, limit)

        if not ids:
            print("Nothing to profile.")
            return

        print(f"Profiling {len(ids)} book(s) ...")
        t0, vectors, empty = time.time(), 0, 0
        for n, book_id in enumerate(ids, 1):
            try:
                res = profile.build(conn, book_id)
            except Exception as exc:            # one bad book must not stop the pass
                print(f"  [{book_id}] FAILED: {exc}")
                continue
            vectors += res["vectors"]
            if not res["vectors"]:
                empty += 1
            if n % 100 == 0 or n == len(ids):
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(ids)}  {vectors} vectors  {rate:.0f} books/s")
        el = time.time() - t0
        print(f"Done: {len(ids)} books, {vectors} vectors, {empty} empty, {el:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--limit", type=int, default=None, help="Profile at most N books.")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild every done book, not only the unprofiled ones.")
    p.add_argument("--status", action="store_true", help="Print coverage and exit.")
    args = p.parse_args()

    if args.status:
        run_status()
    else:
        run_build(args.limit, args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(main())
