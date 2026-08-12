"""
Load the Bible into Postgres.

    python -m library_rag.cli.bible --load     # download if needed, parse, load
    python -m library_rag.cli.bible --status   # what is currently in the table

The Berean Standard Bible, public domain, from bereanbible.com. --load is safe
to re-run: it replaces the table's contents rather than adding to them, and it
only downloads when data/bsb.txt is missing.
"""
import argparse
import sys

from library_rag import bible, db


def run_load(force_download: bool) -> int:
    print(f"Reading {bible.BSB_FILE.name} ...")
    path = bible.download(force=force_download)
    rows = bible.parse(path)

    # A published text has a fixed verse count, so a different one means the
    # upstream file changed shape. Refuse rather than load something whose
    # parse we have no reason to trust.
    if len(rows) != bible.EXPECTED_VERSES:
        print(
            f"  ERROR: parsed {len(rows):,} verses, expected "
            f"{bible.EXPECTED_VERSES:,}.\n"
            f"  The file at {bible.BSB_URL} has changed shape. Nothing loaded.",
            file=sys.stderr,
        )
        return 1

    with db.get_conn() as conn:
        written = bible.load(conn, rows)
    print(f"Loaded {written:,} verses.")
    return 0


def run_status() -> int:
    with db.get_conn() as conn:
        if not bible.loaded(conn):
            print("No Bible loaded. Run --load.")
            return 0
        rows = bible.books(conn)
        total = sum(r[3] for r in rows)
        blank = conn.execute(
            "SELECT count(*) FROM bible_verses WHERE text = ''"
        ).fetchone()[0]

    print(f"{len(rows)} books, {total:,} verses ({blank} carried as empty).\n")
    for book, name, chapters, verses in rows:
        print(f"  {book:>2}  {name:<18}{chapters:>4} ch {verses:>6,} verses")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--load", action="store_true",
                        help="Download (if needed), parse and load the Bible.")
    parser.add_argument("--status", action="store_true",
                        help="Print the books and verse counts currently loaded.")
    parser.add_argument("--force-download", action="store_true",
                        help="With --load: re-download even if data/bsb.txt exists.")
    args = parser.parse_args()

    if args.load:
        return run_load(args.force_download)
    if args.status:
        return run_status()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
