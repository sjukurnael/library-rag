"""
Load the Bible map into Postgres.

    python -m library_rag.cli.biblemap --load ../biblemap_project
    python -m library_rag.cli.biblemap --status

SOURCE_DIR is the folder holding the spreadsheets: Places-Data/, People Data/
and "Bible Events with Resolved PersonIDs and PlaceIDs.xlsx". --load is safe to
re-run: it replaces the biblemap schema's contents rather than adding to them.
"""
import argparse
import sys
from pathlib import Path

from library_rag import biblemap, db


def run_load(source_dir: Path) -> int:
    print(f"Reading {source_dir} ...")
    try:
        ds = biblemap.read_sources(source_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"  ERROR: {err}\n  Nothing loaded.", file=sys.stderr)
        return 1

    for label, n in ds.skipped.items():
        if n:
            print(f"  skipped {n} {label} row(s) with no ID")

    # Refuse rather than load part of it: a map with some events silently
    # missing their places looks complete and is wrong.
    problems = biblemap.validate(ds)
    if problems:
        print(f"  ERROR: {len(problems)} problem(s). Nothing loaded.", file=sys.stderr)
        for p in problems[:50]:
            print(f"    {p}", file=sys.stderr)
        return 1

    with db.get_conn() as conn:
        written = biblemap.load(conn, ds)
    print("Loaded " + ", ".join(f"{n:,} {t}" for t, n in written.items()) + ".")
    return 0


def run_status() -> int:
    with db.get_conn() as conn:
        if not biblemap.loaded(conn):
            print("No Bible map loaded. Run --load SOURCE_DIR.")
            return 0
        c = biblemap.counts(conn)
    print(f"{c['events']:,} events, {c['people']:,} people, {c['places']:,} places "
          f"({c['places_without_coords']} without coordinates).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--load", metavar="SOURCE_DIR", type=Path,
                        help="Read, validate and load the spreadsheets in SOURCE_DIR.")
    parser.add_argument("--status", action="store_true",
                        help="Print what is currently loaded.")
    args = parser.parse_args()

    if args.load:
        return run_load(args.load)
    if args.status:
        return run_status()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
