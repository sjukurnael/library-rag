"""
Load the Bible map into Postgres.

    python -m library_rag.cli.biblemap --load ../biblemap_project
    python -m library_rag.cli.biblemap --status

SOURCE_DIR is the folder holding the spreadsheets: Places-Data/, People Data/
and "Bible Events with Resolved PersonIDs and PlaceIDs.xlsx". --load replaces
the schema's contents rather than adding to them, and refuses when the app holds
edits the files do not -- --export writes those back out first.
"""
import argparse
import sys
from pathlib import Path

from library_rag import biblemap, db


def run_load(source_dir: Path, force: bool = False) -> int:
    # The app can edit these rows now (web/static/biblemap_source.html), which
    # makes the database the source of truth and a re-import destructive. Refuse
    # rather than silently throw the edits away.
    if not force:
        with db.get_conn() as conn:
            edited = biblemap.edit_count(conn) if biblemap.loaded(conn) else 0
        if edited:
            print(
                f"  ERROR: {edited} row(s) have been edited in the app since the last\n"
                f"  load, and loading would discard them. Save them first with\n"
                f"    --export DIR\n"
                f"  or pass --force to overwrite them. Nothing loaded.",
                file=sys.stderr,
            )
            return 1

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


def run_export(out_dir: Path) -> int:
    """Write the stored rows back out as spreadsheets, edits included."""
    with db.get_conn() as conn:
        if not biblemap.loaded(conn):
            print("Nothing loaded to export.", file=sys.stderr)
            return 1
        written = biblemap.export(conn, out_dir)
    print(f"Wrote {len(written)} file(s) to {out_dir}:")
    for path in written:
        print(f"  {path.name}")
    return 0


def run_rebuild() -> int:
    """Re-derive the map from the stored rows, for after a parser change."""
    with db.get_conn() as conn:
        try:
            written = biblemap.rebuild(conn)
        except biblemap.Invalid as err:
            conn.rollback()
            print(f"  ERROR: {len(err.problems)} problem(s). Nothing changed.",
                  file=sys.stderr)
            for p in err.problems[:50]:
                print(f"    {p}", file=sys.stderr)
            return 1
        conn.commit()
    print("Rebuilt " + ", ".join(f"{n:,} {t}" for t, n in written.items()) + ".")
    return 0


def run_status() -> int:
    with db.get_conn() as conn:
        if not biblemap.loaded(conn):
            print("No Bible map loaded. Run --load SOURCE_DIR.")
            return 0
        c = biblemap.counts(conn)
        edited = biblemap.edit_count(conn)
    print(f"{c['events']:,} events, {c['people']:,} people, {c['places']:,} places "
          f"({c['places_without_coords']} without coordinates).")
    if edited:
        print(f"{edited} row(s) edited in the app since the last load. "
              f"--export DIR writes them back to spreadsheets.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--load", metavar="SOURCE_DIR", type=Path,
                        help="Read, validate and load the spreadsheets in SOURCE_DIR.")
    parser.add_argument("--status", action="store_true",
                        help="Print what is currently loaded.")
    parser.add_argument("--export", metavar="OUT_DIR", type=Path,
                        help="Write the stored rows back out as spreadsheets.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-derive the map from the stored rows.")
    parser.add_argument("--force", action="store_true",
                        help="With --load: overwrite rows edited in the app.")
    args = parser.parse_args()

    if args.load:
        return run_load(args.load, args.force)
    if args.export:
        return run_export(args.export)
    if args.rebuild:
        return run_rebuild()
    if args.status:
        return run_status()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
