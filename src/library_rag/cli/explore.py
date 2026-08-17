"""
Drive browsing agent CLI: find books in the Drive library worth reading next.

The tool-use loop and system prompt live in exploration/loop.py, the tools in
exploration/tools.py, the estimation constants in exploration/assumptions.py,
and Drive auth/listing in drive/client.py. This file is argument parsing plus
rendering the event stream.

It consumes the SAME generator the web route streams, so the two surfaces cannot
drift apart in what they show.

Run:
    python -m library_rag.cli.explore --ask "books on the Holy Spirit"
    python -m library_rag.cli.explore --ask "the parables" -n 20
    python -m library_rag.cli.explore              # prompts for the interest
"""
import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from library_rag import db  # noqa: E402
from library_rag.drive.client import DriveAuthError  # noqa: E402
from library_rag.exploration.loop import DEFAULT_COUNT, MAX_COUNT, run  # noqa: E402


def _render(event: dict, quiet: bool) -> None:
    kind = event["type"]
    if kind == "tool" and not quiet:
        args = ", ".join(f"{k}={v!r}" for k, v in event["input"].items())
        print(f"[agent] {event['name']}({args})")
    elif kind == "results" and not quiet:
        summary = ", ".join(f"{k}={v}" for k, v in event["summary"].items())
        print(f"        -> {summary}")
    elif kind == "tool_error":
        print(f"[agent] {event['name']} FAILED -- {event['message']}", file=sys.stderr)
    elif kind == "thinking" and not quiet:
        print(f"[agent] {event['text']}")
    elif kind == "recommendations":
        print()
        for i, b in enumerate(event["books"], 1):
            if b.get("unknown"):
                print(f"{i}. (unknown file id {b['file_id']}) -- {b['error']}")
                continue
            size = f"{b['size_mb']} MB" if b["size_mb"] is not None else "size unknown"
            print(f"{i}. {b['title']}  [{size}]")
            print(f"   {b['why']}")
            print(f"   add: python -m library_rag.cli.explore --add {b['file_id']}")
            print(f"   {b['url']}")
        print()
    elif kind == "answer":
        print(event["text"])
    elif kind == "error":
        print(f"\nerror: {event['message']}", file=sys.stderr)


def run_browse(interest: str, quiet: bool, count: int) -> int:
    with db.get_conn() as conn:
        for event in run(interest, conn, count=count):
            _render(event, quiet)
    return 0


def run_add(file_ids: list) -> int:
    """Queue Drive books by file id, then drain them.

    Deliberately the same two steps the web button performs -- upsert at
    'discovered', then run the worker -- so a book added here is
    indistinguishable from one added there.
    """
    from library_rag import ingest
    from library_rag.drive import client as drive_client

    service = drive_client.build_service()
    with db.get_conn() as conn:
        for file_id in file_ids:
            meta = drive_client.get_file(service, file_id)
            size = meta.get("size")
            db.upsert_book(
                conn, file_id, meta.get("name", ""), meta.get("md5Checksum"),
                int(size) if size is not None else None, source="drive",
            )
            book = db.fetch_book_by_source_id(conn, file_id)
            print(f"[{book['id']}] {book['title']}: {book['status']}")
    processed = ingest.process_queue(source="drive")
    print(f"Done. Processed {processed} book(s) this run.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ask", metavar="INTEREST",
        help="What you want to study. Omit to be prompted.",
    )
    parser.add_argument(
        "--add", nargs="+", metavar="FILE_ID",
        help="Skip browsing: queue these Drive file ids and index them.",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=DEFAULT_COUNT, metavar="N",
        help=f"Shortlist up to N books (1-{MAX_COUNT}, default {DEFAULT_COUNT}). "
             "A ceiling, not a quota -- the agent returns fewer when the "
             "collection runs out of relevant titles.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the [agent] tool-call trail."
    )
    args = parser.parse_args()
    # parser.error, not the ValueError from run(): argparse prints usage and
    # exits 2, which is what a shell expects from a bad flag -- a traceback
    # would read as a crash rather than a typo.
    if not 1 <= args.count <= MAX_COUNT:
        parser.error(f"--count must be between 1 and {MAX_COUNT}, got {args.count}")

    try:
        if args.add:
            sys.exit(run_add(args.add))
        interest = args.ask or input("What do you want to study? ").strip()
        if not interest:
            print("Nothing to look for.", file=sys.stderr)
            sys.exit(1)
        sys.exit(run_browse(interest, args.quiet, args.count))
    except DriveAuthError as e:
        print(f"\nDrive auth error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
