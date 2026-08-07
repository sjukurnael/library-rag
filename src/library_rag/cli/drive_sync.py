"""
Sync the Drive metadata mirror that /library browses and searches.

    python -m library_rag.cli.drive_sync --status
    python -m library_rag.cli.drive_sync --sync     # ~66 Drive calls, ~2 min
    python -m library_rag.cli.drive_sync --embed    # title embeddings, resumable
    python -m library_rag.cli.drive_sync --index    # HNSW, after --embed
    python -m library_rag.cli.drive_sync --all      # sync, embed, index

Metadata only -- nothing here downloads a book. The three passes are separate
commands because they fail differently: --sync needs Drive, --embed needs Voyage
and costs money, --index needs neither and only makes sense once the vectors
exist.
"""
import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from library_rag import db  # noqa: E402
from library_rag.drive import client as drive_client  # noqa: E402
from library_rag.drive import mirror  # noqa: E402
from library_rag.drive.client import DriveAuthError  # noqa: E402
from library_rag.pipeline import embed as embed_mod  # noqa: E402


def _say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run_status(conn) -> None:
    s = db.drive_sync_status(conn)
    print(f"folders   {s['folders']:>8,}")
    print(f"files     {s['files']:>8,}")
    print(f"embedded  {s['embedded']:>8,}")
    print(f"pending   {s['pending']:>8,}")
    print(f"synced    {s['synced_at'] or 'never'}")


def run_sync(conn) -> None:
    print("Syncing Drive metadata...")
    service = drive_client.build_service()
    counts = mirror.sync(conn, service, progress=_say)
    print(
        f"Synced {counts['files']:,} PDFs and {counts['folders']:,} folders; "
        f"{counts['removed']:,} stale row(s) removed, "
        f"{counts['paths']:,} path(s) (re)computed."
    )


def run_embed(conn) -> None:
    pending = mirror.pending_count(conn)
    if not pending:
        print("Nothing to embed.")
        return
    print(f"Embedding {pending:,} title(s)...")
    voyage = embed_mod.build_client()
    n = mirror.embed_titles(conn, voyage, progress=_say)
    print(f"Embedded {n:,} title(s).")


def run_index(conn) -> None:
    print("Building HNSW over drive_files.embedding...")
    mirror.build_index(conn)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sync", action="store_true", help="Re-list Drive into the mirror.")
    parser.add_argument("--embed", action="store_true", help="Embed titles lacking a vector.")
    parser.add_argument("--index", action="store_true", help="Build the HNSW index.")
    parser.add_argument("--all", action="store_true", help="--sync, then --embed, then --index.")
    parser.add_argument("--status", action="store_true", help="Print mirror counts.")
    args = parser.parse_args()

    if not any((args.sync, args.embed, args.index, args.all, args.status)):
        parser.print_help()
        return 2

    try:
        with db.get_conn() as conn:
            if args.sync or args.all:
                run_sync(conn)
            if args.embed or args.all:
                run_embed(conn)
            if args.index or args.all:
                run_index(conn)
            if args.status or args.all:
                run_status(conn)
    except DriveAuthError as e:
        print(f"\nDrive auth error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
