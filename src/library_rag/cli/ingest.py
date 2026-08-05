"""
The ingestion commands.

    python -m library_rag.cli.ingest --discover   # enumerate Drive into `books`
    python -m library_rag.cli.ingest              # drain the queue
    python -m library_rag.cli.ingest --local a.pdf b.pdf
    python -m library_rag.cli.ingest --delete 12
    python -m library_rag.cli.ingest --rechunk    # rebuild from local markdown
    python -m library_rag.cli.ingest --index      # build the HNSW index
    python -m library_rag.cli.ingest --status

Everything here prints, parses argv or orchestrates. The logic it calls lives in
library_rag.ingest.
"""
import argparse
import time
from functools import partial
from pathlib import Path

from library_rag import config, db
from library_rag.drive import client as drive_client
from library_rag.ingest import (
    chunk_embed_and_finish,
    discover,
    process_queue,
    purge_book,
    register_upload,
)
from library_rag.pipeline import embed as embed_mod
from library_rag.pipeline import extract as extract_mod


def run_worker(limit) -> None:
    """Discover the Drive folder, then drain the queue."""
    service = drive_client.build_service()
    with db.get_conn() as conn:
        discover(conn, config.PILOT_FOLDER_ID, partial(drive_client.list_children, service))
    processed = process_queue(limit)
    print(f"Done. Processed {processed} book(s) this run.")


def run_local(paths: list) -> None:
    """Ingest PDFs already on disk, bypassing Drive.

    Identical to uploading them through the web UI -- both call register_upload
    and then drain the queue. Keeping one path means the CLI cannot drift from
    what the API does, which is what happened the first time around: --local
    used to key books on `local:<filename>` and process them inline, outside the
    queue, so an interrupted --local run left no claim to recover from.
    """
    registered = []
    with db.get_conn() as conn:
        for raw in paths:
            src = Path(raw).expanduser().resolve()
            if not src.exists():
                print(f"SKIP -- no such file: {src}")
                continue
            book = register_upload(conn, src)
            registered.append(book)
            print(f"[{book['id']}] {book['title']}: registered ({book['status']})")

    if not registered:
        print("Nothing to do.")
        return
    processed = process_queue()
    print(f"Done. Processed {processed} book(s) this run.")


def run_delete(book_ids: list) -> None:
    with db.get_conn() as conn:
        for book_id in book_ids:
            result = purge_book(conn, book_id)
            if result is None:
                print(f"[{book_id}] no such book")
                continue
            files = ", ".join(result["removed"]) or "no files"
            print(f"[{book_id}] deleted {result['book']['title']} -- removed {files}")


def run_rechunk() -> None:
    """Rebuild chunks+embeddings from local markdown only. No Drive or OCR
    calls -- this is the point of keeping markdown as the permanent asset."""
    voyage_client = embed_mod.build_client()
    with db.get_conn() as conn:
        reset = db.reset_done_to_extracted(conn)
        if reset:
            print(f"Reset {reset} done book(s) back to extracted for rechunk.")
        for book_id, title in db.fetch_rechunkable_books(conn):
            md_path = config.MARKDOWN_DIR / f"{book_id}.md"
            if not md_path.exists():
                print(f"[{book_id}] {title}: SKIP -- no local markdown at {md_path}")
                continue
            markdown = md_path.read_text(encoding="utf-8")
            t0 = time.time()
            n_chunks, n_tokens = chunk_embed_and_finish(
                conn, book_id, markdown, voyage_client
            )
            extract_mod.update_manifest(
                book_id,
                chunk_embed_seconds=round(time.time() - t0, 2),
                chunk_count=n_chunks,
            )
            print(f"[{book_id}] {title}: rechunked -> {n_chunks} chunks, {n_tokens} tokens")


def run_index() -> None:
    with db.get_conn() as conn:
        print("Building HNSW index on chunks.embedding (this can take a while) ...")
        db.build_hnsw_index(conn)
        print("Done.")


def run_status() -> None:
    with db.get_conn() as conn:
        rows = db.status_counts(conn)
    print(f"{'status':<12} {'count':>6}")
    print("-" * 20)
    total = 0
    for status, count in rows:
        print(f"{status:<12} {count:>6}")
        total += count
    print("-" * 20)
    print(f"{'total':<12} {total:>6}")


def run_discover_only() -> None:
    service = drive_client.build_service()
    with db.get_conn() as conn:
        discover(conn, config.PILOT_FOLDER_ID, partial(drive_client.list_children, service))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--discover", action="store_true", help="Enumerate the pilot folder, then stop."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N books this run."
    )
    parser.add_argument(
        "--rechunk",
        action="store_true",
        help="Rebuild chunks+embeddings from local markdown only.",
    )
    parser.add_argument(
        "--index", action="store_true", help="Build the HNSW index on chunks.embedding."
    )
    parser.add_argument(
        "--status", action="store_true", help="Print a counts-by-status table."
    )
    parser.add_argument(
        "--local", nargs="+", metavar="PDF",
        help="Ingest PDFs already on disk instead of from Drive.",
    )
    parser.add_argument(
        "--delete", nargs="+", type=int, metavar="ID",
        help="Remove books by id: row, chunks, and every file they own.",
    )
    args = parser.parse_args()

    if args.delete:
        run_delete(args.delete)
    elif args.local:
        run_local(args.local)
    elif args.status:
        run_status()
    elif args.index:
        run_index()
    elif args.rechunk:
        run_rechunk()
    elif args.discover:
        run_discover_only()
    else:
        run_worker(args.limit)


if __name__ == "__main__":
    main()
