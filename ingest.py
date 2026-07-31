"""
Ingestion worker: pilot-folder PDFs -> markdown -> chunks -> embeddings ->
Postgres. Postgres is the work queue over the `books` table (claim via
UPDATE ... FOR UPDATE SKIP LOCKED), so this is safe to run more than once and
safe to kill and restart mid-book.

    python ingest.py --discover     # enumerate the pilot folder into `books`
    python ingest.py --limit 2      # process at most 2 books this run
    python ingest.py                # process all remaining processable books
    python ingest.py --rechunk      # rebuild chunks+embeddings from local markdown
                                     # only (no Drive or OCR calls -- markdown is the asset)
    python ingest.py --index        # build the HNSW index (run once, after ingestion)
    python ingest.py --status       # print a counts-by-status table

The external services (Drive, Voyage, Mistral OCR) are built once at the top of
a run and passed down, so nothing is a module-level client and tests inject
fakes.
"""
import argparse
import hashlib
import time
from functools import partial
from pathlib import Path

import config
import db
from drive import client as drive_client
from pipeline import chunking as chunk_mod
from pipeline import embed as embed_mod
from pipeline import extract as extract_mod

PDF_MIME = "application/pdf"


# ------------------------------------------------------------- discover --

def discover(conn, folder_id: str, list_children) -> int:
    """Enumerate one Drive folder into `books`. `list_children` is a callable
    folder_id -> list[file resource dict]; the real one is bound to a Drive
    service, tests pass a fake. Idempotent (upsert on drive_file_id)."""
    children = list_children(folder_id)
    pdfs = [c for c in children if c.get("mimeType") == PDF_MIME]
    for f in pdfs:
        size_bytes = int(f["size"]) if f.get("size") is not None else None
        db.upsert_book(conn, f["id"], f.get("name", ""), f.get("md5Checksum"), size_bytes)
    print(f"Discovered {len(pdfs)} PDF(s) in folder {folder_id} (upserted, idempotent).")
    return len(pdfs)


# ------------------------------------------------------------- process --

def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chunk_embed_and_finish(conn, book_id: int, markdown: str, voyage_client) -> tuple:
    """Chunk markdown, embed, and commit chunks+done in one transaction.
    Returns (n_chunks, total_tokens)."""
    dropped = []
    chunks = chunk_mod.chunk_markdown(markdown, dropped=dropped)
    if dropped:
        print(f"  dropped {len(dropped)} junk section(s): {', '.join(dropped)}")
    if not chunks:
        # 'failed', never 'done'. A book with no chunks contributes nothing to
        # search, and marking it done retires it from the queue forever -- the
        # corpus quietly loses a book and every status count still reads clean.
        # Real trigger seen: pymupdf4llm's layout model discards text drawn
        # outside the page body, so a malformed PDF extracts to an empty string
        # and every stage downstream reports success.
        msg = "extraction produced no chunks (empty or unparseable markdown)"
        print(f"  WARNING: {msg} for book {book_id}")
        db.set_status(conn, book_id, "failed", msg)
        return 0, 0

    total_tokens = 0
    for c in chunks:
        c["token_count"] = embed_mod.count_tokens(c["content"])
        total_tokens += c["token_count"]

    db.mark_chunked(conn, book_id)

    texts = [c["content"] for c in chunks]
    embeddings, _ = embed_mod.embed_documents(texts, voyage_client)
    for c, emb in zip(chunks, embeddings):
        c["embedding"] = emb

    db.insert_chunks_and_finish(conn, book_id, chunks)
    return len(chunks), total_tokens


def process_book(conn, book, *, download_file, ocr_client, voyage_client) -> None:
    """Run one book from wherever it is through to 'done'. Always reprocesses
    from scratch (idempotent): wipes any chunks and stale local files a prior,
    possibly-killed attempt left behind, then download -> extract -> chunk ->
    embed -> store."""
    book_id = book["id"]
    drive_file_id = book["drive_file_id"]
    title = book["title"] or ""
    t0 = time.time()

    pdf_path = config.PDF_DIR / f"{book_id}.pdf"
    md_path = config.MARKDOWN_DIR / f"{book_id}.md"
    manifest_path = config.MARKDOWN_DIR / f"{book_id}.manifest.json"

    db.delete_chunks_for_book(conn, book_id)
    for p in (pdf_path, md_path, manifest_path):
        p.unlink(missing_ok=True)

    # -- download --
    t_download = time.time()
    download_file(drive_file_id, str(pdf_path))
    if book.get("md5"):
        local_md5 = _md5_file(pdf_path)
        if local_md5 != book["md5"]:
            raise RuntimeError(
                f"md5 mismatch after download: Drive={book['md5']} local={local_md5}"
            )
    db.mark_downloaded(conn, book_id)
    download_s = time.time() - t_download

    # -- extract --
    t_extract = time.time()
    try:
        result = extract_mod.extract(pdf_path, book_id, ocr_client=ocr_client)
    except extract_mod.NeedsOCRError as e:
        db.set_status(conn, book_id, "needs_ocr", str(e))
        print(f"[{book_id}] {title}: needs_ocr ({e})")
        return
    extract_mod.write_outputs(book_id, drive_file_id, pdf_path, result)
    db.mark_extracted(conn, book_id, result.page_count, result.has_text_layer)
    extract_s = time.time() - t_extract

    # -- chunk + embed + store --
    t_chunk_embed = time.time()
    n_chunks, n_tokens = _chunk_embed_and_finish(
        conn, book_id, result.markdown, voyage_client
    )
    chunk_embed_s = time.time() - t_chunk_embed

    total_s = time.time() - t0
    extract_mod.update_manifest(
        book_id,
        download_seconds=round(download_s, 2),
        extract_seconds=round(extract_s, 2),
        chunk_embed_seconds=round(chunk_embed_s, 2),
        total_seconds=round(total_s, 2),
        chunk_count=n_chunks,
    )
    scanned = not result.has_text_layer
    print(
        f"[{book_id}] {title}: pages={result.page_count} scanned={scanned} "
        f"chunks={n_chunks} tokens={n_tokens} time={total_s:.1f}s"
    )


# --------------------------------------------------------------- runs --

def run_worker(limit) -> None:
    service = drive_client.build_service()
    download_file = partial(drive_client.download_file, service)
    ocr_client = extract_mod.build_ocr_client()
    voyage_client = embed_mod.build_client()

    with db.get_conn() as conn:
        discover(conn, config.PILOT_FOLDER_ID, partial(drive_client.list_children, service))

        processed = 0
        while limit is None or processed < limit:
            book = db.claim_next_book(conn)
            if book is None:
                break
            if book["status"] == "failed":
                print(f"[{book['id']}] {book['title']}: giving up (max attempts)")
                continue
            try:
                process_book(
                    conn,
                    book,
                    download_file=download_file,
                    ocr_client=ocr_client,
                    voyage_client=voyage_client,
                )
            except Exception as e:  # noqa: BLE001 -- one bad book must not stop the run
                db.set_status(conn, book["id"], "failed", str(e))
                print(f"[{book['id']}] {book['title']}: FAILED -- {e}")
            processed += 1

        print(f"Done. Processed {processed} book(s) this run.")


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
            n_chunks, n_tokens = _chunk_embed_and_finish(
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
    args = parser.parse_args()

    if args.status:
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
