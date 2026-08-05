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
    python ingest.py --local a.pdf b.pdf   # ingest PDFs already on disk

The external services (Drive, Voyage, Mistral OCR) are built once at the top of
a run and passed down, so nothing is a module-level client and tests inject
fakes.
"""
import argparse
import hashlib
import shutil
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
    service, tests pass a fake. Idempotent (upsert on source_id)."""
    children = list_children(folder_id)
    pdfs = [c for c in children if c.get("mimeType") == PDF_MIME]
    for f in pdfs:
        size_bytes = int(f["size"]) if f.get("size") is not None else None
        db.upsert_book(
            conn, f["id"], f.get("name", ""), f.get("md5Checksum"), size_bytes,
            source="drive",
        )
    print(f"Discovered {len(pdfs)} PDF(s) in folder {folder_id} (upserted, idempotent).")
    return len(pdfs)


# --------------------------------------------------------------- sources --

def upload_path(source_id: str) -> Path:
    """Where an uploaded book's original bytes live.

    source_id is "upload:<md5>", and the md5 IS the filename -- the upload
    directory is content-addressed. Two consequences worth stating: the same
    file uploaded twice occupies one file on disk, and a filename supplied by a
    client never reaches the filesystem, so there is no path to traverse out of.
    """
    return config.UPLOAD_DIR / f"{source_id.split(':', 1)[1]}.pdf"


def register_upload(conn, src: Path, title: str | None = None) -> dict:
    """Take a PDF already on disk into the library and enqueue it.

    Returns the book row. Does NOT process it: the row lands at status
    'discovered', which is precisely what claim_next_book hands out, so an
    upload joins the same queue a Drive book does and inherits the claim,
    heartbeat, retry and failure handling already built around it. Any other
    design would need a second copy of all of that.

    Idempotent on content. Re-registering identical bytes upserts nothing and
    leaves a finished book finished; different bytes are a different source_id
    and therefore a different book.
    """
    md5 = _md5_file(src)
    source_id = f"upload:{md5}"
    dest = upload_path(source_id)
    if not dest.exists():
        shutil.copyfile(src, dest)
    db.upsert_book(
        conn, source_id, title or src.name, md5, dest.stat().st_size, source="upload"
    )
    return db.fetch_book_by_source_id(conn, source_id)


def purge_book(conn, book_id: int):
    """Remove a book completely: the row, its chunks, and every file it owns.

    Returns {"book": row, "removed": [filenames]}, or None if no such book.

    Deleting the row alone leaves up to four orphans -- the working PDF cache,
    the extracted markdown, its manifest, and (for an upload) the original
    bytes. Nothing would ever collect them: they are keyed by book_id, and the
    next upload gets a new id, so they would sit there forever while looking
    like a book that still exists.

    The row goes FIRST. If file removal then fails, the leftovers are harmless
    and re-uploading overwrites them; doing it the other way round would leave a
    book row pointing at files that are gone, which the pipeline would only
    discover mid-run.
    """
    book = db.delete_book(conn, book_id)
    if book is None:
        return None

    paths = [
        config.PDF_DIR / f"{book_id}.pdf",
        config.MARKDOWN_DIR / f"{book_id}.md",
        config.MARKDOWN_DIR / f"{book_id}.manifest.json",
    ]
    # Only an upload owns its original. A Drive book's source_id is a Drive file
    # id with no "upload:" prefix, and Drive still holds the bytes regardless.
    if book["source"] == "upload":
        paths.append(upload_path(book["source_id"]))

    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(path.name)
    return {"book": book, "removed": removed}


def _lazy_drive_downloader():
    """A Drive downloader that builds the API client on first use only.

    Draining the queue must not require Google credentials when the queue holds
    nothing from Drive. Building the service eagerly made OAuth a hard
    dependency of processing an uploaded file, which is a dependency no user of
    the upload path agreed to.
    """
    holder = {}

    def download(source_id, dest_path):
        if "service" not in holder:
            holder["service"] = drive_client.build_service()
        drive_client.download_file(holder["service"], source_id, dest_path)

    return download


def downloader_for(book, drive_download):
    """Pick how this book's bytes arrive. The ONLY place the two sources
    diverge -- everything after the download step is identical, which is the
    whole reason process_book takes the downloader as an argument."""
    if book["source"] == "upload":

        def copy_in(source_id, dest_path):
            src = upload_path(source_id)
            if not src.exists():
                raise RuntimeError(
                    f"uploaded original is missing from {src} -- the book row "
                    "outlived its file"
                )
            shutil.copyfile(src, dest_path)

        return copy_in
    return drive_download


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
    db.touch_claim(conn, book_id)

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
    source_id = book["source_id"]
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
    download_file(source_id, str(pdf_path))
    if book.get("md5"):
        local_md5 = _md5_file(pdf_path)
        if local_md5 != book["md5"]:
            raise RuntimeError(
                f"md5 mismatch after download: Drive={book['md5']} local={local_md5}"
            )
    db.mark_downloaded(conn, book_id)
    db.touch_claim(conn, book_id)
    download_s = time.time() - t_download

    # -- extract --
    t_extract = time.time()
    try:
        result = extract_mod.extract(pdf_path, book_id, ocr_client=ocr_client)
    except extract_mod.NeedsOCRError as e:
        db.set_status(conn, book_id, "needs_ocr", str(e))
        print(f"[{book_id}] {title}: needs_ocr ({e})")
        return
    extract_mod.write_outputs(book_id, source_id, pdf_path, result)
    db.mark_extracted(conn, book_id, result.page_count, result.has_text_layer)
    db.touch_claim(conn, book_id)
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

def process_queue(limit=None, *, source=None, conn=None) -> int:
    """Drain the work queue, whatever each book's source is. Returns the count.

    This is the whole worker, and it is source-agnostic on purpose: it claims a
    book, asks downloader_for how that book's bytes arrive, and runs the same
    pipeline either way. Adding a third source means adding a downloader, not a
    second worker.

    Nothing here builds a Drive client unless a Drive book is actually claimed
    (see _lazy_drive_downloader), so with source="upload" this is safe to call
    with no Google credentials configured at all.
    """
    drive_download = _lazy_drive_downloader()
    ocr_client = extract_mod.build_ocr_client()
    voyage_client = embed_mod.build_client()

    def drain(c):
        processed = 0
        while limit is None or processed < limit:
            book = db.claim_next_book(c, source)
            if book is None:
                break
            if book["status"] == "failed":
                print(f"[{book['id']}] {book['title']}: giving up (max attempts)")
                continue
            try:
                process_book(
                    c,
                    book,
                    download_file=downloader_for(book, drive_download),
                    ocr_client=ocr_client,
                    voyage_client=voyage_client,
                )
            except Exception as e:  # noqa: BLE001 -- one bad book must not stop the run
                db.set_status(c, book["id"], "failed", str(e))
                print(f"[{book['id']}] {book['title']}: FAILED -- {e}")
            processed += 1
        return processed

    if conn is not None:
        return drain(conn)
    with db.get_conn() as own:
        return drain(own)


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
