"""
Ingestion worker: pilot-folder PDFs -> markdown -> chunks -> embeddings ->
Postgres. Postgres is the work queue over the `books` table (claim via
UPDATE ... FOR UPDATE SKIP LOCKED), so this is safe to run more than once and
safe to kill and restart mid-book.

Commands live in library_rag.cli.ingest; this module is the importable half
(discover, register_upload, process_book, process_queue, purge_book) so the API
and the tests can drive ingestion without going through argv.

The external services (Drive, Voyage, Mistral OCR) are built once at the top of
a run and passed down, so nothing is a module-level client and tests inject
fakes.
"""
import hashlib
import shutil
import time
from pathlib import Path

from library_rag import config, db, storage
from library_rag.drive import client as drive_client
from library_rag.pipeline import chunking as chunk_mod
from library_rag.pipeline import embed as embed_mod
from library_rag.pipeline import extract as extract_mod

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

    # The bucket object is shared by every book with the same bytes (same md5
    # via Drive AND via upload = two rows, one object), so it goes only when
    # the LAST reference is gone. The row was already deleted above, so a
    # plain existence check is the right question.
    if book.get("md5") and not db.md5_in_use(conn, book["md5"]):
        if storage.delete_original(book["md5"]):
            removed.append(storage.original_key(book["md5"]))

    # The extraction output goes unconditionally, unlike the original above: it
    # is keyed by book_id, so nothing else can be referencing it, and the book
    # whose id it names no longer exists. Skipping this was the same orphan bug
    # the local files had -- objects keyed to an id that will never be reissued,
    # which nothing would ever collect.
    md_keys = [storage.markdown_key(book_id), storage.manifest_key(book_id)]
    if storage.delete_keys(md_keys):
        removed.extend(md_keys)
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


def chunk_embed_and_finish(conn, book_id: int, markdown: str, voyage_client) -> tuple:
    """Chunk markdown, embed, and commit chunks+done in one transaction.
    Returns (n_chunks, total_tokens).

    Public, not _private: the rechunk command calls it directly to rebuild a
    book from local markdown without re-downloading or re-extracting, which is
    the entire point of keeping markdown as the permanent asset."""
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
        # Drop whatever the previous successful run indexed BEFORE failing.
        # This return is upstream of insert_chunks_and_finish, which is the only
        # thing that clears old chunks on the --rechunk path (process_book's
        # wipe never runs there). Without this, a book that stops extracting
        # cleanly ends up 'failed' while its stale chunks keep being returned by
        # db.search, which has no status filter -- so the library answers
        # questions from a book it simultaneously reports as broken, and every
        # status count still reads clean.
        db.delete_chunks_for_book(conn, book_id)
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
    local_md5 = _md5_file(pdf_path)
    if book.get("md5") and local_md5 != book["md5"]:
        raise RuntimeError(
            f"md5 mismatch after download: Drive={book['md5']} local={local_md5}"
        )
    # Mirror the verified bytes to the bucket. Warn-only inside: a storage
    # outage must not fail a book the pipeline can otherwise finish, and the
    # viewer falls back to the local copy (or a fresh Drive fetch) anyway.
    #
    # The RESULT is kept, unlike before, because it is what makes dropping the
    # local copy below safe: False means "no confirmed second copy", and a book
    # ingested during a storage outage (or with SUPABASE_* unset) must keep the
    # only one it has.
    mirrored = storage.put_original(book.get("md5") or local_md5, pdf_path)
    db.mark_downloaded(conn, book_id)
    db.touch_claim(conn, book_id)
    download_s = time.time() - t_download

    # -- extract --
    t_extract = time.time()
    try:
        result = extract_mod.extract(pdf_path, book_id, ocr_client=ocr_client)
    except extract_mod.NeedsOCRError as e:
        db.set_status(conn, book_id, "needs_ocr", str(e))
        # Same rule as the success path below: the mirror already ran (it is
        # upstream of extraction), so this copy is redundant the moment the
        # bucket confirms. Returning early without it was how a whole class of
        # book -- every scan, on any install with no MISTRAL_API_KEY -- kept
        # accumulating full PDFs locally while every other path cleaned up.
        #
        # The queue page's "Open PDF" is unaffected: it goes through
        # /api/books/{id}/pdf, which reaches for the signed bucket URL first.
        if mirrored:
            pdf_path.unlink(missing_ok=True)
        print(f"[{book_id}] {title}: needs_ocr ({e})")
        return
    extract_mod.write_outputs(book_id, source_id, pdf_path, result)
    db.mark_extracted(conn, book_id, result.page_count, result.has_text_layer)
    db.touch_claim(conn, book_id)
    extract_s = time.time() - t_extract

    # The working PDF has done its job -- extraction is over and write_outputs
    # has already hashed it into the manifest. Nothing downstream reads it:
    # chunking works on the markdown, and --rechunk works on the markdown too.
    #
    # Dropped rather than kept because it is a SECOND copy of bytes that are
    # already in the bucket, and it was never being cleaned up: measured at 197
    # books, data/pdfs held 488 MB against 21 MB of markdown, and every one of
    # those PDFs was in storage as well. Left alone it grows ~2.6 GB per 1000
    # books forever, on every machine that ever indexes anything.
    #
    # Strictly conditional on `mirrored`, which is the conservative choice
    # rather than a strict necessity: a Drive book could always be re-fetched
    # from Drive and an upload still has its original in data/uploads, so this
    # is never the last copy of anything. But an unconfirmed mirror means the
    # remaining copy is one OAuth token or one bucket outage away from being
    # unreachable, and a local file is a free hedge against that.
    #
    # The viewer re-fetches from storage and process_book re-downloads from
    # scratch, so nothing here is depended on later.
    if mirrored:
        pdf_path.unlink(missing_ok=True)

    # -- chunk + embed + store --
    t_chunk_embed = time.time()
    n_chunks, n_tokens = chunk_embed_and_finish(
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
    # Last step, once the manifest is final: push markdown + manifest to the
    # bucket and drop the local pair. With the working PDF already gone above,
    # this is what leaves the run with NO local state -- the machine that
    # extracted a book stops being the only one that can rechunk it.
    extract_mod.sync_outputs(book_id)
    scanned = not result.has_text_layer
    print(
        f"[{book_id}] {title}: pages={result.page_count} scanned={scanned} "
        f"chunks={n_chunks} tokens={n_tokens} time={total_s:.1f}s"
    )


# ---------------------------------------------------------------- worker --

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
