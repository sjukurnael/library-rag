"""
Local web UI over the indexed library.

    ./.venv/bin/uvicorn library_rag.web.api:app --reload --port 8000
    open http://localhost:8000

Every question goes through retrieval/loop.py: the model runs its own search
loop and decides how many searches the question warrants -- one for a plain
factual question, more for a comparative or multi-part one. There is no
separate "simple" path, because a fixed one-search pipeline is just this with
the decision hardcoded, and hardcoding it is what caps a comparative question at
whatever a single embedding of the user's phrasing happens to reach.

Retrieval itself goes through the same db.search + pipeline/embed.embed_query
that search.py uses, so this is a window onto the pipeline rather than a second
implementation of it. search.py remains the deterministic control: same question
-> same vector -> same passages, which is what you need to tell whether a
chunking or embedding change actually helped. The agent varies run to run, so it
is the better product and the worse measuring instrument.
"""
import json
import os
import secrets
import tempfile
import time
from html import escape
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from library_rag import bible, config, db, ingest, jobs, storage
from library_rag.drive import client as drive_client
from library_rag.drive import mirror
from library_rag.drive import store as drive_store
from library_rag.exploration import loop as browse_loop
from library_rag.exploration import tools as browse_tools
from library_rag.pipeline import embed as embed_mod
from library_rag.retrieval import research
from library_rag.web import auth

app = FastAPI(title="library-rag")
_STATIC = Path(__file__).resolve().parent / "static"

# Order matters: SessionMiddleware must be added AFTER AuthMiddleware so that it
# runs OUTSIDE it. Starlette applies middleware in reverse order of addition, and
# AuthMiddleware reads request.session -- which does not exist until
# SessionMiddleware has decoded the cookie.
app.add_middleware(auth.AuthMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET,
    max_age=config.SESSION_MAX_AGE_SECONDS,
    same_site="lax",          # survives the top-level navigation back from /login
    https_only=config.SESSION_COOKIE_SECURE,
)

if not config.auth_enabled():
    # Loud, because the failure mode is silent: an app deployed without
    # GOOGLE_CLIENT_ID serves every route, including DELETE /api/books/{id}, to
    # anyone who finds the URL.
    print(
        "\n  WARNING: GOOGLE_CLIENT_ID is not set -- sign-in is DISABLED and\n"
        "  every route is public. Fine on localhost; never on a public URL.\n"
    )


@app.exception_handler(drive_client.DriveAuthError)
def _drive_auth_handler(request, exc):
    """Surface Drive auth failures with the instructions they carry.

    DriveAuthError exists to say exactly what to fix -- "delete token.json and
    re-run to re-authorize". Letting it fall through to FastAPI's default
    handler turned that into a bare 500, so the browser showed "500 Internal
    Server Error" for a problem the user could have solved in thirty seconds.

    503, not 500: nothing here is broken. An upstream credential expired, the
    condition is temporary, and the fix is the user's to make.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class BrowseRequest(BaseModel):
    interest: str = Field(min_length=1, max_length=2000)
    # How many books to shortlist -- a ceiling the agent underspends when the
    # collection runs out of genuinely relevant titles (see exploration/loop.py,
    # system_prompt). Bounded here as well as in the loop because this is the
    # boundary that faces the network: 51 is a 422 with a field error naming the
    # limit, not a clamp, so a caller learns the ceiling exists.
    count: int = Field(
        default=browse_loop.DEFAULT_COUNT, ge=1, le=browse_loop.MAX_COUNT
    )


class AddDriveRequest(BaseModel):
    # Bounded: this queues real downloads, and an unbounded list is an easy way
    # to put the whole 130 GB drive into flight from one request.
    file_ids: list[str] = Field(min_length=1, max_length=20)


def _page(name: str) -> FileResponse:
    """Serve one of the HTML pages, revalidated every time.

    Same reasoning as static_file below, and it belongs here MORE, not less:
    each of these pages carries its own inline <script>, so a heuristically
    cached page is stale JS that no amount of reloading app.js can fix. The
    symptom is a UI change that is live on the server and invisible in the
    browser, which reads as a broken feature rather than a stale cache.
    """
    return FileResponse(_STATIC / name, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return _page("index.html")


@app.get("/library")
def library():
    """The Drive browser. A separate page from the research chat because it is a
    different task -- deciding what to read, rather than reading."""
    return _page("library.html")


@app.get("/queue")
def queue():
    """The processing dashboard: everything mid-pipeline, stage by stage, plus
    upload. A third page because it is a third task -- watching work happen,
    rather than reading or choosing what to read."""
    return _page("queue.html")


@app.get("/static/{name}")
def static_file(name: str):
    """Shared CSS/JS. Whitelisted rather than mounted as a directory: this
    process also holds credentials.json and token.json, and a path parameter
    that reaches the filesystem is the wrong thing to be casual about."""
    if name not in {"app.css", "app.js"}:
        raise HTTPException(404, "No such asset.")
    # no-cache means "revalidate every time", not "never cache": the browser
    # still keeps the bytes and the ETag makes revalidation a cheap 304. Without
    # it, browsers heuristically cache these and users keep running old JS for
    # hours after a deploy.
    return FileResponse(
        _STATIC / name, headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/books")
def books():
    """What is actually searchable, plus what is on its way.

    `books` is the searchable set -- the UI must never imply coverage we lack.
    `pending` is everything else, so an upload the user just made is visible
    while it works rather than vanishing until it finishes. A book that failed
    stays in `pending` with its error: silently dropping it would tell the user
    their upload succeeded.
    """
    with db.get_conn() as conn:
        done = conn.execute(
            """
            SELECT b.id, b.title, b.page_count, count(c.id) AS chunks
            FROM books b JOIN chunks c ON c.book_id = b.id
            WHERE b.status = 'done'
            GROUP BY b.id, b.title, b.page_count
            ORDER BY b.title
            """
        ).fetchall()
        # claim_age_s is computed HERE rather than shipping claimed_at for the
        # browser to subtract from Date.now(). Both timestamps are Postgres's,
        # so the answer cannot be wrong because a client's clock is; a skewed
        # browser would otherwise render "-2m" or "7h" and make a clock bug look
        # like a pipeline bug. NULL when the book has never been claimed.
        pending = conn.execute(
            """
            SELECT id, title, status, source, error, page_count,
                   EXTRACT(EPOCH FROM now() - claimed_at) AS claim_age_s,
                   source_id, size_bytes
            FROM books
            WHERE status <> 'done'
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return {
        "books": [
            {"id": r[0], "title": r[1], "pages": r[2], "chunks": r[3]} for r in done
        ],
        "total_chunks": sum(r[3] for r in done),
        "pending": [
            {
                "id": r[0], "title": r[1], "status": r[2],
                "source": r[3], "error": r[4],
                # page_count is written by mark_extracted, so it is NULL until
                # the book has actually been read.
                "pages": r[5],
                "claim_age_s": None if r[6] is None else int(r[6]),
                "url": _original_url(r[0], r[3], r[7]),
                # Shipped so a terminal card can SAY "0 bytes" instead of making
                # someone click a link and interpret a blank PDF viewer. An
                # empty source is the one failure whose cause is fully knowable
                # without opening anything.
                "size_bytes": r[8],
            }
            for r in pending
        ],
    }


def _original_url(book_id: int, source: str, source_id: str) -> str | None:
    """Where to send someone who wants to see the file itself, for a book that
    is still in the queue.

    Deliberately NOT /api/books/{id}/pdf for a Drive book: that route falls
    through to a full in-request download when nothing is cached, so on the
    processing page -- where by definition the bytes have not arrived yet -- it
    would block a click for as long as the file takes, and re-download a file
    the worker is already downloading. Drive's own viewer costs us nothing and
    is live from the moment the row exists, because source_id IS the file id.

    An upload has no such page, but it does have its original on disk from the
    moment it was registered, so the local route is right there and cheap.
    """
    if source == "drive":
        return f"https://drive.google.com/file/d/{source_id}/view"
    return f"/api/books/{book_id}/pdf"


def _drain_queue(source: str):
    """Process what is queued for ONE source. Runs in a background thread.

    Drains the QUEUE rather than the one book just added: the queue is the
    coordination point, claim_next_book already stops two workers touching the
    same row, and processing "my" book directly would be a second scheduling
    path with none of the claim, heartbeat or retry behaviour. Two uploads
    racing therefore cooperate instead of colliding.

    `source` is required rather than defaulted, and that part is not cosmetic.
    Draining everything meant one uploaded PDF put the entire Drive backlog into
    flight behind it, and the first Drive book claimed blocked this thread in an
    OAuth flow with no console to prompt -- so the upload the user actually asked
    for never ran, and nothing said why. Every caller now states which queue it
    means, so that can never be re-introduced by omission.

    The Drive caller is safe for the same reason it once was not: by the time a
    book is added from the browsing agent, that agent has already completed Drive
    API calls in this process, so credentials are proven and token.json is warm.

    A real deployment moves this into its own process -- cli/ingest.py is already
    exactly that worker, and _start_ingest below prefers it whenever
    INGEST_JOB_NAME is set. This remains the fallback, so a single-user local
    install with no Cloud project still does something when you press the button.
    """
    try:
        ingest.process_queue(source=source)
    except Exception as e:  # noqa: BLE001 -- a background task has nowhere to raise
        print(f"background ingest failed ({source}): {e}")


def _start_ingest(background: BackgroundTasks, source: str) -> None:
    """Get the queue drained, by whichever worker this deployment has.

    The Job is tried FIRST and SYNCHRONOUSLY -- inside the request, before the
    response is flushed. That ordering is the entire point. A BackgroundTask
    starts at the moment Cloud Run stops allocating CPU, so scheduling the
    trigger there would make the call that fixes the throttling problem the
    first casualty of it. It is one POST with a 10s ceiling.

    A failed trigger is not an error the user should see: the books are already
    queued and committed, so the fallback simply does the work here, exactly as
    before. Only WHO drains changes.
    """
    if jobs.run_ingest_job():
        return
    background.add_task(_drain_queue, source)


@app.post("/api/books/upload")
async def upload_book(background: BackgroundTasks, file: UploadFile = File(...)):
    """Accept a PDF, enqueue it, and start the pipeline.

    The response returns as soon as the book is queued. Ingestion takes minutes
    -- holding the request open for it would tie the answer to the browser
    staying on the page, and give the user no way to see progress. The client
    polls /api/books instead.
    """
    if not config.VOYAGE_API_KEY:
        raise HTTPException(500, "VOYAGE_API_KEY is not set; ingestion would fail")

    filename = os.path.basename(file.filename or "").strip() or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    # Stream to a temp file, enforcing the size cap as it arrives. Reading the
    # upload into memory to check its length IS the denial of service, and
    # Content-Length is a claim by the client rather than a fact about the body.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            tmp = Path(fh.name)
            written = 0
            first = b""
            while chunk := await file.read(1 << 20):
                if not first:
                    first = chunk[: len(config.PDF_MAGIC)]
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"File is larger than the "
                        f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                fh.write(chunk)

        if written == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        # Magic bytes, not the extension or the browser-supplied content type --
        # the client picks both of those.
        if first != config.PDF_MAGIC:
            raise HTTPException(400, "That file is not a PDF.")

        with db.get_conn() as conn:
            book = ingest.register_upload(conn, tmp, title=filename)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    already_indexed = book["status"] == "done"
    if not already_indexed:
        _start_ingest(background, "upload")

    return {
        "id": book["id"],
        "title": book["title"],
        "status": book["status"],
        # Content-addressed identity means an identical re-upload is recognised
        # rather than duplicated; say so instead of implying work is happening.
        "already_indexed": already_indexed,
    }


@app.get("/api/books/{book_id}/pdf")
def book_pdf(book_id: int):
    """The book's ORIGINAL bytes, for the citation panel's "open the real
    page" link. The caller appends #page=N; fragments are client-side, so they
    survive the redirect below.

    Three sources, tried in order of cheapness:
      1. a signed Supabase Storage URL (no bytes through this process),
      2. a local copy (an upload's original, or the working PDF cache),
      3. for a Drive book, a one-time fetch into the cache -- also mirrored to
         the bucket so the next click takes path 1.
    """
    with db.get_conn() as conn:
        book = db.fetch_book(conn, book_id)
    if book is None:
        raise HTTPException(404, f"No book with id {book_id}.")

    url = storage.signed_original_url(book["md5"])
    if url:
        return RedirectResponse(url, status_code=302)

    inline = {"Content-Disposition": "inline"}
    candidates = []
    if book["source"] == "upload":
        candidates.append(ingest.upload_path(book["source_id"]))
    candidates.append(config.PDF_DIR / f"{book_id}.pdf")
    for path in candidates:
        if path.exists():
            return FileResponse(path, media_type="application/pdf", headers=inline)

    if book["source"] == "drive":
        service = drive_client.build_service()
        dest = config.PDF_DIR / f"{book_id}.pdf"
        drive_client.download_file(service, book["source_id"], str(dest))
        storage.put_original(book["md5"], dest)
        return FileResponse(dest, media_type="application/pdf", headers=inline)

    raise HTTPException(
        404,
        "The original file for this book is gone from disk and storage. "
        "Re-upload it to restore the PDF view; search still works.",
    )


@app.delete("/api/books/{book_id}")
def delete_book(book_id: int):
    """Remove a book and everything it owns.

    Hard delete, not a soft one: the point of removing a book is that it stops
    answering questions, and a flag that has to be honoured by every future
    query is a flag someone will eventually forget in one of them.
    """
    with db.get_conn() as conn:
        result = ingest.purge_book(conn, book_id)
    if result is None:
        raise HTTPException(404, f"No book with id {book_id}.")
    return {
        "id": book_id,
        "title": result["book"]["title"],
        "removed_files": result["removed"],
    }


# A research run is detached from the HTTP connection that started it. The old
# shape -- POST holds one SSE stream open for the whole run -- meant navigating
# to another page killed the agent mid-thought, because closing the response
# closed the generator. Now POST only starts the run; events accumulate in this
# registry whether or not anyone is watching, and the events route below can be
# attached, dropped and re-attached freely. In memory, like _sync_state and
# _auth_flows: a restart loses in-flight runs, which is correct for a
# single-user local app (and the page tells the user so via the 404).
_research_runs: dict[str, dict] = {}
_RESEARCH_KEEP = 20  # finished runs kept for late re-attach, oldest pruned


def _run_research(run_id: str, question: str) -> None:
    """The agent loop, feeding the registry instead of a response. Runs on the
    BackgroundTasks threadpool; list.append is atomic under the GIL, so the
    streaming reader needs no lock to tail it."""
    record = _research_runs[run_id]
    try:
        voyage = embed_mod.build_client()
        with db.get_conn() as conn:
            for event in research.run(question, conn, voyage):
                record["events"].append(event)
    except Exception as e:  # noqa: BLE001 -- a background task has nowhere to raise
        record["events"].append({"type": "error", "message": str(e)})
    finally:
        record["done"] = True


def _research_event_frames(record: dict, after: int):
    """SSE frames for one run, starting at event index `after`.

    Tail-follows the record: drain what is buffered, then poll until the run
    is done AND drained -- both, because `done` can flip while events are still
    unread. The 0.25s poll is imperceptible next to a loop that thinks in
    tens of seconds, and costs nothing while blocked in sleep.
    """
    i = max(0, after)
    while True:
        events = record["events"]
        while i < len(events):
            yield f"data: {json.dumps(events[i])}\n\n"
            i += 1
        if record["done"] and i >= len(record["events"]):
            return
        time.sleep(0.25)


@app.post("/api/research")
def research_start(req: AskRequest, background: BackgroundTasks):
    """Start a research run and return its id -- the events route streams it.

    Split from the stream so the run survives the client: the chat page can
    navigate away mid-run and re-attach to the same run_id when it returns.
    """
    if not config.VOYAGE_API_KEY:
        raise HTTPException(500, "VOYAGE_API_KEY is not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set")

    # Prune finished runs first; an in-flight run is never evicted, because
    # its worker would keep appending to a record nobody can reach.
    while len(_research_runs) >= _RESEARCH_KEEP:
        stale = next((k for k, r in _research_runs.items() if r["done"]), None)
        if stale is None:
            break
        del _research_runs[stale]

    run_id = secrets.token_hex(16)
    _research_runs[run_id] = {"question": req.question, "events": [], "done": False}
    background.add_task(_run_research, run_id, req.question)
    return {"run_id": run_id}


@app.get("/api/research/{run_id}/events")
def research_events(run_id: str, after: int = 0):
    """Server-sent events for one run, resumable via `after`.

    SSE rather than one JSON blob for the same reason as ever: the loop takes
    tens of seconds and the trace is what makes the page a diagnostic instead
    of a demo. `after` is how a returning page skips what it already rendered
    from its saved copy and picks up live at the first unseen event.
    """
    record = _research_runs.get(run_id)
    if record is None:
        raise HTTPException(
            404, "That run is gone — the server may have restarted. Ask again."
        )
    return StreamingResponse(
        _research_event_frames(record, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/browse")
def browse_stream(req: BrowseRequest):
    """Browse the Drive library for books worth reading next.

    Streamed for the same reason /api/research is: the agent runs several Drive
    searches over tens of seconds, and seeing which ones it chose is what
    separates a shortlist you can trust from a list of plausible-looking titles.

    Read-only. The agent can look at anything and write nothing; adding a book is
    a separate, explicit POST that the user triggers by pressing a button.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set")

    def stream():
        try:
            with db.get_conn() as conn:
                for event in browse_loop.run(req.interest, conn, count=req.count):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # noqa: BLE001 -- surface it in the stream, not a 500
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/library/drive")
def add_drive_books(req: AddDriveRequest, background: BackgroundTasks):
    """Add Drive books to the library by file id, and start indexing them.

    The metadata is re-fetched from Drive rather than taken from the request
    body. A client-supplied title, md5 or size would be a claim about a file the
    server can verify in one call -- and md5 is load-bearing: process_book
    compares it against the downloaded bytes, so a wrong one fails every
    ingestion with a confusing mismatch error.

    Idempotent. upsert_book's guard means re-adding an indexed book writes
    nothing and does not reset it; the response says which ids were already held
    so the UI can say so rather than implying work started.
    """
    service = drive_client.build_service()
    added, already = [], []
    with db.get_conn() as conn:
        held = browse_tools.indexed_ids(conn)
        for file_id in req.file_ids:
            if file_id in held:
                already.append({"file_id": file_id, **held[file_id]})
                continue
            meta = drive_client.get_file(service, file_id)
            if meta.get("mimeType") != drive_client.PDF_MIME:
                raise HTTPException(400, f"{file_id} is not a PDF.")
            size = meta.get("size")
            db.upsert_book(
                conn,
                file_id,
                meta.get("name", ""),
                meta.get("md5Checksum"),
                int(size) if size is not None else None,
                source="drive",
            )
            added.append(db.fetch_book_by_source_id(conn, file_id))

    if added:
        _start_ingest(background, "drive")

    return {
        "added": [{"id": b["id"], "title": b["title"], "status": b["status"]}
                  for b in added],
        "already_indexed": already,
    }


# ----------------------------------------------------------- drive mirror --

@app.get("/api/drive/children")
def drive_children(parent: str | None = None):
    """One folder's contents, from the local mirror.

    No Drive call. A full Drive listing is ~110 seconds and expanding a folder
    live is ~0.8s each, neither of which a file browser can spend -- so the tree
    is an indexed SQL lookup on drive_files(parent_id).

    parent omitted means the roots.
    """
    with db.get_conn() as conn:
        result = db.drive_children(conn, parent)
        result["breadcrumb"] = db.drive_breadcrumb(conn, parent) if parent else []
        # The current folder itself, so the header's "index all here" button
        # can exist when you are inside a folder and no row for it is visible.
        result["folder"] = db.drive_file(conn, parent) if parent else None
    # Where the bulk-index line is, so the page can disable oversized folders'
    # buttons up front instead of letting them discover the 413.
    result["folder_limit_mb"] = config.FOLDER_INDEX_LIMIT_BYTES // (1024 * 1024)
    return result


class AddDriveFilesRequest(BaseModel):
    # 100 matches the search route's maximum k: the intended caller is "index
    # everything this search returned". The real guard is the byte limit below.
    file_ids: list[str] = Field(min_length=1, max_length=100)


@app.post("/api/drive/files/index")
def index_drive_files(req: AddDriveFilesRequest, background: BackgroundTasks):
    """Queue a list of mirror PDFs -- the "index all results" of a search.

    Unlike POST /api/library/drive (hand-picked ids, re-verified against Drive
    one call each), this trusts the mirror like the folder route does, so a
    whole result page queues in one statement. The same byte ceiling applies,
    measured over what is not already queued.
    """
    counts = None
    with db.get_conn() as conn:
        counts = db.queue_drive_files(
            conn, req.file_ids, limit_bytes=config.FOLDER_INDEX_LIMIT_BYTES
        )
    if counts["over_limit"]:
        raise HTTPException(
            413,
            f"These results hold {counts['pending_bytes'] // (1024 * 1024)} MB of "
            f"unindexed PDFs, over the "
            f"{config.FOLDER_INDEX_LIMIT_BYTES // (1024 * 1024)} MB bulk-index limit.",
        )
    if counts["queued"]:
        _start_ingest(background, "drive")
    return {
        "queued": counts["queued"],
        "already_indexed": counts["matched"] - counts["queued"],
    }


@app.post("/api/drive/folders/{folder_id}/index")
def index_drive_folder(folder_id: str, background: BackgroundTasks):
    """Queue every PDF in one folder's subtree, if the folder is small enough.

    The size gate is enforced HERE, not just hidden in the UI: one request
    naming the root would otherwise put ~143 GB into flight. subtree_bytes is
    the mirror's recursive PDF total, so the comparison is a column read.

    413 rather than 400 for an oversized folder: the request is well-formed,
    the payload it implies is what is too large.
    """
    limit = config.FOLDER_INDEX_LIMIT_BYTES
    with db.get_conn() as conn:
        folder = db.drive_file(conn, folder_id)
        if folder is None:
            raise HTTPException(404, "No such folder in the mirror.")
        if folder["mime_type"] != db.FOLDER_MIME:
            raise HTTPException(400, f"{folder['title']} is a file, not a folder.")
        total = folder["subtree_bytes"] or 0
        if total > limit:
            raise HTTPException(
                413,
                f"{folder['title']} holds {total // (1024 * 1024)} MB of PDFs, "
                f"over the {limit // (1024 * 1024)} MB bulk-index limit.",
            )
        counts = db.queue_drive_folder(conn, folder_id)

    if counts["queued"]:
        _start_ingest(background, "drive")

    return {
        "folder": folder["title"],
        "queued": counts["queued"],
        "already_indexed": counts["total_pdfs"] - counts["queued"],
    }


@app.get("/api/drive/search")
def drive_search(q: str, mode: str | None = None, k: int = 40):
    """Rank the whole mirror by meaning and by words.

    This is the thing Drive itself cannot do. Drive's `name contains` matches a
    word PREFIX -- measured on this corpus, 'parab' hits and 'arable' returns
    zero -- so it can only find a book if you already know a word in its title.
    Here "the end times" reaches Revelation and eschatology.
    """
    q = q.strip()
    if not q:
        raise HTTPException(400, "Empty query.")
    if not config.VOYAGE_API_KEY:
        raise HTTPException(500, "VOYAGE_API_KEY is not set")
    k = max(1, min(k, 100))
    voyage = embed_mod.build_client()
    vec = embed_mod.embed_query(q, voyage)
    with db.get_conn() as conn:
        rows = db.search_drive_files(conn, vec, q, k, mode=mode)
    return {"query": q, "mode": mode or config.SEARCH_MODE, "results": rows}


@app.get("/api/drive/sync")
def drive_sync_status():
    with db.get_conn() as conn:
        return db.drive_sync_status(conn)


_sync_state = {"running": False, "message": ""}


def _run_sync():
    """Full mirror refresh in a background thread.

    Guarded by a flag rather than a queue: two concurrent syncs would both
    re-list all 57k files to write identical rows, and the second would only
    slow the first down. `finally` clears it, so a failure does not wedge the
    button forever.
    """
    if _sync_state["running"]:
        return
    _sync_state.update(running=True, message="listing Drive…")
    try:
        service = drive_client.build_service()
        with db.get_conn() as conn:
            counts = mirror.sync(conn, service,
                                 progress=lambda m: _sync_state.update(message=m))
            _sync_state["message"] = f"embedding titles ({counts['files']:,} files)…"
            if config.VOYAGE_API_KEY:
                mirror.embed_titles(
                    conn, embed_mod.build_client(),
                    progress=lambda m: _sync_state.update(message=m),
                )
                mirror.build_index(conn)
        _sync_state["message"] = "done"
    except Exception as e:  # noqa: BLE001 -- a background task has nowhere to raise
        _sync_state["message"] = f"failed: {e}"
        print(f"drive sync failed: {e}")
    finally:
        _sync_state["running"] = False


@app.post("/api/drive/sync")
def drive_sync_start(background: BackgroundTasks):
    if _sync_state["running"]:
        return {"started": False, **_sync_state}
    background.add_task(_run_sync)
    return {"started": True, "running": True, "message": "queued"}


@app.get("/api/drive/sync/progress")
def drive_sync_progress():
    with db.get_conn() as conn:
        return {**db.drive_sync_status(conn), **_sync_state}


# ------------------------------------------------------------ drive auth --
#
# Reconnecting Google from the page rather than the terminal. The app is its own
# OAuth callback: drive/client.py explains why run_local_server() (the CLI path)
# cannot be used inside a request handler.

# state -> (redirect_uri, code_verifier) for flows started but not yet
# completed. Both halves are needed at the exchange and neither can be
# recomputed: the redirect_uri must match byte-for-byte, and the PKCE verifier
# exists only on the Flow instance that built the consent URL.
#
# In memory, so a restart mid-consent means starting over. Correct for a
# single-user local app, and better than writing a CSRF token and a PKCE secret
# to disk next to the credentials they protect.
_auth_flows: dict[str, tuple] = {}


@app.get("/api/drive/auth/status")
def drive_auth_status():
    """Is Drive usable right now. Safe to poll -- never opens a consent flow.

    `connection` says who connected it and when, present only when the token
    came from the database. Two people can reconnect this, and a shared
    credential nobody can attribute is a shared credential nobody owns.
    """
    status = drive_client.credentials_status()
    info = drive_store.connection_info()
    if info:
        status["connection"] = info
    return status


@app.get("/api/drive/auth/start")
def drive_auth_start(request: Request):
    """Begin consent: returns the Google URL for the page to open.

    The redirect_uri is derived from the incoming request rather than
    configured, so this works on whatever host and port uvicorn was started
    with. It must match byte-for-byte at the exchange, which is why it is
    stored rather than recomputed.
    """
    redirect_uri = str(request.url_for("drive_auth_callback"))
    state = secrets.token_urlsafe(24)
    try:
        url, verifier = drive_client.auth_url(redirect_uri, state)
    except FileNotFoundError:
        raise HTTPException(503, f"{drive_client.CREDENTIALS_FILE} is missing.")
    _auth_flows[state] = (redirect_uri, verifier)
    return {"url": url}


@app.get("/api/drive/auth/callback", name="drive_auth_callback")
def drive_auth_callback(request: Request, state: str = "", code: str = "",
                        error: str = ""):
    """Where Google sends the user back. Renders a page, not JSON -- a browser
    lands here directly, so it must be readable by a person."""
    if error:
        return _auth_page("Google declined the request", error, ok=False)
    # Reject a callback we did not start. Without this, anything that can make
    # the browser hit this URL could hand us a code of its choosing.
    pending = _auth_flows.pop(state, None)
    if not pending:
        return _auth_page(
            "That sign-in link has expired",
            "Start again from the library page. (The server may also have "
            "restarted mid-flow, which clears pending sign-ins.)",
            ok=False,
        )
    redirect_uri, verifier = pending
    try:
        # Attribute the connection. Either allowlisted person may reconnect
        # -- Google revokes these every 7 days -- and knowing which of them
        # did it is the difference between "whose Drive are we reading" being
        # answerable and not.
        drive_client.exchange_code(
            code, redirect_uri, verifier,
            connected_by=auth.current_user(request) or "unknown",
        )
    except Exception as e:  # noqa: BLE001 -- rendered for a human, not re-raised
        return _auth_page("Could not complete sign-in", str(e), ok=False)
    return _auth_page(
        "Connected to Google Drive",
        "You can close this tab and go back to the library.", ok=True,
    )


def _auth_page(title: str, detail: str, *, ok: bool) -> HTMLResponse:
    # Deliberately not echoing `code` or `state` back into the page: an
    # authorization code in rendered HTML is a credential in the browser
    # history, in a screenshot, and in anything that scrapes the tab title.
    return HTMLResponse(
        f"""<!doctype html><meta charset="utf-8">
        <title>{escape(title)}</title>
        <link rel="stylesheet" href="/static/app.css">
        <main><div class="card">
          <h1 style="color:var({'--good' if ok else '--bad'})">{escape(title)}</h1>
          <p class="sub">{escape(detail)}</p>
          <p><a href="/library">Back to the library</a></p>
        </div></main>""",
        status_code=200 if ok else 400,
    )


# ------------------------------------------------------------------- bible --
# The Bible reader. English only, one table, no embeddings -- see
# migrations/0006_bible_verses.sql and library_rag/bible.py.


@app.get("/bible")
def bible_page():
    """The Bible reader. A fourth page because it is a fourth task -- reading
    Scripture, which shares no state with the PDF library."""
    return _page("bible.html")


@app.get("/api/bible/books")
def bible_books():
    """The 66 books with their chapter counts.

    Also reports `loaded`, so a database that has not run the migration gets a
    page saying which command to run instead of a 500. This is the page's first
    call and the only one that has to be able to describe an empty install as
    data rather than as an error.
    """
    with db.get_conn() as conn:
        if not bible.loaded(conn):
            return {"loaded": False, "books": []}
        rows = bible.books(conn)
    return {
        "loaded": True,
        "books": [
            {"book": b, "name": name, "chapters": chapters, "verses": verses}
            for b, name, chapters, verses in rows
        ],
    }


@app.get("/api/bible/chapter")
def bible_chapter(book: int, chapter: int):
    with db.get_conn() as conn:
        verses = bible.chapter(conn, book, chapter)
    if not verses:
        raise HTTPException(404, f"No chapter {chapter} in book {book}.")
    return {
        "book": book,
        "chapter": chapter,
        # `text` is "" for the 16 placeholder verses. Passed through as-is; the
        # page decides how to show an absence, because that is a display
        # decision and this is the data.
        "verses": [{"verse": v, "text": t} for v, t in verses],
    }


# The cap is here rather than in bible.search so that `truncated` below can be
# computed against the same number the query used.
SEARCH_LIMIT = 200


@app.get("/api/bible/search")
def bible_search(q: str):
    """Verses containing `q`, in Bible order.

    Literal substring matching, not semantic -- typing "love" finds the letters
    l-o-v-e. `truncated` exists so the page can say a result was cut off; a
    silently capped list reads as a complete answer.
    """
    q = q.strip()
    if not q:
        raise HTTPException(400, "Empty search.")
    with db.get_conn() as conn:
        rows = bible.search(conn, q, limit=SEARCH_LIMIT)
    return {
        "query": q,
        "truncated": len(rows) == SEARCH_LIMIT,
        "results": [
            {"book": b, "name": name, "chapter": c, "verse": v, "text": t}
            for b, name, c, v, t in rows
        ],
    }


# -------------------------------------------------------------------- auth --
# Google sign-in. See web/auth.py for the two-questions split (who are you /
# may you in) and for why this is not Supabase Auth.


@app.get("/api/health")
def health():
    """Liveness. Touches nothing on purpose -- a health check that queries
    Postgres turns a momentary database blip into a restart loop.

    NOT /healthz, which is the obvious name and does not work: Google's
    frontend intercepts that exact path in front of Cloud Run and answers its
    own 404, so the request never reaches this process. Verified against the
    deployed service -- /health, /healthz/ and /nonexistent all arrive here and
    get a 401 from the gate, while /healthz alone returns a Google error page
    and appears nowhere in the container logs.
    """
    return {"ok": True}


@app.get("/login")
def login_page():
    return _page("login.html")


@app.get("/api/auth/config")
def auth_config():
    """What the login page needs to render Google's button. The client id is
    public by design -- it ships in the HTML, and the security comes from Google
    signing the token and this server checking the signature and audience."""
    return {"enabled": config.auth_enabled(), "client_id": config.GOOGLE_CLIENT_ID}


class GoogleCredential(BaseModel):
    credential: str = Field(min_length=1, max_length=8192)


@app.post("/api/auth/google")
def auth_google(req: GoogleCredential, request: Request):
    """Exchange a Google ID token for a session on this app.

    Two independent gates, and the order matters: verify the token FIRST, so an
    unverifiable credential never causes a database lookup, and only then ask
    whether that proven identity is on the allowlist.

    403 rather than 401 for a non-allowlisted address: we know exactly who they
    are, and they still may not in. Saying so plainly beats a generic failure
    that reads like a bug in the sign-in button.
    """
    if not config.auth_enabled():
        raise HTTPException(400, "Sign-in is not configured on this server.")
    try:
        email = auth.verify_google_token(req.credential)
    except auth.AuthError as e:
        raise HTTPException(401, str(e)) from e

    with db.get_conn() as conn:
        allowed = auth.is_allowed(conn, email)
    if not allowed:
        raise HTTPException(
            403,
            f"{email} is not on the allowlist for this app. "
            "Ask the owner to add it.",
        )

    auth.sign_in(request, email)
    return {"email": email}


@app.get("/logout")
def logout(request: Request):
    auth.sign_out(request)
    return RedirectResponse("/login", status_code=302)


@app.get("/api/auth/me")
def auth_me(request: Request):
    """Who is signed in -- drives the sidebar footer. Returns null rather than
    401 when auth is off, so the pages render identically in local development."""
    return {"enabled": config.auth_enabled(), "email": auth.current_user(request)}
