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
from html import escape
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from library_rag import config, db, ingest
from library_rag.drive import client as drive_client
from library_rag.drive import mirror
from library_rag.exploration import loop as browse_loop
from library_rag.exploration import tools as browse_tools
from library_rag.pipeline import embed as embed_mod
from library_rag.retrieval import research

app = FastAPI(title="library-rag")
_STATIC = Path(__file__).resolve().parent / "static"


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


class AddDriveRequest(BaseModel):
    # Bounded: this queues real downloads, and an unbounded list is an easy way
    # to put the whole 130 GB drive into flight from one request.
    file_ids: list[str] = Field(min_length=1, max_length=20)


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/library")
def library():
    """The Drive browser. A separate page from the research chat because it is a
    different task -- deciding what to read, rather than reading."""
    return FileResponse(_STATIC / "library.html")


@app.get("/queue")
def queue():
    """The processing dashboard: everything mid-pipeline, stage by stage, plus
    upload. A third page because it is a third task -- watching work happen,
    rather than reading or choosing what to read."""
    return FileResponse(_STATIC / "queue.html")


@app.get("/static/{name}")
def static_file(name: str):
    """Shared CSS/JS. Whitelisted rather than mounted as a directory: this
    process also holds credentials.json and token.json, and a path parameter
    that reaches the filesystem is the wrong thing to be casual about."""
    if name not in {"app.css", "app.js"}:
        raise HTTPException(404, "No such asset.")
    return FileResponse(_STATIC / name)


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
                   EXTRACT(EPOCH FROM now() - claimed_at) AS claim_age_s
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
            }
            for r in pending
        ],
    }


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
    exactly that worker. It lives here so a single-user local install does
    something when you press the button.
    """
    try:
        ingest.process_queue(source=source)
    except Exception as e:  # noqa: BLE001 -- a background task has nowhere to raise
        print(f"background ingest failed ({source}): {e}")


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
        background.add_task(_drain_queue, "upload")

    return {
        "id": book["id"],
        "title": book["title"],
        "status": book["status"],
        # Content-addressed identity means an identical re-upload is recognised
        # rather than duplicated; say so instead of implying work is happening.
        "already_indexed": already_indexed,
    }


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


@app.post("/api/research")
def research_stream(req: AskRequest):
    """Server-sent events rather than one JSON blob: the loop takes tens of
    seconds, and the trace -- which queries it chose, what each returned -- is
    what makes the page a diagnostic instead of a demo. Waiting in silence would
    hide exactly what is worth watching."""
    if not config.VOYAGE_API_KEY:
        raise HTTPException(500, "VOYAGE_API_KEY is not set")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set")

    def stream():
        try:
            voyage = embed_mod.build_client()
            with db.get_conn() as conn:
                for event in research.run(req.question, conn, voyage):
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # noqa: BLE001 -- surface it in the stream, not a 500
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
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
                for event in browse_loop.run(req.interest, conn):
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
        background.add_task(_drain_queue, "drive")

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
    return result


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
    """Is Drive usable right now. Safe to poll -- never opens a consent flow."""
    return drive_client.credentials_status()


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
def drive_auth_callback(state: str = "", code: str = "", error: str = ""):
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
        drive_client.exchange_code(code, redirect_uri, verifier)
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
