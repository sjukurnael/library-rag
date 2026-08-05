"""
Local web UI over the indexed library.

    ./.venv/bin/uvicorn api:app --reload --port 8000
    open http://localhost:8000

Every question goes through agent/research.py: the model runs its own search
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
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
import db
import ingest
from agent import research
from pipeline import embed as embed_mod

app = FastAPI(title="library-rag")
_here = os.path.dirname(os.path.abspath(__file__))


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def index():
    return FileResponse(os.path.join(_here, "static", "index.html"))


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
        pending = conn.execute(
            """
            SELECT id, title, status, source, error
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
            }
            for r in pending
        ],
    }


def _drain_queue():
    """Process whatever is queued. Runs in a background thread after an upload.

    Drains the QUEUE rather than the one book just uploaded: the queue is the
    coordination point, claim_next_book already stops two workers touching the
    same row, and processing "my" book directly would be a second scheduling
    path with none of the claim, heartbeat or retry behaviour. Two uploads
    racing therefore cooperate instead of colliding.

    Scoped to source="upload", though, and that part is not cosmetic. Draining
    everything meant one uploaded PDF put the entire Drive backlog into flight
    behind it, and the first Drive book claimed blocked this thread in an OAuth
    flow with no console to prompt -- so the upload the user actually asked for
    never ran, and nothing said why. Drive ingestion stays a deliberate act:
    `python ingest.py`.

    A real deployment moves this into its own process -- ingest.py is already
    exactly that worker. It lives here so a single-user local install does
    something when you press the button.
    """
    try:
        ingest.process_queue(source="upload")
    except Exception as e:  # noqa: BLE001 -- a background task has nowhere to raise
        print(f"background ingest failed: {e}")


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
        background.add_task(_drain_queue)

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
