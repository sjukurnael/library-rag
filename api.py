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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
import db
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
    """What is actually searchable -- so the UI never implies coverage we lack."""
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT b.id, b.title, b.page_count, count(c.id) AS chunks
            FROM books b JOIN chunks c ON c.book_id = b.id
            WHERE b.status = 'done'
            GROUP BY b.id, b.title, b.page_count
            ORDER BY b.title
            """
        ).fetchall()
    return {
        "books": [
            {"id": r[0], "title": r[1], "pages": r[2], "chunks": r[3]} for r in rows
        ],
        "total_chunks": sum(r[3] for r in rows),
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
