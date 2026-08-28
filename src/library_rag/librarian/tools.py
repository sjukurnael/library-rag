"""
The librarian's tools. Kept deliberately dumb -- judgment belongs in loop.py's
system prompt.

Two populations, and keeping them distinct is the job:

  INDEXED books  -- already chunked and embedded. Free to add to a classroom,
                    answerable the moment they land on the shelf, and readable
                    RIGHT NOW via look_inside.
  DRIVE files    -- the other ~48,000 PDFs in the mirror. Adding one means an
                    ingest: minutes of download and extraction, Voyage tokens,
                    and possibly OCR that is not configured.

Blur them and a reader ends up with a classroom that cannot answer anything for
the next hour, which is why `kind` travels with every recommendation rather than
being inferred at the last moment by the page.
"""
from library_rag import config, db
from library_rag.drive import tools as drive_tools
from library_rag.pipeline import embed as embed_mod

# Re-exported so the loop has one import and the Drive half stays exactly the
# code the browse agent used -- it was already right, and rewriting a working
# Drive lister to move it into a new package is how a rename becomes a bug.
browse_folder = drive_tools.browse_folder
search_drive = drive_tools.search_drive
indexed_ids = drive_tools.indexed_ids
DEFAULT_LIMIT = drive_tools.DEFAULT_LIMIT
MAX_LIMIT = drive_tools.MAX_LIMIT

FIND_K = 12
MAX_FIND_K = 30
LOOK_K = 3
# Words of a passage handed back per look_inside hit. Enough to judge and to
# quote, short enough that verifying thirty candidates does not fill the
# context window with book.
PASSAGE_WORDS = 60


def find_books(conn, topic: str, k: int = FIND_K, *, voyage=None,
               classroom_ids=()) -> dict:
    """Rank INDEXED books by what they are about.

    This is the tool the old browse agent did not have. It reads book-level
    topic centroids, so "the atonement" finds a systematic theology whose title
    never says so -- measured at hit@10 96% against 68% for a single whole-book
    vector, which is why book_topic_vectors is multi-row.

    `thin` is the honest-emptiness signal. A nearest-neighbour ranker always
    fills its page, so "returned 12" says nothing about whether the library
    holds the subject at all. Absolute distance does: real briefs reach
    0.26-0.36 and subjects this corpus lacks bottom out at 0.51-0.58
    (cli/eval_librarian.py). Past the floor, the model is told rather than left
    to infer it from titles that merely look plausible.
    """
    voyage = voyage or embed_mod.build_client()
    k = max(1, min(int(k or FIND_K), MAX_FIND_K))
    vec = embed_mod.embed_query(topic, voyage)
    rows = db.search_book_profiles(conn, vec, topic, k=k + len(classroom_ids))

    books, nearest = [], None
    for r in rows:
        if len(books) >= k:
            break
        d = float(r["distance"]) if r["distance"] is not None else None
        if d is not None and (nearest is None or d < nearest):
            nearest = d
        books.append({
            "book_id": r["book_id"],
            "title": r["title"],
            "pages": r["page_count"],
            "covers": r["best_label"],
            "matching_topics": r["matched_topics"],
            "distance": None if d is None else round(d, 4),
            "in_classroom": r["book_id"] in set(classroom_ids),
            "kind": "indexed",
        })
    return {
        "topic": topic,
        "returned": len(books),
        "nearest": None if nearest is None else round(nearest, 4),
        "thin": nearest is None or nearest > config.PROFILE_RELEVANCE_FLOOR,
        "books": books,
    }


def look_inside(conn, book_id: int, question: str, k: int = LOOK_K,
                *, voyage=None) -> dict:
    """Read a book before recommending it. ~12ms, so verify freely.

    Passages come back trimmed. The librarian needs enough to judge relevance
    and to quote one line as evidence, not the chapter -- and thirty untrimmed
    verifications would cost more context than the whole rest of the run.
    """
    voyage = voyage or embed_mod.build_client()
    vec = embed_mod.embed_query(question, voyage)
    rows = db.look_inside(conn, book_id, vec, k=k)
    title = conn.execute(
        "SELECT title FROM books WHERE id = %s", (book_id,)
    ).fetchone()
    return {
        "book_id": book_id,
        "book": title[0] if title else None,
        "returned": len(rows),
        "nearest": round(float(rows[0]["distance"]), 4) if rows else None,
        "passages": [
            {
                "pages": _pages(r["page_start"], r["page_end"]),
                "heading": r["heading_trail"] or None,
                "distance": round(float(r["distance"]), 4),
                "text": " ".join((r["content"] or "").split()[:PASSAGE_WORDS]),
            }
            for r in rows
        ],
    }


def _pages(start, end):
    if start is None:
        return "p.?"
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


def recommend(conn, picks: list, seen_books: dict, seen_files: dict) -> dict:
    """The shortlist. Writes NOTHING -- the reader's button does that.

    Every pick is validated against what this run actually saw, for the same
    reason the browse agent validated file_ids: a recommendation the reader
    cannot trace is indistinguishable from a hallucination, and silently
    dropping one hides the failure instead of showing it.

    An indexed pick is additionally checked for status='done'. A book still
    being ingested has no chunks, so recommending it would put something on the
    shelf that cannot answer -- exactly the two-population confusion `kind`
    exists to prevent.
    """
    out = []
    for pick in picks:
        why = pick.get("why", "")
        passage = pick.get("passage")
        bid, fid = pick.get("book_id"), pick.get("file_id")

        if bid is not None:
            known = seen_books.get(int(bid))
            if known is None:
                out.append({"book_id": bid, "why": why, "unknown": True,
                            "error": "no search this run returned that book_id"})
                continue
            row = conn.execute(
                "SELECT title, page_count, status::text FROM books WHERE id = %s",
                (int(bid),),
            ).fetchone()
            if row is None or row[2] != "done":
                out.append({"book_id": bid, "title": known.get("title"), "why": why,
                            "unknown": True,
                            "error": f"book is not ready to read (status "
                                     f"{row[2] if row else 'missing'})"})
                continue
            # Where the quote sits, found by matching the words rather than
            # asking the model to report it. Only for indexed books: a Drive
            # pick has no chunks to look in, which is the same reason it has no
            # passage. None when the words cannot be located -- see
            # db.page_of_passage on why that is the right answer.
            span = db.page_of_passage(conn, int(bid), passage) if passage else None
            out.append({
                "kind": "indexed", "book_id": int(bid), "title": row[0],
                "pages": row[1], "covers": known.get("covers"),
                "why": why, "passage": passage,
                "passage_page": span[0] if span else None,
                "passage_pages": _pages(*span) if span else None,
            })
            continue

        if fid is not None:
            item = seen_files.get(fid)
            if item is None:
                out.append({"file_id": fid, "why": why, "unknown": True,
                            "error": "no search this run returned that file_id"})
                continue
            out.append({**item, "kind": "drive", "why": why})
            continue

        out.append({"why": why, "unknown": True,
                    "error": "a pick needs either book_id or file_id"})

    return {"recommendations": out}
