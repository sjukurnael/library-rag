"""
The research agent's tools. Kept deliberately dumb: hybrid search and book
listing. Judgment (what to search, when to stop, how to phrase) belongs in
loop.py's system prompt / the LLM, not here.
"""
from library_rag import db
from library_rag.pipeline import embed as embed_mod

MAX_SEARCHES = 12
DEFAULT_K = 8
MAX_K = 15

# Same cutoff the rest of the stack uses. Measured over this corpus: real hits
# land ~0.33-0.62, an off-topic query lands 0.77+. Passages past it are reported
# to the agent as weak rather than silently dropped, so it can tell "nothing
# here" apart from "my query was badly phrased" and reformulate instead of
# giving up.
RELEVANCE_CUTOFF = 0.70


def _page_label(start, end):
    if start is None:
        return "p.?"
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


class Session:
    """Holds the running source list so citation numbers stay stable across
    searches, and de-duplicates chunks a later search surfaces again."""

    def __init__(self, conn, voyage):
        self.conn = conn
        self.voyage = voyage
        self.sources = []
        self.by_chunk_id = {}
        self.searches = 0

    def list_books(self):
        rows = self.conn.execute(
            """
            SELECT b.id, b.title, b.page_count, count(c.id)
            FROM books b JOIN chunks c ON c.book_id = b.id
            WHERE b.status = 'done'
            GROUP BY b.id, b.title, b.page_count ORDER BY b.title
            """
        ).fetchall()
        return {
            "books": [
                {"book_id": r[0], "title": r[1], "pages": r[2], "chunks": r[3]}
                for r in rows
            ]
        }

    def search(self, query, k=DEFAULT_K, book_id=None):
        if self.searches >= MAX_SEARCHES:
            return {"error": f"search budget exhausted ({MAX_SEARCHES}); answer now."}
        self.searches += 1
        k = max(1, min(int(k or DEFAULT_K), MAX_K))

        vec = embed_mod.embed_query(query, self.voyage)
        # query_text is passed as well as the vector even though the shipping
        # mode over chunks is DENSE (config.SEARCH_MODE), not hybrid. Two
        # reasons, and neither is that the lexical leg is running:
        #   - db.search picks its mode from config, so passing the text is what
        #     makes flipping SEARCH_MODE to "hybrid" a one-line change rather
        #     than a change to every caller.
        #   - --compare in cli/evaluate.py scores modes side by side against
        #     these same call sites.
        # Hybrid IS the default over drive_files (title search), where it was
        # re-measured and does win. See config.SEARCH_MODE for the chunk
        # measurement and config.DRIVE_RRF_LEXICAL_WEIGHT for the title one.
        rows = db.search(self.conn, vec, k, book_id, query_text=query)

        hits = []
        for r in rows:
            cid, dist = r["chunk_id"], float(r["distance"])
            known = self.by_chunk_id.get(cid)
            if known is None:
                known = {
                    "n": len(self.sources) + 1,
                    "chunk_id": cid,
                    "book": r["title"],
                    "book_id": r["book_id"],
                    "heading_trail": r["heading_trail"],
                    "pages": _page_label(r["page_start"], r["page_end"]),
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "ordinal": r["ordinal"],
                    "total_chunks": r["total_chunks"],
                    "book_pages": r["page_count"],
                    "tokens": r["token_count"],
                    "content": r["content"],
                    "distance": round(dist, 4),
                }
                self.sources.append(known)
                self.by_chunk_id[cid] = known
            hits.append(
                {
                    "n": known["n"],
                    "book": r["title"],
                    "pages": known["pages"],
                    "heading": r["heading_trail"] or None,
                    "distance": round(dist, 4),
                    "weak": dist > RELEVANCE_CUTOFF,
                    "text": r["content"],
                }
            )

        strong = sum(1 for h in hits if not h["weak"])
        return {
            "query": query,
            "returned": len(hits),
            "strong_matches": strong,
            "searches_used": self.searches,
            # The budget line is repeated in EVERY tool result on purpose. A
            # system-prompt instruction is read once and stops competing with
            # the model's own momentum by iteration three; state in the
            # conversation is re-read every turn. This is what actually holds
            # effort proportionate to the question.
            "stop_check": (
                f"You have now made {self.searches} search(es). "
                + (
                    "If these passages answer the question, ANSWER NOW. Only "
                    "search again if you can name the specific gap that remains "
                    "-- write the gap down in your reply before searching."
                    if strong > 0
                    else "No strong matches: retry with the vocabulary the books "
                    "would use before concluding the library is silent."
                )
            ),
            "passages": hits,
        }
