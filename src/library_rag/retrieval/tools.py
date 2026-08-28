"""
The tutor's tools. Kept deliberately dumb: scoped search and book listing.
Judgment (what to search, when to stop, how to phrase) belongs in loop.py's
system prompt / the LLM, not here.

With ONE exception, and it is the reason this module is worth reading: the
classroom boundary is enforced here, in Python, and never in the prompt. A
Session is built with the books it may see and cannot be talked out of them --
search() refuses a book_id outside its own scope without issuing a query, and
suggest_books() reads only book-level vectors so no passage from an unchosen
book can reach the model's context at all. "The tutor never answers from
outside the classroom" is meant to be a property of the plumbing, not a promise
the system prompt makes and the model may forget under pressure.
"""
from library_rag import db
from library_rag.pipeline import embed as embed_mod

MAX_SEARCHES = 12
DEFAULT_K = 8
MAX_K = 15

# How many times the tutor may ask what it is missing. Small on purpose: this
# is the escape hatch for "the shelf cannot answer this", not a second search
# tool, and a model that can call it freely will reach for it instead of
# rephrasing a query that would have worked.
MAX_SUGGESTS = 2
# Candidate books returned per suggestion.
SUGGEST_K = 5

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

    def __init__(self, conn, voyage, book_ids):
        """`book_ids` is the classroom. Required, and there is no value meaning
        "everything" -- a Session that could express the whole library would be
        one refactor away from doing it by accident."""
        self.conn = conn
        self.voyage = voyage
        self.book_ids = list(book_ids)
        self.sources = []
        self.by_chunk_id = {}
        self.searches = 0
        self.suggests = 0
        self.suggested = {}

    def list_books(self):
        """The shelf. Scoped, and it reports books that are still arriving.

        An arriving book has no chunks yet, so it is excluded from search but
        must still be visible here: without it the tutor reports an absence that
        is really a delay, and the reader sees the book on their shelf and the
        tutor claiming it is not there.
        """
        rows = self.conn.execute(
            """
            SELECT b.id, b.title, b.page_count, count(c.id), b.status = 'done'
            FROM books b LEFT JOIN chunks c ON c.book_id = b.id
            WHERE b.id = ANY(%s)
            GROUP BY b.id, b.title, b.page_count, b.status ORDER BY b.title
            """,
            (self.book_ids,),
        ).fetchall()
        return {
            "books": [
                {"book_id": r[0], "title": r[1], "pages": r[2], "chunks": r[3],
                 "ready": r[4]}
                for r in rows
            ]
        }

    def search(self, query, k=DEFAULT_K, book_id=None):
        if not self.book_ids:
            return {
                "error": "This classroom has no books yet. Say so, and offer to "
                         "look for some with suggest_books."
            }
        # Refused before any query is issued, and refused by identity rather
        # than by filtering: db.search would compose the scope and return an
        # empty result, which reads to the model as "nothing in that book"
        # rather than "that book is not yours to read".
        if book_id is not None and int(book_id) not in self.book_ids:
            return {
                "error": f"Book {book_id} is not in this classroom. You may "
                         f"narrow to a book on the shelf, not outside it."
            }
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
        rows = db.search(self.conn, vec, k, book_id,
                         book_ids=self.book_ids, query_text=query)

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

    def suggest_books(self, topic, k=SUGGEST_K):
        """Books that might help, for a topic this classroom cannot answer.

        The containment property of the whole design lives in one fact about
        this method: it calls db.search_book_profiles, which reads
        book_topic_vectors and drive_files and NEVER TOUCHES `chunks`. There is
        therefore no path by which a passage from an unchosen book reaches the
        model's context -- not a discouraged path, not one the prompt asks it to
        avoid, none. The model learns that a book exists and what its chapters
        are called; it cannot learn what the book says.

        That distinction is why this is not a nested librarian run. The
        librarian's job includes look_inside, which quotes real text; running it
        inside the tutor would put out-of-classroom passages one token
        prediction away from the answer, and no system prompt reliably survives
        that.

        Suggested books never enter self.sources. They have no chunk id, so
        there is nothing to cite and nothing for a citation to point at.
        """
        if self.suggests >= MAX_SUGGESTS:
            return {
                "error": f"suggestion budget exhausted ({MAX_SUGGESTS}). Answer "
                         f"with what the classroom has, and say what it lacks."
            }
        self.suggests += 1
        vec = embed_mod.embed_query(topic, self.voyage)
        rows = db.search_book_profiles(self.conn, vec, topic, k=k + len(self.book_ids))

        out = []
        for r in rows:
            if r["book_id"] in self.book_ids:
                continue          # already on the shelf; suggesting it is noise
            out.append({
                "book_id": r["book_id"],
                "title": r["title"],
                "pages": r["page_count"],
                # The chapter heading the match landed on -- a label the author
                # wrote, not text from the book. Safe to show, and it is what
                # lets the reader judge the suggestion.
                "covers": r["best_label"],
                "matching_topics": r["matched_topics"],
            })
            self.suggested[r["book_id"]] = out[-1]
            if len(out) >= k:
                break

        return {
            "topic": topic,
            "returned": len(out),
            "books": out,
            "reminder": (
                "These books are NOT in this classroom and you have not read "
                "them. Name them and say what they appear to cover. Do not "
                "describe their argument, quote them, or answer from them -- "
                "cite only [n] from search_library."
            ),
        }
