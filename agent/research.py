"""
Agentic retrieval over the indexed library.

The difference from api.py's one-shot path: there, code decides the search (one
query, top-k, done) and the model only writes prose. Here the model runs the
loop. It can decompose a question into sub-questions, issue several searches,
reformulate one that came back useless, chase a reference it spotted in a
retrieved passage, scope a search to a single book, and judge for itself when it
has enough to answer.

That matters because the one-shot path has a hard ceiling: whatever the single
embedding of the raw question retrieves is all the model will ever see. A
question like "how do these two books differ on the Holy Spirit" is not one
lookup -- it is at least two, and the phrasing that finds each is not the user's
phrasing.

run() is a generator of events so callers can stream the trace. Seeing which
queries the agent issued, and what each returned, is the whole diagnostic value
-- an answer alone hides whether good output came from good retrieval.

Sources accumulate into one numbered list across every search in a run, so a
citation [7] means the same passage no matter which search surfaced it.
"""
import json
import os

from anthropic import Anthropic

import db
from pipeline import embed as embed_mod

MODEL = "claude-opus-5"
MAX_ITERATIONS = 8
MAX_SEARCHES = 12
DEFAULT_K = 8
MAX_K = 15

# Same cutoff api.py uses. Measured over this corpus: real hits land ~0.33-0.62,
# an off-topic query lands 0.77+. Passages past it are reported to the agent as
# weak rather than silently dropped, so it can tell "nothing here" apart from
# "my query was badly phrased" and reformulate instead of giving up.
RELEVANCE_CUTOFF = 0.70

SYSTEM_PROMPT = """You research questions against a library of Bible study guides \
and doctrine books, using ONLY what search returns.

You control the searching, and the skill being tested is knowing when NOT to \
search again. Effort should be proportionate to the question.

Start with ONE search. Then look at what came back and ask: is there a specific \
gap I can name? Search again only if you can name it. "I could probably find \
more" is not a gap. "I have book 93's position but not book 94's" is.

As a calibration:
- A plain factual or definitional question ("who is X", "what does Y mean") is \
usually one search, occasionally two. Answer from it.
- A comparative question needs one search per thing compared -- that is a \
nameable gap for each.
- A multi-part question needs one search per part.

Searching more than the question requires is a cost, not thoroughness. It makes \
you slower and more expensive without making the answer better.

Do not call list_books unless the question names or compares particular books, \
or you need a book_id to scope a search. For a general question it tells you \
nothing you need.

When you do search again:
- Rephrase for the corpus, not the user. These are printed books; they say \
"the Spirit of holiness", not "is the Holy Spirit real". If a search returns \
weak matches (distance above 0.70), that usually means your phrasing missed, \
not that the library is silent. Retry with the vocabulary the books would use \
before concluding anything.
- Scope with book_id when the question is about one book, or to compare two.
- Follow a lead only if answering the question actually depends on it.

Then answer:

- Ground every claim in retrieved passages. Cite inline as [1], [2] using the \
numbers search gave you.
- If the library does not answer the question, say so plainly. Never fill the \
gap from your own knowledge of the Bible or theology -- the point of this system \
is to surface what these particular books say.
- If it partially answers, say what is covered and what is not.
- These are study guides: they often pose questions to the reader rather than \
assert answers. Represent that faithfully; do not convert a prompt into a claim.
- Where two books differ, say so and cite both.

Write the answer as clean prose, the way a knowledgeable person would explain it in conversation. Short paragraphs. No headings unless the answer genuinely has two or more distinct parts, and never for an answer under three paragraphs. Use bold only where a term is genuinely the subject being defined -- not to decorate. Prefer sentences over bullet lists; use a list only for things that are actually a list. Do not restate the question before answering."""

TOOL_SCHEMAS = [
    {
        "name": "search_library",
        "description": (
            "Semantic search over every indexed book. Returns passages ranked by "
            "cosine distance (lower is closer; under 0.70 is a real match). "
            "Call repeatedly with different phrasings and sub-questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to search for, phrased the way the books would put "
                        "it rather than the way the user did."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": f"How many passages (1-{MAX_K}, default {DEFAULT_K}).",
                },
                "book_id": {
                    "type": "integer",
                    "description": "Restrict to one book. Omit to search everything.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_books",
        "description": (
            "The indexed books with their ids, page counts and chunk counts. Call "
            "this first when the question names or compares particular books."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _page_label(start, end):
    if start is None:
        return "p.?"
    return f"p.{start}" if start == end else f"pp.{start}-{end}"


class _Session:
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
        rows = db.search(self.conn, vec, k, book_id)

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


def run(question: str, conn, voyage, *, max_iterations: int = MAX_ITERATIONS):
    """Yield events: {"type": "search"|"results"|"thinking"|"answer"|"done", ...}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    session = _Session(conn, voyage)
    client = Anthropic()
    messages = [{"role": "user", "content": question}]

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            yield {"type": "answer", "text": answer}
            yield {
                "type": "done",
                "sources": session.sources,
                "searches": session.searches,
                "iterations": iteration,
            }
            return

        # Narration the model emitted alongside its tool calls -- this is where
        # it says what it is about to look for and why.
        note = "".join(b.text for b in response.content if b.type == "text").strip()
        if note:
            yield {"type": "thinking", "text": note}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "list_books":
                result = session.list_books()
                yield {"type": "search", "tool": "list_books", "query": None}
            else:
                q = block.input.get("query", "")
                bid = block.input.get("book_id")
                yield {
                    "type": "search",
                    "tool": "search_library",
                    "query": q,
                    "book_id": bid,
                    "k": block.input.get("k", DEFAULT_K),
                }
                result = session.search(q, block.input.get("k", DEFAULT_K), bid)
                yield {
                    "type": "results",
                    "query": q,
                    "returned": result.get("returned", 0),
                    "strong": result.get("strong_matches", 0),
                    "hits": [
                        {
                            "n": h["n"], "book": h["book"], "pages": h["pages"],
                            "heading": h["heading"], "distance": h["distance"],
                            "weak": h["weak"],
                        }
                        for h in result.get("passages", [])
                    ],
                }
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "answer",
        "text": (
            f"Stopped after {max_iterations} iterations without settling on an "
            "answer. The passages found so far are listed below."
        ),
    }
    yield {
        "type": "done",
        "sources": session.sources,
        "searches": session.searches,
        "iterations": max_iterations,
        "exhausted": True,
    }
