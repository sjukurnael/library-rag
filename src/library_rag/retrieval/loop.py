"""
The hand-written Anthropic tool-use loop: system prompt, tool schemas, and
the run() function that drives the conversation until the model stops calling
tools and produces its final answer.

The difference from a one-shot path: there, code decides the search (one query,
top-k, done) and the model only writes prose. Here the model runs the loop. It
can decompose a question into sub-questions, issue several searches, reformulate
one that came back useless, chase a reference it spotted in a retrieved passage,
scope a search to a single book, and judge for itself when it has enough to
answer.

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

from library_rag.retrieval import tools
from library_rag import prompt_cache

# Sonnet, like the librarian. The tutor is the cheaper of the two loops -- three
# turns against the librarian's thirty -- but it is also the more constrained:
# it answers only from passages search handed it, and every citation has to
# come from one of them. That is a job with the reasoning bounded by the
# evidence in front of it, which is where Sonnet holds up. Overridable so the
# model can be changed on a running deployment without a rebuild; a separate
# name from the librarian's, so raising one does not silently raise the other.
MODEL = os.environ.get("TUTOR_MODEL", "claude-sonnet-5")
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """You are a tutor. A reader has assembled a classroom -- a small \
shelf of books they chose -- and you answer their questions using ONLY what \
search returns from those books.

The shelf is the whole world for this conversation. You cannot read anything \
outside it, and you should not pretend to: if the classroom does not cover \
something, that is a fact about the shelf worth saying plainly, not a gap to \
paper over from memory.

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

Call list_books when you need to know what is on the shelf -- it is a handful \
of titles, not a library. A book marked "ready": false is still being prepared \
and cannot be searched yet; say it is still arriving rather than that it is \
missing.

When the shelf genuinely cannot answer:

- Say so first, plainly, and say what it DOES cover that is adjacent.
- Then call suggest_books with the topic. It returns books from the wider \
library that are not on this shelf. You have NOT read them. Name them, say what \
they appear to cover, and leave the choice to the reader -- they decide what \
joins their classroom.
- Never describe a suggested book's argument, quote it, or answer from it. \
Every citation you write must come from search_library.
- Do not reach for suggest_books to avoid rephrasing. Weak matches usually mean \
your wording missed, not that the shelf is silent.

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
            "Semantic search over THIS CLASSROOM'S books only. Returns passages "
            "ranked by cosine distance (lower is closer; under 0.70 is a real "
            "match). Call repeatedly with different phrasings and sub-questions. "
            "This is the only tool whose results you may cite."
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
                    "description": (
                        f"How many passages (1-{tools.MAX_K}, "
                        f"default {tools.DEFAULT_K})."
                    ),
                },
                "book_id": {
                    "type": "integer",
                    "description": (
                        "Narrow to ONE book on this shelf. Omit to search the "
                        "whole classroom. A book not in this classroom is refused."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_books",
        "description": (
            "This classroom's shelf: ids, titles, page counts, and whether each "
            "book is ready to search yet. A handful of titles, cheap to call."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "suggest_books",
        "description": (
            "Books from the WIDER LIBRARY that are not in this classroom, for a "
            "topic the shelf cannot answer. Returns titles and the chapter "
            "headings they match -- NOT their text. You have not read these "
            "books: name them and say what they appear to cover so the reader "
            "can decide whether to add them. Never cite or answer from them. "
            f"At most {tools.MAX_SUGGESTS} calls per question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "The subject the classroom is missing, phrased as a "
                        "reader would describe what they want to study."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
]


def _dispatch(session, block):
    """Run one tool call. Returns (events to yield, result for the model).

    Split out of the loop so the dispatch can be wrapped in one try/except
    rather than three, and so a new tool is one branch rather than another
    nested block inside an already-long generator.
    """
    name = block.name
    if name == "list_books":
        return [{"type": "search", "tool": "list_books", "query": None}], \
               session.list_books()

    if name == "search_library":
        q = block.input.get("query", "")
        bid = block.input.get("book_id")
        k = block.input.get("k", tools.DEFAULT_K)
        events = [{"type": "search", "tool": "search_library",
                   "query": q, "book_id": bid, "k": k}]
        result = session.search(q, k, bid)
        events.append({
            "type": "results",
            "query": q,
            "returned": result.get("returned", 0),
            "strong": result.get("strong_matches", 0),
            "hits": [
                {"n": h["n"], "book": h["book"], "pages": h["pages"],
                 "heading": h["heading"], "distance": h["distance"],
                 "weak": h["weak"]}
                for h in result.get("passages", [])
            ],
        })
        return events, result

    if name == "suggest_books":
        topic = block.input.get("topic", "")
        events = [{"type": "search", "tool": "suggest_books", "query": topic}]
        result = session.suggest_books(topic)
        # A separate event type from `results`, because these are not sources
        # and the page must not render them as citable. They become
        # add-to-classroom cards instead.
        events.append({"type": "suggestions", "topic": topic,
                       "books": result.get("books", [])})
        return events, result

    return [], {"error": f"unknown tool: {name}"}


def run(
    question: str,
    conn,
    voyage,
    book_ids,
    *,
    client=None,
    max_iterations: int = MAX_ITERATIONS,
    model: str | None = None,
):
    """Yield events: {"type": "search"|"results"|"thinking"|"answer"|"done", ...}.

    `book_ids` is the classroom -- the only books this run may read. Positional
    and required, with no value meaning "everything": since migration 0010 there
    is no global chunk index, so an unscoped run is not merely against the
    design, it is a sequential scan of 1.76M chunks that does not finish inside
    a request. Making it impossible to express is cheaper than documenting it.

    `client` is injectable for the same reason conn/voyage/download_file are
    everywhere else in this codebase: tests drive the loop with a scripted fake
    and no network. Left None it builds a real Anthropic client.
    """
    session = tools.Session(conn, voyage, book_ids)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic()
    messages = [{"role": "user", "content": question}]

    # Carried to the 'done' event so a run's cost and ending are recorded rather
    # than inferred. Nothing else in this system costs money per call, and these
    # counters were previously discarded with the response object.
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    stop_reason = None
    model = model or MODEL
    system = prompt_cache.cacheable_system(SYSTEM_PROMPT)

    for iteration in range(1, max_iterations + 1):
        prompt_cache.move_breakpoint(messages)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        stop_reason = response.stop_reason
        # Summed across turns, because one question is several API calls and the
        # per-call figure answers nothing anyone asks. getattr rather than
        # attribute access: the scripted client the tests drive this with has no
        # usage object, and an absent counter is zero rather than a crash.
        turn_usage = getattr(response, "usage", None)
        if turn_usage is not None:
            for key in usage:
                usage[key] += getattr(turn_usage, key, 0) or 0

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            yield {"type": "answer", "text": answer}
            yield {
                "type": "done",
                "sources": session.sources,
                # Deliberately NOT merged into sources. A suggestion has no
                # chunk id, so there is nothing for a citation to point at, and
                # a page that rendered the two lists together would let a
                # reader click through to a passage that was never retrieved.
                "suggestions": list(session.suggested.values()),
                "searches": session.searches,
                "iterations": iteration,
                # 'end_turn' is a real answer; 'max_tokens' is one cut off
                # mid-sentence and 'refusal' is an empty string. The branch
                # above treats all three as success -- reporting which it was
                # is what makes that visible before it is changed.
                "stop_reason": stop_reason,
                "usage": usage,
                "model": model,
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
            # Wrapped, as the librarian loop already wraps its dispatch. Without
            # this a raising tool kills the generator mid-stream and the page
            # reports "the connection ended before the run finished" -- which
            # blames the transport for what is usually a recoverable tool error
            # the model could have worked around if it had been told.
            try:
                events, result = _dispatch(session, block)
            except Exception as e:  # noqa: BLE001 -- see above
                result = {"error": f"{type(e).__name__}: {e}"}
                events = [{"type": "tool_error", "name": block.name,
                           "message": str(e)}]
            for event in events:
                yield event
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
        "stop_reason": stop_reason,
        "usage": usage,
        "exhausted": True,
    }
