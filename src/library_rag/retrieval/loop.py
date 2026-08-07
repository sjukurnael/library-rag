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

MODEL = "claude-opus-5"
MAX_ITERATIONS = 8

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
                    "description": (
                        f"How many passages (1-{tools.MAX_K}, "
                        f"default {tools.DEFAULT_K})."
                    ),
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


def run(
    question: str,
    conn,
    voyage,
    *,
    client=None,
    max_iterations: int = MAX_ITERATIONS,
):
    """Yield events: {"type": "search"|"results"|"thinking"|"answer"|"done", ...}.

    `client` is injectable for the same reason conn/voyage/download_file are
    everywhere else in this codebase: tests drive the loop with a scripted fake
    and no network. Left None it builds a real Anthropic client.
    """
    session = tools.Session(conn, voyage)
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
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
            elif block.name == "search_library":
                q = block.input.get("query", "")
                bid = block.input.get("book_id")
                yield {
                    "type": "search",
                    "tool": "search_library",
                    "query": q,
                    "book_id": bid,
                    "k": block.input.get("k", tools.DEFAULT_K),
                }
                result = session.search(
                    q, block.input.get("k", tools.DEFAULT_K), bid
                )
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
            else:
                result = {"error": f"unknown tool: {block.name}"}
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
