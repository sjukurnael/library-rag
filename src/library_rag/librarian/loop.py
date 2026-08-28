"""
The librarian's tool-use loop: system prompt, tool schemas, and a run() that
drives the conversation until the model produces a shortlist.

The event contract is deliberately the one exploration/loop.py already yielded
-- thinking / tool / results / tool_error / recommendations / answer / done --
so the page's existing trail renderer keeps working. What changed is underneath:
this agent can read the books.

run() is a generator of events, the same shape the tutor uses, so the web route
can stream the trail and a CLI can print it without a second implementation.
Watching which searches ran is most of the value: a list of book titles with no
visible provenance is indistinguishable from a hallucination.
"""
import json
import os

from anthropic import Anthropic

from library_rag import config
from library_rag.drive import client as drive_client
from library_rag.librarian import tools
from library_rag import prompt_cache

# Sonnet rather than Opus, at the reader's request, because this is the loop
# that costs real money: a run is thirty-odd turns and the conversation is
# re-sent on every one of them. The env override is kept so the model can be
# changed on a running deployment without a rebuild.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Higher than the browse agent's 12, because the shape of the work changed. A
# real run is a broad sweep, a gap-filling sweep, then twenty-odd look_inside
# calls -- and at ~5ms a search and ~12ms a read, the cost of that funnel is
# noise next to one model turn. The leash is here to stop a loop, not to ration
# searching.
MAX_ITERATIONS = 40
DEFAULT_COUNT = 12
MAX_COUNT = 30


def system_prompt(count: int = DEFAULT_COUNT, shelf: str = "") -> str:
    return f"""You are a librarian. A reader describes what they want to study \
and you assemble the shelf they will work from -- a classroom of up to {count} \
books.

You are NOT answering their question. You are choosing the books their future \
questions will be answered from, and you will not be there when they ask.

SEARCHING IS CHEAP. find_books is milliseconds; look_inside is about twelve. \
You have budget for dozens of both. Use it, and NEVER recommend a book you have \
not opened.

RECALL BEATS PRECISION, ASYMMETRICALLY.
An extra book costs the reader almost nothing -- it sits on the shelf unused. A \
missing book silently breaks every future question that needed it, and they \
will never know it was missing. When you are unsure, include it.

VERIFY BEFORE RECOMMENDING.
find_books tells you a book MIGHT cover the topic. look_inside tells you it \
DOES. Every pick needs a quoted passage and its page as the reason -- that is \
what lets the reader overrule you, and a recommendation they cannot check is \
worth nothing.

COVER THE SPACE, NOT THE TOP OF A RANKING.
Twelve books making the same argument is a bad shelf. After your first sweep, \
ask what position is missing and go find it. A reader studying the atonement \
needs Christus Victor and moral influence, not five more penal substitution \
volumes.

DECOMPOSE. "Compare Reformed and Wesleyan sanctification" is two searches, not \
one. No single book profile matches a comparative brief.

TWO KINDS OF BOOK, AND THE DIFFERENCE MATTERS.
- find_books returns INDEXED books. They are ready now: the reader adds one and \
can ask about it immediately.
- search_drive and browse_folder also reach files that have never been indexed. \
Adding one of those means waiting minutes for it to be processed first.
Prefer indexed books. Propose an unindexed file only when the library genuinely \
lacks the subject, and say plainly that it will need indexing before it can \
answer.

SAY WHEN IT IS NOT THERE. If find_books comes back `thin`, the library probably \
does not hold this subject. Say so and recommend fewer. A padded shelf produces \
a classroom that fails every question and looks like someone else's bug.

FINISHING. Call recommend with your picks -- each with `why` and the `passage` \
you quoted -- then write a short closing paragraph in plain language for a \
reader who is not technical. Do not describe your searches; they can see them.
{shelf}"""


TOOL_SCHEMAS = [
    {
        "name": "find_books",
        "description": (
            "Rank INDEXED books by what they are actually about, using their "
            "contents rather than their filenames. Your primary tool. Returns "
            "each book's best-matching chapter heading and how many of its "
            "topics matched, plus `thin`: true when even the nearest book is "
            "far from the topic, which means the library probably does not "
            "hold it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "A subject, phrased the way the books would put it. "
                        "One facet per call -- decompose a comparison."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": f"How many books (1-{tools.MAX_FIND_K}, "
                                   f"default {tools.FIND_K}).",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "look_inside",
        "description": (
            "Read the best passages inside ONE indexed book. Use it to confirm "
            "a candidate really covers the topic, and to get the quote that "
            "justifies recommending it. Fast -- verify every book you intend "
            "to recommend."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "book_id": {"type": "integer"},
                "question": {
                    "type": "string",
                    "description": "What you want this book to say about the topic.",
                },
            },
            "required": ["book_id", "question"],
        },
    },
    {
        "name": "search_drive",
        "description": (
            "Search all ~57,000 PDFs in the drive by filename and folder path. "
            "Reaches files that have NOT been indexed and cannot be read. Use "
            "it for author names, series titles, and the folder taxonomy a "
            "human curated -- things a book's contents do not say."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer",
                          "description": f"1-{tools.MAX_LIMIT}, default "
                                         f"{tools.DEFAULT_LIMIT}."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "browse_folder",
        "description": (
            "List one Drive folder: its subfolders and PDFs. Use when the "
            "reader names a place rather than a subject."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_id": {"type": "string"},
                "name_contains": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["folder_id"],
        },
    },
    {
        "name": "recommend",
        "description": (
            "Your shortlist. Each pick carries EITHER book_id (indexed, "
            "preferred) or file_id (needs indexing first), plus `why` in one "
            "sentence and `passage` -- the line you quoted from look_inside. "
            "This does not add anything to the classroom; the reader chooses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "book_id": {"type": "integer"},
                            "file_id": {"type": "string"},
                            "why": {"type": "string"},
                            "passage": {"type": "string"},
                        },
                        "required": ["why"],
                    },
                },
            },
            "required": ["picks"],
        },
    },
]

_NEEDS_LIVE_DRIVE = frozenset({"browse_folder"})


def available_tools() -> list:
    """browse_folder needs live Drive credentials; the rest do not.

    find_books and look_inside read Postgres only, so a server with no Drive
    connection still has a working librarian over everything already indexed --
    which after a full ingest is the common case, not the degraded one.
    """
    if drive_client.credentials_status()["ok"]:
        return TOOL_SCHEMAS
    return [t for t in TOOL_SCHEMAS if t["name"] not in _NEEDS_LIVE_DRIVE]


def run(brief, conn, voyage=None, *, count=DEFAULT_COUNT, classroom_ids=(),
        client=None, max_iterations=None):
    """Validate eagerly, then return the generator.

    Split for the same reason the browse agent split it: a raise inside a
    generator surfaces mid-SSE, long after the request was accepted with a 200
    and headers the route can no longer take back.
    """
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError(f"count must be an integer, got {type(count).__name__}")
    if not 1 <= count <= MAX_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}, got {count}")
    return _run(brief, conn, voyage, count=count, classroom_ids=list(classroom_ids),
                client=client, max_iterations=max_iterations or MAX_ITERATIONS)


def _summarize(name, result):
    """The one-line trail the page renders. Shaped per tool, never the raw
    result -- a passage in the trail would be unreadable and is already on the
    card."""
    if name == "find_books":
        books = result.get("books", [])
        return {"returned": result.get("returned", 0),
                "nearest": result.get("nearest"),
                "thin": result.get("thin", False),
                # The ids, so the page can count DISTINCT books across a whole
                # run. Every search returns k of something, so summing
                # `returned` over twelve searches counts the same book up to
                # twelve times -- "84 books" is only true if you can dedupe,
                # and only the caller sees all the searches at once.
                "book_ids": [b["book_id"] for b in books if b.get("book_id")],
                "already_on_shelf": sum(
                    1 for b in books if b.get("in_classroom")
                )}
    if name == "look_inside":
        return {"book": result.get("book"), "passages": result.get("returned", 0),
                "nearest": result.get("nearest")}
    if name == "search_drive":
        return {"returned": result.get("returned", 0),
                "already_indexed": result.get("already_indexed", 0)}
    if name == "browse_folder":
        return {"folder": result.get("folder"),
                "subfolders": len(result.get("subfolders", [])),
                "returned": result.get("returned", 0)}
    if name == "recommend":
        return {"picks": len(result.get("recommendations", []))}
    return {}


def _run(brief, conn, voyage, *, count, classroom_ids, client, max_iterations):
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic()

    # What this run actually saw, so recommend can reject a pick it did not.
    seen_books, seen_files = {}, {}
    recommendations = []
    shelf = ""
    if classroom_ids:
        shelf = (f"\n\nThis classroom already holds {len(classroom_ids)} book(s). "
                 f"Do not recommend those again; find what is missing.")

    messages = [{"role": "user", "content": brief}]
    tool_schemas = available_tools()
    # Built once. It was being rebuilt every turn, which is harmless for a
    # string but would defeat the cache breakpoint sitting on it.
    system = prompt_cache.cacheable_system(system_prompt(count, shelf))

    # Summed across turns and reported on 'done', the way the tutor already
    # reports its own. This is the loop that costs money -- thirty-odd turns,
    # each re-sending the conversation -- and until now the only way to know
    # what a run cost was to read the bill afterwards and guess which run it
    # was. The two cache counters are the ones worth watching: reads should
    # dominate by the third turn, and if they ever fall to zero the caching
    # above has silently stopped working.
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }

    for iteration in range(1, max_iterations + 1):
        prompt_cache.move_breakpoint(messages)
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system,
            tools=tool_schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # getattr throughout: the scripted client the tests drive this with has
        # no usage object, and an absent counter is zero rather than a crash.
        turn_usage = getattr(response, "usage", None)
        if turn_usage is not None:
            for key in usage:
                usage[key] += getattr(turn_usage, key, 0) or 0

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            yield {"type": "answer", "text": answer}
            yield {"type": "done", "recommendations": recommendations,
                   "iterations": iteration, "usage": usage}
            return

        note = "".join(b.text for b in response.content if b.type == "text").strip()
        if note:
            yield {"type": "thinking", "text": note}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            yield {"type": "tool", "name": block.name, "input": block.input}
            try:
                result = _execute(block.name, block.input, conn, voyage,
                                  seen_books, seen_files, classroom_ids)
            except Exception as e:  # noqa: BLE001 -- the model can recover from a
                # tool error if it is told; killing the run gives it no chance.
                result = {"error": f"{type(e).__name__}: {e}"}
                yield {"type": "tool_error", "name": block.name, "message": str(e)}
            else:
                yield {"type": "results", "name": block.name,
                       "summary": _summarize(block.name, result)}
                if block.name == "recommend":
                    recommendations = result["recommendations"]
                    yield {"type": "recommendations", "books": recommendations}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "answer", "text": (
        f"I stopped after {max_iterations} rounds without settling on a "
        f"shortlist. Anything I found is listed below."
    )}
    yield {"type": "done", "recommendations": recommendations,
           "iterations": max_iterations, "exhausted": True, "usage": usage}


def _remember_files(seen_files, items):
    """A Drive file the model has actually been shown, keyed by file_id.

    An indexed Drive hit is remembered as a BOOK too: search_drive reports
    `indexed` and `book_id` for files already in the library, and a reader who
    found something that way should get the free, ready-now recommendation
    rather than an offer to index what is already indexed.
    """
    for f in items:
        seen_files[f["file_id"]] = f
        if f.get("indexed") and f.get("book_id") is not None:
            seen_files[f["file_id"]] = {**f, "kind": "indexed"}


def _execute(name, inp, conn, voyage, seen_books, seen_files, classroom_ids):
    if name == "find_books":
        out = tools.find_books(conn, inp.get("topic", ""), inp.get("k", tools.FIND_K),
                               voyage=voyage, classroom_ids=classroom_ids)
        for b in out["books"]:
            seen_books[b["book_id"]] = b
        return out

    if name == "look_inside":
        out = tools.look_inside(conn, int(inp["book_id"]), inp.get("question", ""),
                                voyage=voyage)
        # Seeing a book's insides counts as having seen it: the model can reach
        # look_inside from a find_books hit, and a pick it verified must not
        # then be rejected as unknown.
        seen_books.setdefault(int(inp["book_id"]),
                              {"book_id": int(inp["book_id"]),
                               "title": out.get("book")})
        return out

    # The two Drive tools name their file list differently -- `matches` for a
    # search, `pdfs` for a folder listing -- and both shapes are the browse
    # agent's, kept verbatim rather than harmonised, because the page's
    # renderers already read them.
    if name == "search_drive":
        out = tools.search_drive(conn, inp.get("query", ""),
                                 inp.get("limit", tools.DEFAULT_LIMIT),
                                 voyage=voyage)
        _remember_files(seen_files, out.get("matches", []))
        return out

    if name == "browse_folder":
        out = tools.browse_folder(conn, inp["folder_id"], inp.get("name_contains"),
                                  inp.get("limit", tools.DEFAULT_LIMIT))
        _remember_files(seen_files, out.get("pdfs", []))
        return out

    if name == "recommend":
        return tools.recommend(conn, inp.get("picks", []), seen_books, seen_files)

    return {"error": f"unknown tool: {name}"}


__all__ = ["MAX_ITERATIONS", "MODEL", "TOOL_SCHEMAS", "available_tools",
           "run", "system_prompt", "DEFAULT_COUNT", "MAX_COUNT", "config"]
