"""
The browsing agent's tool-use loop: system prompt, tool schemas, and a run()
that drives the conversation until the model stops calling tools.

This is the "walk into a bookstore" agent. The Drive behind this library holds
~57,500 PDFs across ~8,200 folders; the indexed library holds a few dozen.
Deciding what to read next used to mean scrolling Drive by hand.

Phase 0 used this same loop to answer a one-time infrastructure question -- pick
a pilot folder under 5 GB and $5 -- so its tools returned size statistics and 8
sample filenames. Browsing inverts that: titles are the payload. estimate_pipeline
survives unchanged because "what would ingesting this cost" is still a real
question, just no longer the only one.

run() is a generator of events, the same shape retrieval/loop.py uses, so the web
route can stream the trail and the CLI can print it without a second
implementation. Watching which searches it ran is most of the value -- a list of
book titles with no visible provenance is indistinguishable from a hallucination.
"""
import json
import os

from anthropic import Anthropic

from library_rag import config
from library_rag.drive import client as drive_client
from library_rag.exploration import assumptions, tools

ROOT_FOLDER_ID = config.DRIVE_ROOT_FOLDER_ID
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
MAX_ITERATIONS = 12

TOOL_SCHEMAS = [
    {
        "name": "search_drive",
        "description": (
            "Search every book in the drive by MEANING and by words together. "
            "Ranks ~57,000 titles and their folder paths -- a search for 'the "
            "end times' surfaces Revelation and eschatology even though neither "
            "phrase appears in the query. Titles and folders only; it does NOT "
            "read inside the books. Instant. Prefer this over browsing folder by "
            "folder. Returns each match with its title, size, folder path, Drive "
            "URL, and whether it is already in the reader's library."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Describe the SUBJECT, in whatever words are natural -- "
                        "'the end times', 'how to pray', 'Paul writing from "
                        "prison'. You do not need to guess words from the title. "
                        "An author's surname works too, and is the reliable way "
                        "to find a specific known book."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max results (1-{tools.MAX_LIMIT}, default "
                                   f"{tools.DEFAULT_LIMIT}).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "browse_folder",
        "description": (
            "List ONE level of a Drive folder: its subfolders, and the PDFs "
            "directly inside it by title. Never recursive. Use this to explore a "
            "shelf when you do not yet know what words to search for. Folders "
            "here are large -- always check `truncated` and `total_pdfs` before "
            "concluding what a folder contains."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_id": {"type": "string", "description": "Drive folder ID."},
                "name_contains": {
                    "type": "string",
                    "description": "Optional filter on title within this folder.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max PDFs to list (1-{tools.MAX_LIMIT}, "
                                   f"default {tools.DEFAULT_LIMIT}).",
                },
            },
            "required": ["folder_id"],
        },
    },
    {
        "name": "estimate_pipeline",
        "description": (
            "Deterministically estimate cost and time to ingest a candidate set "
            "of PDFs (download -> extract/OCR -> chunk -> embed -> store). All "
            "arithmetic happens here -- never compute costs yourself. Call this "
            "ONLY when the user asks about cost, time, or size; a reader "
            "choosing what to study does not need it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pdf_count": {"type": "integer"},
                "total_mb": {"type": "number"},
                "scanned_ratio": {
                    "type": "number",
                    "description": (
                        "Fraction (0.0-1.0) that is scanned images vs digital "
                        "text, inferred from size: median 50MB+ suggests scans, "
                        "under 10MB suggests digital-native text."
                    ),
                },
            },
            "required": ["pdf_count", "total_mb", "scanned_ratio"],
        },
    },
    {
        "name": "recommend",
        "description": (
            "Submit your final picks. Call this ONCE, after you have finished "
            "looking and BEFORE you write your closing summary. Every file_id "
            "must be one that search_drive or browse_folder actually returned to "
            "you in this conversation -- do not construct or guess ids. This is "
            "how the picks reach the user's screen as something they can add to "
            "their library."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_id": {"type": "string"},
                            "why": {
                                "type": "string",
                                "description": (
                                    "One sentence on why THIS book fits what the "
                                    "reader asked for. Concrete, not generic."
                                ),
                            },
                        },
                        "required": ["file_id", "why"],
                    },
                }
            },
            "required": ["picks"],
        },
    },
]

SYSTEM_PROMPT = f"""You help someone choose their next book from a large Google \
Drive library of Bible study guides, commentaries, doctrine books, dissertations \
and articles. Think of yourself as a librarian who knows the shelves: they tell \
you what they want to study, you go find the candidates.

The root folder is {ROOT_FOLDER_ID}. It holds ~57,500 PDFs across ~8,200 \
folders. Top level: Books (the main collection), Dissertations, Articles (mostly \
journal issues, so short pieces rather than book-length treatments), Transcripts. \
Expect .DS_Store and index.html throughout; the tools already filter those out.

HOW TO LOOK

Start with search_drive. It ranks titles and folder paths by MEANING as well as \
by words, so describe the subject rather than guessing filename words -- "the end \
times" reaches Revelation and eschatology on its own. Run two or three searches \
from different angles (the subject, a narrower aspect, an author you expect) \
rather than one long one; each gives you a different slice of a 57,000-book \
collection.

Every search returns the top `returned` matches ranked out of the whole \
collection (`ranked_over`), so a full-looking result list is NORMAL and is not a \
sign you missed anything -- there is nothing better further down, only worse. Do \
not re-search to cover a shortfall that a full list does not indicate. The one \
number worth reacting to is `word_and_meaning`, the titles that matched on words \
AND ranked near you by meaning: a healthy topical search scores at least 2, and \
0 means the collection has nothing on this and the list you got is just nearest \
neighbours. React to a 0 by rewording or by browsing a likely shelf -- not by \
trying another synonym of the same idea.

Use browse_folder when a search comes back thin and you need to see what is \
actually on a shelf, or when the reader names a topic area rather than a subject.

Stop when you have enough to choose well. Three or four searches is usually \
plenty. Exhaustiveness is not the goal -- a good short list is.

WHAT TO RECOMMEND

3 to 6 books. For each, say concretely why it fits what they asked, based on the \
title and what you can reasonably infer from it. If a title is ambiguous, say so \
rather than inventing a description of contents you have not read.

You can only see filenames and sizes. You have NOT read these books. Do not \
describe their arguments, chapters or conclusions as if you had.

Books already in the reader's library come back with `indexed: true`. Do NOT \
recommend those -- they already have them. If a search turns up an indexed book \
that is genuinely the best fit, mention it in your prose as something they \
already own, but keep it out of your picks.

Prefer variety across your picks: a study guide and a fuller treatment beats two \
near-identical study guides. Note when something looks like a scan (very large \
file, tens of MB) since those cost more to index and extract less cleanly.

If nothing good turns up, say so plainly and suggest what to search instead. \
Never pad the list to reach a number.

FINISHING

Call `recommend` with your picks, then write a short closing paragraph: what you \
searched, what the collection turned out to have, and which one you would start \
with and why. Plain prose, conversational, no headings, no bullet lists. Do not \
restate the request back to them.

Cost estimation via estimate_pipeline is available but off by default -- use it \
only if they ask about cost, time, or storage. Pilot caps of \
${assumptions.PILOT_MAX_COST_USD} / {assumptions.PILOT_MAX_GB} GB were a Phase 0 \
constraint and no longer apply to choosing a book to read."""


# Tools that call Google Drive live rather than reading the local mirror.
# search_drive is deliberately NOT here: it ranks the mirror, and only falls
# back to Drive when the mirror is empty.
_NEEDS_LIVE_DRIVE = frozenset({"browse_folder"})


def available_tools() -> list:
    """The tool schemas to offer this run.

    Drops the live-Drive tools when there are no Drive credentials -- which is
    the normal state of the deployed app, where indexing happens on a laptop and
    no `token.json` ships in the image (see README, "Deploying").

    Offering a tool that can only ever return an error is worse than not
    offering it: the model spends turns discovering that, and the trail fills
    with failures that look like the app is broken. search_drive and recommend
    remain, and search_drive is the one that does the real work anyway.
    """
    if drive_client.credentials_status()["ok"]:
        return TOOL_SCHEMAS
    return [t for t in TOOL_SCHEMAS if t["name"] not in _NEEDS_LIVE_DRIVE]


def _summarize(tool_name: str, result: dict) -> dict:
    """A compact, renderable summary of a tool result for the event stream."""
    if tool_name == "search_drive":
        return {
            "returned": result["returned"],
            "already_indexed": sum(1 for m in result["matches"] if m["indexed"]),
            # Both optional: the two branches of search_drive report different
            # things, because only one of them HAS a real ceiling (see there).
            **({"hit_limit": result["hit_limit"]} if "hit_limit" in result else {}),
            **({"word_and_meaning": result["word_and_meaning"]}
               if "word_and_meaning" in result else {}),
        }
    if tool_name == "browse_folder":
        return {
            "folder": result["folder"]["name"],
            "subfolders": len(result["subfolders"]),
            "returned": len(result["pdfs"]),
            "total_pdfs": result["total_pdfs"],
            "truncated": result["truncated"],
        }
    if tool_name == "estimate_pipeline":
        t = result["totals"]
        return {
            "api_path": t["api_path"]["cost_label"],
            "gpu_path": t["gpu_path"]["cost_label"],
        }
    if tool_name == "recommend":
        return {"picks": len(result["recommendations"])}
    return {}


def run(interest: str, conn, *, client=None, max_iterations: int = MAX_ITERATIONS):
    """Yield events: {"type": "tool"|"results"|"thinking"|"recommendations"
    |"answer"|"done", ...}.

    `client` is injectable for the same reason conn/voyage/download_file are
    everywhere else here: tests drive the loop with a scripted fake and no
    network.
    """
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        client = Anthropic()

    # Every Drive item this run has actually seen, file_id -> raw resource.
    # recommend() checks picks against it, which is what turns a hallucinated
    # file_id into a flagged pick instead of a dead Drive link on the user's
    # screen.
    seen = {}
    recommendations = []
    messages = [{"role": "user", "content": interest}]
    tool_schemas = available_tools()

    for iteration in range(1, max_iterations + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tool_schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            yield {"type": "answer", "text": answer}
            yield {
                "type": "done",
                "recommendations": recommendations,
                "iterations": iteration,
            }
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
                result = _execute(block.name, block.input, conn, seen)
            except Exception as e:  # noqa: BLE001 -- the model can recover from a
                # tool error if it is told; killing the run gives it no chance.
                result = {"error": f"{type(e).__name__}: {e}"}
                yield {"type": "tool_error", "name": block.name, "message": str(e)}
            else:
                yield {
                    "type": "results",
                    "name": block.name,
                    "summary": _summarize(block.name, result),
                }
                if block.name == "recommend":
                    recommendations = result["recommendations"]
                    yield {"type": "recommendations", "books": recommendations}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    yield {
        "type": "answer",
        "text": (
            f"Stopped after {max_iterations} rounds without settling on a "
            "shortlist. Anything found so far is listed below."
        ),
    }
    yield {
        "type": "done",
        "recommendations": recommendations,
        "iterations": max_iterations,
        "exhausted": True,
    }


def _execute(name: str, tool_input: dict, conn, seen: dict) -> dict:
    if name == "search_drive":
        result = tools.search_drive(
            conn, tool_input["query"], tool_input.get("limit", tools.DEFAULT_LIMIT)
        )
        _remember(seen, result["matches"])
        return result
    if name == "browse_folder":
        result = tools.browse_folder(
            conn,
            tool_input["folder_id"],
            name_contains=tool_input.get("name_contains"),
            limit=tool_input.get("limit", tools.DEFAULT_LIMIT),
        )
        _remember(seen, result["pdfs"])
        return result
    if name == "estimate_pipeline":
        return tools.estimate_pipeline(
            tool_input["pdf_count"],
            tool_input["total_mb"],
            tool_input["scanned_ratio"],
        )
    if name == "recommend":
        return tools.recommend(conn, tool_input.get("picks", []), seen)
    return {"error": f"unknown tool: {name}"}


def _remember(seen: dict, pdfs: list) -> None:
    """Record the Drive resources a tool returned, in the raw shape _as_pdf()
    consumes, so recommend() can re-derive a full card from a file_id alone."""
    for p in pdfs:
        size = p.get("size_mb")
        seen[p["file_id"]] = {
            "id": p["file_id"],
            "name": p["title"],
            "webViewLink": p["url"],
            "size": None if size is None else str(int(size * 1024 * 1024)),
        }
