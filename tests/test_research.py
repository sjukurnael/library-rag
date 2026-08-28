"""
retrieval/loop.py + retrieval/tools.py: the tool loop, citation numbering
across searches, the search budget, and the iteration leash.

The Anthropic client is a scripted fake -- each test states exactly what the
model "decides" so the assertions are about our loop, not about the model's
judgement. Retrieval is faked too, so these run with no Postgres and no network.
"""
import json
import types

import pytest

from library_rag.retrieval import research, tools

# ------------------------------------------------------------- fakes --

class _Block:
    """Mimics an SDK content block: .type plus whichever fields that type has."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _text(s):
    return _Block(type="text", text=s)


def _tool(name, tid, **inp):
    return _Block(type="tool_use", name=name, id=tid, input=inp)


class FakeResponse:
    def __init__(self, blocks):
        self.content = blocks
        self.stop_reason = (
            "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        )


class ScriptedClient:
    """Returns the next scripted response per call, and records what it was sent
    so tests can assert on what the loop fed back to the model."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.seen.append(kw["messages"])
        if not self.script:
            return FakeResponse([_text("ran out of script")])
        return FakeResponse(self.script.pop(0))

    def last_tool_results(self):
        """The tool_result payloads handed back on the most recent call."""
        out = []
        for msg in self.seen[-1]:
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(json.loads(block["content"]))
        return out


class FakeConn:
    """Only list_books touches conn directly; search goes through db.search,
    which tests monkeypatch."""

    def __init__(self, books=()):
        self._books = books

    def execute(self, *_a, **_k):
        return types.SimpleNamespace(fetchall=lambda: list(self._books))


def _row(chunk_id, ordinal=0, title="Book A", book_id=1, dist=0.4, trail="Ch 1",
         content="Body text about the covenant."):
    return {
        "chunk_id": chunk_id, "book_id": book_id, "ordinal": ordinal,
        "heading_trail": trail, "page_start": 10, "page_end": 11,
        "content": content, "token_count": 42, "title": title,
        "page_count": 100, "total_chunks": 50, "distance": dist,
    }


@pytest.fixture
def fake_search(monkeypatch):
    """Route db.search to a queue of canned result sets."""
    queued = []

    def _search(_conn, _vec, k, book_id=None, **kw):
        return queued.pop(0) if queued else []

    monkeypatch.setattr(tools.db, "search", _search)
    monkeypatch.setattr(
        tools.embed_mod, "embed_query", lambda text, client: [0.0] * 8
    )
    return queued


# The classroom these tests run inside. Book 1 is what _row() attributes its
# passages to, so a scope containing it is what lets the canned results through
# -- an empty classroom short-circuits before db.search is ever consulted.
CLASSROOM = [1, 2]


def _run(script, conn=None, queued_note=None, book_ids=None, **kw):
    return list(
        research.run("a question", conn or FakeConn(), object(),
                     CLASSROOM if book_ids is None else book_ids,
                     client=ScriptedClient(script), **kw)
    )


# ------------------------------------------------------------- tests --

def test_one_search_then_answer(fake_search):
    fake_search.append([_row(101), _row(102, ordinal=1)])
    events = _run([
        [_text("Looking this up."), _tool("search_library", "t1", query="covenant")],
        [_text("The covenant is [1].")],
    ])

    kinds = [e["type"] for e in events]
    assert kinds == ["thinking", "search", "results", "answer", "done"]
    assert events[1]["query"] == "covenant"
    assert events[-1]["searches"] == 1
    assert events[-1]["iterations"] == 2
    assert [s["n"] for s in events[-1]["sources"]] == [1, 2]


def test_citation_numbers_are_stable_and_deduped_across_searches(fake_search):
    # Second search re-surfaces chunk 101 and adds 103.
    fake_search.append([_row(101), _row(102, ordinal=1)])
    fake_search.append([_row(101), _row(103, ordinal=2)])
    events = _run([
        [_tool("search_library", "t1", query="first")],
        [_tool("search_library", "t2", query="second")],
        [_text("done [1][3]")],
    ])

    sources = events[-1]["sources"]
    assert [s["chunk_id"] for s in sources] == [101, 102, 103], "duplicate stored twice"
    assert [s["n"] for s in sources] == [1, 2, 3]

    second = [e for e in events if e["type"] == "results"][1]
    assert [h["n"] for h in second["hits"]] == [1, 3], (
        "a re-surfaced chunk must keep its original citation number"
    )


def test_weak_results_tell_the_model_to_rephrase(fake_search):
    fake_search.append([_row(101, dist=0.82)])          # above RELEVANCE_CUTOFF
    client = ScriptedClient([
        [_tool("search_library", "t1", query="badly phrased")],
        [_text("nothing here")],
    ])
    list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    payload = client.last_tool_results()[0]
    assert payload["strong_matches"] == 0
    assert "vocabulary the books would use" in payload["stop_check"]
    assert payload["passages"][0]["weak"] is True


def test_stop_check_reports_the_running_search_count(fake_search):
    fake_search.append([_row(101)])
    fake_search.append([_row(102, ordinal=1)])
    client = ScriptedClient([
        [_tool("search_library", "t1", query="one")],
        [_tool("search_library", "t2", query="two")],
        [_text("ok")],
    ])
    list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    # The budget line is re-sent every turn, with a rising count -- that is what
    # actually holds effort proportionate; the same instruction placed in the
    # system prompt changed nothing, because a system prompt is read once.
    checks = [r["stop_check"] for r in client.last_tool_results()]
    assert len(checks) == 2, "earlier tool results should still be in the transcript"
    assert "made 1 search(es)" in checks[0]
    assert "made 2 search(es)" in checks[1]
    assert all("ANSWER NOW" in c for c in checks)


def test_search_budget_is_enforced(fake_search, monkeypatch):
    monkeypatch.setattr(tools, "MAX_SEARCHES", 2)
    for _ in range(4):
        fake_search.append([_row(101)])
    client = ScriptedClient([
        [_tool("search_library", f"t{i}", query=f"q{i}")] for i in range(4)
    ] + [[_text("fine")]])
    events = list(research.run("q", FakeConn(), object(), CLASSROOM, client=client,
                               max_iterations=6))

    assert events[-1]["searches"] == 2, "budget exceeded"
    assert any("budget exhausted" in json.dumps(r)
               for r in client.last_tool_results()), "model was not told it ran out"


def test_iteration_leash_still_returns_an_answer_and_sources(fake_search):
    for _ in range(10):
        fake_search.append([_row(101)])
    events = _run(
        [[_tool("search_library", f"t{i}", query="loop")] for i in range(10)],
        max_iterations=3,
    )

    assert events[-1]["exhausted"] is True
    assert events[-1]["iterations"] == 3
    assert events[-2]["type"] == "answer", "must still emit an answer, not just stop"
    assert events[-1]["sources"], "passages found before giving up must be returned"


def test_list_books_is_offered_and_reported(fake_search):
    # 5-tuples: list_books now also reports whether each book is ready to
    # search, so a book still being ingested reads as arriving rather than as
    # missing from the shelf.
    conn = FakeConn(books=[(1, "Book A", 100, 50, True),
                           (2, "Book B", 60, 30, True)])
    client = ScriptedClient([
        [_tool("list_books", "t1")],
        [_text("Two books.")],
    ])
    events = list(research.run("q", conn, object(), CLASSROOM, client=client))

    assert events[0] == {"type": "search", "tool": "list_books", "query": None}
    payload = client.last_tool_results()[0]
    assert [b["book_id"] for b in payload["books"]] == [1, 2]
    assert events[-1]["searches"] == 0, "list_books must not spend search budget"


def test_unknown_tool_is_rejected_without_searching(fake_search):
    client = ScriptedClient([
        [_tool("hack_the_planet", "t1")],
        [_text("ok")],
    ])
    events = list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    assert events[-1]["searches"] == 0
    assert client.last_tool_results()[0] == {"error": "unknown tool: hack_the_planet"}
    assert "results" not in {e["type"] for e in events}


def test_book_id_is_passed_through_to_retrieval(monkeypatch):
    seen = {}
    seen_scope = []

    def _search(_conn, _vec, k, book_id=None, **kw):
        seen["k"], seen["book_id"] = k, book_id
        seen["query_text"] = kw.get("query_text")
        seen_scope.append(kw.get("book_ids"))
        return [_row(101)]

    monkeypatch.setattr(tools.db, "search", _search)
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)
    # Book 2 is on the shelf. Narrowing to a book NOT on it is refused before
    # any query is issued -- see the containment test below.
    events = _run([
        [_tool("search_library", "t1", query="scoped", k=3, book_id=2)],
        [_text("ok")],
    ])

    assert seen == {"k": 3, "book_id": 2, "query_text": "scoped"}, (
        "the raw query text must reach retrieval, not just its embedding -- "
        "the lexical leg of the hybrid search has nothing to match without it"
    )
    assert seen_scope == [CLASSROOM], (
        "every search must carry the classroom, not only the narrowed book"
    )
    assert events[0]["book_id"] == 2


def test_k_is_clamped_to_the_allowed_range(monkeypatch):
    seen = []
    monkeypatch.setattr(
        tools.db, "search",
        lambda _c, _v, k, book_id=None, **kw: (seen.append(k), [_row(101)])[1],
    )
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)
    _run([
        [_tool("search_library", "t1", query="huge", k=999)],
        [_tool("search_library", "t2", query="zero", k=0)],
        [_text("ok")],
    ])
    assert seen == [tools.MAX_K, tools.DEFAULT_K]


def test_missing_api_key_is_reported_only_when_building_a_real_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        list(research.run("q", FakeConn(), object(), CLASSROOM))


# ------------------------------------------- detached runs over HTTP --
# The chat page can navigate away mid-run and come back: POST only starts a
# run, events accumulate in Postgres, and the events route replays from any
# offset. Postgres and not memory because a run outlives both the request that
# started it and the container that served it -- see migrations/0009. These
# tests fake research.run itself; the loop's behaviour is the rest of this
# file's business.
#
# They take the real `conn` fixture rather than stubbing db.get_conn, which is
# the point: the behaviour under test IS the persistence.

from fastapi.testclient import TestClient  # noqa: E402

from library_rag import db as db_mod  # noqa: E402
from library_rag.web import api  # noqa: E402

EVENTS = [
    {"type": "search", "tool": "search_library", "query": "q1", "book_id": None, "k": 8},
    {"type": "answer", "text": "It is written [1]."},
    {
        "type": "done",
        "sources": [],
        "searches": 1,
        "iterations": 1,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 900,
            "output_tokens": 120,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 0,
        },
    },
]


@pytest.fixture
def chat(conn, monkeypatch):
    monkeypatch.setattr(api.config, "VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(api.embed_mod, "build_client", lambda: object())
    monkeypatch.setattr(api.research, "run",
                        lambda q, c, voyage, book_ids: iter(EVENTS))
    return TestClient(api.app)


def _frames(text):
    return [json.loads(p[len("data: "):]) for p in text.split("\n\n") if p.startswith("data: ")]


def _start(chat, question="q", classroom_id=None):
    """Every tutor run belongs to a classroom, so the durability tests make one.

    It is a real row rather than a stub id: research_start refuses an unknown
    classroom with a 404, which is the behaviour that keeps a question from
    being asked against a shelf that does not exist.
    """
    if classroom_id is None:
        with db_mod.get_conn() as c:
            classroom_id = db_mod.create_classroom(c, None, "Test classroom")
    return chat.post("/api/research",
                     json={"question": question,
                           "classroom_id": classroom_id}).json()["run_id"]


def test_a_run_survives_its_starter_and_replays_in_full(chat):
    """TestClient executes background tasks before returning, so by the time
    POST answers, the whole run is on disk -- exactly the state a returning
    page finds after navigating away for the run's duration."""
    run_id = _start(chat)

    r = chat.get(f"/api/research/{run_id}/events")
    assert r.status_code == 200
    assert _frames(r.text) == EVENTS
    # And again: attaching consumes nothing, so a second visitor gets it all.
    assert _frames(chat.get(f"/api/research/{run_id}/events").text) == EVENTS


def test_after_skips_what_the_page_already_rendered(chat):
    run_id = _start(chat)
    assert _frames(chat.get(f"/api/research/{run_id}/events?after=2").text) == EVENTS[2:]


def test_an_unknown_run_is_a_helpful_404(chat):
    r = chat.get("/api/research/deadbeef/events")
    assert r.status_code == 404
    assert "No run with that id" in r.json()["detail"]


def test_a_finished_run_is_a_row_that_outlives_the_process(chat, conn):
    """The reason the table exists. A run that has ended leaves a queryable
    record of what it cost and how it ended -- not just frames that were
    rendered once and discarded."""
    run_id = _start(chat, "who is the Spirit of holiness?")

    row = db_mod.fetch_research_run(conn, run_id)
    assert row["status"] == "done"
    assert row["question"] == "who is the Spirit of holiness?"
    assert row["answer"] == "It is written [1]."
    assert row["stop_reason"] == "end_turn"
    assert (row["iterations"], row["searches"]) == (1, 1)
    # Token counters, which is how "did prompt caching help" becomes answerable.
    assert row["input_tokens"] == 900
    assert row["cache_read_tokens"] == 700
    assert row["finished_at"] is not None


def test_a_crashing_run_ends_as_an_error_event_not_a_wedge(chat, conn, monkeypatch):
    def boom(q, c, voyage, book_ids):
        raise RuntimeError("model fell over")
        yield  # unreachable -- its presence makes this a generator

    monkeypatch.setattr(api.research, "run", boom)
    run_id = _start(chat)

    frames = _frames(chat.get(f"/api/research/{run_id}/events").text)
    assert frames[-1]["type"] == "error"
    assert "model fell over" in frames[-1]["message"]
    # And the failure is countable afterwards, which is the half that used to
    # be destroyed by the same event that caused it.
    row = db_mod.fetch_research_run(conn, run_id)
    assert row["status"] == "failed"
    assert "model fell over" in row["error"]


def test_the_stream_tails_a_live_run_lazily(conn):
    """No threads needed: the frame generator polls on demand, so writing to
    the table between next() calls is exactly a run that progressed while the
    watcher was mid-read."""
    db_mod.create_research_run(conn, "live", "q")
    db_mod.append_research_event(conn, "live", EVENTS[0])

    gen = api._research_event_frames("live", 0)
    assert json.loads(next(gen)[len("data: "):-2]) == EVENTS[0]

    db_mod.append_research_event(conn, "live", EVENTS[2])
    db_mod.finish_research_run(conn, "live", status="done")
    assert json.loads(next(gen)[len("data: "):-2]) == EVENTS[2]
    with pytest.raises(StopIteration):
        next(gen)


def test_a_run_whose_worker_died_ends_instead_of_hanging_forever(conn):
    """The failure that only exists once the reader can outlive the writer.

    A run killed by a redeploy leaves its row at 'running' and stops
    heartbeating. Without the staleness check every watcher would poll until
    the connection dropped; with it, the stream ends and the row records what
    happened.
    """
    db_mod.create_research_run(conn, "orphan", "q")
    db_mod.append_research_event(conn, "orphan", EVENTS[0])
    conn.execute(
        "UPDATE research_runs SET heartbeat_at = now() - interval '1 hour' "
        "WHERE run_id = 'orphan'"
    )
    conn.commit()

    frames = list(api._research_event_frames("orphan", 0))

    assert frames[0].startswith("data: ")
    last = json.loads(frames[-1][len("data: "):-2])
    assert last["type"] == "error"
    assert "stopped reporting" in last["message"]
    # No 'done' frame, so the page's missing-terminal check reports a failure.
    assert not any('"done"' in f for f in frames)
    assert db_mod.fetch_research_run(conn, "orphan")["status"] == "interrupted"


def test_a_late_finish_beats_the_reaper(conn):
    """mark_research_interrupted is guarded on status='running', so a worker
    that finished in the gap between the staleness check and the update keeps
    the result it earned."""
    db_mod.create_research_run(conn, "racer", "q")
    db_mod.finish_research_run(conn, "racer", status="done", answer="done in time")
    db_mod.mark_research_interrupted(conn, "racer")

    row = db_mod.fetch_research_run(conn, "racer")
    assert row["status"] == "done"
    assert row["answer"] == "done in time"


# --------------------------------------------------- classroom containment --
# The tutor may not read outside its classroom. These assert that as a property
# of the plumbing -- what db.search was actually called with, and what JSON the
# model was actually handed -- rather than as something the prompt asks for.

def test_narrowing_to_a_book_outside_the_classroom_never_reaches_retrieval(monkeypatch):
    """Refused by identity, and refused BEFORE a query is issued.

    Letting it through and relying on the scope to return nothing would be
    almost as safe and much worse to read: an empty result says "nothing in that
    book" where the truth is "that book is not yours".
    """
    called = []
    monkeypatch.setattr(tools.db, "search",
                        lambda *a, **kw: called.append(kw) or [_row(101)])
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)

    client = ScriptedClient([
        [_tool("search_library", "t1", query="anything", book_id=999)],
        [_text("ok")],
    ])
    list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    assert called == [], "a query was issued for a book outside the classroom"
    payload = client.last_tool_results()[0]
    assert "not in this classroom" in payload["error"]


def test_an_empty_classroom_answers_without_searching(fake_search):
    """An empty shelf is a real state a reader can create, not a crash."""
    client = ScriptedClient([
        [_tool("search_library", "t1", query="anything")],
        [_text("This classroom has no books yet.")],
    ])
    events = list(research.run("q", FakeConn(), object(), [], client=client))

    payload = client.last_tool_results()[0]
    assert "no books yet" in payload["error"]
    assert events[-1]["type"] == "done"
    assert events[-1]["searches"] == 0


def test_suggested_books_carry_no_passage_text(monkeypatch):
    """The structural half of "never answers from outside the classroom".

    suggest_books reads book_topic_vectors and drive_files; it must never reach
    `chunks`. Asserted on the JSON actually handed to the model, because that is
    the only surface where a leak would matter -- a passage the model cannot see
    cannot be quoted, however the prompt is worded.
    """
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)
    monkeypatch.setattr(tools.db, "search_book_profiles", lambda *a, **kw: [
        {"book_id": 41, "title": "Off-Shelf Book", "page_count": 300,
         "best_label": "Ch 3: The Atonement", "matched_topics": 4,
         "distance": 0.31, "best_members": 12},
        {"book_id": 1, "title": "Already Here", "page_count": 100,
         "best_label": "Ch 1", "matched_topics": 2,
         "distance": 0.30, "best_members": 5},
    ])
    monkeypatch.setattr(tools.db, "search",
                        lambda *a, **kw: pytest.fail("suggest_books touched chunks"))

    client = ScriptedClient([
        [_tool("suggest_books", "t1", topic="the atonement")],
        [_text("The shelf does not cover this. You could add Off-Shelf Book.")],
    ])
    events = list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    payload = client.last_tool_results()[0]
    books = payload["books"]
    assert [b["book_id"] for b in books] == [41], (
        "a book already on the shelf must not be suggested back"
    )
    forbidden = {"content", "text", "passage", "chunk_id", "passages"}
    for b in books:
        assert not (forbidden & set(b)), f"a suggestion leaked passage text: {b}"

    # Suggestions are never citable: no chunk id means nothing for [n] to reach.
    done = events[-1]
    assert done["sources"] == []
    assert [b["book_id"] for b in done["suggestions"]] == [41]
    assert any(e["type"] == "suggestions" for e in events)


def test_the_suggestion_budget_is_enforced(monkeypatch):
    """A cheap escape hatch becomes a second search tool if it is unlimited."""
    monkeypatch.setattr(tools, "MAX_SUGGESTS", 1)
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)
    monkeypatch.setattr(tools.db, "search_book_profiles", lambda *a, **kw: [])

    client = ScriptedClient([
        [_tool("suggest_books", "t1", topic="one")],
        [_tool("suggest_books", "t2", topic="two")],
        [_text("ok")],
    ])
    list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    assert "budget exhausted" in json.dumps(client.last_tool_results())


def test_a_raising_tool_yields_tool_error_and_the_run_continues(monkeypatch):
    """Previously a raising tool killed the generator mid-stream, and the page
    blamed the transport: "the connection ended before the run finished"."""
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)

    def _boom(*_a, **_k):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(tools.db, "search", _boom)
    client = ScriptedClient([
        [_tool("search_library", "t1", query="anything")],
        [_text("I could not search just then.")],
    ])
    events = list(research.run("q", FakeConn(), object(), CLASSROOM, client=client))

    kinds = [e["type"] for e in events]
    assert "tool_error" in kinds
    assert kinds[-1] == "done", "the run must finish rather than die mid-stream"
    assert "retrieval exploded" in json.dumps(client.last_tool_results())
