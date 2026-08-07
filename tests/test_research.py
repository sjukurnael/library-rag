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


def _run(script, conn=None, queued_note=None, **kw):
    return list(
        research.run("a question", conn or FakeConn(), object(),
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
    list(research.run("q", FakeConn(), object(), client=client))

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
    list(research.run("q", FakeConn(), object(), client=client))

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
    events = list(research.run("q", FakeConn(), object(), client=client,
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
    conn = FakeConn(books=[(1, "Book A", 100, 50), (2, "Book B", 60, 30)])
    client = ScriptedClient([
        [_tool("list_books", "t1")],
        [_text("Two books.")],
    ])
    events = list(research.run("q", conn, object(), client=client))

    assert events[0] == {"type": "search", "tool": "list_books", "query": None}
    payload = client.last_tool_results()[0]
    assert [b["book_id"] for b in payload["books"]] == [1, 2]
    assert events[-1]["searches"] == 0, "list_books must not spend search budget"


def test_unknown_tool_is_rejected_without_searching(fake_search):
    client = ScriptedClient([
        [_tool("hack_the_planet", "t1")],
        [_text("ok")],
    ])
    events = list(research.run("q", FakeConn(), object(), client=client))

    assert events[-1]["searches"] == 0
    assert client.last_tool_results()[0] == {"error": "unknown tool: hack_the_planet"}
    assert "results" not in {e["type"] for e in events}


def test_book_id_is_passed_through_to_retrieval(monkeypatch):
    seen = {}

    def _search(_conn, _vec, k, book_id=None, **kw):
        seen["k"], seen["book_id"] = k, book_id
        seen["query_text"] = kw.get("query_text")
        return [_row(101)]

    monkeypatch.setattr(tools.db, "search", _search)
    monkeypatch.setattr(tools.embed_mod, "embed_query", lambda t, c: [0.0] * 8)
    events = _run([
        [_tool("search_library", "t1", query="scoped", k=3, book_id=7)],
        [_text("ok")],
    ])

    assert seen == {"k": 3, "book_id": 7, "query_text": "scoped"}, (
        "the raw query text must reach retrieval, not just its embedding -- "
        "the lexical leg of the hybrid search has nothing to match without it"
    )
    assert events[0]["book_id"] == 7


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
        list(research.run("q", FakeConn(), object()))
