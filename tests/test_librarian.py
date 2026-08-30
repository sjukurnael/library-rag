"""librarian/loop.py + librarian/tools.py: the shelf-building agent.

The Anthropic client is a scripted fake, so every assertion is about our loop
rather than the model's judgement. Retrieval is real Postgres with
lexical_vector embeddings, because the thing worth testing about find_books is
that it ranks the right book first -- a mocked ranker would only prove the
plumbing calls itself.
"""
import json
import types

from library_rag import db
from library_rag.librarian import loop, tools

from .conftest import lexical_vector, make_book, make_chunks, make_profile

# ------------------------------------------------------------- fakes --


class _Block:
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
        out = []
        for msg in self.seen[-1]:
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(json.loads(block["content"]))
        return out


class FakeVoyage:
    """embed_query is monkeypatched, so the client itself is only a token."""


def _shelf(conn):
    """Three books that differ in subject, embedded so ranking is meaningful."""
    atonement = make_book(conn, "aton", "The Cross of Christ")
    make_chunks(conn, atonement, ["atonement sacrifice propitiation the cross",
                                  "penal substitution and the wrath of God"])
    make_profile(conn, atonement, [("Ch 4: The Self-Substitution of God",
                                    "atonement sacrifice propitiation")])

    psalms = make_book(conn, "psalm", "The Editing of the Hebrew Psalter")
    make_chunks(conn, psalms, ["psalter editing arrangement of the five books"])
    make_profile(conn, psalms, [("Ch 1: Shape of the Psalter",
                                 "psalter editing arrangement")])
    return atonement, psalms


def _run(conn, script, brief="the atonement", monkeypatch=None, **kw):
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    return list(loop.run(brief, conn, FakeVoyage(),
                         client=ScriptedClient(script), **kw))


# ------------------------------------------------------------- tools --

def test_find_books_ranks_by_contents_not_filename(conn, monkeypatch):
    """The whole reason this agent replaced the browse agent."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    atonement, _ = _shelf(conn)

    out = tools.find_books(conn, "atonement sacrifice propitiation",
                           voyage=FakeVoyage())
    assert out["books"][0]["book_id"] == atonement
    assert out["books"][0]["kind"] == "indexed"
    assert out["books"][0]["covers"].startswith("Ch 4")


def test_find_books_flags_a_topic_the_library_does_not_hold(conn, monkeypatch):
    """`thin` is the honest-emptiness signal: a nearest-neighbour ranker always
    fills its page, so a count can never say whether the subject is here."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    _shelf(conn)

    covered = tools.find_books(conn, "atonement sacrifice propitiation",
                               voyage=FakeVoyage())
    absent = tools.find_books(conn, "formula one aerodynamics ground effect",
                              voyage=FakeVoyage())
    assert covered["thin"] is False
    assert absent["thin"] is True, "an absent subject must not read as covered"


def test_find_books_marks_what_is_already_on_the_shelf(conn, monkeypatch):
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    atonement, _ = _shelf(conn)

    out = tools.find_books(conn, "atonement", voyage=FakeVoyage(),
                           classroom_ids=[atonement])
    marked = {b["book_id"]: b["in_classroom"] for b in out["books"]}
    assert marked[atonement] is True


def test_the_search_summary_carries_the_book_ids_it_found(conn, monkeypatch):
    """The page counts DISTINCT books across a whole run, and cannot without
    ids: every search returns k of something, so summing `returned` over a
    dozen searches counts the same book a dozen times. `returned` restates the
    request; the ids are what let the count mean anything.

    The model never sees this -- it gets the raw tool result -- so the summary
    is free to carry what the UI needs and nothing the model would spend
    context on."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    atonement, _ = _shelf(conn)

    result = tools.find_books(conn, "atonement", voyage=FakeVoyage(),
                              classroom_ids=[atonement])
    summary = loop._summarize("find_books", result)

    assert summary["book_ids"] == [b["book_id"] for b in result["books"]]
    assert len(summary["book_ids"]) == summary["returned"]
    assert atonement in summary["book_ids"]
    assert summary["already_on_shelf"] == 1


def test_summaries_stay_small_enough_to_stream(conn, monkeypatch):
    """One `results` frame per tool call, 20-40 of them a run. A summary that
    quietly grew to hold passages would put a book through the event stream."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    book = make_book(conn, "inside", "A Book")
    make_chunks(conn, book, ["the atonement is the heart of the matter"] * 4)

    raw = tools.look_inside(conn, book, "atonement", voyage=FakeVoyage())
    summary = loop._summarize("look_inside", raw)

    # A count of passages, never their text. The key set is the contract; the
    # size check is what would catch someone adding a field that carries prose.
    # book_id joined the set when the run page needed to mark which candidates
    # were actually opened -- an integer, which is the point.
    assert set(summary) == {"book", "book_id", "passages", "nearest"}
    assert len(json.dumps(summary)) < 200
    assert len(json.dumps(raw)) > len(json.dumps(summary)) * 2, \
        "the summary should be much smaller than the result it summarises"


def test_look_inside_trims_the_passage(conn, monkeypatch):
    """Verifying thirty candidates must not cost thirty chapters of context."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    book = make_book(conn, "long", "A Long Book")
    make_chunks(conn, book, [" ".join(["atonement"] * 400)])

    out = tools.look_inside(conn, book, "atonement", voyage=FakeVoyage())
    assert out["book"] == "A Long Book"
    words = out["passages"][0]["text"].split()
    assert len(words) <= tools.PASSAGE_WORDS


# ------------------------------------------------------- recommend --

def test_recommend_rejects_a_book_no_search_returned(conn):
    """A recommendation the reader cannot trace is indistinguishable from a
    hallucination, so it is surfaced as unusable rather than dropped."""
    out = tools.recommend(conn, [{"book_id": 4242, "why": "trust me"}], {}, {})
    pick = out["recommendations"][0]
    assert pick["unknown"] is True
    assert "no search this run" in pick["error"]


def test_recommend_rejects_a_book_that_is_not_ready(conn):
    """A book mid-ingest has no chunks. Putting it on a shelf would produce a
    classroom that cannot answer from a book the reader believes is there."""
    arriving = make_book(conn, "arr", "Still Arriving", status="discovered")
    out = tools.recommend(conn, [{"book_id": arriving, "why": "looks good"}],
                          {arriving: {"title": "Still Arriving"}}, {})
    pick = out["recommendations"][0]
    assert pick["unknown"] is True
    assert "not ready" in pick["error"]


def test_recommend_carries_the_kind_so_the_two_populations_stay_distinct(conn):
    """An indexed book is free and answerable now; a Drive file needs minutes of
    ingest first. A card that cannot tell them apart produces a classroom that
    answers nothing for an hour."""
    book = make_book(conn, "ready", "Ready Book")
    seen_books = {book: {"title": "Ready Book", "covers": "Ch 1"}}
    seen_files = {"drive-1": {"file_id": "drive-1", "title": "Unindexed.pdf",
                              "size_mb": 4.0, "indexed": False}}

    out = tools.recommend(conn, [
        {"book_id": book, "why": "already here", "passage": "p.10: 'the cross'"},
        {"file_id": "drive-1", "why": "would need indexing"},
    ], seen_books, seen_files)

    kinds = [r["kind"] for r in out["recommendations"]]
    assert kinds == ["indexed", "drive"]
    assert out["recommendations"][0]["passage"].startswith("p.10")


def test_recommend_writes_nothing(conn):
    """The agent recommends; the reader's button acts. A shelf that assembles
    itself mid-stream is a list of titles with no provenance."""
    book = make_book(conn, "ready", "Ready Book")
    cid = db.create_classroom(conn, None, "Empty")
    tools.recommend(conn, [{"book_id": book, "why": "x"}],
                    {book: {"title": "Ready Book"}}, {})
    assert db.classroom_books(conn, cid) == []


# ------------------------------------------------------------- loop --

def test_the_loop_verifies_then_recommends(conn, monkeypatch):
    atonement, _ = _shelf(conn)
    events = _run(conn, [
        [_text("Looking for books on this."),
         _tool("find_books", "t1", topic="atonement sacrifice")],
        [_tool("look_inside", "t2", book_id=atonement, question="atonement")],
        [_tool("recommend", "t3", picks=[
            {"book_id": atonement, "why": "covers it directly",
             "passage": "p.10: atonement sacrifice"}])],
        [_text("One book, and here is why.")],
    ], monkeypatch=monkeypatch)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "thinking"
    assert "recommendations" in kinds
    assert kinds[-1] == "done"
    recs = events[-1]["recommendations"]
    assert [r["book_id"] for r in recs] == [atonement]
    assert recs[0]["kind"] == "indexed"


def test_look_inside_alone_is_enough_to_recommend(conn, monkeypatch):
    """A book verified but never returned by find_books must not be rejected as
    unknown -- the model can reach look_inside from a Drive hit."""
    atonement, _ = _shelf(conn)
    events = _run(conn, [
        [_tool("look_inside", "t1", book_id=atonement, question="atonement")],
        [_tool("recommend", "t2", picks=[{"book_id": atonement, "why": "read it"}])],
        [_text("done")],
    ], monkeypatch=monkeypatch)

    recs = events[-1]["recommendations"]
    assert recs[0].get("unknown") is not True


def test_a_raising_tool_does_not_kill_the_run(conn, monkeypatch):
    monkeypatch.setattr(tools, "find_books",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    events = _run(conn, [
        [_tool("find_books", "t1", topic="anything")],
        [_text("I could not search just then.")],
    ], monkeypatch=monkeypatch)

    kinds = [e["type"] for e in events]
    assert "tool_error" in kinds
    assert kinds[-1] == "done"


def test_count_is_validated_before_the_stream_opens(conn):
    """A raise inside a generator surfaces mid-SSE, long after the request was
    accepted with a 200 and headers the route can no longer take back."""
    for bad in (0, loop.MAX_COUNT + 1, "eight", True):
        try:
            loop.run("brief", conn, FakeVoyage(), count=bad)
        except ValueError:
            continue
        raise AssertionError(f"count={bad!r} was accepted")


def test_drive_tools_are_dropped_without_credentials_but_reading_is_not(monkeypatch):
    """find_books and look_inside read Postgres only, so a server with no Drive
    connection still has a working librarian over everything already indexed."""
    monkeypatch.setattr(loop.drive_client, "credentials_status", lambda: {"ok": False})
    names = [t["name"] for t in loop.available_tools()]
    assert "browse_folder" not in names
    assert {"find_books", "look_inside", "recommend"} <= set(names)


# ------------------------------------------------------- cost accounting --

class _Usage:
    def __init__(self, **kw):
        self.input_tokens = kw.get("input_tokens", 0)
        self.output_tokens = kw.get("output_tokens", 0)
        self.cache_read_input_tokens = kw.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens = kw.get("cache_creation_input_tokens", 0)


def test_a_run_reports_what_it_cost(conn, monkeypatch):
    """Summed across turns, because one shelf is thirty-odd API calls and the
    per-call figure answers nothing anyone asks. Without this the only way to
    price a run was to read the bill and guess which run it was."""
    _shelf(conn)

    class Billed(ScriptedClient):
        def _create(self, **kw):
            response = super()._create(**kw)
            response.usage = _Usage(input_tokens=10, output_tokens=5,
                                    cache_read_input_tokens=2000,
                                    cache_creation_input_tokens=100)
            return response

    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    events = list(loop.run("the atonement", conn, FakeVoyage(), client=Billed([
        [_tool("find_books", "t1", query="atonement")],
        [_text("Two books will do.")],
    ])))

    done = [e for e in events if e["type"] == "done"][-1]
    assert done["iterations"] == 2
    assert done["usage"] == {"input_tokens": 20, "output_tokens": 10,
                             "cache_read_input_tokens": 4000,
                             "cache_creation_input_tokens": 200}


def test_a_client_that_reports_no_usage_does_not_crash_the_run(conn, monkeypatch):
    """The scripted fakes above have no usage object at all, and neither did the
    real responses before caching was turned on. An absent counter is zero."""
    _shelf(conn)
    events = _run(conn, [[_text("Nothing to add.")]], monkeypatch=monkeypatch)

    done = [e for e in events if e["type"] == "done"][-1]
    assert done["usage"]["input_tokens"] == 0


def test_every_turn_after_the_first_reuses_the_cached_conversation(conn, monkeypatch):
    """The loop must hand the API exactly one cache breakpoint, on the newest
    tool_result. Two would pay for a cache nothing reads again; none would mean
    every turn re-buys the whole conversation at full price. Neither shows up as
    a failure anywhere else -- the answers are identical either way."""
    _shelf(conn)

    marks = []

    class Watching(ScriptedClient):
        def _create(self, **kw):
            found = [(i, j) for i, m in enumerate(kw["messages"])
                     if isinstance(m.get("content"), list)
                     for j, b in enumerate(m["content"])
                     if isinstance(b, dict) and "cache_control" in b]
            marks.append(found)
            assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}, (
                "the system prompt and tool schemas must stay cached"
            )
            return super()._create(**kw)

    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    list(loop.run("the atonement", conn, FakeVoyage(), client=Watching([
        [_tool("find_books", "t1", query="atonement")],
        [_tool("find_books", "t2", query="sacrifice")],
        [_text("Done.")],
    ])))

    assert marks[0] == [], "the opening brief is a bare string, nothing to cache"
    for turn, found in enumerate(marks[1:], start=1):
        assert len(found) == 1, f"turn {turn}: expected one breakpoint, got {found}"


# ------------------------------------------------------ the quoted page --

def test_a_quoted_passage_is_located_on_its_page(conn, monkeypatch):
    """The page comes from matching the words, not from the model reporting it.

    look_inside hands the model a whitespace-collapsed prefix of a chunk, so the
    quote that comes back will not equal the stored text byte for byte. Matching
    has to survive that or the page is silently never found."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["nothing to do with the subject",
                             "the   cross\n\nis   the place where wrath and mercy meet"])
    conn.execute("UPDATE chunks SET page_start = 86, page_end = 87 "
                 "WHERE book_id = %s AND content LIKE '%%wrath%%'", (book,))
    conn.commit()

    span = db.page_of_passage(conn, book, "the cross is the place where wrath and mercy")
    assert span == (86, 87)


def test_a_passage_that_is_not_in_the_book_gets_no_page(conn):
    """None, not a guess. A page number nobody can check is worse than none --
    the whole value of the citation is that a reader can turn to it."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["the cross is the place where wrath and mercy meet"])

    assert db.page_of_passage(conn, book, "a sentence this book never contained") is None
    assert db.page_of_passage(conn, book, "too short") is None, (
        "a two-word 'quote' would match half the library"
    )


def test_like_wildcards_in_a_quote_are_literal(conn):
    """An underscore in a passage is an underscore. Unescaped it means 'any
    character', which turns a failed match into a wrong one."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["the value of x_1 determines the whole series here"])

    assert db.page_of_passage(conn, book, "the value of x_1 determines the whole") is not None
    assert db.page_of_passage(conn, book, "the value of xA1 determines the whole") is None


def test_a_recommendation_carries_the_page_its_quote_came_from(conn, monkeypatch):
    """End to end through recommend, which is what the card renders."""
    book = make_book(conn, "b1", "The Cross of Christ")
    make_chunks(conn, book, ["penal substitution and the wrath of God poured out"])
    conn.execute("UPDATE chunks SET page_start = 142, page_end = 142 WHERE book_id = %s",
                 (book,))
    conn.commit()

    out = tools.recommend(
        conn,
        [{"book_id": book, "why": "central",
          "passage": "penal substitution and the wrath of God poured out"}],
        {book: {"title": "The Cross of Christ"}}, {},
    )
    pick = out["recommendations"][0]
    assert pick["passage_page"] == 142
    assert pick["passage_pages"] == "p.142"


def test_a_drive_pick_has_no_page_and_does_not_crash(conn):
    """A Drive book has no chunks to look in -- the lookup must not be attempted
    rather than returning something misleading."""
    out = tools.recommend(
        conn, [{"file_id": "f1", "why": "looks right"}],
        {}, {"f1": {"file_id": "f1", "title": "Not Indexed", "url": "https://drive/x"}},
    )
    pick = out["recommendations"][0]
    assert pick.get("passage_page") is None


def test_a_quote_stitched_with_an_ellipsis_still_finds_its_page(conn):
    """The failure that measuring a real run exposed. The librarian joins two
    fragments with "..." -- taking a needle off the front runs it straight into
    the ellipsis and matches nothing. Five of twelve recommendations lost their
    page to exactly this."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["the serpent is characterized in various ways by the "
                             "commentators and later Grigor says that the serpent was subtle"])
    conn.execute("UPDATE chunks SET page_start = 160, page_end = 161 WHERE book_id = %s",
                 (book,))
    conn.commit()

    span = db.page_of_passage(
        conn, book,
        "The serpent is characterized in various ways... Grigor says that the serpent was subtle")
    assert span == (160, 161)


def test_a_quoted_heading_is_found_in_the_heading_trail(conn):
    """Sometimes it quotes a section title rather than prose. Those live in
    their own column, so a chunk-body-only search returns nothing for a quote
    that is genuinely in the book."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["body text that does not contain the title at all"])
    conn.execute("""UPDATE chunks SET page_start = 43, page_end = 44,
                    heading_trail = 'Genesis 3 and the Alleged Satan Figure'
                    WHERE book_id = %s""", (book,))
    conn.commit()

    span = db.page_of_passage(conn, book, "Genesis 3 and the Alleged Satan Figure")
    assert span == (43, 44)


def test_a_quote_trimmed_short_of_ten_words_still_resolves(conn):
    """A ten-word needle fails when the model dropped the eleventh word. The
    fragment is retried shorter -- down to five, which is still specific inside
    one book."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["was the snake good the Naassenes regarded it as wisdom itself"])
    conn.execute("UPDATE chunks SET page_start = 53, page_end = 53 WHERE book_id = %s",
                 (book,))
    conn.commit()

    assert db.page_of_passage(conn, book, "was the snake good the Naassenes") == (53, 53)


def test_a_paraphrase_still_gets_no_page(conn):
    """The looser matching must not become guessing. Words that are not in the
    book resolve to nothing, however many attempts are made."""
    book = make_book(conn, "b1", "A Book")
    make_chunks(conn, book, ["the serpent is characterized in various ways by the commentators"])

    assert db.page_of_passage(
        conn, book, "the author argues at length that the snake represents wisdom") is None


def test_the_candidate_list_carries_titles_and_never_prose(conn, monkeypatch):
    """find_books summaries DO carry every candidate now -- the run page exists
    to show which books were considered, and a count cannot answer that.

    What must not change is the kind of thing they carry. Titles and centroid
    labels are short and bounded by k; a passage is a paragraph, and twenty of
    those per search is a book going through the event stream. The list is also
    capped at what the tool returned, so it cannot outgrow k."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    _shelf(conn)

    raw = tools.find_books(conn, "atonement sacrifice propitiation",
                           voyage=FakeVoyage())
    summary = loop._summarize("find_books", raw)

    assert set(summary) == {"returned", "nearest", "thin", "topic",
                            "candidates", "book_ids", "already_on_shelf"}
    assert len(summary["candidates"]) == summary["returned"] <= tools.MAX_FIND_K

    for c in summary["candidates"]:
        assert set(c) == {"book_id", "title", "distance", "covers", "pages",
                          "in_classroom"}
        # The label is a heading, not a paragraph. Anything approaching prose
        # length here means a passage found its way in.
        assert len(c["covers"] or "") < 200, "covers should be a label, not prose"


def test_the_angle_is_recorded_with_its_results(conn, monkeypatch):
    """The topic string is the most readable thing in a recorded run -- it is
    what the librarian decided to go looking for, in its own words. Losing it
    would leave the run page with rankings and no reason for them."""
    monkeypatch.setattr(tools.embed_mod, "embed_query",
                        lambda text, client: lexical_vector(text))
    _shelf(conn)

    raw = tools.find_books(conn, "penal substitution", voyage=FakeVoyage())
    assert loop._summarize("find_books", raw)["topic"] == "penal substitution"
