"""The classroom HTTP surface: CRUD, the cap, and what may go on a shelf.

These assert on status codes and messages, not just on the database, because
the refusals ARE the feature here -- a 409 that names the books still being
prepared is the difference between "you cannot add that yet" and a shelf that
silently cannot answer.
"""
import pytest
from fastapi.testclient import TestClient

from library_rag import config, db
from library_rag.web import api

from .conftest import make_book, make_chunks


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setattr(api.config, "VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return TestClient(api.app)


def _ready(conn, source_id, title):
    book = make_book(conn, source_id, title)
    make_chunks(conn, book, ["some text about the covenant"])
    return book


# -------------------------------------------------------------- lifecycle --

def test_a_classroom_is_created_listed_renamed_and_deleted(client, conn):
    made = client.post("/api/classrooms",
                       json={"name": "Atonement", "brief": "why the cross saves"})
    assert made.status_code == 200
    cid = made.json()["classroom"]["id"]

    listed = client.get("/api/classrooms").json()
    assert [c["name"] for c in listed["classrooms"]] == ["Atonement"]
    assert listed["max_books"] == config.CLASSROOM_MAX_BOOKS

    client.patch(f"/api/classrooms/{cid}", json={"name": "The Cross"})
    assert client.get(f"/api/classrooms/{cid}").json()["classroom"]["name"] == "The Cross"

    assert client.delete(f"/api/classrooms/{cid}").status_code == 200
    assert client.get(f"/api/classrooms/{cid}").status_code == 404


def test_a_classroom_can_be_deleted_after_the_tutor_has_been_used(client, conn):
    """The delete test above only ever deleted an unused classroom -- which was
    the only kind that worked. 0012 added research_runs.classroom_id as a bare
    REFERENCES, so it defaulted to ON DELETE NO ACTION and the first question a
    reader asked made the shelf permanently undeletable with a 500. Found by
    hand against a live server, not by that test.

    The run must survive the delete, unfiled. It holds the answer text, and
    deleting a shelf should no more destroy the answers taken off it than it
    destroys the books that were on it."""
    cid = client.post("/api/classrooms", json={"name": "Atonement"}).json()["classroom"]["id"]
    db.create_research_run(conn, "run-abc", "what does hilasmos mean?", classroom_id=cid)
    db.finish_research_run(conn, "run-abc", status="done", answer="Expiation.")

    assert client.delete(f"/api/classrooms/{cid}").status_code == 200
    assert client.get(f"/api/classrooms/{cid}").status_code == 404

    kept = db.fetch_research_run(conn, "run-abc")
    assert kept is not None, "deleting a classroom destroyed its answer history"
    assert kept["answer"] == "Expiation."
    assert kept["classroom_id"] is None


def test_a_classroom_reports_how_heavy_its_shelf_is(client, conn):
    """Books and bytes, so the card can say what is on a shelf without the page
    fetching every book to add them up. books is already joined for `arriving`,
    so both sums come free -- and an empty shelf must read 0, not blank: sum()
    over no rows is NULL, which reaches JSON as null and renders as nothing."""
    cid = client.post("/api/classrooms", json={"name": "Paul"}).json()["classroom"]["id"]

    empty = client.get("/api/classrooms").json()["classrooms"][0]
    assert empty["book_count"] == 0
    assert empty["size_bytes"] == 0, "an empty shelf must report 0, not null"
    assert empty["pages"] == 0

    a = _ready(conn, "a", "One")
    b = _ready(conn, "b", "Two")
    conn.execute("UPDATE books SET size_bytes = %s, page_count = %s WHERE id = %s",
                 (1_500_000, 120, a))
    conn.execute("UPDATE books SET size_bytes = %s, page_count = %s WHERE id = %s",
                 (2_500_000, 80, b))
    conn.commit()
    client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [a, b]})

    row = client.get("/api/classrooms").json()["classrooms"][0]
    assert row["book_count"] == 2
    assert row["size_bytes"] == 4_000_000
    assert row["pages"] == 200

    # And per book, for the header on the classroom page itself.
    books = client.get(f"/api/classrooms/{cid}").json()["books"]
    assert sum(bk["size_bytes"] for bk in books) == 4_000_000


def test_adding_a_batch_keeps_each_book_its_own_reason(client, conn):
    """"Add all" puts a whole shortlist on the shelf in one call. The shelf shows
    why each book is there, so one rationale stamped across twelve of them would
    make eleven lie about themselves -- the librarian's evidence is per book."""
    a = _ready(conn, "a", "Christus Victor")
    b = _ready(conn, "b", "Cur Deus Homo")
    c = _ready(conn, "c", "No Reason Given")
    cid = client.post("/api/classrooms", json={"name": "Atonement"}).json()["classroom"]["id"]

    r = client.post(f"/api/classrooms/{cid}/books", json={
        "book_ids": [a, b, c],
        "added_by": "librarian",
        "rationale": "fallback",
        "rationales": {str(a): "victory over the powers",
                       str(b): "satisfaction of divine honour"},
    })
    assert r.status_code == 200

    got = {bk["title"]: bk["rationale"] for bk in r.json()["books"]}
    assert got["Christus Victor"] == "victory over the powers"
    assert got["Cur Deus Homo"] == "satisfaction of divine honour"
    # A book with no entry of its own falls back rather than ending up blank.
    assert got["No Reason Given"] == "fallback"


def test_an_unknown_classroom_is_404_everywhere(client, conn):
    assert client.get("/api/classrooms/9999").status_code == 404
    assert client.patch("/api/classrooms/9999", json={"name": "x"}).status_code == 404
    assert client.delete("/api/classrooms/9999").status_code == 404
    assert client.post("/api/classrooms/9999/books",
                       json={"book_ids": [1]}).status_code == 404


def test_deleting_a_classroom_keeps_its_books(client, conn):
    book = _ready(conn, "a", "Book A")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [book]})

    client.delete(f"/api/classrooms/{cid}")
    assert conn.execute("SELECT count(*) FROM books WHERE id = %s",
                        (book,)).fetchone()[0] == 1


# ------------------------------------------------------------ adding books --

def test_only_ready_books_may_join_a_shelf(client, conn):
    """A book mid-ingest has no chunks. Adding it would put something on the
    shelf the tutor cannot read, and the reader would meet that later as a
    missing answer rather than now as a refusal."""
    arriving = make_book(conn, "arr", "Still Arriving", status="discovered")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]

    r = client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [arriving]})
    assert r.status_code == 409
    assert "Still Arriving" in r.json()["detail"], (
        "the refusal must name the book, not just the rule"
    )
    assert client.get(f"/api/classrooms/{cid}").json()["books"] == []


def test_adding_an_unknown_book_is_404_naming_it(client, conn):
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    r = client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [4242]})
    assert r.status_code == 404
    assert "4242" in r.json()["detail"]


def test_the_shelf_reports_readiness_per_book(client, conn):
    """A book added while ready can stop being ready -- a re-ingest resets it --
    so the flag is computed on read rather than stored at add time."""
    book = _ready(conn, "a", "Book A")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [book]})

    assert client.get(f"/api/classrooms/{cid}").json()["books"][0]["ready"] is True
    conn.execute("UPDATE books SET status = 'discovered' WHERE id = %s", (book,))
    conn.commit()
    assert client.get(f"/api/classrooms/{cid}").json()["books"][0]["ready"] is False


def test_the_cap_is_a_409_that_says_the_numbers(client, conn):
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    books = [_ready(conn, f"b{i}", f"Book {i}")
             for i in range(config.CLASSROOM_MAX_BOOKS)]
    assert client.post(f"/api/classrooms/{cid}/books",
                       json={"book_ids": books}).status_code == 200

    over = _ready(conn, "over", "One Too Many")
    r = client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [over]})
    assert r.status_code == 409
    assert str(config.CLASSROOM_MAX_BOOKS) in r.json()["detail"]


def test_a_request_past_the_cap_is_refused_at_the_edge(client, conn):
    """Bounded in the model as well as in the database, so an oversized list is
    a 422 naming the field rather than a 409 after the work of looking every
    book up."""
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    r = client.post(f"/api/classrooms/{cid}/books",
                    json={"book_ids": list(range(config.CLASSROOM_MAX_BOOKS + 5))})
    assert r.status_code == 422


def test_removing_a_book_returns_the_new_shelf(client, conn):
    book = _ready(conn, "a", "Book A")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    client.post(f"/api/classrooms/{cid}/books", json={"book_ids": [book]})

    r = client.delete(f"/api/classrooms/{cid}/books/{book}")
    assert r.status_code == 200
    assert r.json()["books"] == []


def test_the_librarian_rationale_is_stored_with_the_pick(client, conn):
    """The evidence for a recommendation outlives the run that produced it --
    otherwise a shelf is a list of titles with no provenance."""
    book = _ready(conn, "a", "Book A")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    client.post(f"/api/classrooms/{cid}/books",
                json={"book_ids": [book], "added_by": "librarian",
                      "rationale": "p.12: 'the cross...'"})

    shelf = client.get(f"/api/classrooms/{cid}").json()["books"]
    assert shelf[0]["added_by"] == "librarian"
    assert shelf[0]["rationale"].startswith("p.12")


def test_added_by_is_constrained(client, conn):
    book = _ready(conn, "a", "Book A")
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]
    r = client.post(f"/api/classrooms/{cid}/books",
                    json={"book_ids": [book], "added_by": "the-tooth-fairy"})
    assert r.status_code == 422


# ------------------------------------------------------------- the tutor --

def test_asking_without_a_classroom_is_refused_not_answered(client, conn):
    """Since 0010 there is no global chunk index, so an unscoped question is not
    a broader search -- it is a sequential scan that exhausts the request
    timeout. A 422 naming the field is the honest failure."""
    assert client.post("/api/research", json={"question": "anything"}).status_code == 422


def test_asking_an_unknown_classroom_is_404(client, conn):
    r = client.post("/api/research",
                    json={"question": "anything", "classroom_id": 9999})
    assert r.status_code == 404


def test_a_tutor_run_is_recorded_against_its_classroom(client, conn, monkeypatch):
    monkeypatch.setattr(api.embed_mod, "build_client", lambda: object())
    monkeypatch.setattr(api.research, "run",
                        lambda q, c, voyage, book_ids, model=None: iter([{"type": "done"}]))
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]

    run_id = client.post("/api/research",
                         json={"question": "q", "classroom_id": cid}).json()["run_id"]
    row = db.fetch_research_run(conn, run_id)
    assert row["classroom_id"] == cid


# ------------------------------------------------------- choosing a model --

def test_a_model_label_reaches_the_tutor_as_an_id(client, conn, monkeypatch):
    """The wire carries "opus"; the loop is given claude-opus-5. The mapping is
    the protection -- see the next test for what it protects against."""
    monkeypatch.setattr(api.embed_mod, "build_client", lambda: object())
    seen = {}

    def fake_run(q, c, voyage, book_ids, model=None):
        seen["model"] = model
        return iter([{"type": "done"}])

    monkeypatch.setattr(api.research, "run", fake_run)
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]

    client.post("/api/research",
                json={"question": "q", "classroom_id": cid, "model": "opus"})
    assert seen["model"] == "claude-opus-5"


def test_an_arbitrary_model_string_is_refused(client, conn):
    """A field carrying a model ID would let a caller point this project's key
    at any model they named. The closed set is why it cannot."""
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]

    for bad in ("claude-opus-4-1", "gpt-4", "", "OPUS"):
        r = client.post("/api/research",
                        json={"question": "q", "classroom_id": cid, "model": bad})
        assert r.status_code == 422, f"{bad!r} was accepted"


def test_omitting_the_model_leaves_the_deployment_default(client, conn, monkeypatch):
    """No choice means the server's own default, not a hardcoded one here --
    LIBRARIAN_MODEL / TUTOR_MODEL still decide what a plain request runs."""
    monkeypatch.setattr(api.embed_mod, "build_client", lambda: object())
    monkeypatch.setattr(api.retrieval_loop, "MODEL", "claude-from-the-environment")
    seen = {}

    def fake_run(q, c, voyage, book_ids, model=None):
        seen["model"] = model
        return iter([{"type": "done"}])

    monkeypatch.setattr(api.research, "run", fake_run)
    cid = client.post("/api/classrooms", json={"name": "C"}).json()["classroom"]["id"]

    client.post("/api/research", json={"question": "q", "classroom_id": cid})
    assert seen["model"] == "claude-from-the-environment"


# ------------------------------------------------------------- activity --

def test_runs_are_listed_newest_first_and_filtered_by_agent(client, conn):
    """Both agents write to one table now, so the list has to be able to
    separate them -- and 0016 backfilled every pre-existing row as 'tutor',
    which was the only thing it could have been."""
    db.create_research_run(conn, "r-tutor", "a question", agent="tutor")
    db.create_research_run(conn, "r-lib", "a brief", agent="librarian",
                           model="claude-sonnet-5")

    everything = client.get("/api/runs").json()
    assert everything["total"] == 2
    assert everything["runs"][0]["run_id"] == "r-lib", "newest first"

    only_lib = client.get("/api/runs?agent=librarian").json()
    assert [r["run_id"] for r in only_lib["runs"]] == ["r-lib"]
    assert only_lib["runs"][0]["model"] == "claude-sonnet-5"

    assert client.get("/api/runs?agent=nonsense").status_code == 422


def test_a_run_comes_back_with_its_events_in_order(client, conn):
    """The page rebuilds the whole run from these -- which angles were searched,
    what each returned, what was opened. Order is the run."""
    db.create_research_run(conn, "r1", "a brief", agent="librarian")
    for e in ({"type": "tool", "name": "find_books", "input": {"topic": "one"}},
              {"type": "results", "name": "find_books",
               "summary": {"topic": "one", "candidates": [{"book_id": 7}]}},
              {"type": "done", "iterations": 2}):
        db.append_research_event(conn, "r1", e)

    body = client.get("/api/runs/r1").json()
    assert body["run"]["agent"] == "librarian"
    assert [e["payload"]["type"] for e in body["events"]] == \
        ["tool", "results", "done"]
    assert body["events"][1]["payload"]["summary"]["candidates"] == [{"book_id": 7}]


def test_an_unknown_run_is_404(client, conn):
    assert client.get("/api/runs/nope").status_code == 404


def test_a_run_names_the_classroom_it_was_taken_off(client, conn):
    """Joined, not stored: 0014 made classroom_id SET NULL when a shelf is
    deleted, so the name must be absent rather than wrong once it is gone."""
    cid = client.post("/api/classrooms", json={"name": "Atonement"}).json()["classroom"]["id"]
    db.create_research_run(conn, "r1", "q", classroom_id=cid, agent="tutor")

    assert client.get("/api/runs").json()["runs"][0]["classroom_name"] == "Atonement"
    assert client.get("/api/runs/r1").json()["classroom"]["name"] == "Atonement"

    client.delete(f"/api/classrooms/{cid}")
    assert client.get("/api/runs").json()["runs"][0]["classroom_name"] is None
    assert client.get("/api/runs/r1").json()["classroom"] is None
