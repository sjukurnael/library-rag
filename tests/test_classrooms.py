"""Classrooms: the shelf, and the promise that nothing outside it is reachable.

The containment test below is the one that matters. Everything else in this
design -- dropping the global chunk index, scope-as-signal, the tutor's whole
premise -- rests on a scoped search being unable to return a chunk from a book
the reader did not choose. So it is asserted against real Postgres with a chunk
that deliberately matches BETTER than anything in scope, not against a mock.
"""
import pytest

from library_rag import config, db

from .conftest import lexical_vector, make_book, make_chunks

# ------------------------------------------------------------- containment --

def test_a_scoped_search_cannot_reach_a_better_chunk_outside_the_classroom(conn):
    """The security property of the entire design.

    `outside` holds the passage that matches the query best. If scoping is ever
    wrong -- a dropped WHERE, a NULL guard inverted -- this is the test that
    fails, because the wrong answer is also the most attractive one.
    """
    inside = make_book(conn, "in", "On the Shelf")
    outside = make_book(conn, "out", "Not on the Shelf")
    make_chunks(conn, inside, ["atonement sacrifice"])
    make_chunks(conn, outside, ["atonement sacrifice cross blood ransom propitiation"])

    query = lexical_vector("atonement sacrifice cross blood ransom propitiation")

    # Unscoped, the outside book wins -- establishing that it really is nearer.
    unscoped = db.search(conn, query, 5, query_text="atonement", mode="dense")
    assert unscoped[0]["book_id"] == outside, "the bait must actually be the best match"

    for mode in ("dense", "hybrid", "lexical"):
        rows = db.search(conn, query, 5, book_ids=[inside],
                         query_text="atonement sacrifice", mode=mode)
        assert {r["book_id"] for r in rows} <= {inside}, f"{mode} leaked outside the scope"


def test_an_empty_classroom_matches_nothing_rather_than_everything(conn):
    """[] and None are different questions. A shelf with no books on it must
    return no passages, not the whole library."""
    a = make_book(conn, "a", "Book A")
    make_chunks(conn, a, ["atonement sacrifice"])

    assert db.search(conn, lexical_vector("atonement"), 5, book_ids=[]) == []
    assert db.search(conn, lexical_vector("atonement"), 5) != []


def test_narrowing_to_a_book_outside_the_scope_returns_nothing(conn):
    """book_id and book_ids compose rather than override. The tutor's Session
    refuses this case before it gets here; this asserts the second lock."""
    inside = make_book(conn, "in", "On the Shelf")
    outside = make_book(conn, "out", "Not on the Shelf")
    make_chunks(conn, inside, ["atonement sacrifice"])
    make_chunks(conn, outside, ["atonement sacrifice"])

    assert db.search(conn, lexical_vector("atonement"), 5, outside,
                     book_ids=[inside]) == []
    got = db.search(conn, lexical_vector("atonement"), 5, inside, book_ids=[inside])
    assert {r["book_id"] for r in got} == {inside}


# -------------------------------------------------------------------- CRUD --

def test_a_classroom_round_trips(conn):
    cid = db.create_classroom(conn, "reader@example.com", "Atonement",
                              brief="why Christ's death saves")
    row = db.fetch_classroom(conn, cid)
    assert row["name"] == "Atonement"
    assert row["brief"] == "why Christ's death saves"

    a = make_book(conn, "a", "Book A")
    db.add_classroom_books(conn, cid, [(a, "librarian", "p.12: 'the cross...'")])
    shelf = db.classroom_books(conn, cid)
    assert [s["book_id"] for s in shelf] == [a]
    assert shelf[0]["added_by"] == "librarian"
    assert shelf[0]["rationale"].startswith("p.12")


def test_adding_the_same_book_twice_is_a_no_op(conn):
    cid = db.create_classroom(conn, None, "C")
    a = make_book(conn, "a", "Book A")
    db.add_classroom_books(conn, cid, [(a, "reader", None)])
    db.add_classroom_books(conn, cid, [(a, "reader", None)])
    assert len(db.classroom_books(conn, cid)) == 1


def test_the_cap_refuses_rather_than_truncating(conn):
    """Adding a subset would be the worse failure: the reader is told 'added'
    and discovers the gap later, from the tutor, as a missing answer."""
    cid = db.create_classroom(conn, None, "C")
    books = [make_book(conn, f"b{i}", f"Book {i}") for i in range(config.CLASSROOM_MAX_BOOKS)]
    db.add_classroom_books(conn, cid, [(b, "reader", None) for b in books])

    one_too_many = make_book(conn, "over", "One Too Many")
    with pytest.raises(db.ClassroomFull) as e:
        db.add_classroom_books(conn, cid, [(one_too_many, "reader", None)])
    assert str(config.CLASSROOM_MAX_BOOKS) in str(e.value)
    assert len(db.classroom_books(conn, cid)) == config.CLASSROOM_MAX_BOOKS


def test_re_adding_at_the_cap_is_allowed(conn):
    """The cap counts NEW books. A shelf that is full must still tolerate the
    reader clicking 'add' on something already on it."""
    cid = db.create_classroom(conn, None, "C")
    books = [make_book(conn, f"b{i}", f"Book {i}") for i in range(config.CLASSROOM_MAX_BOOKS)]
    db.add_classroom_books(conn, cid, [(b, "reader", None) for b in books])
    db.add_classroom_books(conn, cid, [(books[0], "reader", None)])  # must not raise
    assert len(db.classroom_books(conn, cid)) == config.CLASSROOM_MAX_BOOKS


def test_removing_a_book_leaves_the_book_alone(conn):
    cid = db.create_classroom(conn, None, "C")
    a = make_book(conn, "a", "Book A")
    db.add_classroom_books(conn, cid, [(a, "reader", None)])

    assert db.remove_classroom_book(conn, cid, a) is True
    assert db.classroom_books(conn, cid) == []
    assert conn.execute("SELECT count(*) FROM books WHERE id = %s", (a,)).fetchone()[0] == 1


def test_deleting_a_classroom_never_deletes_its_books(conn):
    """The cascade runs one way only. Emptying a shelf must not touch the
    library -- the books are shared with every other classroom."""
    cid = db.create_classroom(conn, None, "C")
    a = make_book(conn, "a", "Book A")
    make_chunks(conn, a, ["some text"])
    db.add_classroom_books(conn, cid, [(a, "reader", None)])

    assert db.delete_classroom(conn, cid) is True
    assert db.fetch_classroom(conn, cid) is None
    assert conn.execute("SELECT count(*) FROM books WHERE id = %s", (a,)).fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM chunks WHERE book_id = %s", (a,)).fetchone()[0] == 1


def test_deleting_a_book_removes_it_from_every_shelf(conn):
    """The other direction of the cascade, and the reason classrooms_holding()
    exists -- the confirm dialog has to be able to say how many shelves lose it."""
    a = make_book(conn, "a", "Book A")
    one = db.create_classroom(conn, None, "One")
    two = db.create_classroom(conn, None, "Two")
    for cid in (one, two):
        db.add_classroom_books(conn, cid, [(a, "reader", None)])

    assert {c["name"] for c in db.classrooms_holding(conn, a)} == {"One", "Two"}
    conn.execute("DELETE FROM books WHERE id = %s", (a,))
    conn.commit()
    assert db.classroom_books(conn, one) == []
    assert db.classroom_books(conn, two) == []


def test_the_tutor_scope_excludes_books_still_arriving(conn):
    """A book mid-ingest has no chunks. Including it would widen the scope by a
    book that cannot answer, and the tutor would report an absence that is
    really a delay."""
    cid = db.create_classroom(conn, None, "C")
    ready = make_book(conn, "ready", "Ready")
    arriving = make_book(conn, "arriving", "Arriving", status="discovered")
    make_chunks(conn, ready, ["atonement"])
    db.add_classroom_books(conn, cid, [(ready, "reader", None), (arriving, "reader", None)])

    assert db.classroom_book_ids(conn, cid) == [ready]
    assert sorted(db.classroom_book_ids(conn, cid, ready_only=False)) == sorted([ready, arriving])
    # The shelf still shows both -- "arriving" is a state, not an omission.
    assert len(db.classroom_books(conn, cid)) == 2


def test_the_list_reports_counts_and_orders_by_last_used(conn):
    old = db.create_classroom(conn, "me@example.com", "Old")
    db.create_classroom(conn, "me@example.com", "New")
    a = make_book(conn, "a", "Book A")
    arriving = make_book(conn, "arr", "Arriving", status="discovered")
    db.add_classroom_books(conn, old, [(a, "reader", None), (arriving, "reader", None)])

    rows = db.list_classrooms(conn, "me@example.com")
    by_name = {r["name"]: r for r in rows}
    assert by_name["Old"]["book_count"] == 2
    assert by_name["Old"]["arriving"] == 1
    assert by_name["New"]["book_count"] == 0
    # `old` was touched by the add, so it floats above `new`.
    assert [r["name"] for r in rows] == ["Old", "New"]


def test_listing_without_an_owner_returns_everything(conn):
    """auth_enabled() can be false, and then there is no owner to filter by.
    Returning nothing in that mode would hide every classroom on a laptop."""
    db.create_classroom(conn, "a@example.com", "A")
    db.create_classroom(conn, None, "B")
    assert {r["name"] for r in db.list_classrooms(conn)} == {"A", "B"}
    assert {r["name"] for r in db.list_classrooms(conn, "a@example.com")} == {"A"}
