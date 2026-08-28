"""search_book_profiles() and look_inside(): the librarian's two retrieval tools.

Both are new, and both are the kind of code that fails quietly -- a scoping bug
returns plausible passages from the wrong book, and a fusion bug reorders
results without ever erroring. Vectors come from conftest's lexical_vector, so
texts sharing words really are near each other and these assert on retrieval
behaviour rather than on a round-trip.
"""

from library_rag import config, db

from .conftest import lexical_vector, make_book, make_chunks, make_profile

# ----------------------------------------------------------- look_inside --

def test_look_inside_only_returns_the_requested_book(conn):
    """The scoping IS the security property here. Without it the librarian
    quotes a passage from a book it is not recommending."""
    a = make_book(conn, "a", "Book A")
    b = make_book(conn, "b", "Book B")
    make_chunks(conn, a, ["atonement and the cross of christ"])
    make_chunks(conn, b, ["atonement and the cross of christ"])   # identical text

    rows = db.look_inside(conn, a, lexical_vector("atonement cross"), k=10)
    assert rows, "the matching chunk should be found"
    assert {r["id"] for r in rows} == {
        r[0] for r in conn.execute("SELECT id FROM chunks WHERE book_id = %s", (a,))
    }


def test_look_inside_orders_by_distance_and_honours_k(conn):
    a = make_book(conn, "a", "Book A")
    make_chunks(conn, a, ["psalms hebrew psalter editing",
                      "psalms hebrew",
                      "entirely unrelated maritime navigation"])

    rows = db.look_inside(conn, a, lexical_vector("psalms hebrew psalter editing"), k=2)
    assert len(rows) == 2, "k must cap the result set"
    assert rows[0]["distance"] <= rows[1]["distance"], "nearest first"
    assert "maritime" not in rows[0]["content"]


def test_look_inside_on_a_book_with_no_chunks_is_empty(conn):
    a = make_book(conn, "a", "Book A", status="discovered")
    assert db.look_inside(conn, a, lexical_vector("anything")) == []


# -------------------------------------------------- search_book_profiles --

def test_one_row_per_book_however_many_centroids_match(conn):
    """A book scores on its BEST centroid. Several matching topics must collapse
    to a single recommendation, not flood the list with one book."""
    a = make_book(conn, "a", "Book A")
    make_chunks(conn, a, ["placeholder"])
    make_profile(conn, a, [("Ch1", "atonement sacrifice"),
                       ("Ch2", "atonement propitiation"),
                       ("Ch3", "atonement substitution")])

    rows = db.search_book_profiles(conn, lexical_vector("atonement"), "atonement", k=10)
    assert [r["book_id"] for r in rows].count(a) == 1


def test_matched_topics_counts_the_centroids_that_matched(conn):
    """The signal that separates 'covers the subject' from 'mentions it once'."""
    broad = make_book(conn, "broad", "Broad Book")
    narrow = make_book(conn, "narrow", "Narrow Book")
    for b in (broad, narrow):
        make_chunks(conn, b, ["placeholder"])
    make_profile(conn, broad, [("A", "atonement sacrifice"), ("B", "atonement blood"),
                           ("C", "atonement ransom")])
    make_profile(conn, narrow, [("A", "atonement sacrifice"),
                            ("B", "maritime navigation charts"),
                            ("C", "beekeeping in winter")])

    rows = {r["book_id"]: r for r in
            db.search_book_profiles(conn, lexical_vector("atonement"), "atonement", k=10)}
    assert rows[broad]["matched_topics"] > rows[narrow]["matched_topics"]


def test_best_label_names_the_nearest_centroid(conn):
    """The label is what the librarian quotes as its rationale, so it has to be
    the topic that actually matched -- not just any topic of that book."""
    a = make_book(conn, "a", "Book A")
    make_chunks(conn, a, ["placeholder"])
    make_profile(conn, a, [("Wrong Chapter", "beekeeping in winter"),
                       ("Right Chapter", "psalms hebrew psalter editing")])

    rows = db.search_book_profiles(
        conn, lexical_vector("psalms hebrew psalter editing"), "psalter", k=5)
    assert rows[0]["best_label"] == "Right Chapter"


def test_a_book_matching_both_legs_outranks_one_matching_only_contents(conn):
    """The whole point of fusing the title leg: agreement between two different
    kinds of evidence beats a single leg's confidence."""
    both = make_book(conn, "both", "Atonement Studies.pdf")
    contents_only = make_book(conn, "contents", "Untitled Volume 7.pdf")
    for b in (both, contents_only):
        make_chunks(conn, b, ["placeholder"])
        make_profile(conn, b, [("Ch1", "atonement sacrifice cross")])

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO drive_files (file_id, name, mime_type, size_bytes, embedding)"
            " VALUES (%s, %s, 'application/pdf', 100, %s)",
            [("both", "Atonement Studies.pdf", db.HalfVector(lexical_vector("atonement studies"))),
             ("contents", "Untitled Volume 7.pdf", db.HalfVector(lexical_vector("untitled volume")))],
        )
    conn.commit()

    rows = db.search_book_profiles(conn, lexical_vector("atonement"), "atonement", k=10)
    order = [r["book_id"] for r in rows]
    assert order.index(both) < order.index(contents_only)


def test_unfinished_books_are_not_recommended(conn):
    """A book still in the queue has no chunks to answer from, so recommending
    it would build a session around something unreadable."""
    pending = make_book(conn, "pending", "Atonement Pending.pdf", status="discovered")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO drive_files (file_id, name, mime_type, size_bytes, embedding)"
            " VALUES ('pending', 'Atonement Pending.pdf', 'application/pdf', 100, %s)",
            (db.HalfVector(lexical_vector("atonement pending")),),
        )
    conn.commit()

    rows = db.search_book_profiles(conn, lexical_vector("atonement"), "atonement", k=10)
    assert pending not in [r["book_id"] for r in rows]


def test_shortlist_never_exceeds_ef_search(conn):
    """An HNSW scan silently returns fewer rows than asked for when ef_search is
    below the LIMIT. That misconfiguration invalidated a whole evaluation before
    anyone noticed, because raising the LIMIT changed nothing."""
    assert config.PROFILE_SHORTLIST <= config.PROFILE_EF_SEARCH
