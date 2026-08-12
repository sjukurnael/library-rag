"""The Bible table, its loader, and the three read queries.

Zero network: every test runs against the fixture in tests/fixtures/, never
against bereanbible.com. The parser tests need no database either -- parse() is
deliberately a pure function from a file to a list of tuples.
"""
from pathlib import Path

import pytest

from library_rag import bible

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bsb_sample.txt"


@pytest.fixture
def rows():
    return bible.parse(FIXTURE)


@pytest.fixture
def loaded(conn, rows):
    """The fixture verses, in the database. Returns the connection."""
    bible.load(conn, rows)
    return conn


# ------------------------------------------------------- parsing (no database --

def test_the_three_header_lines_are_skipped(rows):
    """The file opens with two copyright notices and a column header. Parsing
    them as verses would put 'Verse' in the table as a book name."""
    assert len(rows) == 10
    assert rows[0][1] == "Genesis"


def test_a_verse_becomes_the_tuple_the_table_expects(rows):
    assert rows[0] == (
        1, "Genesis", 1, 1,
        "In the beginning God created the heavens and the earth.",
    )


@pytest.mark.parametrize(
    "index, book_num, book_name",
    [
        (3, 9, "1 Samuel"),          # leading digit
        (4, 22, "Song of Solomon"),  # three words
        (8, 64, "3 John"),           # digit AND the book is one chapter
    ],
)
def test_multi_word_book_names_parse(rows, index, book_num, book_name):
    """The single most likely bug in this feature.

    'Genesis 1:1' splits on spaces harmlessly; '1 Samuel 1:1' does not, and
    neither do the other seventeen. The regex takes the book name greedily up
    to the LAST ' <digits>:<digits>', which is what makes these work.
    """
    assert rows[index][0] == book_num
    assert rows[index][1] == book_name


def test_an_empty_verse_is_kept_not_skipped(rows):
    """16 verses in the real file carry no text -- they are in the KJV and
    absent from the earliest manuscripts. Dropping them would make Matthew 17
    jump from verse 20 to 22, which reads as a loader bug rather than as the
    textual fact it is."""
    matthew = [r for r in rows if r[1] == "Matthew"]
    assert [r[3] for r in matthew] == [20, 21, 22]
    assert matthew[1][4] == ""


def test_a_line_that_will_not_parse_raises(tmp_path):
    """Skipping it would drop a verse from the Bible and report success."""
    bad = tmp_path / "bad.txt"
    bad.write_text("h1\nh2\nh3\nthis is not a verse line\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        bible.parse(bad)
    assert "line 4" in str(e.value)


def test_an_unknown_book_name_raises(tmp_path):
    """A name that disagrees by one character would otherwise become a KeyError
    somewhere far from the cause."""
    bad = tmp_path / "bad.txt"
    bad.write_text("h1\nh2\nh3\nNarnia 1:1\tOnce upon a time.\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        bible.parse(bad)
    assert "unknown book" in str(e.value)


def test_the_canon_is_66_books_numbered_in_order():
    assert len(bible.BOOKS) == 66
    assert bible.BOOK_NUMBER["Genesis"] == 1
    assert bible.BOOK_NUMBER["Malachi"] == 39
    assert bible.BOOK_NUMBER["Matthew"] == 40
    assert bible.BOOK_NUMBER["Revelation"] == 66
    # The source spells it 'Psalm', singular. Correcting it here would stop
    # every lookup for that book from matching.
    assert "Psalm" in bible.BOOK_NUMBER
    assert "Psalms" not in bible.BOOK_NUMBER


# ------------------------------------------------------------------ loading --

def test_load_writes_every_row(loaded, rows):
    n = loaded.execute("SELECT count(*) FROM bible_verses").fetchone()[0]
    assert n == len(rows) == 10


def test_loading_twice_replaces_rather_than_accumulates(conn, rows):
    """A fixed published text, so a re-run after a parser fix should be
    trivially correct rather than doubling every verse."""
    bible.load(conn, rows)
    bible.load(conn, rows)
    assert conn.execute("SELECT count(*) FROM bible_verses").fetchone()[0] == 10


def test_round_trip_through_the_database(loaded):
    """parse -> load -> chapter() returns exactly what went in."""
    assert bible.chapter(loaded, 1, 1) == [
        (1, "In the beginning God created the heavens and the earth."),
        (2, "Now the earth was formless and void."),
    ]


def test_loaded_reports_an_empty_table_as_not_loaded(conn, rows):
    """The page needs 'run the loader' to be distinguishable from 'your search
    found nothing'."""
    assert bible.loaded(conn) is False
    bible.load(conn, rows)
    assert bible.loaded(conn) is True


# ------------------------------------------------------------------ reading --

def test_books_reports_chapter_counts_for_the_pickers(loaded):
    """The page computes the chapter dropdown AND the arrow wrap from this, so
    a wrong chapter count is a navigation bug."""
    listing = {name: (chapters, verses) for _, name, chapters, verses in bible.books(loaded)}
    assert listing["Genesis"] == (2, 3)      # chapters 1 and 2, three verses
    assert listing["3 John"] == (1, 1)
    assert listing["Revelation"] == (22, 1)  # one verse, but it is in chapter 22


def test_books_come_back_in_canonical_order(loaded):
    numbers = [b for b, _, _, _ in bible.books(loaded)]
    assert numbers == sorted(numbers)
    assert numbers[0] == 1 and numbers[-1] == 66


def test_a_missing_chapter_is_empty_not_an_error(loaded):
    assert bible.chapter(loaded, 1, 99) == []


def test_search_is_case_insensitive_and_in_bible_order(loaded):
    hits = bible.search(loaded, "THE HEAVENS")
    assert [(h[1], h[2], h[3]) for h in hits] == [
        ("Genesis", 1, 1), ("Genesis", 2, 1),
    ]


def test_search_escapes_like_wildcards(loaded):
    """% and _ are LIKE wildcards. Unescaped, a search for '%' becomes the
    pattern '%%%' and matches the entire Bible -- a wrong answer that looks
    like a working feature."""
    assert bible.search(loaded, "%") == []
    assert bible.search(loaded, "_") == []
    # And a literal underscore IS findable when it is really there.
    assert len(bible.search(loaded, "Ramathaim-zophim")) == 1


def test_search_respects_its_limit(loaded):
    assert len(bible.search(loaded, "the", limit=2)) == 2
