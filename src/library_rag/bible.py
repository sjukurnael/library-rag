"""The Bible, one row per verse. English only.

The Berean Standard Bible, dedicated to the public domain in April 2023. One
4.3 MB text file from bereanbible.com becomes 31,102 rows in `bible_verses`,
and the web page reads nothing else.

Three stages, deliberately separated, so each is understandable and testable on
its own:

    download()   network -> data/bsb.txt      (runs ONCE; then the file is there)
    parse()      data/bsb.txt -> list of tuples   (no database)
    load()       tuples -> Postgres               (no filesystem)

Nothing here knows about Greek, morphology, syntax or embeddings. It is an
English Bible and nothing more.
"""
import re
import urllib.request

from library_rag import config

# The one network call in this feature. Not an API -- a plain HTTP GET of a
# static text file, no key and no auth. The running web app never touches it;
# only the CLI does, and only when data/bsb.txt is absent.
BSB_URL = "https://bereanbible.com/bsb.txt"
BSB_FILE = config.DATA_DIR / "bsb.txt"

# Every line of the file looks like:
#     Genesis 1:1<TAB>In the beginning God created the heavens and the earth.
#
# `(.+)` is GREEDY, which is the whole trick: it takes the book name up to the
# LAST " <digits>:<digits>", so "1 Samuel 1:1" and "Song of Solomon 2:3" parse
# correctly. Splitting on spaces breaks on all eighteen multi-word book names.
VERSE_RE = re.compile(r"^(.+) (\d+):(\d+)\t(.*)$")

# The first three lines are two copyright notices and a
# "Verse<TAB>Berean Standard Bible" column header.
HEADER_LINES = 3

# The count is fixed and published, so it doubles as a checksum: a different
# number means the upstream file changed shape and the parse is not trustworthy.
EXPECTED_VERSES = 31102

# Canonical order, spelled exactly as bsb.txt spells them -- note "Psalm",
# singular, and "Song of Solomon". This list IS the book numbering: Genesis is
# 1, Revelation is 66. Copied from the file itself rather than typed from
# memory, because a name that disagrees by one character stops matching.
BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalm", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]
BOOK_NUMBER = {name: i for i, name in enumerate(BOOKS, start=1)}


# ---------------------------------------------------------------- download --

def download(force: bool = False):
    """Fetch bsb.txt into data/, unless it is already there.

    Returns the path either way, so callers do not branch. `force` re-downloads
    over the top, for the day the upstream text is revised.
    """
    if BSB_FILE.exists() and not force:
        return BSB_FILE
    BSB_FILE.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BSB_URL, BSB_FILE)
    return BSB_FILE


# ------------------------------------------------------------------- parse --

def parse(path) -> list:
    """The text file -> [(book, book_name, chapter, verse, text), ...].

    utf-8-sig rather than utf-8: the file begins with a byte-order mark, and
    reading it as plain utf-8 makes the first book name "﻿Genesis", which
    then fails the BOOK_NUMBER lookup for one row and no others.

    A line that will not parse RAISES. Skipping it would drop a verse from the
    Bible and report success, which is the failure nobody notices until they
    look up a verse that is not there.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()[HEADER_LINES:]
    rows = []
    for n, line in enumerate(lines, start=HEADER_LINES + 1):
        # The file uses CRLF; splitlines() leaves the \r on some platforms.
        match = VERSE_RE.match(line.rstrip("\r"))
        if not match:
            raise ValueError(f"line {n}: cannot parse {line[:60]!r}")
        name, chapter, verse, text = match.groups()
        if name not in BOOK_NUMBER:
            raise ValueError(f"line {n}: unknown book {name!r}")
        # text is '' for the 16 placeholder verses. Kept, not skipped -- see
        # the migration's comment.
        rows.append(
            (BOOK_NUMBER[name], name, int(chapter), int(verse), text.strip())
        )
    return rows


# -------------------------------------------------------------------- load --

def load(conn, rows) -> int:
    """Replace the table's contents with `rows`. Returns how many were written.

    TRUNCATE-and-reload rather than upsert: the Bible is a fixed published text,
    not accumulating state, so a re-run after a parser fix is trivially correct
    instead of having to reconcile against whatever the previous run wrote.

    COPY rather than 31,102 INSERTs, because this crosses the network to
    Supabase and an INSERT each would be 31,102 round trips.

    One transaction: the TRUNCATE and every row commit together, so a failure
    part-way leaves the old Bible intact rather than half of a new one.
    """
    conn.execute("TRUNCATE bible_verses")
    with conn.cursor() as cur:
        with cur.copy(
            "COPY bible_verses (book, book_name, chapter, verse, text) FROM STDIN"
        ) as copy:
            for row in rows:
                copy.write_row(row)
    conn.commit()
    return len(rows)


# ----------------------------------------------------------------- reading --

def loaded(conn) -> bool:
    """Whether the table exists AND has anything in it.

    to_regclass returns NULL for a missing table instead of raising, which
    matters twice: the app's database may not have run the migration yet, and a
    failed statement would poison the transaction for everything after it.
    """
    if conn.execute("SELECT to_regclass('bible_verses')").fetchone()[0] is None:
        return False
    return conn.execute("SELECT EXISTS (SELECT 1 FROM bible_verses)").fetchone()[0]


def books(conn) -> list:
    """Every book with its chapter count, in canonical order.

    Drives both pickers AND the chapter arrows: the page computes "what comes
    after Revelation 22" from this rather than hardcoding it, so the wrap stays
    correct without the frontend knowing anything about the canon.
    """
    return conn.execute(
        """
        SELECT book, book_name, max(chapter) AS chapters, count(*) AS verses
        FROM bible_verses
        GROUP BY book, book_name
        ORDER BY book
        """
    ).fetchall()


def chapter(conn, book: int, chapter: int) -> list:
    """One chapter's verses, in order. [] if there is no such chapter."""
    return conn.execute(
        """
        SELECT verse, text
        FROM bible_verses
        WHERE book = %s AND chapter = %s
        ORDER BY verse
        """,
        (book, chapter),
    ).fetchall()


def search(conn, q: str, limit: int = 200) -> list:
    """Verses containing `q` as literal text, in Bible order.

    ILIKE is case-insensitive LIKE. % and _ are LIKE wildcards, so without the
    escaping below a search for "100%" would become the pattern "%100%%" and
    match every verse in the Bible; ESCAPE names the character that turns them
    back into literals. The backslash itself is escaped first, or it would eat
    the escapes added after it.

    A sequential scan over 31k rows -- there is no index for this and does not
    need to be one. `limit` is a hard cap; the caller is expected to say when it
    was hit rather than present a truncated list as complete.
    """
    pattern = (
        q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return conn.execute(
        """
        SELECT book, book_name, chapter, verse, text
        FROM bible_verses
        WHERE text ILIKE %s ESCAPE '\\'
        ORDER BY book, chapter, verse
        LIMIT %s
        """,
        (f"%{pattern}%", limit),
    ).fetchall()
