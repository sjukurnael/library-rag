"""Every page route serves its HTML, and the static whitelist matches reality.

These are shallow on purpose. What they buy is the class of mistake that costs
an hour to diagnose in a browser and a second to catch here: a page renamed on
disk but not in the route, a new page whose file is never wired up, and -- the
one that has actually bitten -- a <script src> for a file the static handler
refuses to serve, which yields a 404 the page cannot report and a screen that
just sits there.

Boot behaviour is covered separately by tests/frontend/pages_boot.mjs; these
only assert the bytes arrive.
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from library_rag.web import api

STATIC = Path(api.__file__).resolve().parent / "static"

PAGES = {
    "/": "classrooms.html",
    "/library": "library.html",
    "/queue": "queue.html",
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api.config, "VOYAGE_API_KEY", "test-key")
    return TestClient(api.app)


@pytest.fixture
def bible_on(monkeypatch):
    monkeypatch.setattr(api.config, "BIBLE_ENABLED", True)


@pytest.mark.parametrize("path,filename", sorted(PAGES.items()))
def test_each_page_route_serves_its_html(client, path, filename):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    assert r.headers["content-type"].startswith("text/html")
    # Served from the file we think it is, not some other page.
    assert (STATIC / filename).read_text()[:200] in r.text


def test_the_classroom_page_renders_without_the_classroom_existing(client):
    """The id is read from the path by JavaScript and resolved over the API, so
    the route itself must not touch the database -- otherwise a stale bookmark
    is a 500 instead of a page that says the classroom is gone."""
    r = client.get("/classroom/999999")
    assert r.status_code == 200
    assert (STATIC / "classroom.html").read_text()[:200] in r.text


def test_every_script_and_stylesheet_a_page_asks_for_is_actually_served(client):
    """The static handler serves an explicit whitelist. Adding tutor.js to a
    page without adding it there is a 404 on a file the browser needed, which
    looks like a page that half-loaded and reports nothing."""
    missing = []
    for filename in sorted(set(PAGES.values()) | {"classroom.html", "login.html"}):
        html = (STATIC / filename).read_text()
        refs = set(re.findall(r'(?:src|href)=["\'](/static/[\w.\-]+)["\']', html))
        for ref in sorted(refs):
            if client.get(ref).status_code != 200:
                missing.append(f"{filename} -> {ref}")
    assert not missing, "referenced but not served: " + ", ".join(missing)


def test_the_static_handler_refuses_to_walk_out_of_its_directory(client):
    for attempt in ("/static/../api.py", "/static/%2e%2e/api.py", "/static/../../.env"):
        assert client.get(attempt).status_code != 200, attempt


# ------------------------------------------------------------ the Bible flag --
#
# What these pin is the GATE, not the Bible. Whether the verses are loaded is a
# separate question with its own tests; here the only thing that matters is
# whether config.BIBLE_ENABLED opens and closes the door.

BIBLE_API = ["/api/bible/books",
             "/api/bible/chapter?book=1&chapter=1",
             "/api/bible/search?q=love"]

# The refusal _require_bible raises. Distinguishable from the ordinary 404s the
# Bible routes raise on their own ("No chapter 99 in book 1"), which is what
# lets the enabled case below assert the gate is open without needing verses.
GATE_404 = "Not found."


@pytest.mark.parametrize("route", ["/bible"] + BIBLE_API)
def test_the_bible_is_off_by_default(client, route):
    """Every entry point, page and API alike. The gate runs before any database
    work, so this holds with no database at all -- which is also the proof that
    a disabled feature cannot be reached by keeping the URL."""
    r = client.get(route)
    assert r.status_code == 404, route
    assert r.json()["detail"] == GATE_404, route


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_bible_nav_entry_is_gone_when_it_is_off(client, page):
    """Stripped from the document, not hidden with CSS. A link that is only
    invisible is still focusable by keyboard and still points at a 404."""
    assert 'href="/bible"' not in client.get(page).text


def test_the_flag_reopens_the_page(client, bible_on):
    """The page route touches no database, so this needs nothing running."""
    r = client.get("/bible")
    assert r.status_code == 200
    assert (STATIC / "bible.html").read_text()[:200] in r.text


@pytest.mark.parametrize("route", BIBLE_API)
def test_the_flag_reopens_the_api(client, conn, bible_on, route):
    """Past the gate, not necessarily to data: an empty bible_verses table is a
    legitimate 404 from the handler, and asserting 200 here would be testing
    whether Scripture was imported rather than whether the flag works."""
    r = client.get(route)
    if r.status_code == 404:
        assert r.json()["detail"] != GATE_404, f"{route} still refused by the gate"


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_flag_brings_the_nav_entry_back(client, bible_on, page):
    """The page and the link are gated together. A nav entry over a dead route
    -- or a live route with no way to reach it -- is the failure this pins."""
    assert 'href="/bible"' in client.get(page).text


def test_the_other_nav_entries_survive_the_strip(client):
    """The strip is a regex over served HTML, so the thing to prove is that it
    takes out the Bible entry and nothing adjacent to it. The Bible link sits
    between /queue and the end of the nav, so those are the neighbours at risk."""
    html = client.get("/").text
    for href in ('href="/library"', 'href="/queue"'):
        assert href in html, href


@pytest.mark.parametrize("page", sorted(PAGES) + ["/classroom/1"])
def test_every_page_can_get_back_to_the_classrooms(client, page):
    """The classroom pages dropped "Classrooms" from the nav when the sidebar
    started listing the rooms themselves, which left no link to the list at all.
    The brand carries it now -- and a shell with no way home is the kind of
    thing that only shows up when someone is already lost."""
    assert 'href="/"' in client.get(page).text, page


# ------------------------------------------------ where each action may live --

def test_only_the_drive_page_can_start_an_index(client):
    """Indexing is minutes of download, extraction and Voyage tokens. It belongs
    beside the coverage bars that say what it costs, not on a button in a dialog
    whose job is picking books off a shelf. A classroom adds books that are
    already in the library; it never pulls new ones in."""
    index_endpoints = ("/api/library/drive", "/api/drive/folders")
    for page in ("classrooms.html", "classroom.html"):
        html = (STATIC / page).read_text()
        for ep in index_endpoints:
            assert ep not in html, f"{page} can reach {ep}"
    assert any(ep in (STATIC / "library.html").read_text() for ep in index_endpoints), \
        "the Drive page must still be able to index"


def test_there_is_exactly_one_new_classroom_button(client):
    """Two of them -- one in the sidebar, one in the header -- read as two
    different actions, and the sidebar copy offered to start a new classroom
    while you were sitting inside a different one."""
    assert (STATIC / "classrooms.html").read_text().count("New classroom") == 1
    assert "New classroom" not in (STATIC / "classroom.html").read_text()


def test_classrooms_is_reachable_from_the_nav_not_only_the_wordmark(client):
    """A section you can only reach by clicking the logo is a section people do
    not know is there."""
    for page in ("/", "/classroom/1", "/library", "/queue"):
        html = client.get(page).text
        nav = html[html.index('<nav class="snav">'):html.index("</nav>")]
        assert 'href="/"' in nav, f"{page} has no Classrooms entry in its nav"


# ------------------------------------------------------------ the book list --

def test_the_indexed_list_is_paged(client, conn):
    """8,679 books arrived as one 1.1 MB response that took 9.3 seconds to
    build, because the count came from joining chunks and grouping over all
    1.76M rows. The page could not paint until it finished.

    `total_books` is the real size; `books` is one window onto it."""
    from .conftest import make_book, make_chunks
    for i in range(5):
        make_chunks(conn, make_book(conn, f"b{i}", f"Book {i:02d}"), ["text"] * (i + 1))

    first = client.get("/api/books?limit=2").json()
    assert len(first["books"]) == 2
    assert first["total_books"] == 5, "the count must be the library, not the page"
    assert first["limit"] == 2 and first["offset"] == 0

    # total_chunks counts every passage, not the ones on this page.
    assert first["total_chunks"] == 1 + 2 + 3 + 4 + 5

    rest = client.get("/api/books?limit=50&offset=2").json()
    assert [b["id"] for b in first["books"]] + [b["id"] for b in rest["books"]] \
        == [b["id"] for b in client.get("/api/books?limit=50").json()["books"]], \
        "paging must not repeat or skip a row"


def test_the_page_size_is_clamped(client, conn):
    """A limit is a promise about response size, so it cannot be a number the
    caller chooses freely."""
    assert client.get("/api/books?limit=99999").json()["limit"] == 200
    assert client.get("/api/books?limit=0").json()["limit"] == 1
    assert client.get("/api/books?offset=-5").json()["offset"] == 0


def test_pending_is_never_paged(client, conn):
    """It is bounded by work in flight, not by library size, and an upload that
    vanished from the queue until it finished would read as a failed upload."""
    from .conftest import make_book
    for i in range(3):
        make_book(conn, f"p{i}", f"Pending {i}", status="discovered")
    assert len(client.get("/api/books?limit=1").json()["pending"]) == 3


# --------------------------------------------------------- clearing the queue --

def _stuck(conn, source_id, title, status, hours_ago):
    from .conftest import make_book
    book = make_book(conn, source_id, title, status=status)
    conn.execute("UPDATE books SET updated_at = now() - make_interval(hours => %s) "
                 "WHERE id = %s", (hours_ago, book))
    conn.commit()
    return book


def test_clearing_stopped_books_never_touches_the_library(client, conn):
    """The one property that matters. This control sits on a maintenance page,
    and a reader is one mis-click from it -- so `done` books must be untouchable
    from here no matter what span is asked for."""
    from .conftest import make_book, make_chunks
    keeper = make_book(conn, "good", "A Real Book")
    make_chunks(conn, keeper, ["the covenant"])
    conn.execute("UPDATE books SET updated_at = now() - make_interval(hours => 900) "
                 "WHERE id = %s", (keeper,))
    conn.commit()
    _stuck(conn, "f1", "Failed One", "failed", 100)

    r = client.post("/api/books/stuck/clear", json={"older_than_hours": None})
    assert r.status_code == 200
    assert r.json()["deleted"] == 1

    assert conn.execute("SELECT count(*) FROM books WHERE id = %s",
                        (keeper,)).fetchone()[0] == 1, "an indexed book was deleted"
    assert conn.execute("SELECT count(*) FROM chunks WHERE book_id = %s",
                        (keeper,)).fetchone()[0] == 1


def test_the_span_selects_by_when_the_book_stopped(client, conn):
    """Hours are measured against updated_at -- when it gave up -- and computed
    by Postgres, so a browser with a skewed clock cannot widen the blast."""
    _stuck(conn, "old", "Gave up last week", "failed", 200)
    _stuck(conn, "new", "Gave up an hour ago", "needs_ocr", 1)

    assert client.get("/api/books/stuck").json()["total"] == 2
    assert client.get("/api/books/stuck?older_than_hours=24").json()["total"] == 1

    client.post("/api/books/stuck/clear", json={"older_than_hours": 24})
    left = [r[0] for r in conn.execute("SELECT title FROM books ORDER BY title").fetchall()]
    assert left == ["Gave up an hour ago"], "the recent one should have survived"


def test_both_stopped_statuses_are_cleared(client, conn):
    _stuck(conn, "a", "Failed", "failed", 5)
    _stuck(conn, "b", "Needs OCR", "needs_ocr", 5)
    assert client.post("/api/books/stuck/clear", json={}).json()["deleted"] == 2
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 0


# ------------------------------------------------------- invisible controls --

def test_no_row_action_is_rendered_invisible(client):
    """The Index button used to sit under a "Not ready" badge at opacity 0,
    revealed on hover. A transparent element still takes clicks, so clicking
    what looked like a label really did queue the book -- and the "Queued"
    confirmation faded out the moment the pointer left the row. It looked
    broken while working perfectly.

    A control that is invisible but clickable is the specific mistake; this
    pins the CSS that caused it rather than the markup that happened to use it."""
    css = client.get("/static/app.css").text
    assert ".fswap" not in css, "the badge-over-button stack is back"

    # Find the .fact .btn rule and check it does not start transparent.
    import re
    for m in re.finditer(r"\.fact\s+\.btn[^{]*\{([^}]*)\}", css):
        body = m.group(1)
        assert "opacity: 0" not in body.replace(" ", " "), \
            f"a row action starts invisible: {m.group(0)[:80]}"


def test_a_drive_row_offers_index_or_ready_and_nothing_else(client):
    """Two states. "Not ready" described what the app had not done rather than
    what the reader could do about it."""
    html = (STATIC / "library.html").read_text()
    # `>Not ready<`, not the bare words: the comment above statusCell explains
    # why the badge went, and a test that forbids naming the thing you removed
    # is a test against writing down why.
    assert ">Not ready<" not in html
    assert 'class="btn ix"' in html, "the Index control must still be a button"
    assert "badge good\">Ready" in html


def test_indexing_a_book_is_confirmed_first(client):
    """It is a download, a full text extraction and Voyage tokens per passage --
    not a toggle."""
    html = (STATIC / "library.html").read_text()
    ix = html[html.index("const btn = e.target.closest('.ix');"):]
    ix = ix[:ix.index("fetchJSON")]
    assert "confirmDialog" in ix, "the Index button must confirm before queueing"


def test_clearing_does_not_purge_one_book_at_a_time(client, conn, monkeypatch):
    """The bug that made this look broken.

    purge_book commits per row and makes up to two object-store calls each. At
    353 books that was ~700 round trips and over two minutes -- past the
    browser's own timeout -- so the page said "Could not clear them" about a
    deletion that had already succeeded. Destructive work finishing after the
    UI reports failure is the worst version of slow.

    Asserting on the SHAPE rather than the clock: a timing test against a
    hosted database measures the weather. The single-book purge is the right
    call for the one book someone clicks and the wrong one for a batch, so the
    test is simply that a clear never reaches for it."""
    from library_rag import ingest

    for i in range(25):
        _stuck(conn, f"s{i}", f"Stopped {i}", "failed", 5)

    singles = []
    real_single = ingest.purge_book

    def counted(conn_, book_id):
        singles.append(book_id)
        return real_single(conn_, book_id)

    monkeypatch.setattr(ingest, "purge_book", counted)
    out = client.post("/api/books/stuck/clear", json={})

    assert out.json()["deleted"] == 25
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 0
    assert singles == [], (
        f"purge_book was called {len(singles)} times -- this is the per-row "
        f"path whose round trips outlasted the browser's timeout"
    )


def test_the_clear_failure_path_checks_what_is_left(client):
    """A client timeout is not evidence the work did not happen, and this call
    deletes things. The page must ask what remains instead of reporting the
    request."""
    html = (STATIC / "queue.html").read_text()
    catch = html[html.index("async function clearStuck"):]
    catch = catch[catch.index("} catch"):]
    assert "countStuck(hours)" in catch, (
        "the failure path must re-check the count rather than assume failure")


def test_the_librarian_lives_on_the_bookshelf_not_over_the_conversation(client):
    """The tutor may read only this shelf; the librarian reads all 8,676 books.
    Stacking them in one column made them look like two halves of one feature.
    The librarian is reached from the panel it fills, as a dialog."""
    html = (STATIC / "classroom.html").read_text()
    # Search for the closing tag AFTER the rail opens: the sidebar is also an
    # <aside> and it comes first, so a bare index() sliced backwards to nothing
    # and the assertions below all passed against an empty string.
    start = html.index('<aside class="rail"')
    shelf = html[start:html.index("</aside>", start)]
    assert ">Bookshelf<" in shelf
    assert 'id="asklib"' in shelf, "the librarian must be reachable from the shelf"
    assert 'id="addbooks"' in shelf

    # ...and it is a dialog you can close, not a panel wedged above the chat.
    assert 'id="libmodal"' in html
    chat = html[html.index('<div class="chatcol">'):html.index('<aside class="rail"')]
    assert "librarian" not in chat.lower(), "the librarian is back over the conversation"


def test_every_phase_style_the_librarian_renders_exists(client):
    """The phase list is built by classroom.html and styled in app.css, and the
    two live in different files -- so a rewrite of one section silently dropped
    .phases/.ph and the trail rendered as a run-on sentence. Nothing else in
    the suite looks at whether a class has any rules."""
    css = client.get("/static/app.css").text
    for sel in (".phases", ".ph.done", ".ph.now", ".ph .dot", ".ph .det",
                ".rawtrail", ".thinnote", ".libmark", ".pick"):
        assert sel in css, f"{sel} is rendered but has no styles"


def test_indexing_a_book_never_redraws_the_whole_list(client):
    """Clicking Index used to call open(current) 1.2s later, and the poller
    called it again every 3 seconds while the book worked -- each one replacing
    every row in the folder. In a 4,032-file folder that reads as the page
    reloading itself: scroll position, hover and any in-flight click all go.

    A status is one cell. It gets written into the row that owns it."""
    html = (STATIC / "library.html").read_text()

    start = html.index("const btn = e.target.closest('.ix');")
    handler = html[start:html.index("// ---", start)] if "// ---" in html[start:] \
        else html[start:start + 2500]
    assert "open(current)" not in handler, \
        "the Index click still reloads the folder"
    assert "btn.closest('.fact')" in handler, \
        "the click must update its own cell"

    # And the poller patches, falling back to a redraw only when the row SET
    # changed -- which is the one case a patch cannot express.
    assert "function patchStatuses" in html
    poll = html[html.index("function schedulePoll"):]
    poll = poll[:poll.index("\n}")]
    assert "patchStatuses" in poll and "render(" in poll, \
        "the poll must patch first and redraw only as a fallback"


def test_every_drive_row_can_be_addressed_on_its_own(client):
    """patchStatuses finds a row by file id. Without a stable handle on the
    element there is no way to update one row, and the only option left is
    replacing all of them."""
    html = (STATIC / "library.html").read_text()
    row = html[html.index("function fileRow"):]
    row = row[:row.index("\n}")]
    assert 'data-file="${esc(f.file_id)}"' in row


def test_the_book_count_is_typed_and_bounded(client):
    """A free number with a stated ceiling, not a clamp. Silently turning 50
    into 20 answers a different question from the one asked and gives the reader
    no way to notice."""
    html = (STATIC / "classroom.html").read_text()
    assert 'id="lc" type="number"' in html
    assert 'min="1"' in html and 'max="20"' in html
    assert "const MAX_BOOKS = 20" in html

    fn = html[html.index("function readCount()"):]
    fn = fn[:fn.index("\n}")]
    assert "Number.isInteger" in fn, "a decimal or blank must be refused"
    assert "MAX_BOOKS" in fn and "return null" not in fn.split("problem")[0]

    # The submit path must refuse rather than send a bad count.
    sub = html[html.index("$('#lf').addEventListener('submit'"):]
    sub = sub[:sub.index("\n});")]
    assert "if (count === null)" in sub, "an invalid count must not start a run"


def test_removing_a_book_from_a_shelf_is_confirmed(client):
    """The cost of this one is invisible: the shelf just gets shorter, and what
    actually changed is every answer from then on. Every other destructive
    control in the app asks first; this one is the easiest to hit by accident,
    because the × sits on the card you are reading."""
    html = (STATIC / "classroom.html").read_text()
    handler = html[html.index("$('#shelf').addEventListener('click'"):]
    handler = handler[:handler.index("\n});")]

    assert "confirmDialog" in handler, "removal must be confirmed"
    # And the confirmation has to come BEFORE the request, not after.
    assert handler.index("confirmDialog") < handler.index("method: 'DELETE'"), \
        "the book is deleted before the reader is asked"
    assert "book.title" in handler, "the dialog must name the book"


def test_recommended_books_link_to_their_original_pdf(client):
    """The librarian quotes one passage from a book that may run 400 pages.
    Deciding on that alone means trusting a sentence; the link is how you check
    it. Both card types carry one -- an indexed book through our own route, a
    Drive pick straight to Drive, since it has no book row yet."""
    room = (STATIC / "classroom.html").read_text()
    card = room[room.index("function pickCard"):]
    card = card[:card.index("\n}")]
    assert "/api/books/${b.book_id}/source" in card, "indexed picks need a link out"
    assert "b.url" in card, "a Drive pick has no book row; it must link to Drive"
    assert 'target="_blank"' in card, "opening in place would lose the shortlist"

    # The tutor's suggestions are the same card and matter more: it has NOT read
    # those books, so the PDF is the only way to check one before shelving it.
    tut = (STATIC / "tutor.js").read_text()
    sug = tut[tut.index("function paintSuggestions"):]
    sug = sug[:sug.index("\n}")]
    assert "/api/books/${b.book_id}/source" in sug


def test_the_pdf_route_is_reachable_and_404s_cleanly(client, conn):
    """The link points at a route that already existed for the citation panel.
    An unknown id must be a 404 that names it, not a 500."""
    r = client.get("/api/books/999999/pdf")
    assert r.status_code == 404
    assert "999999" in r.json()["detail"]


def test_open_in_drive_goes_to_drive_not_our_copy(client, conn):
    """The reader wants the whole book where it lives, with its folder one click
    away. /pdf serves OUR bytes -- right for a citation landing on page 87,
    wrong for "let me look at this book before I shelve it"."""
    from .conftest import make_book
    book = make_book(conn, "drive-file-1", "A Drive Book")
    conn.execute("""INSERT INTO drive_files (file_id, name, mime_type, web_view_link)
                    VALUES ('drive-file-1', 'A Drive Book', 'application/pdf',
                            'https://drive.google.com/file/d/xyz/view')""")
    conn.commit()

    r = client.get(f"/api/books/{book}/source", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "https://drive.google.com/file/d/xyz/view"


def test_an_upload_falls_back_to_our_copy(client, conn):
    """An upload has no Drive file. A link that works for 8,681 books and dead-
    ends on four is worse than one that always opens something."""
    from .conftest import make_book
    book = make_book(conn, "upload:abc", "An Uploaded Book")
    conn.execute("UPDATE books SET source = 'upload' WHERE id = %s", (book,))
    conn.commit()

    r = client.get(f"/api/books/{book}/source", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"/api/books/{book}/pdf"


def test_the_source_route_404s_on_an_unknown_book(client, conn):
    assert client.get("/api/books/999999/source").status_code == 404
