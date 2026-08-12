"""
The Drive browsing agent: drive-wide search, folder browsing, the
already-indexed cross-reference, the recommend output channel, and the
add-to-library route.

Drive is a fake service throughout -- these assertions are about our query
construction and our bookkeeping, not about Google's. The Anthropic client is
scripted for the same reason (see tests/test_research.py).
"""
import json
import types

import pytest
from fastapi.testclient import TestClient

from library_rag import config, db
from library_rag.drive import client as drive_client
from library_rag.exploration import loop as browse_loop
from library_rag.exploration import tools
from library_rag.web import api

FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"


# ------------------------------------------------------------------ fakes --

def _pdf(fid, name, size_mb=2.0, md5=None):
    return {
        "id": fid, "name": name, "mimeType": PDF_MIME,
        "size": str(int(size_mb * 1024 * 1024)),
        "md5Checksum": md5 or f"md5-{fid}",
        "webViewLink": f"https://drive.google.com/file/d/{fid}/view",
    }


class FakeFiles:
    """Records every files().list(q=...) so tests can assert on the query we
    built, which is the part of the Drive call that is actually ours."""

    def __init__(self, listing=None, by_id=None):
        self.listing = listing or []
        self.by_id = by_id or {}
        self.queries = []

    def list(self, **kw):
        self.queries.append(kw["q"])
        return types.SimpleNamespace(
            execute=lambda: {"files": list(self.listing), "nextPageToken": None}
        )

    def get(self, **kw):
        item = self.by_id.get(kw["fileId"], {})
        return types.SimpleNamespace(execute=lambda: item)


class FakeService:
    def __init__(self, files):
        self._files = files

    def files(self):
        return self._files


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


@pytest.fixture
def drive(monkeypatch):
    """A fake Drive whose listing and get() the test controls."""
    files = FakeFiles()
    monkeypatch.setattr(drive_client, "build_service", lambda: FakeService(files))
    monkeypatch.setattr(
        drive_client, "get_folder_name",
        lambda service, fid: {"name": "Books", "webViewLink": "u"},
    )
    return files


# ------------------------------------------------------------ search_files --

def test_search_builds_a_pdf_scoped_query(drive):
    drive_client.search_files(FakeService(drive), "Romans", limit=5)
    q = drive.queries[0]
    assert "name contains 'Romans'" in q
    assert f"mimeType = '{PDF_MIME}'" in q
    assert "trashed = false" in q


def test_search_escapes_quotes_in_the_query():
    """Drive's `q` grammar delimits literals with single quotes. An apostrophe
    in a title -- "Paul's Letters" -- would end the literal early and 400."""
    assert drive_client._escape_q("Paul's Letters") == "Paul\\'s Letters"
    assert drive_client._escape_q("back\\slash") == "back\\\\slash"


def test_search_passes_the_term_through_verbatim(drive):
    """Drive matches a WORD PREFIX, not a substring -- 'arable' finds nothing
    inside "parable". Lowercasing or stemming the term here would change what
    matches, so the caller's word must reach the query untouched."""
    drive_client.search_files(FakeService(drive), "Parab", limit=5)
    assert "name contains 'Parab'" in drive.queries[0]


def test_search_respects_its_limit(drive):
    drive.listing = [_pdf(f"f{i}", f"Book {i}") for i in range(50)]
    got = drive_client.search_files(FakeService(drive), "Book", limit=7)
    assert len(got) == 7


# ------------------------------------------------- the indexed cross-ref --

def test_a_drive_book_we_already_hold_comes_back_indexed(conn, drive):
    db.upsert_book(conn, "owned", "Romans Study.pdf", "md5-owned", 100, source="drive")
    conn.commit()
    drive.listing = [_pdf("owned", "Romans Study.pdf"), _pdf("new", "Romans Deeper.pdf")]

    result = tools.search_drive(conn, "Romans")
    by_id = {m["file_id"]: m for m in result["matches"]}

    assert by_id["owned"]["indexed"] is True
    assert by_id["owned"]["book_id"] == db.fetch_book_by_source_id(conn, "owned")["id"]
    assert by_id["new"]["indexed"] is False
    assert "book_id" not in by_id["new"], (
        "an absent book_id must be unambiguous, not a null to interpret"
    )


def test_an_upload_never_counts_as_an_indexed_drive_book(conn, drive):
    """indexed_ids is keyed on source_id, which only means a Drive file id when
    source='drive'. An upload's source_id is 'upload:<md5>' and must not shadow
    a Drive file that happens to collide."""
    db.upsert_book(conn, "upload:abc", "Some Upload.pdf", "abc", 10, source="upload")
    conn.commit()
    assert tools.indexed_ids(conn) == {}


# ---------------------------------------------------------- browse_folder --

def test_browse_reports_truncation_and_the_true_total(conn, drive):
    drive.listing = [_pdf(f"f{i}", f"Book {i:03d}") for i in range(80)]

    result = tools.browse_folder(conn, "folder", limit=10)

    assert len(result["pdfs"]) == 10
    assert result["total_pdfs"] == 80
    assert result["truncated"] is True, (
        "a silently-cut list reads as 'that is all there is'"
    )


def test_browse_filters_by_name_within_the_folder(conn, drive):
    drive.listing = [
        _pdf("a", "Romans.pdf"), _pdf("b", "Galatians.pdf"), _pdf("c", "romans-2.pdf"),
    ]
    result = tools.browse_folder(conn, "folder", name_contains="romans")

    assert {p["file_id"] for p in result["pdfs"]} == {"a", "c"}, "match is case-insensitive"
    assert result["total_pdfs"] == 2
    assert result["truncated"] is False


def test_browse_separates_subfolders_from_junk(conn, drive):
    drive.listing = [
        _pdf("a", "Romans.pdf"),
        {"id": "sub", "name": "Commentaries", "mimeType": FOLDER_MIME,
         "webViewLink": "u"},
        {"id": "j", "name": ".DS_Store", "mimeType": "application/octet-stream",
         "size": "6"},
    ]
    result = tools.browse_folder(conn, "folder")

    assert [s["name"] for s in result["subfolders"]] == ["Commentaries"]
    assert [p["title"] for p in result["pdfs"]] == ["Romans.pdf"]
    assert result["junk_files"] == 1


def test_browse_caches_by_folder_id(conn, drive):
    drive.listing = [_pdf("a", "Romans.pdf")]
    tools.browse_folder(conn, "folder")
    n_after_first = len(drive.queries)
    tools.browse_folder(conn, "folder")

    assert len(drive.queries) == n_after_first, "second browse should hit the cache"
    assert config.DRIVE_CACHE_FILE.exists()


# -------------------------------------------------------------- recommend --

def test_recommend_enriches_picks_from_what_the_run_saw(conn, drive):
    drive.listing = [_pdf("a", "Romans.pdf", size_mb=3.5)]
    seen = {}
    browse_loop._remember(seen, tools.search_drive(conn, "Romans")["matches"])

    out = tools.recommend(conn, [{"file_id": "a", "why": "verse-by-verse"}], seen)
    pick = out["recommendations"][0]

    assert pick["title"] == "Romans.pdf"
    assert pick["size_mb"] == 3.5
    assert pick["url"].endswith("/view")
    assert pick["why"] == "verse-by-verse"
    assert pick["indexed"] is False


def test_recommend_flags_a_file_id_the_agent_never_saw(conn):
    """A hallucinated id must surface, not render as a dead Drive link."""
    out = tools.recommend(conn, [{"file_id": "made-up", "why": "trust me"}], seen={})
    pick = out["recommendations"][0]

    assert pick["unknown"] is True
    assert "no such file_id" in pick["error"]
    assert "url" not in pick


# ------------------------------------------------------------- the loop --

def test_the_loop_streams_tools_then_recommendations_then_the_answer(conn, drive):
    drive.listing = [_pdf("a", "Romans.pdf")]
    client = ScriptedClient([
        [_text("Let me look."), _tool("search_drive", "t1", query="Romans")],
        [_tool("recommend", "t2", picks=[{"file_id": "a", "why": "fits"}])],
        [_text("Start with Romans.")],
    ])

    events = list(browse_loop.run("paul's letters", conn, client=client))
    kinds = [e["type"] for e in events]

    assert kinds == [
        "thinking", "tool", "results",
        "tool", "results", "recommendations",
        "answer", "done",
    ]
    assert events[-2]["text"] == "Start with Romans."
    assert events[-1]["recommendations"][0]["title"] == "Romans.pdf"


def test_a_tool_error_is_reported_to_the_model_instead_of_killing_the_run(conn,
                                                                         monkeypatch):
    """The model can recover from a bad call if it is told. Raising out of the
    loop gives it no chance and loses everything found so far."""
    def boom(*a, **kw):
        raise RuntimeError("drive is down")

    monkeypatch.setattr(tools, "search_drive", boom)
    client = ScriptedClient([
        [_tool("search_drive", "t1", query="Romans")],
        [_text("Could not reach Drive.")],
    ])

    events = list(browse_loop.run("anything", conn, client=client))

    assert [e["type"] for e in events] == ["tool", "tool_error", "answer", "done"]

    # Searched rather than indexed: ScriptedClient records the same list object
    # every call, and the loop keeps appending to it.
    results = [
        block
        for m in client.seen[-1]
        for block in (m["content"] if isinstance(m["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert "drive is down" in json.loads(results[0]["content"])["error"]


def test_the_loop_stops_at_its_iteration_leash(conn, drive):
    drive.listing = [_pdf("a", "Romans.pdf")]
    client = ScriptedClient(
        [[_tool("search_drive", f"t{i}", query="Romans")] for i in range(10)]
    )

    events = list(browse_loop.run("x", conn, client=client, max_iterations=3))

    assert events[-1]["exhausted"] is True
    assert events[-1]["iterations"] == 3


# ------------------------------------------------ POST /api/library/drive --

@pytest.fixture
def web(conn, test_database_url, monkeypatch, drive):
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    monkeypatch.setattr(config, "VOYAGE_API_KEY", "test-key-not-used")
    drained = []
    monkeypatch.setattr(api, "_drain_queue", drained.append)
    c = TestClient(api.app)
    c.drained = drained
    c.drive = drive
    return c


def test_adding_a_drive_book_queues_it_and_drains_the_drive_queue(web, conn):
    web.drive.by_id = {"a": _pdf("a", "Romans.pdf", size_mb=4.0, md5="realmd5")}

    r = web.post("/api/library/drive", json={"file_ids": ["a"]})

    assert r.status_code == 200
    assert r.json()["added"][0]["status"] == "discovered"
    row = db.fetch_book_by_source_id(conn, "a")
    assert row["source"] == "drive"
    assert row["md5"] == "realmd5", "md5 must come from Drive, not the request"
    assert web.drained == ["drive"], (
        "a drain scoped to uploads leaves the book at 'discovered' forever"
    )


def test_adding_a_book_already_held_does_not_requeue_it(web, conn):
    db.upsert_book(conn, "a", "Romans.pdf", "md5-a", 100, source="drive")
    conn.execute("UPDATE books SET status = 'done' WHERE source_id = 'a'")
    conn.commit()

    r = web.post("/api/library/drive", json={"file_ids": ["a"]})

    assert r.json()["added"] == []
    assert r.json()["already_indexed"][0]["file_id"] == "a"
    assert db.fetch_book_by_source_id(conn, "a")["status"] == "done"
    assert web.drained == [], "nothing to do, so nothing should have been started"


def test_adding_a_non_pdf_is_refused(web):
    web.drive.by_id = {
        "d": {"id": "d", "name": "notes", "mimeType": "application/vnd.google-apps.document"}
    }
    r = web.post("/api/library/drive", json={"file_ids": ["d"]})

    assert r.status_code == 400
    assert web.drained == []


def test_the_add_route_bounds_how_many_books_one_request_can_queue(web):
    r = web.post("/api/library/drive", json={"file_ids": [f"f{i}" for i in range(21)]})
    assert r.status_code == 422, "an unbounded list puts the whole drive in flight"


def test_a_drive_auth_failure_reports_how_to_fix_it(web, monkeypatch):
    """DriveAuthError carries instructions ("delete token.json and re-run to
    re-authorize"). Falling through to FastAPI's default handler turned that
    into a bare "500 Internal Server Error" -- a thirty-second fix presented as
    an unexplained crash. Seen live when the OAuth refresh token expired."""
    def expired():
        raise drive_client.DriveAuthError(
            "token.json exists but refresh failed. Fix: delete token.json."
        )

    monkeypatch.setattr(drive_client, "build_service", expired)
    r = web.post("/api/library/drive", json={"file_ids": ["a"]})

    assert r.status_code == 503, "an expired credential is not a server fault"
    assert "delete token.json" in r.json()["detail"]


# --------------------------------------- POST /api/drive/folders/…/index --

MB = 1024 * 1024
PDF_MIME = "application/pdf"


def _mirror_tree(conn):
    """lib/ holds One.pdf and sub/, which holds Two.pdf and Three.pdf --
    10 MB of PDFs across two levels, with subtree_bytes as a sync leaves it."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO drive_files (file_id, name, mime_type, parent_id,"
            " size_bytes, md5, subtree_bytes) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [
                ("lib", "Library", db.FOLDER_MIME, None, None, None, 10 * MB),
                ("sub", "Sub", db.FOLDER_MIME, "lib", None, None, 8 * MB),
                ("f1", "One.pdf", PDF_MIME, "lib", 2 * MB, "md5-f1", None),
                ("f2", "Two.pdf", PDF_MIME, "sub", 3 * MB, "md5-f2", None),
                ("f3", "Three.pdf", PDF_MIME, "sub", 5 * MB, "md5-f3", None),
            ],
        )
    conn.commit()


def test_indexing_a_folder_queues_its_whole_subtree(web, conn):
    _mirror_tree(conn)

    r = web.post("/api/drive/folders/lib/index")

    assert r.status_code == 200
    assert r.json() == {"folder": "Library", "queued": 3, "already_indexed": 0}
    book = db.fetch_book_by_source_id(conn, "f3")
    assert book["status"] == "discovered"
    assert book["md5"] == "md5-f3", "metadata must come from the mirror row"
    assert web.drained == ["drive"]


def test_folder_indexing_leaves_held_books_alone(web, conn):
    _mirror_tree(conn)
    db.upsert_book(conn, "f2", "Two.pdf", "md5-f2", 3 * MB, source="drive")
    conn.execute("UPDATE books SET status = 'done' WHERE source_id = 'f2'")
    conn.commit()

    r = web.post("/api/drive/folders/lib/index")

    assert r.json()["queued"] == 2
    assert r.json()["already_indexed"] == 1
    assert db.fetch_book_by_source_id(conn, "f2")["status"] == "done", (
        "bulk-queueing must never reset a book that is done or mid-flight"
    )


def test_an_oversized_folder_is_refused_and_nothing_is_queued(web, conn, monkeypatch):
    """The frontend disables the button, but THIS is the guard: one request
    naming the root would otherwise put the whole drive into flight."""
    _mirror_tree(conn)
    monkeypatch.setattr(config, "FOLDER_INDEX_LIMIT_BYTES", 9 * MB)

    r = web.post("/api/drive/folders/lib/index")

    assert r.status_code == 413
    assert "bulk-index limit" in r.json()["detail"]
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 0
    assert web.drained == []


def test_a_subfolder_under_the_limit_still_qualifies(web, conn, monkeypatch):
    _mirror_tree(conn)
    monkeypatch.setattr(config, "FOLDER_INDEX_LIMIT_BYTES", 9 * MB)

    r = web.post("/api/drive/folders/sub/index")

    assert r.status_code == 200
    assert r.json()["queued"] == 2


def test_folder_indexing_refuses_files_and_unknown_ids(web, conn):
    _mirror_tree(conn)
    assert web.post("/api/drive/folders/f1/index").status_code == 400
    assert web.post("/api/drive/folders/nope/index").status_code == 404
    assert web.drained == []


def test_folder_indexing_is_idempotent(web, conn):
    _mirror_tree(conn)
    web.post("/api/drive/folders/lib/index")

    r = web.post("/api/drive/folders/lib/index")

    assert r.json() == {"folder": "Library", "queued": 0, "already_indexed": 3}
    assert web.drained == ["drive"], (
        "the second click queued nothing, so it must not start a second drain"
    )


def test_children_payload_carries_the_limit_and_the_current_folder(web, conn):
    """What the page's gate runs on: the ceiling, and the viewed folder's own
    size for the breadcrumb's index-all button."""
    _mirror_tree(conn)

    d = web.get("/api/drive/children", params={"parent": "sub"}).json()

    assert d["folder_limit_mb"] == config.FOLDER_INDEX_LIMIT_BYTES // MB
    assert d["folder"]["file_id"] == "sub"
    assert d["folder"]["size_mb"] == 8.0


# ------------------------------------------- POST /api/drive/files/index --

def test_indexing_a_list_of_results_queues_the_new_ones(web, conn):
    """The "index all results" of a search: held books are skipped, folders
    and unknown ids are ignored, everything else queues from mirror metadata."""
    _mirror_tree(conn)
    db.upsert_book(conn, "f1", "One.pdf", "md5-f1", 2 * MB, source="drive")
    conn.commit()

    r = web.post("/api/drive/files/index",
                 json={"file_ids": ["f1", "f2", "f3", "sub", "nope"]})

    assert r.status_code == 200
    assert r.json() == {"queued": 2, "already_indexed": 1}
    assert db.fetch_book_by_source_id(conn, "f2")["status"] == "discovered"
    assert web.drained == ["drive"]


def test_result_indexing_gates_on_the_unindexed_bytes_only(web, conn, monkeypatch):
    """40 results where 39 are already yours is a small job, not a big one --
    the ceiling applies to what would actually be queued."""
    _mirror_tree(conn)
    monkeypatch.setattr(config, "FOLDER_INDEX_LIMIT_BYTES", 6 * MB)
    db.upsert_book(conn, "f3", "Three.pdf", "md5-f3", 5 * MB, source="drive")
    conn.commit()

    # f1 + f2 pending = 5 MB, under the 6 MB limit even though all three = 10.
    r = web.post("/api/drive/files/index", json={"file_ids": ["f1", "f2", "f3"]})
    assert r.status_code == 200
    assert r.json()["queued"] == 2


def test_oversized_result_lists_are_refused_with_nothing_queued(web, conn, monkeypatch):
    _mirror_tree(conn)
    monkeypatch.setattr(config, "FOLDER_INDEX_LIMIT_BYTES", 4 * MB)

    r = web.post("/api/drive/files/index", json={"file_ids": ["f1", "f2", "f3"]})

    assert r.status_code == 413
    assert "bulk-index limit" in r.json()["detail"]
    assert conn.execute("SELECT count(*) FROM books").fetchone()[0] == 0
    assert web.drained == []


def test_result_indexing_bounds_the_id_list(web):
    r = web.post("/api/drive/files/index",
                 json={"file_ids": [f"f{i}" for i in range(101)]})
    assert r.status_code == 422


# ------------------------------------------------------------ drive auth --

def test_auth_status_never_starts_a_consent_flow(web, monkeypatch, tmp_path):
    """The page polls this. A status check that opened a browser tab would be a
    booby trap, so it must never touch the interactive flow."""
    called = []
    monkeypatch.setattr(drive_client, "auth_url",
                        lambda *a, **k: called.append(1) or ("http://x", "v"))
    # BOTH files, not just the client secrets. Isolating only CREDENTIALS_FILE
    # left this reading the developer's real token.json from the repo root --
    # harmless while the code checked the client first, and an instant failure
    # once it checked the token first. A test that depends on whether the
    # machine running it happens to be connected to Drive is not a test.
    monkeypatch.setattr(drive_client, "CREDENTIALS_FILE", str(tmp_path / "nope.json"))
    monkeypatch.setattr(drive_client, "TOKEN_FILE", str(tmp_path / "no-token.json"))

    r = web.get("/api/drive/auth/status")
    assert r.status_code == 200
    # No token AND no way to obtain one -- the state a deployed container is in.
    assert r.json()["reason"] == "cannot_connect"
    assert called == []


def test_auth_start_returns_a_consent_url_and_remembers_the_state(web, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        drive_client, "auth_url",
        lambda redirect_uri, state: (
            seen.update(uri=redirect_uri, state=state) or ("https://g/x", "pkce-verifier")
        ),
    )
    d = web.get("/api/drive/auth/start").json()

    assert d["url"] == "https://g/x"
    assert seen["uri"].endswith("/api/drive/auth/callback"), (
        "the redirect must point back at this app, and must match at exchange"
    )
    # The PKCE verifier is stored WITH the state. It exists only on the Flow
    # instance that built the consent URL, and the exchange runs in a later
    # request with a different instance -- losing it means Google rejects the
    # code with "invalid_grant: Missing code verifier".
    assert api._auth_flows.get(seen["state"]) == (seen["uri"], "pkce-verifier")


def test_the_callback_rejects_a_state_it_never_issued(web, monkeypatch):
    """Without this, anything that can make the browser hit this URL could hand
    us an authorization code of its choosing."""
    exchanged = []
    monkeypatch.setattr(drive_client, "exchange_code",
                        lambda *a, **k: exchanged.append(1))

    r = web.get("/api/drive/auth/callback", params={"state": "forged", "code": "c"})

    assert r.status_code == 400
    assert exchanged == [], "a code from an unknown flow must never be exchanged"


def test_a_completed_callback_exchanges_once_and_burns_the_state(web, monkeypatch):
    monkeypatch.setattr(drive_client, "auth_url",
                        lambda redirect_uri, state: ("https://g/x", "pkce-verifier"))
    calls = []
    monkeypatch.setattr(drive_client, "exchange_code",
                        lambda code, redirect_uri, verifier, connected_by=None:
                            calls.append((code, redirect_uri, verifier, connected_by)))

    web.get("/api/drive/auth/start")
    state = next(iter(api._auth_flows))
    ok = web.get("/api/drive/auth/callback", params={"state": state, "code": "abc"})

    assert ok.status_code == 200
    assert calls[0][0] == "abc"
    assert calls[0][1].endswith("/api/drive/auth/callback")
    assert calls[0][2] == "pkce-verifier", "the exchange must reuse the PKCE verifier"
    # Who reconnected is recorded with the token. Either allowlisted person can
    # do it -- Google revokes these weekly -- so an unattributable shared
    # credential is one nobody owns. "unknown" here because the gate is off in
    # tests, so there is no signed-in user to name.
    assert calls[0][3] == "unknown"
    # Replaying the same link must not exchange again.
    again = web.get("/api/drive/auth/callback", params={"state": state, "code": "abc"})
    assert again.status_code == 400
    assert len(calls) == 1


def test_the_callback_page_never_echoes_the_code(web, monkeypatch):
    """An authorization code rendered into HTML is a credential in the browser
    history, in a screenshot, and in anything that reads the page."""
    monkeypatch.setattr(drive_client, "auth_url",
                        lambda redirect_uri, state: ("https://g/x", "v"))
    monkeypatch.setattr(drive_client, "exchange_code",
                        lambda code, redirect_uri, verifier: None)
    web.get("/api/drive/auth/start")
    state = next(iter(api._auth_flows))

    body = web.get("/api/drive/auth/callback",
                   params={"state": state, "code": "SECRET-CODE"}).text
    assert "SECRET-CODE" not in body
    assert state not in body


def test_a_google_denial_is_reported_without_exchanging(web, monkeypatch):
    calls = []
    monkeypatch.setattr(drive_client, "exchange_code", lambda *a, **k: calls.append(1))
    r = web.get("/api/drive/auth/callback", params={"error": "access_denied"})
    assert r.status_code == 400
    assert "declined" in r.text
    assert calls == []


def test_the_consent_url_asks_for_a_refresh_token(tmp_path, monkeypatch):
    """Builds the real URL -- no network, authorization_url() only formats a
    string. The other auth tests mock this away, which let a dropped
    prompt="consent" pass everything.

    prompt="consent" is the load-bearing one: Google returns a refresh_token
    only on FIRST consent unless consent is forced, so without it a reconnect
    after expiry yields an access token that dies in an hour with nothing to
    renew it -- exactly the failure this flow exists to fix. Dropping it passes
    every other test in this file.

    access_type="offline" is asserted too, though google_auth_oauthlib already
    defaults to it. The assertion is on the OUTCOME, so it holds whether the
    value comes from us or the library, and would catch a future default change.
    """
    import json
    from urllib.parse import parse_qs, urlparse

    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"installed": {
        "client_id": "cid.apps.googleusercontent.com", "client_secret": "shh",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }}))
    monkeypatch.setattr(drive_client, "CREDENTIALS_FILE", str(creds))

    url, verifier = drive_client.auth_url(
        "http://localhost:8000/api/drive/auth/callback", "st8")
    q = parse_qs(urlparse(url).query)

    # PKCE: the URL carries the S256 challenge, and the verifier that answers it
    # comes back for the caller to keep until the exchange.
    assert q["code_challenge_method"] == ["S256"]
    assert verifier and len(verifier) > 20

    assert q["access_type"] == ["offline"]
    assert q["prompt"] == ["consent"], "without this a re-consent returns no refresh token"
    assert q["state"] == ["st8"]
    assert q["redirect_uri"] == ["http://localhost:8000/api/drive/auth/callback"]
    assert "drive.readonly" in q["scope"][0]


def test_exchange_code_applies_the_verifier_before_fetching(tmp_path, monkeypatch):
    """The exchange must set code_verifier on ITS flow, not merely accept the
    argument. Every other auth test mocks exchange_code away, so silently
    dropping the assignment passes all of them -- and fails in production with
    "invalid_grant: Missing code verifier", which is how this was found.

    No network: fetch_token is intercepted and the flow's state inspected.
    """
    import json

    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"installed": {
        "client_id": "cid", "client_secret": "shh",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }}))
    monkeypatch.setattr(drive_client, "CREDENTIALS_FILE", str(creds))
    monkeypatch.setattr(drive_client, "TOKEN_FILE", str(tmp_path / "token.json"))

    captured = {}

    def fake_fetch(self, **kw):
        captured["verifier"] = self.code_verifier
        captured["code"] = kw.get("code")

    monkeypatch.setattr(InstalledAppFlow, "fetch_token", fake_fetch)
    monkeypatch.setattr(
        InstalledAppFlow, "credentials",
        property(lambda self: type("C", (), {"to_json": lambda s: "{}"})()),
    )

    drive_client.exchange_code("the-code", "http://localhost:8000/cb", "the-verifier")

    assert captured["code"] == "the-code"
    assert captured["verifier"] == "the-verifier"
