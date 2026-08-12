"""The sign-in gate.

Zero network: Google's token verification is stubbed, because the thing worth
testing is what THIS code does with the answer, not whether google-auth can
check a signature. The one call we must never stub away is the audience check --
see test_a_token_minted_for_another_app_is_refused.
"""
import pytest
from fastapi.testclient import TestClient

from library_rag import config, db
from library_rag.web import api, auth

CLIENT_ID = "test-client-id.apps.googleusercontent.com"


@pytest.fixture
def client(conn, test_database_url, monkeypatch):
    """A TestClient whose requests hit the throwaway test database.

    Same reasoning as tests/test_upload.py: api.py calls db.get_conn() with no
    argument, which reads config.DATABASE_URL at call time, so pointing that at
    the test database is what keeps these routes off the real one.
    """
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    return TestClient(api.app)


@pytest.fixture
def allowlisted(conn):
    auth.add_user(conn, "allowed@example.com", "test")
    return conn


@pytest.fixture
def auth_on(monkeypatch):
    """Turn the gate on for one test. config is read at call time by
    auth_enabled(), so monkeypatching the module attribute is enough."""
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", CLIENT_ID)
    return CLIENT_ID


@pytest.fixture
def google(monkeypatch):
    """Stub Google's verifier. Records what audience it was asked to check, so
    a test can assert we passed one at all."""
    calls = {}

    def fake_verify(credential, request, audience):
        calls["audience"] = audience
        if credential == "bad":
            raise ValueError("Token has wrong signature")
        if credential == "unverified":
            return {"email": "someone@example.com", "email_verified": False}
        return {"email": credential, "email_verified": True}

    monkeypatch.setattr(auth.id_token, "verify_oauth2_token", fake_verify)
    return calls


# ------------------------------------------------------------ the gate off --

def test_with_no_client_id_configured_everything_stays_open(client):
    """Local development and the existing suite must be unaffected. This is the
    default, so if it ever breaks, it breaks everything."""
    assert config.auth_enabled() is False
    assert client.get("/api/bible/books").status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200


# ------------------------------------------------------------- the gate on --

def test_a_browser_is_redirected_to_the_login_page(auth_on, client):
    r = client.get("/bible", headers={"accept": "text/html"}, follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/login"


def test_a_fetch_gets_401_json_not_a_login_page(auth_on, client):
    """A 302 to an HTML page would reach the frontend as a JSON parse error,
    which reads as a bug rather than as "your session ended"."""
    r = client.get("/api/bible/books", headers={"accept": "application/json"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Not signed in."


@pytest.mark.parametrize("path", ["/login", "/healthz", "/static/app.css",
                                  "/api/auth/config"])
def test_the_open_paths_stay_reachable(auth_on, client, path):
    """Otherwise signing in would require being signed in."""
    assert client.get(path).status_code == 200


def test_every_mutating_route_is_closed_by_default(auth_on, client):
    """The eight routes that change state, all denied without a session.

    Written as a sweep over the app's own route table rather than a hand-listed
    set: a route added next month is covered without anyone remembering to add
    it here, which is the same property the middleware itself has.
    """
    checked = 0
    for route in api.app.routes:
        methods = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        if not (methods & {"POST", "DELETE", "PUT", "PATCH"}):
            continue
        if path in auth.OPEN_PATHS:
            continue
        # Substitute anything for path params; it must not get far enough to matter.
        url = path.replace("{book_id}", "1").replace("{folder_id}", "x")
        method = "DELETE" if "DELETE" in methods else "POST"
        r = client.request(method, url, headers={"accept": "application/json"})
        assert r.status_code == 401, f"{method} {path} returned {r.status_code}"
        checked += 1
    assert checked >= 8, f"expected to sweep at least 8 mutating routes, saw {checked}"


# -------------------------------------------------------------- signing in --

def test_an_allowlisted_account_gets_a_session(auth_on, google, allowlisted, client):
    r = client.post("/api/auth/google", json={"credential": "allowed@example.com"})
    assert r.status_code == 200
    assert r.json()["email"] == "allowed@example.com"
    # And the session actually works on a gated route.
    assert client.get("/api/bible/books").status_code == 200
    assert client.get("/api/auth/me").json()["email"] == "allowed@example.com"


def test_an_account_not_on_the_list_is_refused(auth_on, google, allowlisted, client):
    """403, not 401: we know exactly who they are and they still may not in.
    The message names the address so they can ask for the right one."""
    r = client.post("/api/auth/google", json={"credential": "stranger@example.com"})
    assert r.status_code == 403
    assert "stranger@example.com" in r.json()["detail"]
    assert client.get("/api/bible/books").status_code == 401


def test_a_token_google_cannot_verify_is_refused(auth_on, google, allowlisted, client):
    r = client.post("/api/auth/google", json={"credential": "bad"})
    assert r.status_code == 401
    assert client.get("/api/bible/books").status_code == 401


def test_an_unverified_google_email_is_refused(auth_on, google, conn, client):
    """Google will assert an unverified address on some account types, and an
    unverified email is not evidence of anything."""
    auth.add_user(conn, "someone@example.com")
    r = client.post("/api/auth/google", json={"credential": "unverified"})
    assert r.status_code == 401


def test_a_token_minted_for_another_app_is_refused(auth_on, google, allowlisted, client):
    """The audience check, which is the one that is easy to omit and fatal.

    Without passing `audience`, any valid Google ID token verifies -- including
    one an attacker obtained by signing in to a completely different app. This
    asserts we hand google-auth OUR client id to check against.
    """
    client.post("/api/auth/google", json={"credential": "allowed@example.com"})
    assert google["audience"] == CLIENT_ID


def test_the_email_is_matched_case_insensitively(auth_on, google, conn, client):
    """'Allowed@Example.com' and 'allowed@example.com' are one Google account;
    a capital letter must never be the reason someone is locked out."""
    auth.add_user(conn, "MiXeD@Example.COM")
    r = client.post("/api/auth/google", json={"credential": "mixed@example.com"})
    assert r.status_code == 200


def test_signing_out_closes_the_gate_again(auth_on, google, allowlisted, client):
    client.post("/api/auth/google", json={"credential": "allowed@example.com"})
    assert client.get("/api/bible/books").status_code == 200
    client.get("/logout", follow_redirects=False)
    assert client.get("/api/bible/books").status_code == 401


def test_an_expired_session_stops_working(auth_on, google, allowlisted, client, monkeypatch):
    """The cookie carries its own issue time and the server re-checks it, so a
    cookie captured off the wire cannot be replayed forever by something that
    is not a browser honouring max-age."""
    client.post("/api/auth/google", json={"credential": "allowed@example.com"})
    assert client.get("/api/bible/books").status_code == 200
    monkeypatch.setattr(config, "SESSION_MAX_AGE_SECONDS", -1)
    assert client.get("/api/bible/books").status_code == 401


# --------------------------------------------------------------- allowlist --

def test_the_allowlist_round_trips(conn):
    assert auth.add_user(conn, "a@example.com", "first") is True
    assert auth.add_user(conn, "a@example.com") is False       # already there
    assert auth.is_allowed(conn, "a@example.com") is True
    assert auth.is_allowed(conn, "A@EXAMPLE.COM") is True      # case-folded
    assert auth.is_allowed(conn, "b@example.com") is False
    assert auth.remove_user(conn, "a@example.com") is True
    assert auth.remove_user(conn, "a@example.com") is False
    assert auth.is_allowed(conn, "a@example.com") is False


def test_addresses_are_stored_lowercased(conn):
    """The table CHECKs it too, so a row written by hand in the Supabase editor
    with a capital letter is rejected rather than silently never matching."""
    auth.add_user(conn, "  MiXeD@Example.COM  ")
    assert [r[0] for r in auth.list_users(conn)] == ["mixed@example.com"]
    with pytest.raises(Exception):
        conn.execute("INSERT INTO allowed_users (email) VALUES ('Nope@X.com')")


def test_an_empty_allowlist_locks_everyone_out(auth_on, google, conn, client):
    """The intended default. A seeded address would be a backdoor that survives
    into every deployment."""
    assert auth.list_users(conn) == []
    assert client.post("/api/auth/google",
                  json={"credential": "anyone@example.com"}).status_code == 403


def test_healthz_touches_no_database(auth_on, client, monkeypatch):
    """Cloud Run probes this. If it needed Postgres, a momentary database blip
    would fail the health check and turn into a restart loop."""
    def explode(*a, **k):
        raise AssertionError("/healthz must not open a database connection")
    monkeypatch.setattr(db, "get_conn", explode)
    assert client.get("/healthz").json() == {"ok": True}
