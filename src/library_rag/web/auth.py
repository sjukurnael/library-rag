"""Google sign-in, and the gate in front of every route.

Two separate questions, answered by two separate things:

  WHO ARE YOU?   Google. The browser gets a signed ID token (a JWT) from
                 Google and hands it to us; we verify Google's signature.
  MAY YOU IN?    The `allowed_users` table. Anyone on earth has a Google
                 account, so proving identity is not authorisation.

This is deliberately NOT Supabase Auth. The app serves server-rendered pages, so
a plain navigation to /bible arrives with no JavaScript having run and no header
to inspect -- a browser-held token cannot gate that. A server-set cookie can.
Supabase hosts the Postgres this reads; nothing here uses its auth API.

It is also NOT the Drive OAuth in drive/client.py. That is the OWNER's
read-only access to their own Drive, with a restricted scope, from a Desktop
client. This is `openid email profile` from a Web client, for whoever is
visiting. Conflating them would ask every visitor to hand over their own Drive.
"""
import time

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

from library_rag import config

# Reachable without a session. Everything else is denied by default -- see
# AuthMiddleware. Kept as an explicit short list rather than a pattern, because
# "which URLs are public" should be readable in one glance.
#
# /healthz is here so Cloud Run's probe never needs a session, and it must not
# touch the database either: a health check that fails when Postgres blinks
# turns a recoverable blip into a restart loop.
# /api/auth/config is open because the login page fetches it BEFORE anyone is
# signed in -- gating it would mean needing a session to learn how to get one.
# It returns only the client id, which is public by design and ships in the HTML.
OPEN_PATHS = frozenset({
    "/login", "/api/auth/google", "/api/auth/config", "/logout", "/healthz",
})
OPEN_PREFIXES = ("/static/",)

SESSION_KEY = "email"
_ISSUED_AT = "at"


class AuthError(Exception):
    """Sign-in refused. The message is shown to the person trying to sign in,
    so it says what to do rather than what went wrong internally."""


# ---------------------------------------------------------------- verifying --

def verify_google_token(credential: str) -> str:
    """A Google ID token -> the verified email address. Raises AuthError.

    google-auth does the work that matters: fetches Google's public keys,
    checks the signature, the issuer and the expiry. Two things it only does
    because we ask, and both are load-bearing:

    - `audience`. Without it, ANY valid Google ID token is accepted -- including
      one minted for somebody else's app entirely, which an attacker can obtain
      simply by signing in to that app. The audience check is what ties a token
      to THIS application.
    - `email_verified`. Google will happily assert an unverified address on some
      account types, and an unverified email is not evidence of anything.
    """
    if not config.GOOGLE_CLIENT_ID:
        raise AuthError("Sign-in is not configured on this server.")
    try:
        claims = id_token.verify_oauth2_token(
            credential, google_requests.Request(), config.GOOGLE_CLIENT_ID
        )
    except Exception as e:  # noqa: BLE001 -- every failure is the same answer
        # Deliberately not echoing the library's message: it distinguishes
        # expired from malformed from wrong-audience, and that is a detail
        # nobody signing in needs and an attacker would enjoy.
        raise AuthError("That Google sign-in could not be verified.") from e

    if not claims.get("email_verified"):
        raise AuthError("That Google account has no verified email address.")
    email = (claims.get("email") or "").strip().lower()
    if not email:
        raise AuthError("That Google account did not provide an email address.")
    return email


def is_allowed(conn, email: str) -> bool:
    """Is this address on the allowlist? One SELECT, run once per sign-in --
    not per request, which is what the session cookie buys."""
    return conn.execute(
        "SELECT EXISTS (SELECT 1 FROM allowed_users WHERE email = %s)",
        (email.strip().lower(),),
    ).fetchone()[0]


def add_user(conn, email: str, note: str | None = None) -> bool:
    """Grant access. True if added, False if already there."""
    row = conn.execute(
        "INSERT INTO allowed_users (email, note) VALUES (%s, %s) "
        "ON CONFLICT (email) DO NOTHING RETURNING email",
        (email.strip().lower(), note),
    ).fetchone()
    conn.commit()
    return row is not None


def remove_user(conn, email: str) -> bool:
    row = conn.execute(
        "DELETE FROM allowed_users WHERE email = %s RETURNING email",
        (email.strip().lower(),),
    ).fetchone()
    conn.commit()
    return row is not None


def list_users(conn) -> list:
    return conn.execute(
        "SELECT email, note, added_at FROM allowed_users ORDER BY added_at"
    ).fetchall()


# ----------------------------------------------------------------- sessions --

def sign_in(request, email: str) -> None:
    """Record a verified, allowlisted email in the signed session cookie."""
    request.session[SESSION_KEY] = email
    request.session[_ISSUED_AT] = int(time.time())


def sign_out(request) -> None:
    request.session.clear()


def current_user(request) -> str | None:
    """The signed-in email, or None. Also enforces the session's own maximum
    age: SessionMiddleware's cookie max_age stops a browser SENDING an old
    cookie, but a cookie captured off the wire could be replayed by something
    that is not a browser, so the age is re-checked here on the server."""
    if not config.auth_enabled():
        return None
    email = request.session.get(SESSION_KEY)
    if not email:
        return None
    issued = request.session.get(_ISSUED_AT, 0)
    if int(time.time()) - issued > config.SESSION_MAX_AGE_SECONDS:
        request.session.clear()
        return None
    return email


# --------------------------------------------------------------- the gate --

def _is_open(path: str) -> bool:
    return path in OPEN_PATHS or path.startswith(OPEN_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    """Deny by default.

    Every path not in OPEN_PATHS needs a session. Stated as a gate rather than
    a per-route dependency on purpose: a route added next month is protected
    without anyone remembering to protect it, whereas a forgotten decorator is
    a silently public endpoint. The eight mutating routes in api.py are exactly
    the ones you would forget.

    Answers in the shape the caller can use. A browser typing a URL gets a 302
    to the login page; a fetch() gets 401 JSON, because a login page delivered
    into `await r.json()` is a parse error rather than a message.
    """

    async def dispatch(self, request, call_next):
        if not config.auth_enabled() or _is_open(request.url.path):
            return await call_next(request)

        if current_user(request):
            return await call_next(request)

        # "Is this a browser navigating?" -- Accept: text/html is the honest
        # signal. fetch() sends */* or application/json unless told otherwise.
        wants_html = "text/html" in request.headers.get("accept", "")
        if wants_html:
            return RedirectResponse("/login", status_code=302)
        return JSONResponse({"detail": "Not signed in."}, status_code=401)
