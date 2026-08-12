"""Where Drive credentials come from.

Two things have to be found before the app can read Drive, and each has two
possible homes:

    the OAuth CLIENT SECRETS   env var DRIVE_CLIENT_SECRETS_JSON, else
                               credentials.json on disk
    the USER TOKEN             the drive_credentials table, else
                               token.json on disk

Both exist because the deployed container has no filesystem worth writing to.
Cloud Run's disk is in-memory and discarded on every restart, so a token written
there survives until the next cold start -- which for a scale-to-zero service is
minutes. Postgres is the only thing in this system that persists.

PRECEDENCE IS DELIBERATE, and it is "database first, but only if there is
actually a row". That gives three behaviours from one rule:

  - The container has a row and no files, so it uses the database.
  - A laptop with no row keeps using token.json exactly as it always has --
    which is what stops this change touching the CLI or the 81 Drive tests.
  - A laptop that HAS connected through the web UI shares the one token with
    the deployed app, which is the point of putting it in Postgres at all.

The table holds one row (see migrations/0008). One shared credential, not one
per user: everyone with access to this app is looking at the SAME Drive folder,
so a second token would add another person's whole-Drive read access to the
blast radius while granting the app nothing it could not already reach.
"""
import json
import os

from library_rag import config

# The OAuth client secrets, as JSON text, for deployments with no file. Same
# content as credentials.json -- it is a secret, and belongs wherever the
# Supabase key and Anthropic key are kept.
CLIENT_SECRETS_ENV = "DRIVE_CLIENT_SECRETS_JSON"


def client_config(credentials_file: str) -> dict | None:
    """The OAuth client secrets as a dict, or None if unavailable.

    A dict rather than a path because google-auth-oauthlib offers
    from_client_config() alongside from_client_secrets_file(), and only the
    former works without a filesystem.
    """
    raw = os.environ.get(CLIENT_SECRETS_ENV, "").strip()
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # A malformed secret is worth failing loudly on -- silently falling
            # through to the file would make a deployment look "not connected"
            # when it is actually misconfigured.
            raise ValueError(
                f"{CLIENT_SECRETS_ENV} is set but is not valid JSON."
            ) from None
    if os.path.exists(credentials_file):
        with open(credentials_file) as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ token --

def _table_exists(conn) -> bool:
    return conn.execute(
        "SELECT to_regclass('drive_credentials')"
    ).fetchone()[0] is not None


def load_token(token_file: str) -> dict | None:
    """The stored user token as a dict, or None if there is none anywhere.

    Never raises on a database problem. A Drive token is not worth taking the
    whole app down for: every caller treats None as "not connected", which is a
    state the UI already renders.
    """
    row = None
    try:
        from library_rag import db

        with db.get_conn() as conn:
            if _table_exists(conn):
                row = conn.execute(
                    "SELECT token_json FROM drive_credentials WHERE id = 1"
                ).fetchone()
    except Exception:  # noqa: BLE001 -- absence is a normal answer here
        row = None

    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    if os.path.exists(token_file):
        try:
            with open(token_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def save_token(token_file: str, token: dict, connected_by: str = "cli") -> str:
    """Persist a token. Returns "database" or "file", so a caller can say which.

    Writes to whichever home the token would be READ from next, so a save and
    the following load never disagree: the database when the table is there,
    the file otherwise.
    """
    payload = json.dumps(token) if isinstance(token, dict) else str(token)
    try:
        from library_rag import db

        with db.get_conn() as conn:
            if _table_exists(conn):
                conn.execute(
                    """
                    INSERT INTO drive_credentials (id, token_json, connected_by)
                    VALUES (1, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                       SET token_json = EXCLUDED.token_json,
                           connected_by = EXCLUDED.connected_by,
                           connected_at = now()
                    """,
                    (payload, connected_by),
                )
                conn.commit()
                return "database"
    except Exception:  # noqa: BLE001 -- fall through to the file
        pass

    with open(token_file, "w") as f:
        f.write(payload)
    return "file"


def connection_info() -> dict | None:
    """Who connected Drive and when, for the UI. None when it came from a file
    or is absent -- a file has no such story to tell."""
    try:
        from library_rag import db

        with db.get_conn() as conn:
            if not _table_exists(conn):
                return None
            row = conn.execute(
                "SELECT connected_by, connected_at FROM drive_credentials WHERE id = 1"
            ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    return {"connected_by": row[0], "connected_at": row[1].isoformat()}


def clear_token() -> None:
    """Forget the stored token. Used by a disconnect action, and by tests."""
    try:
        from library_rag import db

        with db.get_conn() as conn:
            if _table_exists(conn):
                conn.execute("DELETE FROM drive_credentials WHERE id = 1")
                conn.commit()
    except Exception:  # noqa: BLE001
        pass


# `config` is imported for symmetry with the rest of the package and to keep the
# module importable in isolation; the database URL it holds is read by db.
_ = config
