"""
Shared test fixtures. Strategy: real Postgres, fake externals, zero network.

A session fixture creates a throwaway `library_test` database from DATABASE_URL
and runs the migration runner against it; a function fixture truncates tables
between tests. If Postgres is unreachable the whole DB-backed suite skips with a
clear message (pure-function tests like chunking still run).

Nothing here touches the Voyage / Drive / Mistral APIs -- embedders and Drive
listings are injected as deterministic fakes.
"""
import hashlib
import math
import os
import random
import re
import uuid

import psycopg
import pytest
from psycopg import conninfo

from library_rag import config, migrate
from library_rag import db as db_mod

TEST_DB = "library_test"

# Tests CREATE and DROP whole databases, so they must never aim at a shared
# server. Now that DATABASE_URL can point at Supabase, the suite takes its own
# TEST_DATABASE_URL (falling back to DATABASE_URL for the pre-cloud setup) and
# refuses anything non-local outright -- a skip message beats discovering that
# `pytest` dropped a database on a production cluster.
TEST_BASE_URL = os.environ.get("TEST_DATABASE_URL", config.DATABASE_URL)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", None, ""}


def _require_local(url: str) -> None:
    host = conninfo.conninfo_to_dict(url).get("host")
    if host not in _LOCAL_HOSTS:
        pytest.skip(
            f"Refusing to create/drop test databases on non-local host {host!r}. "
            "Set TEST_DATABASE_URL to a local Postgres (e.g. the docker-compose one)."
        )


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch):
    """No test may write to the real data/ dirs. process_book / write_outputs
    key their filenames off book_id (config.PDF_DIR / f"{book_id}.pdf"), so a
    worker test whose book_id collides with a real book will otherwise overwrite
    a genuinely-downloaded PDF and its markdown. Redirect all three to a
    per-test tmp dir for every test.

    UPLOAD_DIR especially: it is content-addressed and holds the ONLY copy of an
    uploaded book's bytes, so a test writing there is not overwriting a cache,
    it is adding to the real library."""
    pdfs = tmp_path / "pdfs"
    markdown = tmp_path / "markdown"
    uploads = tmp_path / "uploads"
    for d in (pdfs, markdown, uploads):
        d.mkdir()
    monkeypatch.setattr(config, "PDF_DIR", pdfs)
    monkeypatch.setattr(config, "MARKDOWN_DIR", markdown)
    monkeypatch.setattr(config, "UPLOAD_DIR", uploads)
    # A browse test must not poison the real cache with fake Drive folders --
    # the agent would then recommend books that do not exist.
    monkeypatch.setattr(config, "DRIVE_CACHE_FILE", tmp_path / "drive_cache.json")
    # No mirror test may depend on the real library's folder id. Left set, the
    # sync's prune step deletes every synthetic fixture row (none descend from a
    # folder that exists only in the user's Drive) and the failure looks like a
    # broken sync rather than a misconfigured test. Pruning gets its own test,
    # which sets this deliberately.
    monkeypatch.setattr(config, "DRIVE_ROOT_FOLDER_ID", None)
    # Storage must be OFF in tests: config loads the real .env, so without this
    # a purge test would delete a real object from the real bucket. Blanking
    # the config makes storage.enabled() False and every call a no-op; tests
    # that exercise storage behaviour monkeypatch the module explicitly.
    monkeypatch.setattr(config, "SUPABASE_URL", "")
    monkeypatch.setattr(config, "SUPABASE_SERVICE_KEY", "")

    # Sign-in must be OFF in tests, for the same reason and with a sharper
    # edge: config loads the real .env, so the moment a developer sets
    # GOOGLE_CLIENT_ID to work on auth, auth_enabled() becomes True for the
    # WHOLE suite and every API test gets a 401 from the gate. Measured when it
    # happened: 49 failures across upload, storage, browse and research, none
    # of which have anything to do with authentication.
    #
    # Tests that exercise the gate turn it back on explicitly (see the
    # `auth_on` fixture in test_auth.py), which is also what keeps this from
    # hiding a regression in the gate itself.
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")

    # Fail CLOSED toward production. Library code does not always take a
    # connection -- drive/store.py calls db.get_conn() with no argument -- so
    # any test touching it would otherwise open config.DATABASE_URL, which on
    # this machine is Supabase. That is exactly how a test run wrote a row into
    # the real database while asserting against the test one.
    #
    # Pointed at a local address nothing listens on rather than at the test
    # database, because THIS fixture must not require Postgres: the pure
    # chunking tests run without it. A test that genuinely needs a database
    # takes `conn`, which overrides this with the real test URL.
    #
    # _require_local already refuses to CREATE or DROP remotely. It cannot stop
    # an ordinary INSERT issued by the code under test; this can.
    monkeypatch.setattr(
        config, "DATABASE_URL",
        "postgresql://unused@127.0.0.1:1/tests-must-take-the-conn-fixture",
    )


def _url_with_db(base_url: str, dbname: str) -> str:
    params = conninfo.conninfo_to_dict(base_url)
    params["dbname"] = dbname
    return conninfo.make_conninfo(**params)


def _admin_connect():
    """Connect to the `postgres` maintenance DB (autocommit) to CREATE/DROP test
    databases. Skips the suite if Postgres isn't reachable."""
    _require_local(TEST_BASE_URL)
    admin_url = _url_with_db(TEST_BASE_URL, "postgres")
    try:
        return psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable at {TEST_BASE_URL!r}: {e}")


def _drop_db(admin, dbname: str) -> None:
    admin.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = %s AND pid <> pg_backend_pid()",
        (dbname,),
    )
    admin.execute(f'DROP DATABASE IF EXISTS "{dbname}"')


@pytest.fixture(scope="session")
def test_database_url():
    """A migrated, session-scoped test database URL."""
    with _admin_connect() as admin:
        _drop_db(admin, TEST_DB)
        admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    url = _url_with_db(TEST_BASE_URL, TEST_DB)
    migrate.run(url)
    yield url
    with _admin_connect() as admin:
        _drop_db(admin, TEST_DB)


@pytest.fixture
def conn(test_database_url, monkeypatch):
    """A connection to the test DB with tables truncated to a clean slate.

    ALSO points config.DATABASE_URL at the test database, which matters for any
    library code that opens its OWN connection rather than taking one. Found
    the hard way: drive/store.py calls db.get_conn() with no argument, so a
    test exercising it wrote a row into the real Supabase database while
    reading assertions from the test one -- every assertion failed, and the
    damage was in the other database entirely.

    _require_local already refuses to CREATE or DROP on a remote host. It does
    not, and cannot, stop an ordinary INSERT issued by the code under test.
    This closes that gap for every test that takes `conn`.

    drive_files is listed explicitly. It has no foreign key to books -- a Drive
    file exists whether or not we hold it -- so CASCADE does not reach it, and
    leaving it out let one test's mirror (and its embeddings) leak into the
    next's counts.
    """
    monkeypatch.setattr(config, "DATABASE_URL", test_database_url)
    with db_mod.get_conn(test_database_url) as c:
        c.execute(
            "TRUNCATE books, drive_files, bible_verses, allowed_users, "
            "drive_credentials "
            "RESTART IDENTITY CASCADE"
        )
        c.commit()
        yield c


@pytest.fixture
def fresh_database():
    """A uniquely-named EMPTY database (no migrations applied), for tests that
    exercise the migration runner itself. Dropped on teardown."""
    name = f"lr_test_{uuid.uuid4().hex[:8]}"
    with _admin_connect() as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    try:
        yield _url_with_db(TEST_BASE_URL, name)
    finally:
        with _admin_connect() as admin:
            _drop_db(admin, name)


# ------------------------------------------------------------- fakes --

def deterministic_vector(text: str, dim: int = None) -> list:
    """A stable pseudo-random unit-ish vector seeded by the text, so the same
    string always embeds to the same vector (no network, no model)."""
    dim = dim or config.EMBED_DIM
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def lexical_vector(text: str, dim: int = None) -> list:
    """A hashed bag-of-words vector: hash each lowercased word to a dimension,
    count it, then L2-normalise so cosine is well behaved.

    deterministic_vector is seeded by the WHOLE string, so two texts about the
    same subject are as far apart as two unrelated ones -- fine for "did this
    row round-trip", useless for "did retrieval return the right passage". This
    one has the single property an evaluation needs: texts sharing words are
    close, texts sharing none are far. It is a crude embedder, not a good one,
    and that is deliberate -- tests/test_eval.py measures whether the HARNESS
    and the SQL rank correctly, and a real model would put network calls and
    non-determinism inside the thing being measured.
    """
    dim = dim or config.EMBED_DIM
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class _FakeEmbedResult:
    def __init__(self, embeddings, total_tokens):
        self.embeddings = embeddings
        self.total_tokens = total_tokens


class FakeVoyageClient:
    """Mimics the tiny slice of voyageai.Client that pipeline/embed.py uses."""

    def __init__(self):
        self.calls = []

    def embed(self, texts, model, input_type):
        self.calls.append((tuple(texts), model, input_type))
        embeddings = [deterministic_vector(t) for t in texts]
        return _FakeEmbedResult(embeddings, sum(len(t) for t in texts))


@pytest.fixture
def fake_voyage():
    return FakeVoyageClient()
