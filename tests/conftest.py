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
import random
import re
import uuid

import psycopg
import pytest
from psycopg import conninfo

import config
import db as db_mod
import migrate

TEST_DB = "library_test"


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch):
    """No test may write to the real data/ dirs. process_book / write_outputs
    key their filenames off book_id (config.PDF_DIR / f"{book_id}.pdf"), so a
    worker test whose book_id collides with a real book will otherwise overwrite
    a genuinely-downloaded PDF and its markdown. Redirect both to a per-test
    tmp dir for every test."""
    pdfs = tmp_path / "pdfs"
    markdown = tmp_path / "markdown"
    pdfs.mkdir()
    markdown.mkdir()
    monkeypatch.setattr(config, "PDF_DIR", pdfs)
    monkeypatch.setattr(config, "MARKDOWN_DIR", markdown)


def _url_with_db(base_url: str, dbname: str) -> str:
    params = conninfo.conninfo_to_dict(base_url)
    params["dbname"] = dbname
    return conninfo.make_conninfo(**params)


def _admin_connect():
    """Connect to the `postgres` maintenance DB (autocommit) to CREATE/DROP test
    databases. Skips the suite if Postgres isn't reachable."""
    admin_url = _url_with_db(config.DATABASE_URL, "postgres")
    try:
        return psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable at {config.DATABASE_URL!r}: {e}")


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
    url = _url_with_db(config.DATABASE_URL, TEST_DB)
    migrate.run(url)
    yield url
    with _admin_connect() as admin:
        _drop_db(admin, TEST_DB)


@pytest.fixture
def conn(test_database_url):
    """A connection to the test DB with tables truncated to a clean slate."""
    with db_mod.get_conn(test_database_url) as c:
        c.execute("TRUNCATE books RESTART IDENTITY CASCADE")
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
        yield _url_with_db(config.DATABASE_URL, name)
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
