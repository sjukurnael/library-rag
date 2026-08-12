"""Where Drive credentials are read from and written to.

The module is small; its precedence rules are not obvious, and getting them
wrong is invisible until a deployment silently keeps using a stale token or a
laptop stops finding one it has. Each rule gets a test.
"""
import json

import pytest

from library_rag.drive import client as drive_client
from library_rag.drive import store


@pytest.fixture
def token_file(tmp_path):
    """A token.json on disk, as a laptop has."""
    p = tmp_path / "token.json"
    p.write_text(json.dumps({"marker": "from-file", "refresh_token": "r"}))
    return str(p)


# ------------------------------------------------------------- precedence --

def test_with_no_row_the_file_is_used(conn, token_file):
    """The laptop's behaviour, unchanged. This is what keeps the CLI and the 81
    existing Drive tests working -- conftest truncates drive_credentials, so
    every one of them takes this path."""
    assert store.load_token(token_file)["marker"] == "from-file"


def test_a_stored_row_beats_the_file(conn, token_file):
    """The container's behaviour: it has a row and no file worth reading."""
    store.save_token(token_file, {"marker": "from-db"}, "someone@example.com")
    assert store.load_token(token_file)["marker"] == "from-db"


def test_clearing_the_row_falls_back_to_the_file(conn, token_file):
    store.save_token(token_file, {"marker": "from-db"}, "someone@example.com")
    store.clear_token()
    assert store.load_token(token_file)["marker"] == "from-file"


def test_nothing_anywhere_is_None_not_an_error(conn, tmp_path):
    """Every caller treats None as "not connected", which the UI renders. A
    Drive token is not worth taking the app down for."""
    assert store.load_token(str(tmp_path / "absent.json")) is None


def test_a_save_lands_where_the_next_load_will_look(conn, token_file):
    """The property that matters more than either branch: save and load must
    never disagree about which home they are using."""
    where = store.save_token(token_file, {"marker": "roundtrip"}, "x@example.com")
    assert where == "database"
    assert store.load_token(token_file)["marker"] == "roundtrip"


def test_a_corrupt_stored_token_reads_as_absent(conn, token_file):
    """Garbage in the column must mean "reconnect", not a 500 on every page
    that polls Drive status."""
    conn.execute(
        "INSERT INTO drive_credentials (id, token_json, connected_by) "
        "VALUES (1, 'not json', 'x') ON CONFLICT (id) DO UPDATE "
        "SET token_json = EXCLUDED.token_json"
    )
    conn.commit()
    assert store.load_token(token_file) is None


# ---------------------------------------------------------------- the row --

def test_only_one_row_can_exist(conn):
    """A shared credential with two rows is one where nothing knows which is
    current. The CHECK makes that impossible rather than merely unlikely."""
    store.save_token("unused", {"a": 1}, "first@example.com")
    with pytest.raises(Exception):
        conn.execute(
            "INSERT INTO drive_credentials (id, token_json, connected_by) "
            "VALUES (2, '{}', 'second@example.com')"
        )


def test_reconnecting_replaces_rather_than_duplicates(conn):
    store.save_token("unused", {"v": 1}, "first@example.com")
    store.save_token("unused", {"v": 2}, "second@example.com")
    assert conn.execute("SELECT count(*) FROM drive_credentials").fetchone()[0] == 1
    assert store.connection_info()["connected_by"] == "second@example.com"


def test_who_connected_is_recorded(conn):
    """Two people can reconnect this; an unattributable shared credential is
    one nobody owns."""
    assert store.connection_info() is None
    store.save_token("unused", {"a": 1}, "stephen@example.com")
    info = store.connection_info()
    assert info["connected_by"] == "stephen@example.com"
    assert info["connected_at"]


# ----------------------------------------------------------- client config --

def test_the_env_var_beats_the_file(monkeypatch, tmp_path):
    """How the container gets client secrets with no filesystem."""
    f = tmp_path / "credentials.json"
    f.write_text(json.dumps({"installed": {"client_id": "from-file"}}))
    monkeypatch.setenv(store.CLIENT_SECRETS_ENV,
                       json.dumps({"installed": {"client_id": "from-env"}}))
    assert store.client_config(str(f))["installed"]["client_id"] == "from-env"


def test_the_file_is_used_when_the_env_var_is_absent(monkeypatch, tmp_path):
    f = tmp_path / "credentials.json"
    f.write_text(json.dumps({"installed": {"client_id": "from-file"}}))
    monkeypatch.delenv(store.CLIENT_SECRETS_ENV, raising=False)
    assert store.client_config(str(f))["installed"]["client_id"] == "from-file"


def test_neither_is_None(monkeypatch, tmp_path):
    monkeypatch.delenv(store.CLIENT_SECRETS_ENV, raising=False)
    assert store.client_config(str(tmp_path / "absent.json")) is None


def test_malformed_client_secrets_raise_rather_than_fall_through(monkeypatch, tmp_path):
    """Falling back to the file would make a misconfigured deployment report
    "not connected", sending someone to reconnect a thing that cannot work."""
    f = tmp_path / "credentials.json"
    f.write_text(json.dumps({"installed": {"client_id": "from-file"}}))
    monkeypatch.setenv(store.CLIENT_SECRETS_ENV, "{not json")
    with pytest.raises(ValueError) as e:
        store.client_config(str(f))
    assert store.CLIENT_SECRETS_ENV in str(e.value)


# ------------------------------------------------- what the app reports --

def test_status_says_cannot_connect_when_there_is_no_way_in(
    conn, monkeypatch, tmp_path
):
    """The deployed container's normal state: no token, no client secrets.

    Distinct from "not connected", which means a reconnect button would help.
    The old code reported the missing client FIRST and told a reader to go
    download OAuth secrets -- advice for whoever set the server up, not for
    them.
    """
    monkeypatch.delenv(store.CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setattr(drive_client, "TOKEN_FILE", str(tmp_path / "none.json"))
    monkeypatch.setattr(drive_client, "CREDENTIALS_FILE", str(tmp_path / "none.json"))
    s = drive_client.credentials_status()
    assert s["ok"] is False
    assert s["reason"] == "cannot_connect"


def test_status_says_not_connected_when_a_reconnect_would_work(
    conn, monkeypatch, tmp_path
):
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps({"installed": {"client_id": "x"}}))
    monkeypatch.delenv(store.CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setattr(drive_client, "TOKEN_FILE", str(tmp_path / "none.json"))
    monkeypatch.setattr(drive_client, "CREDENTIALS_FILE", str(creds))
    assert drive_client.credentials_status()["reason"] == "not_connected"
