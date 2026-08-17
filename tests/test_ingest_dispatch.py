"""_start_ingest(): who actually drains the queue.

The queueing half of every index endpoint is unaffected by this -- rows are
inserted and committed before any of it runs. What is pinned here is only the
handoff: when a Cloud Run Job is configured the app must NOT also drain in its
own process (that is the throttled path the Job exists to replace), and when
the trigger fails for any reason it MUST, because the books are already queued
and something has to pick them up.
"""
import pytest

from library_rag import config, jobs
from library_rag.web import api


class _Background:
    """Stand-in for FastAPI's BackgroundTasks that just records."""

    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args):
        self.tasks.append((fn, args))


def test_falls_back_to_in_process_when_no_job_is_configured(monkeypatch):
    """The local / test / single-user path: no Cloud project, same behaviour
    the app has always had."""
    monkeypatch.setattr(config, "INGEST_JOB_NAME", "")
    bg = _Background()

    api._start_ingest(bg, "upload")

    assert [args for _, args in bg.tasks] == [("upload",)]


def test_does_not_drain_in_process_when_the_job_starts(monkeypatch):
    monkeypatch.setattr(api.jobs, "run_ingest_job", lambda: True)
    bg = _Background()

    api._start_ingest(bg, "drive")

    assert bg.tasks == [], "queue drained in-process despite the Job starting"


def test_falls_back_when_the_trigger_fails(monkeypatch):
    """A failed trigger must not strand the rows. They are committed already;
    the only question is who drains them."""
    monkeypatch.setattr(api.jobs, "run_ingest_job", lambda: False)
    bg = _Background()

    api._start_ingest(bg, "drive")

    assert [args for _, args in bg.tasks] == [("drive",)]


@pytest.mark.parametrize(
    "boom",
    [
        RuntimeError("metadata server unreachable"),
        ValueError("malformed credentials"),
    ],
)
def test_trigger_never_raises_into_the_request(monkeypatch, boom):
    """run_ingest_job swallows everything on purpose: the user's click already
    succeeded at the part that matters, so a broken trigger degrades to the
    fallback rather than turning a queued folder into a 500."""
    monkeypatch.setattr(config, "INGEST_JOB_NAME", "library-rag-ingest")

    def _explode(*a, **k):
        raise boom

    monkeypatch.setattr(jobs.google.auth, "default", _explode)

    assert jobs.run_ingest_job() is False


def test_unconfigured_trigger_does_not_touch_credentials(monkeypatch):
    """The config check has to come FIRST. Reaching for credentials in an
    environment that has none is how a test suite or a laptop picks up a
    multi-second timeout against a metadata server that is not there."""
    monkeypatch.setattr(config, "INGEST_JOB_NAME", "")

    def _explode(*a, **k):
        raise AssertionError("credentials looked up despite no job configured")

    monkeypatch.setattr(jobs.google.auth, "default", _explode)

    assert jobs.run_ingest_job() is False
