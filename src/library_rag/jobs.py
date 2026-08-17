"""
Starting the ingestion worker that runs OUTSIDE this process.

Why this module exists at all: ingestion inside the web container is work with
no request attached. Cloud Run allocates CPU per request, so a BackgroundTask
begins exactly when the response is flushed and the instance stops being given
CPU -- and the instance may be evicted mid-book. The claim/lease in
db.claim_next_book keeps that SAFE (a stale claim is reaped after
CLAIM_STALE_MINUTES) but it cannot make it FINISH: the book sits in 'processing'
with no error until something drains again.

So the app stops doing the work and starts asking for it instead. It queues rows
-- which is fast, in-request, and the part it is actually good at -- then POSTs
to the Cloud Run Admin API to start a Job that has full un-throttled CPU and no
request lifecycle to outlive.

Unconfigured (INGEST_JOB_NAME empty) this returns False and the caller falls
back to the in-process drain, which is what keeps `make serve` and the test
suite working with no Google project at all.
"""
import google.auth
from google.auth.transport.requests import AuthorizedSession

from library_rag import config

# Cloud Run's v1 (Knative-shaped) Admin API. The regional host matters: the
# global run.googleapis.com endpoint does not serve the namespaces/... paths.
_RUN_API = "https://{region}-run.googleapis.com/apis/run.googleapis.com/v1"
_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def run_ingest_job() -> bool:
    """Start one execution of the ingest Job. True if it started.

    Never raises. Every failure -- not configured, no credentials, API refusing
    -- returns False, because the caller's fallback (drain in-process) is
    strictly better than a 500 on a button that has already queued the books
    successfully. The rows are in the queue either way; this only decides who
    picks them up.

    Deliberately does NOT pass a source. The Job drains everything, which is
    safe because concurrent workers cooperate through FOR UPDATE SKIP LOCKED --
    the reason the API's own drains are source-scoped was a Drive book blocking
    a REQUEST thread on an OAuth prompt, and neither half of that applies to a
    Job (the token lives in Postgres now, and a Job is not a request).

    Nor does it check whether an execution is already running. Two executions
    racing is not a correctness problem for the same SKIP LOCKED reason, and the
    check would cost an extra API call on every click to save a fraction of a
    cent of the 1-minute minimum billing.
    """
    if not config.INGEST_JOB_NAME:
        return False

    try:
        creds, default_project = google.auth.default(scopes=[_SCOPE])
        project = config.INGEST_JOB_PROJECT or default_project
        if not project:
            print("ingest job: no project id available; falling back to in-process")
            return False

        base = _RUN_API.format(region=config.INGEST_JOB_REGION)
        url = f"{base}/namespaces/{project}/jobs/{config.INGEST_JOB_NAME}:run"

        # Short timeout: this is on the request path, and a slow Admin API must
        # degrade to the in-process drain rather than hold the user's response.
        resp = AuthorizedSession(creds).post(url, timeout=10)
        if resp.status_code >= 400:
            print(f"ingest job: {resp.status_code} {resp.text[:200]}")
            return False
        return True
    except Exception as e:  # noqa: BLE001 -- see docstring: never raise here
        print(f"ingest job: could not start ({e}); falling back to in-process")
        return False
