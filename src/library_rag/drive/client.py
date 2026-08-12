"""
Google Drive API v3 auth + raw listing/download, read-only.

Expects credentials.json (OAuth client secrets) in the repo root. Caches the
resulting token in token.json. Both are gitignored -- never commit them.

The authenticated service object is built explicitly (build_service()) and
passed into every call, rather than kept as a module-level singleton, so tests
and the pipeline can inject fakes and callers control the client's lifetime.
"""
import os
import random
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
PDF_MIME = "application/pdf"
# `parents` is what makes the tree reconstructible: Drive stores containment as
# an array on the child, and nothing on a folder says what it holds. Without it
# a listing is a flat pile with no edges.
FIELDS = (
    "nextPageToken, "
    "files(id, name, mimeType, size, md5Checksum, webViewLink, parents)"
)

RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}
MAX_RETRIES = 5


class DriveAuthError(RuntimeError):
    """Raised with a message that tells the user exactly what to fix."""


def _load_credentials():
    if not os.path.exists(CREDENTIALS_FILE):
        raise DriveAuthError(
            f"Missing {CREDENTIALS_FILE} in the current directory.\n"
            "Fix: download your OAuth client secrets from Google Cloud Console "
            "(APIs & Services > Credentials > OAuth client ID > Desktop app), "
            f"save the JSON as {CREDENTIALS_FILE} in this repo, and re-run.\n"
            "See README.md for step-by-step setup."
        )

    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            return creds
        except Exception as e:
            raise DriveAuthError(
                f"token.json exists but refresh failed ({e}).\n"
                f"Fix: delete {TOKEN_FILE} and re-run to re-authorize via browser."
            )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    except Exception as e:
        raise DriveAuthError(
            f"OAuth flow failed to complete ({e}).\n"
            f"Fix: check that {CREDENTIALS_FILE} is a valid Desktop-app OAuth "
            "client secrets file, and that you completed the browser consent "
            "screen. See README.md for setup."
        )

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    return creds


def credentials_status() -> dict:
    """Whether Drive is usable right now, WITHOUT triggering a consent flow.

    Read-only on purpose: the UI polls this to show a "reconnect" banner before
    an action fails, and a status check that opens a browser tab would be a
    booby trap. It does attempt a refresh, because that is the only way to tell
    a token that still works from one Google has revoked -- expiry alone says
    nothing, since a valid refresh token silently renews.
    """
    if not os.path.exists(CREDENTIALS_FILE):
        return {"ok": False, "reason": "missing_client",
                "detail": f"{CREDENTIALS_FILE} is not present. Download OAuth "
                          "client secrets from Google Cloud Console."}
    if not os.path.exists(TOKEN_FILE):
        return {"ok": False, "reason": "not_connected",
                "detail": "Not connected to Google Drive yet."}
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    except Exception as e:  # noqa: BLE001 -- a corrupt token is "reconnect", not a crash
        return {"ok": False, "reason": "unreadable", "detail": str(e)}
    if creds.valid:
        return {"ok": True, "reason": "valid", "detail": ""}
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
            return {"ok": True, "reason": "refreshed", "detail": ""}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "reason": "revoked", "detail": str(e)}
    return {"ok": False, "reason": "no_refresh_token",
            "detail": "The stored token cannot be refreshed."}


# --------------------------------------------------------- browser OAuth --
#
# _load_credentials()'s InstalledAppFlow.run_local_server() is right for a CLI
# and wrong for the web app: it BLOCKS the calling thread and starts a SECOND
# http server on a random port to catch the redirect. Inside a request handler
# that is the "OAuth flow with no console to prompt" failure documented in
# api._drain_queue -- the request hangs and nothing says why.
#
# These two functions let the app be its own callback instead. Google permits
# any http://localhost:<port> redirect for an "installed" (Desktop) client
# without registering it in the Cloud console -- which is exactly the mechanism
# run_local_server relies on, used deliberately rather than incidentally.

def auth_url(redirect_uri: str, state: str) -> tuple:
    """The Google consent URL, and the PKCE verifier the exchange will need.

    Returns (url, code_verifier). Both must be kept: authorization_url()
    generates a PKCE code_verifier, puts its S256 challenge in the URL, and
    stores the verifier ON THIS FLOW INSTANCE. exchange_code() necessarily
    builds a different instance -- it runs in a later request -- so unless the
    verifier is carried across, Google rejects the exchange with
    "invalid_grant: Missing code verifier".

    Carried rather than disabled: PKCE is what stops an authorization code
    intercepted in transit from being redeemable by anyone else, and the code
    travels through a browser redirect.
    """
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, SCOPES, redirect_uri=redirect_uri
    )
    url, _ = flow.authorization_url(
        access_type="offline",
        # Force the consent screen even when the user has approved before.
        # Google returns a refresh_token only on FIRST consent otherwise, so a
        # reconnect after expiry would hand back an access token that dies in an
        # hour with nothing to renew it -- reproducing the failure being fixed.
        prompt="consent",
        include_granted_scopes="true",
        state=state,
    )
    return url, flow.code_verifier


def exchange_code(code: str, redirect_uri: str, code_verifier: str | None = None) -> None:
    """Trade the callback's code for tokens and persist them.

    `code_verifier` must be the one auth_url() returned for this flow; see
    there for why it cannot be regenerated here.
    """
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, SCOPES, redirect_uri=redirect_uri
    )
    flow.code_verifier = code_verifier
    flow.fetch_token(code=code)
    with open(TOKEN_FILE, "w") as f:
        f.write(flow.credentials.to_json())


def build_service():
    """Build an authenticated Drive v3 service. Runs the OAuth flow / token
    refresh via _load_credentials(). Call once and pass the result into the
    listing/download functions below."""
    creds = _load_credentials()
    return build("drive", "v3", credentials=creds)


def _request_with_retry(request):
    for attempt in range(MAX_RETRIES):
        try:
            return request.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            if status == 403:
                raise DriveAuthError(
                    "Drive API returned 403 (permission denied) after retries.\n"
                    "Fix: confirm this Google account has at least read access "
                    "to the shared folder, and that the Drive API is enabled "
                    "for your Cloud project."
                )
            if status == 404:
                raise DriveAuthError(
                    "Drive API returned 404 (not found) for this folder ID.\n"
                    "Fix: confirm the folder ID is correct and that it has been "
                    "shared with the authenticated account."
                )
            raise
    raise DriveAuthError("Drive API request failed after max retries.")


def _list(service, q: str, limit: int | None = None) -> list:
    """Run one Drive `files.list` query to exhaustion (or to `limit`).

    The paginated request loop, shared by every listing call. `limit` stops
    paging early -- a `name contains` query against a 130 GB drive can match
    thousands of files, and nothing downstream wants them all.
    """
    items = []
    page_token = None
    while True:
        request = service.files().list(
            q=q,
            fields=FIELDS,
            pageToken=page_token,
            pageSize=1000 if limit is None else min(1000, limit - len(items)),
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        response = _request_with_retry(request)
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token or (limit is not None and len(items) >= limit):
            break
    return items[:limit] if limit is not None else items


def list_children(service, folder_id: str) -> list:
    """Return the full (paginated) list of immediate children of folder_id
    as raw Drive API file resources: {id, name, mimeType, size, md5Checksum,
    webViewLink}. size is a string in bytes when present, absent for e.g.
    Google-native docs; md5Checksum is present for binary files like PDFs.
    """
    return _list(service, f"'{folder_id}' in parents and trashed = false")


def _escape_q(value: str) -> str:
    r"""Escape a user string for embedding in a Drive `q` clause.

    Drive's query grammar delimits string literals with single quotes and uses
    backslash escapes inside them. A title like "Paul's Letters" would otherwise
    terminate the literal early and produce a 400 -- and since these strings come
    from an LLM reading real book titles, apostrophes are routine, not exotic.
    Backslash must be escaped FIRST or it would re-escape the quote we just added.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def search_files(service, name_contains: str, limit: int = 40) -> list:
    """PDFs anywhere in the drive whose name contains `name_contains`.

    One API call instead of walking the tree: the `Books` folder alone holds 204
    subfolders and 5,024 PDFs, so crawling to find a title is thousands of
    requests to answer a question Drive can answer in one.

    Matching is Drive's own, and it is NOT a substring match. Drive tokenises the
    filename into words and `contains` matches a WORD PREFIX, case-insensitively:
    measured against this drive, 'parab' returns 100+ hits and 'arable' returns
    zero, though both sit inside "parable". Callers must pass whole words or
    their beginnings; a fragment from the middle of one silently finds nothing.

    Filenames only -- it does not look inside the PDFs. Drive's `fullText
    contains` does that, and is a different tool with different tradeoffs.

    Returns raw file resources, same shape as list_children.
    """
    q = (
        f"name contains '{_escape_q(name_contains)}' "
        f"and mimeType = '{PDF_MIME}' and trashed = false"
    )
    return _list(service, q, limit=limit)


def download_file(service, file_id: str, dest_path: str) -> None:
    """Stream a file's binary content to dest_path, with retries on
    retryable errors for each chunk (supports Shared Drives).

    get_media() is the SAME REST endpoint as get() -- GET /drive/v3/files/{id}
    -- differing only in `alt`: get() sends alt=json and returns metadata,
    get_media() sends alt=media and returns the bytes. The Python client splits
    them into two methods, which is worth keeping rather than "unifying":

      - Do NOT try service.files().get(fileId=..., alt="media"). The
        discovery-built get() accepts that argument, silently ignores it and
        emits alt=json anyway -- so it would write a JSON metadata blob into a
        file named .pdf, with no error until something tried to parse it.
      - MediaIoBaseDownload reads request.uri to issue ranged chunk requests,
        so it needs a URI that already carries alt=media.

    Also unset here: acknowledgeAbuse. Drive refuses alt=media with a 403 for a
    file it has flagged as malware unless that flag is passed. 403 is in
    RETRYABLE_STATUS (Drive also uses it for rate limiting), so such a file
    would retry five times and then surface as a permissions error, which is
    the wrong explanation. Rare enough to leave; noted so it is not a mystery.
    """
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            for attempt in range(MAX_RETRIES):
                try:
                    _, done = downloader.next_chunk()
                    break
                except HttpError as e:
                    status = getattr(e.resp, "status", None)
                    if status in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
                        delay = (2**attempt) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    raise


def get_folder_name(service, folder_id: str) -> str:
    request = service.files().get(
        fileId=folder_id, fields="id, name, webViewLink", supportsAllDrives=True
    )
    return _request_with_retry(request)


def get_file(service, file_id: str) -> dict:
    """One file resource, same fields listing returns.

    Used when adding a book by id: title, size and md5 come from Drive rather
    than from whoever sent the id. md5 especially -- process_book verifies the
    downloaded bytes against it, so accepting a client's value would turn a
    wrong guess into an ingestion that fails with a confusing mismatch.
    """
    request = service.files().get(
        fileId=file_id,
        fields="id, name, mimeType, size, md5Checksum, webViewLink",
        supportsAllDrives=True,
    )
    return _request_with_retry(request)
