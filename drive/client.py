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
FIELDS = "nextPageToken, files(id, name, mimeType, size, md5Checksum, webViewLink)"

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


def list_children(service, folder_id: str) -> list:
    """Return the full (paginated) list of immediate children of folder_id
    as raw Drive API file resources: {id, name, mimeType, size, md5Checksum,
    webViewLink}. size is a string in bytes when present, absent for e.g.
    Google-native docs; md5Checksum is present for binary files like PDFs.
    """
    items = []
    page_token = None
    while True:
        request = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields=FIELDS,
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        response = _request_with_retry(request)
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items


def download_file(service, file_id: str, dest_path: str) -> None:
    """Stream a file's binary content to dest_path, with retries on
    retryable errors for each chunk (supports Shared Drives)."""
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
