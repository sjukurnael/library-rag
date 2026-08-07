"""
Remote home for a book's original bytes: Supabase Storage.

One bucket (config.SUPABASE_BUCKET), content-addressed keys -- an original
lives at originals/<md5>.pdf, so the same bytes uploaded twice occupy one
object and a client-supplied filename never becomes a storage path.

Everything here is OPTIONAL and warn-only by design. The pipeline's job is to
make a book searchable; mirroring its bytes to the bucket is durability, not
correctness, and a storage outage must not fail an ingest that otherwise
succeeded. The one caller that needs to know whether storage is live at all
asks enabled() first.

Tests never reach this module's network side: conftest blanks the SUPABASE_*
config, so enabled() is False and every function is a cheap no-op.
"""
import time
from pathlib import Path

from library_rag import config


def enabled() -> bool:
    return bool(config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY)


def original_key(md5: str) -> str:
    return f"originals/{md5}.pdf"


_client = None


def _bucket():
    """The supabase-py storage handle, built once. Imported lazily so that a
    purely local install (or the test suite) never pays for -- or needs --
    the supabase package's import chain."""
    global _client
    if _client is None:
        from supabase import create_client

        _client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    return _client.storage.from_(config.SUPABASE_BUCKET)


def put_original(md5: str, path: Path) -> bool:
    """Mirror a PDF into the bucket. True if it is there afterwards.

    Upsert, because re-processing a book is routine (--rechunk, retries) and
    "already exists" is success, not an error.
    """
    if not (enabled() and md5):
        return False
    # Three tries with a short backoff: multi-MB uploads over home networks
    # hit transient TLS resets (observed: "EOF in violation of protocol",
    # "bad record mac") often enough that one attempt loses real books, and
    # rarely enough that three attempts practically never do.
    for attempt in range(3):
        try:
            with open(path, "rb") as f:
                _bucket().upload(
                    original_key(md5),
                    f.read(),
                    file_options={"content-type": "application/pdf", "upsert": "true"},
                )
            return True
        except Exception as e:  # noqa: BLE001 -- durability, not correctness
            if attempt == 2:
                print(f"  WARNING: could not mirror original to storage: {e}")
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def delete_original(md5: str) -> bool:
    if not (enabled() and md5):
        return False
    try:
        _bucket().remove([original_key(md5)])
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not delete original from storage: {e}")
        return False


def signed_original_url(md5: str, expires_in: int | None = None) -> str | None:
    """A short-lived direct link to the original, or None if storage is off,
    the object is missing, or Supabase is unreachable. Callers treat None as
    "fall back to a local copy", so a storage outage degrades instead of 500s.
    """
    if not (enabled() and md5):
        return None
    try:
        res = _bucket().create_signed_url(
            original_key(md5), expires_in or config.SIGNED_URL_TTL_SECONDS
        )
        url = res.get("signedURL") or res.get("signedUrl")
        # Older storage3 releases return the path relative to /storage/v1.
        if url and url.startswith("/"):
            url = f"{config.SUPABASE_URL}/storage/v1{url}"
        return url
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: could not sign storage URL: {e}")
        return None
