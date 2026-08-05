"""
The filesystem half of 0002_book_sources.sql. Run once, after that migration.

    python scripts/rekey_local_uploads.py

0002 labelled the old `--local` books as source='upload' but left their
source_id as "local:<filename>". Re-keying to the content-addressed
"upload:<md5>" needs two things SQL cannot do: hash the file, and copy it into
data/uploads/ where downloader_for will look for it. Doing the rename in SQL
alone would leave a book whose id points at a file nobody ever put there.

Idempotent: books already keyed "upload:" are skipped, so a second run is a
no-op. The bytes come from data/pdfs/<book_id>.pdf, the working copy the
original ingest left behind -- which is why this must run before anything
prunes that cache.

It lives in scripts/ rather than migrations/ because it is a one-off run by a
human against a specific database, not part of the schema sequence.
"""
import hashlib
import shutil
import sys
from pathlib import Path

from library_rag import config, db


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, source_id, title FROM books WHERE source_id LIKE 'local:%' "
            "ORDER BY id"
        ).fetchall()
        if not rows:
            print("Nothing to re-key.")
            return 0

        for book_id, source_id, title in rows:
            cached = config.PDF_DIR / f"{book_id}.pdf"
            if not cached.exists():
                print(f"[{book_id}] {title}: SKIP -- no working copy at {cached}")
                continue

            md5 = _md5(cached)
            dest = config.UPLOAD_DIR / f"{md5}.pdf"
            if not dest.exists():
                shutil.copyfile(cached, dest)

            conn.execute(
                "UPDATE books SET source_id = %s, md5 = %s, size_bytes = %s "
                "WHERE id = %s",
                (f"upload:{md5}", md5, dest.stat().st_size, book_id),
            )
            print(f"[{book_id}] {title}: {source_id} -> upload:{md5[:12]}...")
        conn.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
