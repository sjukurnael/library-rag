"""
One-off: mirror every existing book's original PDF into Supabase Storage.

Ingestion mirrors at download time from now on, so this exists only to carry
the books indexed BEFORE storage was wired up. Safe to re-run: uploads are
content-addressed upserts, so an object that is already there costs one PUT
of the same bytes and nothing changes.

    ./.venv/bin/python scripts/backfill_storage.py

A book whose bytes exist nowhere locally (Drive book whose working cache was
cleaned) is reported and skipped -- the viewer route fetches those from Drive
on first click and mirrors them then.
"""
from dotenv import load_dotenv

load_dotenv()

from library_rag import config, db, ingest, storage  # noqa: E402


def local_original(book) -> object | None:
    if book["source"] == "upload":
        p = ingest.upload_path(book["source_id"])
        if p.exists():
            return p
    p = config.PDF_DIR / f"{book['id']}.pdf"
    return p if p.exists() else None


def main() -> None:
    if not storage.enabled():
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY are not set.")

    mirrored, skipped, missing = [], [], []
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM books ORDER BY id")
            ids = [r[0] for r in cur.fetchall()]
        for book_id in ids:
            book = db.fetch_book(conn, book_id)
            if not book["md5"]:
                skipped.append((book_id, book["title"], "no md5 recorded"))
                continue
            path = local_original(book)
            if path is None:
                missing.append((book_id, book["title"]))
                continue
            ok = storage.put_original(book["md5"], path)
            (mirrored if ok else skipped).append(
                (book_id, book["title"], "" if ok else "upload failed")
            )
            print(f"  [{book_id}] {'ok  ' if ok else 'FAIL'} {book['title']}")

    print(f"\nMirrored {len(mirrored)} book(s).")
    for book_id, title, why in skipped:
        print(f"  skipped [{book_id}] {title} -- {why}")
    for book_id, title in missing:
        print(f"  no local bytes [{book_id}] {title} -- will mirror on first view")


if __name__ == "__main__":
    main()
