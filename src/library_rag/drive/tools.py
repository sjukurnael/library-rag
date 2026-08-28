"""
Reading Google Drive as a library: list a folder, search it, and know which of
what comes back we already hold.

Kept deliberately dumb -- mechanical listing and a mechanical "do I already own
this" lookup. All judgment about which book is worth reading belongs to the
librarian, which is a caller of this module rather than a part of it.

These live here rather than under the agent that first needed them because the
Drive tree is not an agent's private concern: the librarian searches it, and the
Drive browser page lists it. They outlived the browsing agent that introduced
them, which is exactly why they moved.

TITLES are the payload. The folder tools return names and let the caller page
through them, rather than aggregate statistics over a sample -- an earlier
version answered "how big is this folder", which is the opposite question.
"""
import json
import statistics

from library_rag import config, db
from library_rag.drive import client as drive_client
from library_rag.pipeline import embed as embed_mod

FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"

DEFAULT_LIMIT = 50
MAX_LIMIT = 120


# ---------------------------------------------------------------- caching --

def _load_cache() -> dict:
    path = config.DRIVE_CACHE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    config.DRIVE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ----------------------------------------------------------- own-library --

def indexed_ids(conn) -> dict:
    """Drive file id -> {book_id, title, status} for every Drive book we hold.

    The single most useful thing the agent can know. Without it the obvious
    failure is recommending a book already sitting in the library -- which reads
    as the agent not knowing what it is looking at. Keyed on source_id because
    that IS the Drive file id for source='drive' (see 0002_book_sources.sql).
    """
    rows = conn.execute(
        "SELECT source_id, id, title, status FROM books WHERE source = 'drive'"
    ).fetchall()
    return {r[0]: {"book_id": r[1], "title": r[2], "status": r[3]} for r in rows}


def _as_pdf(item: dict, owned: dict) -> dict:
    """One Drive file resource -> the shape the agent and the UI both consume."""
    size = item.get("size")
    have = owned.get(item["id"])
    return {
        "file_id": item["id"],
        "title": item.get("name", ""),
        "url": item.get("webViewLink", ""),
        "size_mb": round(int(size) / (1024 * 1024), 1) if size is not None else None,
        "indexed": have is not None,
        # Only present when indexed, so an absent key is unambiguous rather than
        # a null the model has to interpret.
        **({"book_id": have["book_id"], "status": have["status"]} if have else {}),
    }


# ---------------------------------------------------------- browse_folder --

def _raw_folder(folder_id: str, fresh: bool = False) -> dict:
    """The cached Drive listing for one folder, normalised but not filtered.

    Cached because folder contents are stable and a shelf gets revisited within
    a single browse; keyed by folder_id.
    """
    cache = _load_cache()
    if not fresh and folder_id in cache:
        return cache[folder_id]

    service = drive_client.build_service()
    folder_meta = drive_client.get_folder_name(service, folder_id)
    children = drive_client.list_children(service, folder_id)

    subfolders, pdfs, other_count = [], [], 0
    for item in children:
        mime = item.get("mimeType", "")
        if mime == FOLDER_MIME:
            subfolders.append({
                "folder_id": item["id"],
                "name": item.get("name", ""),
                "url": item.get("webViewLink", ""),
            })
        elif mime == PDF_MIME and item.get("size") is not None:
            pdfs.append(item)
        else:
            # Non-PDF junk (.DS_Store, index.html), AND PDFs with no size field.
            other_count += 1

    result = {
        "folder": {
            "name": folder_meta.get("name", ""),
            "url": folder_meta.get("webViewLink", ""),
        },
        "subfolders": subfolders,
        "pdfs": pdfs,
        "other_count": other_count,
    }
    cache[folder_id] = result
    _save_cache(cache)
    return result


def browse_folder(
    conn, folder_id: str, name_contains=None, limit=DEFAULT_LIMIT, fresh=False
) -> dict:
    """One shelf: its subfolders, and the PDFs on it by title.

    ONE level, never recursive. `name_contains` filters within the folder and
    `limit` caps the page, because folders here are not small -- `Books` holds
    5,024 PDFs directly alongside 204 subfolders, and returning all of them
    would bury the answer in the question.

    `truncated` and `total_pdfs` are always reported. A silently-cut list is
    indistinguishable from a complete one, and an agent that cannot tell the
    difference will confidently say "that folder only has 50 books".
    """
    raw = _raw_folder(folder_id, fresh=fresh)
    owned = indexed_ids(conn)

    pdfs = raw["pdfs"]
    if name_contains:
        needle = name_contains.lower()
        pdfs = [p for p in pdfs if needle in p.get("name", "").lower()]

    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    shown = sorted(pdfs, key=lambda p: p.get("name", "").lower())[:limit]
    sizes = [int(p["size"]) / (1024 * 1024) for p in raw["pdfs"]]

    return {
        "folder": raw["folder"],
        "subfolders": raw["subfolders"],
        "pdfs": [_as_pdf(p, owned) for p in shown],
        "total_pdfs": len(pdfs),
        "truncated": len(pdfs) > len(shown),
        "junk_files": raw["other_count"],
        # Retained from Phase 0: still the input estimate_pipeline needs, and
        # still the only signal for "are these scans or digital text".
        "size_median_mb": round(statistics.median(sizes), 1) if sizes else 0.0,
        "size_total_mb": round(sum(sizes), 1) if sizes else 0.0,
    }


# ----------------------------------------------------------- search_drive --

def search_drive(conn, query: str, limit=DEFAULT_LIMIT, *, voyage=None) -> dict:
    """Rank the whole drive by meaning and by words, from the local mirror.

    This used to call Drive's `name contains`, which matches a word PREFIX --
    measured on this corpus, 'parab' hit and 'arable' returned zero. That forced
    the agent to guess exact title words, and its own trail showed the workaround:
    after topic searches it fell back to author surnames (Jeremias, Blomberg,
    Bailey) because that was the only way to reach a book whose title it could
    not predict. Against the mirror, "the end times" reaches Revelation.

    Falls back to the live Drive call when the mirror is empty, so a fresh
    install still works before anyone has run drive_sync.
    """
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    mirrored = db.drive_sync_status(conn)["files"]
    if mirrored == 0:
        service = drive_client.build_service()
        items = drive_client.search_files(service, query, limit=limit)
        owned = indexed_ids(conn)
        return {
            "query": query, "source": "drive-live",
            "matches": [_as_pdf(i, owned) for i in items],
            "returned": len(items),
            # Meaningful HERE and only here: a `name contains` match really can
            # come back short, so a full page means Drive had more to give.
            "hit_limit": len(items) >= limit,
        }

    voyage = voyage or embed_mod.build_client()
    vec = embed_mod.embed_query(query, voyage)
    rows = db.search_drive_files(conn, vec, query, limit)
    return {
        "query": query,
        "source": "mirror",
        "matches": [
            {
                "file_id": r["file_id"], "title": r["title"], "url": r["url"],
                "size_mb": r["size_mb"], "path": r["path"],
                "indexed": r["indexed"],
                **({"book_id": r["book_id"], "status": r["status"]}
                   if r["indexed"] else {}),
            }
            for r in rows
        ],
        "returned": len(rows),
        # Deliberately NO hit_limit on this branch, though the live-Drive branch
        # above still reports one.
        #
        # The dense leg is a KNN scan with no relevance threshold: every one of
        # the ~57k embedded titles has a distance to any query vector, so the
        # candidate pool ALWAYS fills and this search ALWAYS returns k rows --
        # for a real topic and for gibberish alike. Measured: "zzzqqqx
        # nonexistent gibberish term" returns a full 50 and would have reported
        # "hit the limit" just as "the end times" did. The flag was true on every
        # search ever run against the mirror, so it carried no information, and
        # the agent -- which is handed this whole dict -- spent extra turns
        # re-searching to cover a shortfall that was never there.
        #
        # What the agent can actually act on: these are the top `returned` of the
        # WHOLE mirror, already ranked, so there is no unseen better match to go
        # looking for -- only worse ones.
        "ranked_over": mirrored,
        # Titles BOTH rankers surfaced -- matched on words and among the nearest
        # by meaning. Agreement is the signal; neither leg alone is.
        #
        # Counting lexical hits alone was tried first and is worthless: the
        # tsquery ORs every term, so one ordinary word carries the whole query.
        # Measured over 11 queries, "purple monetary derivatives arbitrage" and
        # "zzzqqqx nonexistent gibberish term" each matched 10 titles on words
        # (on "term", on "monetary") -- indistinguishable from a real topic.
        # Agreement separated them cleanly: 6/6 real subjects scored 2 or more,
        # 4/5 off-topic scored exactly 0. The lone leak was a contrived query
        # ending in the word "time".
        "word_and_meaning": sum(
            1 for r in rows if r.get("dense_rank") and r.get("lexical_rank")
        ),
    }
