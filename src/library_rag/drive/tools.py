"""
The browsing agent's tools. Kept deliberately dumb: mechanical Drive listing,
a mechanical "do I already own this" lookup, and mechanical arithmetic. All
judgment (which shelf looks promising, which book fits the reader's interest,
what scanned_ratio to assume) belongs in loop.py's system prompt / the LLM.

These used to answer "how big is this folder" -- Phase 0 picked a pilot folder
by size. Browsing asks the opposite question: TITLES are the payload and size is
a footnote, so the folder tools return names and let the caller page through
them rather than returning aggregate statistics over an 8-file sample.
"""
import json
import statistics

from library_rag import config, db
from library_rag.drive import client as drive_client
from library_rag.exploration import assumptions
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


# -------------------------------------------------------------- recommend --

def recommend(conn, picks: list, seen: dict) -> dict:
    """The agent's OUTPUT channel, not a side effect. Nothing is written here.

    The agent supplies {file_id, why}; this re-attaches the Drive metadata
    already fetched during the browse so the UI can render a real card (title,
    size, link, Add button) instead of regex-parsing prose for Drive URLs.

    `seen` is the run's file_id -> Drive item map. A file_id the agent never
    actually looked at comes back flagged rather than silently dropped -- a
    hallucinated id is exactly the failure worth surfacing.
    """
    owned = indexed_ids(conn)
    out = []
    for pick in picks:
        fid = pick.get("file_id", "")
        item = seen.get(fid)
        if item is None:
            out.append({
                "file_id": fid,
                "why": pick.get("why", ""),
                "unknown": True,
                "error": "no such file_id was returned by any search this run",
            })
            continue
        out.append({**_as_pdf(item, owned), "why": pick.get("why", "")})
    return {"recommendations": out}


# -------------------------------------------------------- estimate_pipeline --

def _hours_to_label(low_hr: float, high_hr: float) -> str:
    if high_hr < (1 / 60):
        return "instant"
    if high_hr < 1:
        lo_min, hi_min = round(low_hr * 60), round(high_hr * 60)
        if lo_min == hi_min:
            return f"~{hi_min} min"
        return f"{lo_min}-{hi_min} min"
    lo, hi = round(low_hr, 1), round(high_hr, 1)
    if lo == hi:
        return f"~{hi} hrs"
    return f"{lo}-{hi} hrs"


def _cost_to_label(low: float, high: float) -> str:
    if high == 0:
        return "free"
    if round(low, 2) == round(high, 2):
        return f"${low:.2f}"
    return f"${low:.2f}-${high:.2f}"


def estimate_pipeline(pdf_count: int, total_mb: float, scanned_ratio: float) -> dict:
    """Deterministic stage-by-stage estimate. No LLM arithmetic — this is the
    only place estimation math happens. Returns everything as (low, high)
    ranges, labeled, so the caller never has to do its own math either.
    """
    a = assumptions
    scanned_ratio = max(0.0, min(1.0, scanned_ratio))

    scanned_mb = total_mb * scanned_ratio
    digital_mb = total_mb * (1 - scanned_ratio)

    scanned_pages = scanned_mb / a.MB_PER_SCANNED_PAGE
    digital_pages = digital_mb / a.MB_PER_DIGITAL_PAGE
    total_pages = scanned_pages + digital_pages

    # -- Download (free, time-bound by bandwidth) --
    download_mb_per_s = a.DOWNLOAD_MBPS / 8
    download_hr = (total_mb / download_mb_per_s) / 3600 if download_mb_per_s else 0
    download = {
        "time_hr": (download_hr, download_hr),
        "time_label": _hours_to_label(download_hr, download_hr),
        "cost_usd": (0.0, 0.0),
        "cost_label": "free",
    }

    # -- Extraction: digital is ~free/instant; scanned needs OCR, two paths --
    digital_extract_hr = digital_pages / a.DIGITAL_EXTRACT_PAGES_PER_HR

    ocr_api_hr = (
        scanned_pages / a.OCR_API_PAGES_PER_HR[1],
        scanned_pages / a.OCR_API_PAGES_PER_HR[0],
    )
    ocr_api_cost = (
        (scanned_pages / 1000) * a.OCR_API_COST_PER_1K_PAGES[0],
        (scanned_pages / 1000) * a.OCR_API_COST_PER_1K_PAGES[1],
    )
    extract_api_hr = (digital_extract_hr + ocr_api_hr[0], digital_extract_hr + ocr_api_hr[1])

    gpu_hr = (
        scanned_pages / a.MARKER_PAGES_PER_HR[1],
        scanned_pages / a.MARKER_PAGES_PER_HR[0],
    )
    gpu_cost = (gpu_hr[0] * a.GPU_COST_PER_HR, gpu_hr[1] * a.GPU_COST_PER_HR)
    extract_gpu_hr = (digital_extract_hr + gpu_hr[0], digital_extract_hr + gpu_hr[1])

    extraction = {
        "pages": {"scanned": round(scanned_pages), "digital": round(digital_pages),
                   "total": round(total_pages)},
        "api_path": {
            "time_hr": extract_api_hr,
            "time_label": _hours_to_label(*extract_api_hr),
            "cost_usd": ocr_api_cost,
            "cost_label": _cost_to_label(*ocr_api_cost),
        },
        "gpu_path": {
            "time_hr": extract_gpu_hr,
            "time_label": _hours_to_label(*extract_gpu_hr),
            "cost_usd": gpu_cost,
            "cost_label": _cost_to_label(*gpu_cost),
        },
    }

    # -- Chunk + embed --
    tokens = (total_pages * a.TOKENS_PER_PAGE[0], total_pages * a.TOKENS_PER_PAGE[1])
    embed_cost = (
        tokens[0] * a.EMBED_COST_PER_MTOK[0] / 1_000_000,
        tokens[1] * a.EMBED_COST_PER_MTOK[1] / 1_000_000,
    )
    embed_hr = (tokens[0] / a.EMBED_TOKENS_PER_HR, tokens[1] / a.EMBED_TOKENS_PER_HR)
    chunk_embed = {
        "tokens": tokens,
        "time_hr": embed_hr,
        "time_label": _hours_to_label(*embed_hr),
        "cost_usd": embed_cost,
        "cost_label": _cost_to_label(*embed_cost),
    }

    # -- Storage --
    chunks = (tokens[0] / a.CHUNK_TOKENS, tokens[1] / a.CHUNK_TOKENS)
    pg_bytes_per_chunk = a.PG_BYTES_PER_CHUNK_VECTOR + a.PG_BYTES_PER_CHUNK_HNSW_INDEX
    pg_gb = (chunks[0] * pg_bytes_per_chunk / 1e9, chunks[1] * pg_bytes_per_chunk / 1e9)
    md_gb = total_pages * a.MARKDOWN_BYTES_PER_PAGE / 1e9
    storage = {
        "chunks": (round(chunks[0]), round(chunks[1])),
        "postgres_gb": pg_gb,
        "markdown_gb": (md_gb, md_gb),
        "fits_on_laptop": pg_gb[1] < 20,  # a few GB of vectors is trivially laptop-sized
    }

    # -- Totals, per OCR-path scenario --
    total_cost_api = (ocr_api_cost[0] + embed_cost[0], ocr_api_cost[1] + embed_cost[1])
    total_cost_gpu = (gpu_cost[0] + embed_cost[0], gpu_cost[1] + embed_cost[1])
    total_time_api = (
        download_hr + extract_api_hr[0] + embed_hr[0],
        download_hr + extract_api_hr[1] + embed_hr[1],
    )
    total_time_gpu = (
        download_hr + extract_gpu_hr[0] + embed_hr[0],
        download_hr + extract_gpu_hr[1] + embed_hr[1],
    )

    return {
        "inputs": {
            "pdf_count": pdf_count,
            "total_mb": total_mb,
            "scanned_ratio": scanned_ratio,
        },
        "pages": extraction["pages"],
        "download": download,
        "extraction": extraction,
        "chunk_embed": chunk_embed,
        "storage": storage,
        "totals": {
            "api_path": {
                "cost_usd": total_cost_api,
                "cost_label": _cost_to_label(*total_cost_api),
                "time_hr": total_time_api,
                "time_label": _hours_to_label(*total_time_api),
            },
            "gpu_path": {
                "cost_usd": total_cost_gpu,
                "cost_label": _cost_to_label(*total_cost_gpu),
                "time_hr": total_time_gpu,
                "time_label": _hours_to_label(*total_time_gpu),
            },
        },
    }
