"""
The agent's tools. Kept deliberately dumb: mechanical Drive listing and
mechanical arithmetic. All judgment (which folder looks promising, what
scanned_ratio to assume, which candidate wins) belongs in agent/loop.py's
system prompt / the LLM, not here.
"""
import json
import os
import random
import statistics

from library_rag.drive import client as drive_client
from library_rag.exploration import assumptions

CACHE_FILE = "cache.json"
FOLDER_MIME = "application/vnd.google-apps.folder"
PDF_MIME = "application/pdf"


# ---------------------------------------------------------------- caching --

def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# ------------------------------------------------------------ list_folder --

def list_folder(folder_id: str, fresh: bool = False) -> dict:
    """List ONE level of a Drive folder. Never recurses. Caches to cache.json
    keyed by folder_id; pass fresh=True (--fresh at the CLI) to bypass cache.
    """
    cache = _load_cache()
    if not fresh and folder_id in cache:
        return cache[folder_id]

    service = drive_client.build_service()
    folder_meta = drive_client.get_folder_name(service, folder_id)
    children = drive_client.list_children(service, folder_id)

    subfolders = []
    pdfs = []  # each: {name, id, url, size_mb}
    other_count = 0

    for item in children:
        mime = item.get("mimeType", "")
        if mime == FOLDER_MIME:
            subfolders.append({
                "id": item["id"],
                "name": item.get("name", ""),
                "url": item.get("webViewLink", ""),
            })
        elif mime == PDF_MIME and item.get("size") is not None:
            pdfs.append({
                "name": item.get("name", ""),
                "id": item["id"],
                "url": item.get("webViewLink", ""),
                "size_mb": int(item["size"]) / (1024 * 1024),
            })
        else:
            # Non-PDF junk, AND PDFs missing a size field (can't include them
            # in size stats, so they're bucketed as "other" per spec).
            other_count += 1

    sizes = [p["size_mb"] for p in pdfs]
    result = {
        "folder": {
            "name": folder_meta.get("name", ""),
            "url": folder_meta.get("webViewLink", ""),
        },
        "subfolders": subfolders,
        "files": {
            "pdf_count": len(pdfs),
            "pdf_total_mb": round(sum(sizes), 1) if sizes else 0.0,
            "pdf_size_min_mb": round(min(sizes), 1) if sizes else 0.0,
            "pdf_size_median_mb": round(statistics.median(sizes), 1) if sizes else 0.0,
            "pdf_size_max_mb": round(max(sizes), 1) if sizes else 0.0,
            "other_count": other_count,
            "sample_pdfs": _sample_pdfs(pdfs),
        },
    }

    cache[folder_id] = result
    _save_cache(cache)
    return result


def _sample_pdfs(pdfs: list, n: int = 8) -> list:
    """Up to n PDFs: a mix of the largest (interesting for OCR-cost judgment)
    plus a random sample of the rest, deduped."""
    if len(pdfs) <= n:
        return sorted(pdfs, key=lambda p: -p["size_mb"])

    n_largest = max(1, n // 2)
    by_size = sorted(pdfs, key=lambda p: -p["size_mb"])
    largest = by_size[:n_largest]
    largest_ids = {p["id"] for p in largest}
    remainder = [p for p in pdfs if p["id"] not in largest_ids]
    random_pick = random.sample(remainder, min(n - n_largest, len(remainder)))
    return largest + random_pick


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
