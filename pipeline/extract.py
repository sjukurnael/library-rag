"""
PDF -> markdown, plus the manifest that records how it got there.

Digital-native PDFs (those with a usable text layer) are extracted locally with
PyMuPDF: fast, free, no network call. Image-only (scanned) PDFs need OCR: if an
ocr_client is provided (ingest builds one from MISTRAL_API_KEY) they go through
the Mistral OCR API, otherwise extract() raises NeedsOCRError and the caller
marks the book needs_ocr and moves on -- a scanned book with no OCR configured
must never fail the run.

Text-layer probe (config.TEXT_LAYER_*): a PDF is treated as digital when at
least half of its first 10 pages carry >50 extractable characters.

Every page boundary in the output markdown is marked with `<!-- page: N -->`
so pipeline/chunking.py can recover page_start/page_end per chunk.
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

import config

# pymupdf4llm >= 1.28 defaults to an ML layout model (pymupdf.layout / onnxruntime)
# when it's importable. That model silently drops text on some pages and is slow.
# use_layout(False) selects the classic, deterministic path: document-wide
# font-statistics heading detection (IdentifyHeaders), no ML inference.
pymupdf4llm.use_layout(False)

EXTRACTOR_VERSION = "2.0"


class NeedsOCRError(RuntimeError):
    """Raised when a PDF is scanned and no ocr_client was provided."""


@dataclass
class ExtractionResult:
    markdown: str
    page_count: int
    has_text_layer: bool
    extractor: str  # "pymupdf" | "mistral-ocr"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probe_text_layer(doc: "fitz.Document") -> bool:
    """True when >= TEXT_LAYER_MIN_PAGE_RATIO of the first TEXT_LAYER_PROBE_PAGES
    pages have more than TEXT_LAYER_MIN_CHARS extractable characters."""
    n_probe = min(config.TEXT_LAYER_PROBE_PAGES, doc.page_count)
    if n_probe == 0:
        return False
    texty = sum(
        1
        for i in range(n_probe)
        if len(doc[i].get_text("text").strip()) > config.TEXT_LAYER_MIN_CHARS
    )
    return (texty / n_probe) >= config.TEXT_LAYER_MIN_PAGE_RATIO


def extract(pdf_path: Path, book_id: int, ocr_client=None) -> ExtractionResult:
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count
        has_text_layer = probe_text_layer(doc)
        if has_text_layer:
            markdown = _pymupdf_to_markdown(doc)
            return ExtractionResult(markdown, page_count, True, "pymupdf4llm")
    finally:
        doc.close()

    if ocr_client is None:
        raise NeedsOCRError(
            f"book {book_id} looks scanned (image-only) and no OCR client is "
            "configured (set MISTRAL_API_KEY)"
        )

    markdown, ocr_page_count = _mistral_ocr_to_markdown(ocr_client, pdf_path)
    return ExtractionResult(markdown, ocr_page_count or page_count, False, "mistral-ocr")


def write_outputs(
    book_id: int, drive_file_id: str, pdf_path: Path, result: ExtractionResult
) -> None:
    md_path = config.MARKDOWN_DIR / f"{book_id}.md"
    manifest_path = config.MARKDOWN_DIR / f"{book_id}.manifest.json"
    md_path.write_text(result.markdown, encoding="utf-8")
    manifest = {
        "book_id": book_id,
        "drive_file_id": drive_file_id,
        "extractor": result.extractor,
        "extractor_version": EXTRACTOR_VERSION,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "page_count": result.page_count,
        "has_text_layer": result.has_text_layer,
        "pdf_sha256": sha256_file(pdf_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_manifest(book_id: int, **fields) -> None:
    """Merge extra fields (timing, chunk counts) into an existing manifest.
    Used by ingest.py after chunking/embedding completes, so report.py can read
    wall-clock-per-stage from the manifest without the DB schema needing timing
    columns."""
    manifest_path = config.MARKDOWN_DIR / f"{book_id}.manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.update(fields)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(book_id: int) -> dict:
    manifest_path = config.MARKDOWN_DIR / f"{book_id}.manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------- pymupdf --

def _pymupdf_to_markdown(doc: "fitz.Document") -> str:
    """Digital-native PDF -> markdown via pymupdf4llm. It derives heading levels
    from document-wide font statistics (and renders lists/tables), rather than
    the per-page, max-span, +1pt font guess this module used to hand-roll -- the
    old heuristic promoted almost every slightly-large line to a heading, which
    shattered study-guide PDFs into thousands of tiny chunks downstream.

    page_chunks=True returns one markdown string per page, so we keep the
    `<!-- page: N -->` markers pipeline/chunking.py relies on to recover
    page_start/page_end for each chunk."""
    pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    parts = []
    for i, page in enumerate(pages, start=1):
        parts.append(f"<!-- page: {i} -->")
        parts.append(page["text"])
    return "\n\n".join(parts)


# --------------------------------------------------------- mistral ocr --

def build_ocr_client():
    """Construct a Mistral client for OCR, or return None if MISTRAL_API_KEY is
    unset (scanned books then fall through to needs_ocr)."""
    if not config.MISTRAL_API_KEY:
        return None
    from mistralai import Mistral  # lazy import: optional dependency path

    return Mistral(api_key=config.MISTRAL_API_KEY)


def _mistral_ocr_to_markdown(client, pdf_path: Path) -> tuple:
    uploaded = client.files.upload(
        file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)
    result = client.ocr.process(
        model=config.MISTRAL_OCR_MODEL,
        document={"type": "document_url", "document_url": signed_url.url},
    )

    parts = []
    for i, page in enumerate(result.pages, start=1):
        parts.append(f"<!-- page: {i} -->")
        parts.append(page.markdown)
    return "\n\n".join(parts), len(result.pages)
