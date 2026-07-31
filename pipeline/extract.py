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

# Set explicitly rather than relying on the library default, because this one
# flag decides whether the whole pipeline sees any document structure at all.
#
# The layout model (pymupdf_layout / onnxruntime, a hard dependency of
# pymupdf4llm >= 1.28) segments each page visually -- title, heading, paragraph,
# table -- and reflows text into real paragraphs. The classic path instead reads
# font metadata and infers heading levels from a document-wide size histogram.
#
# That histogram approach collapses on this corpus. Measured 2026-07-30 over 5
# books (3 Jensen guides + 2 doctrine books):
#
#   headings found   classic -> layout    chunks    median chunk chars
#   holy_spirit (436p)    20 -> 527        329->584     3158 -> 1385
#   biblical_doc (248p)  140 -> 549        194->330     2388 ->  932
#   jensen 1 Samuel (120p) 3 -> 142         54->121     2924 -> 1123
#
# On the Jensen guides -- scanned paper with an OCR'd text layer, so font
# metadata is synthetic and uniform -- the classic path found 3 headings in 120
# pages and wrapped 77% of the book in ``` code fences (indented/monospaced runs
# read as "preformatted"). MarkdownHeaderTextSplitter correctly ignores headings
# inside fences, so chunking saw 2 sections in a whole book and fell back to
# slicing at CHUNK_SIZE_CHARS.
#
# A previous comment here justified use_layout(False) by claiming the layout
# model "silently drops text on some pages and is slow". Neither reproduced:
# it yields MORE text on 4 of 5 books (+7.7% on 1 Corinthians, -0.15% on the one
# exception) and is 27% faster (407 vs 554 ms/page on a 100-page guide).
pymupdf4llm.use_layout(True)

EXTRACTOR_VERSION = "3.0"  # bumped: layout model changes extraction output


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
