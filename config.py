"""
Central configuration for the Phase 1 ingestion pipeline. Every tunable
constant lives here; secrets come only from environment variables (see
.env.example). Nothing in db.py / ingest.py / search.py / report.py / the
pipeline/ package should hardcode a number or path that belongs here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
# Markdown is the permanent asset; Postgres is disposable and rebuildable
# from it (see ingest.py --rechunk). PDFs are a local cache of Drive, not
# an asset -- safe to delete and re-download.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
MARKDOWN_DIR = DATA_DIR / "markdown"
PDF_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

# ---- Database ----
# Port 5434 must match the host side of docker-compose.yml's "5434:5432"
# mapping. It is deliberately not 5432/5433 -- both are commonly taken by a
# native Postgres install, and pointing at the wrong server fails with a
# confusing auth/missing-database error rather than "nothing is listening".
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://app:app@localhost:5434/library"
)

# ---- Worker / queue ----
# A book "claimed" (claimed_at set) but not advanced within this window is
# assumed to belong to a dead worker and is re-claimable (the reaper clause of
# claim_next_book). MAX_ATTEMPTS caps retries: a book claimed for the Nth time
# with attempts already >= MAX_ATTEMPTS is marked failed instead of processed.
CLAIM_STALE_MINUTES = 30
MAX_ATTEMPTS = 3

# ---- Drive ----
# Books / Jensen Bible Self Study Guides -- the pilot folder explore.py
# picked (23 PDFs, 117.7 MB, mostly digital-native).
PILOT_FOLDER_ID = os.environ.get(
    "PILOT_FOLDER_ID", "1ZkjfpG7KPve2grlQhJ7ZHXLhuyC9b5tL"
)

# ---- Extraction ----
# Text-layer probe: a PDF has a usable text layer (digital-native, extract with
# PyMuPDF) when at least TEXT_LAYER_MIN_PAGE_RATIO of the first
# TEXT_LAYER_PROBE_PAGES pages carry more than TEXT_LAYER_MIN_CHARS extractable
# characters. Otherwise it is image-only (scanned) and routed to OCR.
TEXT_LAYER_PROBE_PAGES = 10
TEXT_LAYER_MIN_CHARS = 50
TEXT_LAYER_MIN_PAGE_RATIO = 0.5

# ---- Chunking ----
CHUNK_SIZE_CHARS = 3200
CHUNK_OVERLAP_CHARS = 400
# Through h6, not h4. pymupdf4llm maps a document's font sizes onto heading
# levels, and on the scanned Jensen guides almost everything lands at h6 (127 of
# 142 headings in 1 Samuel, 173 of 186 in 1 Corinthians). Stopping at h4 left
# those invisible to MarkdownHeaderTextSplitter, so they stayed in the body and
# went into the embeddings as literal "######". Registering them here took
# 1 Samuel from 57 chunks (median 2915, 50 carrying leaked markup) to 121
# chunks (median 1123, none leaking).
MARKDOWN_HEADERS = [
    ("#", "h1"), ("##", "h2"), ("###", "h3"),
    ("####", "h4"), ("#####", "h5"), ("######", "h6"),
]
# Sections shorter than this are merged with adjacent siblings (same parent
# heading) before chunking. Study guides are full of dense `###` subheadings
# with two lines each; without merging, every one becomes its own ~40-char
# chunk, which embeds noisily and retrieves badly. 500 is well under
# CHUNK_SIZE_CHARS, so merging never forces an immediate re-split.
MIN_CHUNK_CHARS = 500
# Front/back matter, dropped before chunking: a table of contents matches every
# topical query on keyword overlap while containing no prose, and an index is a
# list of names. Matched EXACTLY against a section's OWN heading (not as a
# substring of the full trail) -- "index" as a substring also matches
# "An Index of Divine Names", and matching the trail lets a junk-looking `h1`
# silently drop every real section nested beneath it.
JUNK_HEADINGS = {
    "table of contents",
    "contents",
    "index",
    "bibliography",
    "copyright",
    "about the author",
    "acknowledgments",
    "acknowledgements",
}

# ---- Embedding ----
# Hard requirement: search.py must use the exact same model as ingest.py, or
# query/document vectors live in different spaces. Both import
# pipeline/embed.py's embed_query / embed_documents -- never call the
# Voyage API anywhere else.
# voyage-4-lite: 1024-dim (matches the HALFVEC(1024) schema), and on the free
# 200M-token tier -- voyage-3 lost its free allocation in 2026.
EMBED_MODEL = "voyage-4-lite"
EMBED_DIM = 1024
EMBED_BATCH_SIZE = 128
# USD per 1M tokens, voyage-4-lite non-batch pricing, verified 2026-07.
EMBED_COST_PER_MTOK = 0.02
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

# ---- OCR (optional; only exercised for scanned/image-only PDFs) ----
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
# USD per 1,000 pages, Mistral OCR API non-batch pricing, verified 2026-07 at
# https://mistral.ai/pricing/api/
OCR_COST_PER_1K_PAGES = 4.0
