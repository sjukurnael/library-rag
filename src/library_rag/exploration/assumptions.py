"""
Every constant used by estimate_pipeline() lives here, with a rationale.
Tune these after the pilot run produces real numbers -- that's the point
of running a pilot at all. Nothing in tools.py should hardcode a number
that belongs here.

All costs are USD. All "ranges" are (low, high) tuples.
"""

# ---- Page density (how many MB does one page take, by kind) ----
# A scanned page is a raster image (300dpi photo of a book page); a
# digital-native page is mostly vector text + maybe a figure. Both are
# rough averages across a "typical" academic/library PDF.
MB_PER_SCANNED_PAGE = 1.5   # source: typical 300dpi grayscale scan ~= 1-2MB/page
MB_PER_DIGITAL_PAGE = 0.15  # source: typical text-native PDF page ~= 0.1-0.2MB/page

# ---- Download ----
# Expressed in megaBITS/sec (like an ISP speed rating) since that's how most
# people know their connection speed; tools.py divides by 8 to get MB/s.
DOWNLOAD_MBPS = 15  # conservative sustained throughput for Drive API download over
                    # a home/office connection with retries; not a speed test number

# ---- Extraction / OCR ----
# Digital-native pages: PyMuPDF text extraction is local, CPU-only, and fast
# enough to treat as free and instant at pilot scale.
DIGITAL_EXTRACT_PAGES_PER_HR = 100_000  # effectively "instant"; used only to keep
                                         # the time math honest instead of a hardcoded 0

# Scanned pages, path (a): hosted OCR API (e.g. Mistral OCR).
OCR_API_COST_PER_1K_PAGES = (2.0, 4.0)   # $ per 1,000 pages, low-high, per published
                                          # per-page OCR API pricing as of 2026
OCR_API_PAGES_PER_HR = (150, 400)        # single-key sequential throughput once
                                          # per-minute rate limits and retries on
                                          # dense/low-quality scans are factored in

# Scanned pages, path (b): rent a GPU and run an open OCR model (e.g. Marker).
GPU_COST_PER_HR = 0.40            # ~price of a mid-tier cloud GPU instance (e.g.
                                   # T4/L4 class spot/on-demand) as of 2026
MARKER_PAGES_PER_HR = (800, 2500) # Marker throughput on a single mid-tier GPU,
                                   # scanned/mixed PDFs, published benchmarks range

# ---- Chunk + embed ----
TOKENS_PER_PAGE = (350, 500)          # rough tokens per extracted page of text
EMBED_COST_PER_MTOK = (0.02, 0.13)    # $ per 1M tokens, low (cheap open/batch API)
                                       # to high (premium hosted embedding API)
EMBED_TOKENS_PER_HR = 3_000_000       # throughput bound assuming a single-worker
                                       # client against a rate-limited embeddings API

# ---- Storage ----
CHUNK_TOKENS = 800  # target tokens per chunk before embedding
PG_BYTES_PER_CHUNK_VECTOR = 2 * 1024      # halfvec embedding row, ~2KB/chunk
PG_BYTES_PER_CHUNK_HNSW_INDEX = 2 * 1024  # HNSW index overhead, roughly matches
                                           # vector size for typical graph fan-out
MARKDOWN_BYTES_PER_PAGE = 2 * 1024        # extracted markdown/text stored per page,
                                           # for an S3 (or local) archive copy

# ---- Pilot decision caps (used by the agent's reasoning, not by the math tool) ----
PILOT_MAX_GB = 1.5
PILOT_MAX_COST_USD = 5.0
