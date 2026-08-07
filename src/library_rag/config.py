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
# The repo root, three levels up from src/library_rag/config.py. Data lives
# beside the source tree, not inside it: `data/` is the user's corpus -- their
# uploaded originals and extracted markdown -- and packaging user data into an
# importable package would put it wherever pip happened to install this.
# Overridable so a container or an installed copy can point somewhere writable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LIBRARY_RAG_DATA_DIR", PROJECT_ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
MARKDOWN_DIR = DATA_DIR / "markdown"
# Uploaded originals. This is to an uploaded book what Drive is to a Drive book:
# the only copy of the bytes that exists, so it is NOT disposable the way
# PDF_DIR is. PDF_DIR stays a per-book working cache keyed by book_id, and
# process_book copies an upload into it exactly as it downloads a Drive file --
# which is what lets both sources share the pipeline below the download step.
UPLOAD_DIR = DATA_DIR / "uploads"
PDF_DIR.mkdir(parents=True, exist_ok=True)
MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# The browsing agent's Drive-listing cache. Absolute, under DATA_DIR: it used to
# be the bare relative "cache.json", which resolves against whatever cwd the
# process happened to start in -- so the CLI and the web server could disagree
# about where the cache was, and tests could write one into the repo root.
DRIVE_CACHE_FILE = DATA_DIR / "drive_cache.json"

# ---- Upload limits ----
# Enforced while streaming to disk, not after: reading an unbounded upload into
# memory to measure it is the denial of service, and a Content-Length header is
# a claim by the client, not a fact.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "64")) * 1024 * 1024
# Every PDF begins with this. Checked against the actual first bytes rather than
# the filename or the browser-supplied Content-Type, both of which the client
# chooses freely.
PDF_MAGIC = b"%PDF-"

# ---- Supabase Storage (optional) ----
# When both are set, every successfully downloaded original is mirrored to the
# bucket and the PDF viewer serves signed URLs from it; when either is empty,
# the app is purely local and no network storage is touched. Tests blank these
# (conftest) so the suite can never talk to a real bucket.
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "library-bucket")
# Lifetime of a signed PDF link. Long enough to read a book chapter from it,
# short enough that a leaked link goes stale the same afternoon.
SIGNED_URL_TTL_SECONDS = 3600

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
#
# 5, not 30, because ingest.py now heartbeats (db.touch_claim) at every stage
# boundary. Without a heartbeat this had to exceed the slowest possible book --
# a long OCR job -- or a live worker's book would be stolen mid-flight. With
# one, the only thing that must fit inside the window is the longest single
# stage, so a dead worker's book is recovered in minutes instead of half an hour.
CLAIM_STALE_MINUTES = 5
MAX_ATTEMPTS = 3

# ---- Drive ----
# Books / Jensen Bible Self Study Guides -- the pilot folder explore.py
# picked (23 PDFs, 117.7 MB, mostly digital-native).
PILOT_FOLDER_ID = os.environ.get(
    "PILOT_FOLDER_ID", "1ZkjfpG7KPve2grlQhJ7ZHXLhuyC9b5tL"
)

# The shared library's top folder. The metadata mirror is pruned to this
# subtree, and the browsing agent starts here.
#
# Scoping is not cosmetic. Drive's `files.list` has no "within this subtree"
# filter, so a sync necessarily pulls everything the account can see -- which on
# a personal account is coursework, sheet music, tax documents and a resume. The
# first real sync mirrored 126 such files alongside the 57,401 library ones, and
# they are worse than noise: a browse UI that surfaces someone's personal
# documents is one nobody can show anyone. Set to None to mirror the whole drive.
DRIVE_ROOT_FOLDER_ID = os.environ.get(
    "DRIVE_ROOT_FOLDER_ID", "1jOO-7ZAEosq2mAtuVzTTq9uAekrypVfq"
) or None

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

# ---- Retrieval ----
# Search fuses two rankers over the same chunks: dense (cosine over the Voyage
# embedding) and lexical (Postgres full-text over the generated `tsv` column).
# They fail in opposite directions, which is the whole reason to run both.
# Dense retrieval matches meaning and survives paraphrase, but it blurs rare
# literal tokens -- a proper name, a reference like "2 Tim 3:16", a word that
# appears twice in the corpus -- because those are barely represented in a
# 1024-dim vector trained on general text. Lexical matching nails exactly those
# and is useless the moment the user's wording differs from the book's.
#
# Reciprocal Rank Fusion combines them on RANK, not score: a chunk's fused score
# is sum(1 / (RRF_K + rank_in_that_leg)) over the legs that returned it. Rank is
# the only thing the two legs share. Cosine distance (0-2, and its useful band
# is corpus-dependent) and ts_rank_cd (unbounded, length-normalised) are not on
# a comparable scale, and normalising them per query would mean knowing each
# query's score distribution before ranking it. Ranks need no calibration and no
# per-query tuning, which is why RRF is the default rather than a weighted sum.
#
# RRF_K = 60 is the constant from Cormack et al. (2009) and the de-facto
# default. It damps the head of each list: a chunk that ranks #1 in one leg and
# nowhere in the other scores 1/61 = 0.0164, while a chunk that ranks #3 in both
# scores 2/63 = 0.0317 and wins. Agreement between the legs beats a single
# leg's confidence, which is exactly the behaviour worth having.
RRF_K = 60
# How the user's question becomes a tsquery for the lexical leg.
#   "and" -- websearch_to_tsquery, which ANDs every term. High precision, and
#            frequently zero recall: "What was Paul asking of Philemon?" ANDs to
#            0 of 1,423 chunks, so the lexical leg contributes nothing at all and
#            hybrid silently degenerates to plain dense search.
#   "or"  -- stem and drop stopwords via to_tsvector, then OR the lexemes. The
#            same question matches 427 chunks; ts_rank_cd and the candidate pool
#            do the sorting.
# Set from measurement, not taste -- see `python -m evaluate --compare` and the
# numbers recorded in eval/questions.json's header. The tradeoff is real in both
# directions: Postgres full-text has no IDF, so an OR query cannot tell that
# "Philemon" is rarer than "ask", and its extra recall can be extra noise.
LEXICAL_TSQUERY = "and"
# Relative trust in the lexical leg when fusing. RRF as published is unweighted
# -- it treats every ranker as equally good. That assumption is what breaks
# here: fusing a weak ranker with a strong one does not average them, it drags
# the strong one down, so the weight exists to be able to say "count this leg,
# but less".
RRF_LEXICAL_WEIGHT = 0.25

# ---- Which retrieval mode actually ships ----
# "dense". Measured, not assumed -- and NOT the answer that was expected when
# the lexical leg was built.
#
# Two question sets of 22 questions each over the same 1,423-chunk corpus, same
# ground truth, scored by evaluate.py (eval/questions.json is phrased in the
# books' own vocabulary; eval/questions_paraphrase.json asks the same things the
# way a user would -- "Colossae" not "Colosse", "grandkids" not "grandchildren"):
#
#                       book-voice              user-voice
#                    hit@8      MRR          hit@8      MRR
#   dense           100.0%     0.865         77.3%     0.641
#   hybrid/and      100.0%     0.888         77.3%     0.633
#   hybrid/or       100.0%     0.867         72.7%     0.531
#   lexical/and      63.6%     0.614          9.1%     0.091
#   lexical/or       63.6%     0.286         36.4%     0.161
#   (hybrid rows at RRF_LEXICAL_WEIGHT = 0.25; the full sweep is in the commit)
#
# hit-rate@8 is IDENTICAL between dense and hybrid/and on both sets. Fusion
# only reshuffles ranks within a result set that already contained the right
# passage, and the MRR deltas (+0.023 one set, -0.008 the other) point in
# opposite directions at n=22 -- that is noise, not a win. So there is no
# evidence hybrid helps here, and a default that measurement does not support is
# the exact failure this harness was built to catch. Dense ships.
#
# What would change the answer, in rough order of likelihood:
#   - A much larger corpus. At 1,423 chunks the dense leg is under no pressure;
#     rare literal tokens are still distinctive in embedding space because there
#     is so little to confuse them with. Re-run this at 100K chunks.
#   - IDF. Postgres full-text has none -- ts_rank_cd cannot tell that "Philemon"
#     is rarer than "ask", which is most of why the lexical leg scores so badly.
#     BM25 (via an extension) rather than better fusion is the thing to try.
#   - Reranking a fused candidate pool with a cross-encoder, which fixes ordering
#     with a model instead of with arithmetic over ranks.
#
# The lexical path stays in the codebase because it is what makes that re-run a
# one-flag experiment instead of a rewrite. Flip this to "hybrid" only with a
# fresh evaluate.py run pasted into this comment.
SEARCH_MODE = "dense"
# Candidates pulled from EACH leg before fusing. Over-fetch on purpose: a chunk
# sitting at dense rank 30 and lexical rank 2 should be able to win, and it can
# only do that if both legs are asked for more than the k the caller wants.
HYBRID_CANDIDATES = 50

# ---- OCR (optional; only exercised for scanned/image-only PDFs) ----
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = "mistral-ocr-latest"
# USD per 1,000 pages, Mistral OCR API non-batch pricing, verified 2026-07 at
# https://mistral.ai/pricing/api/
OCR_COST_PER_1K_PAGES = 4.0


# ---- Drive title search ----
# Weight of the lexical leg when fusing search over drive_files (titles + folder
# paths), separate from RRF_LEXICAL_WEIGHT because the two corpora behave
# differently and the measurement says so.
#
# Over CHUNKS, hybrid tied dense and SEARCH_MODE stayed "dense". Over TITLES it
# genuinely helps. Measured on 8 natural-language queries against all 57,527
# titles, scoring hit-rate@5 and MRR against a substring ground truth:
#
#     dense           7/8   MRR 0.719
#     lexical         5/8   MRR 0.448
#     hybrid w=1.0    7/8   MRR 0.719
#     hybrid w=0.5    8/8   MRR 0.812   <- shipped
#     hybrid w=0.25   7/8   MRR 0.812
#     hybrid w=0.1    7/8   MRR 0.812
#
# Read that carefully. Fusing beats either leg alone, but an EQUAL weight does
# not: at 1.0 the lexical leg drags in machine-generated filenames like
# "07_John_Jesus_and_History_Volume_3_Glimpses..." that rank high on token count
# rather than relevance, and "the end times" returns essays on how Mark ends his
# narrative. Held at 0.5 it breaks dense's ties without setting the order.
#
# 0.5 over 0.25/0.1 is worth one question and no MRR -- inside the noise of an
# 8-question set. Re-measure with a larger set before treating the gap as real.
DRIVE_RRF_LEXICAL_WEIGHT = 0.5
