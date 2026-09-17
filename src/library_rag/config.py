"""
Central configuration for the Phase 1 ingestion pipeline. Every tunable
constant lives here; secrets come only from environment variables (see
.env.example). Nothing in db.py / ingest.py / search.py / report.py / the
pipeline/ package should hardcode a number or path that belongs here.
"""
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
# Markdown is the permanent asset; Postgres is disposable and rebuildable from
# it (see ingest.py --rechunk). PDFs are a local cache of Drive, not an asset --
# safe to delete and re-download.
#
# WHERE that permanent asset lives depends on whether SUPABASE_* is configured.
# With storage on, process_book mirrors markdown to the bucket and drops the
# local copy, so the BUCKET holds it and these directories are working space;
# with storage off, nothing is uploaded and nothing is deleted, and the local
# files are the only copy. Either way there is exactly one durable home for the
# markdown, and losing it means re-extracting -- which for a scanned book means
# paying for OCR again.
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

# ---- Folder indexing ----
# The ceiling for "index this whole folder": a folder whose PDFs sum past this
# cannot be bulk-queued, and the button says so. One misclick on the root would
# otherwise put the entire drive (~143 GB, days of OCR and embedding spend)
# into flight. 500 MB is a real subfolder -- roughly 50-150 books -- while
# staying an order of magnitude short of the topic folders.
FOLDER_INDEX_LIMIT_BYTES = int(os.environ.get("FOLDER_INDEX_LIMIT_MB", "500")) * 1024 * 1024

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
# Raised from 5 back to 30 after a parallel run over the Books folder.
#
# The 5-minute window assumed the longest single STAGE fits inside it, which
# holds for the pilot corpus (2.2 MB average) and does not hold here: that
# folder has 486 PDFs over 30 MB and one at 433 MB. Two books failed with
#     [Errno 2] No such file or directory: data/pdfs/<id>.pdf
# and they were the two LARGEST failures (43.6 MB and 10.7 MB) while every
# small failure had an unrelated cause -- a live worker was still downloading
# when the reaper declared it dead, a second worker claimed the same book, and
# they collided on the shared per-book path.
#
# The cost of 30 is that a genuinely dead worker's book waits half an hour to be
# recovered instead of five minutes. That is the cheaper mistake: a slow retry
# is invisible, whereas stealing a live worker's book destroys the file it is
# reading and marks a good book failed.
# Left at 30, not the 90 the large-file phase ran at. 90 was a deliberate,
# temporary widening while the 83 books over 30 MB were ingested three at a time
# -- a 412 MB download can outlast half an hour in a single stage. Steady state
# is a handful of new books a day, where waiting 90 minutes to recover a dead
# worker's book is pure latency for no protection. Raise it again, temporarily,
# for any future bulk run over large files.
CLAIM_STALE_MINUTES = 30
MAX_ATTEMPTS = 3

# The same reaper idea for research runs, in seconds because they are shorter.
# A run whose row still says 'running' but whose heartbeat is older than this is
# treated as dead, and a reader streaming it is told so rather than waiting
# forever -- the failure a Postgres-backed stream has that an in-memory one did
# not, because the reader can now outlive the writer.
#
# Generous on purpose. The heartbeat only ticks when the loop yields an event,
# and it yields nothing while waiting on a model call, so the window has to
# exceed the slowest single turn. Declaring a live run dead is a visible
# regression; taking two minutes to notice a dead one is not.
RESEARCH_STALE_SECONDS = 120

# Where the worker actually runs, when it is not this process. Set
# INGEST_JOB_NAME to the Cloud Run Job's name and the API stops draining the
# queue itself: it queues the rows and asks that Job to do the work (see
# jobs.py for why -- a BackgroundTask on Cloud Run only gets CPU while a
# request is in flight, which ingestion by definition is not).
#
# EMPTY BY DEFAULT, which is the whole point: local development, `make serve`
# and the test suite have no Cloud project, get False from run_ingest_job, and
# fall straight back to the in-process drain that has always been there.
INGEST_JOB_NAME = os.environ.get("INGEST_JOB_NAME", "")
INGEST_JOB_REGION = os.environ.get("INGEST_JOB_REGION", "us-west1")
# Normally inferred from the runtime's own credentials (the metadata server on
# Cloud Run knows its project); set only to point at a different project.
INGEST_JOB_PROJECT = os.environ.get("INGEST_JOB_PROJECT", "")

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


# ---- Who is allowed to use the deployed app (see web/auth.py) ----
# Sign-in with Google, gating the WHOLE app. This is authentication -- proving
# who a visitor is -- and it is deliberately separate from the Drive OAuth in
# drive/client.py, which is the OWNER's read-only access to their own Drive.
# Two different Google clients, two different scopes, two different accounts:
# conflating them would ask every visitor to hand over their own Drive.
#
# A *Web application* OAuth client ID from the Google Cloud console, NOT the
# Desktop-app one in credentials.json. Public by design -- it ships in the login
# page's HTML, and the security comes from Google signing the ID token and this
# server verifying that signature and audience. There is no client secret in
# this flow.
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Comma-separated allowlist. Anyone can sign in with Google; only these get in.
# Compared case-insensitively after trimming, because "Nael@Gmail.com" and
# "nael@gmail.com" are one Google account and a capital letter should not be a
# lockout.
ALLOWED_EMAILS = frozenset(
    e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()
)

# Signs the session cookie. Changing it logs everyone out, which is the intended
# panic button. Generated per-process when unset so local development works with
# no setup -- and deliberately NOT persisted, so a deployment that forgets to set
# it logs everyone out on every restart rather than running on a guessable key.
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(32)

# How long a sign-in lasts before Google must be consulted again.
SESSION_MAX_AGE_SECONDS = int(os.environ.get("SESSION_MAX_AGE_HOURS", "168")) * 3600

# Set only over HTTPS. Off by default so http://localhost still works; the
# deploy docs turn it on, and Cookie: Secure is what stops the session being
# readable over a plain-HTTP connection.
SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "").lower() in {"1", "true", "yes"}


# ---- Feature flags ----
# The Bible reader: its own page, its own three API routes, and a nav entry.
#
# Off by default. It is a genuinely separate product from the librarian ->
# classroom -> tutor path the rest of the app is about, and it was crowding a
# sidebar with four items in it. Gated rather than deleted because the code
# works and the data is loaded; this is a decision about what the app is for,
# and those get reversed more often than they get regretted.
#
# BIBLE_ENABLED=1 brings back the page, the routes and the nav link together --
# hiding the link while leaving /bible reachable would be a menu that lies.
BIBLE_ENABLED = os.environ.get("BIBLE_ENABLED", "").lower() in {"1", "true", "yes"}


# ---- Classrooms ----
# The most books one classroom may hold.
#
# A cap is not tidiness. Migration 0010 dropped the global chunk index BECAUSE
# retrieval was going to be scoped; an uncapped classroom quietly rebuilds the
# unindexed full-corpus scan that decision was predicated on avoiding.
#
# Measured on this corpus (8,679 books, 1.76M chunks, ~203 chunks per book),
# scoped search through chunks_book_id_idx:
#
#     books   chunks   dense   hybrid      (warm)
#         5      951    28ms     29ms
#        15    2,674    41ms     51ms
#        30    5,343    60ms     85ms
#        50   13,577   123ms    191ms
#       100   21,326   187ms    285ms
#
# The plan never changes -- it is an index scan at every size -- so the number
# that matters is how much TOAST the scan must read. A chunk's embedding is
# ~2 KB, so 30 books is ~11 MB and stays resident across a conversation, while
# 100 books is ~43 MB that evicts itself between questions: the same 100-book
# scope measured 187ms warm and 2,745ms cold. The cap is really a promise that
# a classroom fits in cache.
#
# 80 is where that promise still holds and the product argument stops binding.
# It interpolates to ~21,000 chunks and ~34 MB -- inside the band that stayed
# resident, below the 100-book scope that fell off the cliff -- so the cache
# claim survives, with less margin than 50 had. The reason to spend that margin
# is that readers assemble shelves from real selections: a 64-book pick hitting
# a 50 cap is the case that prompted this, and refusing it taught nobody
# anything about scope. Scope is still the signal the whole design runs on;
# 80 of 8,679 books is under 1% of the corpus, so a classroom still means
# something. Past ~100 it stops meaning anything and the numbers above say so.
#
# The librarian's own ceiling (librarian/loop.py MAX_COUNT) stays at 50. It must
# never exceed this one -- a run returning more books than a shelf can hold is a
# 409 after two minutes of work -- but it has no reason to track it upward.
CLASSROOM_MAX_BOOKS = 80

# ---- Book profiles (the librarian's tier) ----
# How many topic centroids a book gets. round(sqrt(chunks)), clamped -- so a
# 5-chunk fragment gets 2 and a 500-chunk systematic theology gets 20.
#
# Scaling to length rather than fixing k matters in both directions. Too few and
# a big book's chapters merge into a blur that matches every query weakly (the
# measured failure of a single whole-book vector: median rank 6 vs 1). Too many
# and a short document is split into clusters that are noise, plus the index
# grows for nothing -- across the first 203 books this rule averages 5.9 vectors
# each, which projects to ~254k vectors and a ~94 MB binary index over the whole
# drive.
PROFILE_MIN_CLUSTERS = 1
PROFILE_MAX_CLUSTERS = 20
# Lloyd's algorithm converges on this data long before 25; the cap only exists
# so a pathological book cannot spin. Init is farthest-first, not random, so a
# rebuild produces byte-identical centroids -- an eval harness comparing two
# runs must be comparing the corpus, not the seed.
PROFILE_KMEANS_ITERS = 25
# Words of chunk text to fall back on when a cluster has no usable heading.
# Short enough to read in a result list, long enough to be recognisable.
PROFILE_LABEL_WORDS = 12


# ---- Librarian search ----
# How many centroids the binary leg shortlists before the exact rerank.
#
# MUST be <= PROFILE_EF_SEARCH. An HNSW scan cannot return more rows than
# ef_search permits, and it does not error when you ask for more -- it silently
# returns fewer. That cost a full evaluation: a LIMIT 800 against the default
# ef_search of 40 shortlisted 40 of 101,385 vectors, raising the LIMIT changed
# nothing, and the flat result read as proof the pool was irrelevant when it was
# proof the pool was never applied.
PROFILE_SHORTLIST = 800
# pgvector's hard ceiling is 1000; the shortlist above is sized under it.
# Measured over 44 questions against 8,680 books: at 40 the binary leg lost 19
# books outright, at 200 it lost 11, and at 1000 it matched an exact scan to the
# decimal -- so this is not a recall/latency tradeoff so much as the price of
# the two-stage design working at all. ~25 ms against ~13 ms at the default.
PROFILE_EF_SEARCH = 1000
# Trust in the filename/path leg when fusing it with content centroids.
# Lower than the 0.5 that config.DRIVE_RRF_LEXICAL_WEIGHT uses over titles
# alone, because here it is competing with a leg that reads what the book
# actually SAYS: measured on the eval questions, the title leg by itself missed
# 15-19 of 22, so it belongs in the fusion as a tiebreaker and a way to catch
# author and series names, not as an equal vote.
PROFILE_TITLE_WEIGHT = 0.3
# How close a centroid must be to a book's OWN best match to count towards
# matched_topics, in cosine-distance units.
#
# Measured relative to the book rather than against a fixed cutoff, because an
# absolute threshold means something different for a narrow query than a broad
# one. The first version simply counted centroids that reached the shortlist,
# which is only meaningful while the shortlist is a small fraction of the index:
# a unit test with six centroids total scored a book covering one topic and a
# book covering three identically, because everything reached the shortlist.
PROFILE_TOPIC_MARGIN = 0.06
# Past this cosine distance, treat the library as not really holding the topic.
#
# A nearest-neighbour ranker always fills its page, so "12 books returned" is
# never evidence that a subject is covered. Absolute distance is: measured by
# cli/eval_librarian.py over this corpus, twelve real briefs reach a best
# passage at 0.26-0.36, while five briefs on subjects the library genuinely
# lacks -- quantum chromodynamics, Japanese woodblock printing, Formula One
# aerodynamics -- bottom out at 0.51-0.58 with no overlap between the two
# populations. 0.50 sits in that gap.
#
# It is a signal to the librarian, not a filter. Books past the floor are still
# returned; the model is simply told the shelf may be thin so it can say so
# rather than recommending five plausible-looking titles about nothing.
PROFILE_RELEVANCE_FLOOR = 0.50


def auth_enabled() -> bool:
    """Whether the sign-in gate is active.

    Driven by GOOGLE_CLIENT_ID rather than a separate flag, so there is one
    thing to set and no way to have a client id configured but ignored. Unset
    means wide open, which is right for `uvicorn --reload` on a laptop and wrong
    for anything with a public URL -- api.py prints a startup warning when a
    server starts unauthenticated.
    """
    return bool(GOOGLE_CLIENT_ID)
