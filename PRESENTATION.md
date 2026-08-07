# Presentation script — library-rag

*A talk script, ~15 minutes at a normal speaking pace. **[bracketed cues]** tell you what
to have on screen. Everything quoted here is real: the numbers come from measurements
recorded in the code, and every claim has a file path you can open live.*

---

## 1. The problem and the shape of the system (1 min)

**[show: the app at localhost:8000, then the Drive page]**

> I have a Google Drive with about 57,000 theology PDFs in 8,000 folders — far more than
> anyone can read, and Drive's own search can't even find a book unless you already know
> a word in its title. I built a system that turns any subset of that Drive into a
> searchable, citable library: you pick books, a pipeline ingests them, and an AI agent
> answers questions **using only what those books actually say**, with citations that
> open the real PDF at the real page.

> Two pieces do the heavy lifting, and they're what I'll focus on: the **ingestion
> pipeline** — download, parse, chunk, embed — and the **two AI agents** built on top.
> Everything lives in one Python package:

```
src/library_rag/
├── ingest.py            ← pipeline orchestrator + work queue logic
├── pipeline/            ← the four stages: extract.py, chunking.py, embed.py
├── retrieval/           ← agent #1: answers questions (loop.py, tools.py)
├── exploration/         ← agent #2: browses the Drive (loop.py, tools.py)
├── db.py                ← Postgres: schema, queue, vector + lexical search
├── drive/               ← Drive client + a local metadata mirror of all 57k titles
├── web/                 ← FastAPI + a three-page frontend (chat / drive / queue)
└── config.py            ← every tunable, each with the measurement that set it
```

---

## 2. The ingestion pipeline (6–7 min)

### 2.1 Postgres is the work queue — `ingest.py`, `db.py`

**[show: `db.py` — `claim_next_book`, around line 107]**

> There's no Celery, no Redis, no queue service. The `books` table IS the queue. A worker
> claims work with one SQL statement — `UPDATE … WHERE id = (SELECT … FOR UPDATE SKIP
> LOCKED LIMIT 1)` — which means multiple workers can drain the same queue with no
> coordinator and no double-processing; Postgres's row locks are the scheduler.

> Three design details worth pointing out in that one query:
>
> 1. **Claims are leases, not locks.** `claimed_at` is a timestamp; a book claimed more
>    than 5 minutes ago is re-claimable (`CLAIM_STALE_MINUTES`, `config.py`). The worker
>    *heartbeats* — `db.touch_claim` at every stage boundary — so the lease only has to
>    outlive the longest single *stage*, not the longest *book*. A dead worker's book is
>    recovered in minutes; a slow OCR job never gets stolen mid-flight.
> 2. **`ORDER BY status DESC`** — the status enum is ordered
>    `discovered → downloaded → extracted → chunked`, so the most-advanced book is
>    claimed first: near-finished work completes before new work starts.
> 3. **Retry cap in the same statement.** `attempts` increments on claim; past
>    `MAX_ATTEMPTS = 3` the book flips to `failed` with the reason stored. One bad book
>    can never wedge the queue or burn API credits in a loop.

> Every book — whether it came from Drive or a direct upload — goes through the same
> queue. The *only* place the two sources differ is one function, `downloader_for`
> (`ingest.py:148`), which picks how the bytes arrive. Uploads are content-addressed:
> the file's md5 IS its filename, so a client-supplied filename never touches the
> filesystem, and re-uploading the same bytes is automatically recognized as a duplicate.

### 2.2 Stage 1 — Download (`ingest.py:244`)

> The stage itself is simple; the checks around it aren't:
>
> - The downloaded bytes are **md5-verified** against what Drive claimed. A mismatch
>   fails the ingest rather than indexing bytes we can't vouch for — and if a file's md5
>   *changes* in Drive, the upsert resets that book to `discovered` and reprocesses it,
>   because chunks describing a document that no longer exists are worse than no chunks.
> - The verified original is **mirrored to Supabase Storage** under `originals/<md5>.pdf`
>   (`storage.py`). That's what lets a citation open the actual PDF at the actual page
>   later. Deliberately *warn-only*: mirroring is durability, not correctness — a storage
>   outage must not fail an ingest that otherwise succeeded.

### 2.3 Stage 2 — Parsing / extraction — `pipeline/extract.py`

**[show: `extract.py` module docstring + the measurement table around line 28]**

> Every PDF becomes **markdown** — and markdown is the permanent asset. Postgres is
> declared disposable in `config.py`: everything downstream can be rebuilt from the
> markdown with `--rechunk`, without re-downloading or re-parsing anything.

> Two extraction paths, chosen by a **text-layer probe**: a PDF is "digital" if at least
> half of its first 10 pages carry more than 50 extractable characters.
>
> - **Digital → PyMuPDF** (pymupdf4llm) — fast, free, local.
> - **Scanned → Mistral OCR API** — and if no OCR key is configured, the book is parked
>   as `needs_ocr` instead of failing; a scanned book must never fail the run.
>
> The interesting engineering here was *measuring* the extractor. pymupdf4llm has a
> layout model, and the code carries the before/after table: on a 436-page book it found
> **20 headings without the layout model and 527 with it**. Headings drive chunking, so
> that one flag took median chunk size from ~3,000 characters of undifferentiated text
> to ~1,200 characters of real sections. The previous comment claiming the layout model
> was slow got retracted with measurements — it's actually 27% faster.

> One small trick with big consequences: every page boundary is written into the
> markdown as `<!-- page: N -->`. That's how a chunk knows its page range — which is how
> a citation can say "p. 54" and open the PDF there.

### 2.4 Stage 3 — Chunking — `pipeline/chunking.py`

**[show: `chunking.py` docstring — the three passes]**

> Chunking is **structure-aware, in three passes**:
>
> 1. Split on markdown headings (h1 through h6) — so chunks align with the book's own
>    sections, not arbitrary character offsets.
> 2. **Merge undersized siblings** up to `MIN_CHUNK_CHARS = 500` — the header splitter
>    only ever splits, so a study guide with dense subheadings would otherwise produce
>    hundreds of ~40-character fragments.
> 3. Recursively split anything still over `CHUNK_SIZE_CHARS = 3200` (with 400 overlap),
>    preferring paragraph > line > sentence boundaries.

> Two details I want to call out:
>
> - **Every chunk is embedded with its heading trail prepended** — the text that gets
>   embedded literally starts with `"Ch 3 > The Exile > Return"`. The section path often
>   carries more meaning than the paragraph itself, and it rides along into the vector.
> - **Why h6 matters:** the OCR'd study guides map almost all their headings to h6
>   (127 of 142 in one book). Registering h1–h6 took that book from 57 chunks with
>   leaked `######` markup to 121 clean, section-shaped chunks. The constant in
>   `config.py` carries those numbers as its justification.
>
> There's also a cleaning layer with measured justifications — e.g. pymupdf4llm's
> "picture text" (OCR-walked map labels in random order) appeared in **19% of chunks**
> and is dropped whole — and a junk-section filter (table of contents, index,
> copyright…) that matches the section's own heading exactly, because `"index"` as a
> substring would also kill "An Index of Divine Names".

> Deliberate boundary: cleaning lives in *chunking*, not extraction. What we embed is a
> policy decision, and changing policy costs a `--rechunk`, not a re-extraction.

### 2.5 Stage 4 — Embedding — `pipeline/embed.py`

**[show: `embed.py` — it's ~80 lines]**

> The whole file exists to enforce one rule: **this is the only place that calls the
> Voyage API.** Documents (`input_type="document"`) and queries (`input_type="query"`)
> must go through the same model — `voyage-4-lite`, 1024 dimensions — or query vectors
> and document vectors live in different spaces and rankings are silently garbage.
>
> Mechanics: batches of 128, retry only on HTTP 429 with exponential backoff + jitter,
> and vectors are stored as **pgvector `halfvec(1024)`** — half-precision, half the RAM,
> with an HNSW index built once after bulk load (building it before loading is wasted
> work, and the migration says so).

> And the finish line is **atomic**: `db.insert_chunks_and_finish` inserts all of a
> book's chunks AND flips it to `done` in one transaction. A crash leaves zero chunks
> and a claimable book — never a half-indexed one. Related guard: a book that extracts
> to zero chunks is marked `failed`, not `done`, because "done with nothing searchable"
> is a book silently lost from the corpus.

### 2.6 What retrieval ships — a measured decision (1 min)

**[show: `config.py` — the eval table around line 220]**

> The search layer supports dense, lexical, and hybrid (reciprocal-rank fusion, k=60,
> the Cormack et al. constant). Which one ships was decided by evaluation, and the eval
> table lives *in the config comment next to the setting*: on this corpus, dense and
> hybrid had **identical hit-rate@8**, and the MRR deltas pointed in opposite directions
> across question sets — noise, not a win. So chunk search ships dense, and the comment
> says "flip this only with a fresh evaluate.py run pasted into this comment."
>
> Interestingly, the *opposite* answer held for the 57k Drive **titles**: there, hybrid
> at lexical weight 0.5 beat dense 8/8 vs 7/8 — short titles reward exact-word evidence
> in a way full passages don't. Same machinery, two different measured answers.

---

## 3. The AI agents (5–6 min)

> Both agents are **hand-written tool loops on the Anthropic API** (`claude-opus-5`) —
> no framework. Each is a generator that yields typed events (`thinking`, `search`,
> `results`, `answer`, `done`), streamed to the browser as SSE, because seeing *which
> queries the agent ran and what came back* is the difference between a diagnostic tool
> and a demo.

### 3.1 Agent #1: the research agent — `retrieval/loop.py` + `retrieval/tools.py`

**[show: `loop.py` docstring, then the chat page answering a question]**

> The core argument for an agent, straight from the docstring: a one-shot RAG pipeline
> has a hard ceiling — whatever the single embedding of the user's raw question
> retrieves is all the model will ever see. *"How do these two books differ on the Holy
> Spirit"* is not one lookup; it's at least two, and the phrasing that finds each is not
> the user's phrasing. So here, **the model runs the loop**: it gets two tools —
> `search_library(query, k, book_id)` and `list_books()` — and decides what to search,
> how often, and when to stop.

> Three design decisions worth showing:
>
> 1. **Leashes, two kinds.** `MAX_ITERATIONS = 8` bounds the conversation loop;
>    `MAX_SEARCHES = 12` is a budget enforced by the *tool*, which returns
>    "budget exhausted — answer now" as a tool result. Exhaustion isn't an error; the
>    agent lands with whatever it has, flagged `exhausted: true`.
>
> 2. **The re-injected stop-check** (`tools.py:109`) — my favorite line in the repo:
>    *"A system-prompt instruction is read once and stops competing with the model's own
>    momentum by iteration three; state in the conversation is re-read every turn."*
>    So every single tool result carries the budget line and a prompt: if these passages
>    answer the question, ANSWER NOW; only search again if you can *name* the gap.
>    That — not the system prompt — is what actually holds effort proportional to the
>    question.
>
> 3. **The Session and stable citations** (`tools.py:27`). Sources accumulate across all
>    searches in a run into one numbered list, de-duplicated by chunk id — so citation
>    [7] means the same passage no matter which of the twelve searches surfaced it.
>    Each source carries the chunk's full provenance: book, heading trail, page range,
>    cosine distance, ordinal ("passage 14 of 330"). The frontend turns those into
>    clickable citations that open a panel with the original markdown and an
>    "Open the PDF at p.54" link — served by a signed Supabase URL with a `#page=54`
>    fragment.

> Grounding is enforced in both directions: the prompt forbids filling gaps from the
> model's own theology knowledge, and retrieval reports a **measured weak-match
> threshold** — real hits on this corpus land at cosine distance 0.33–0.62, off-topic
> lands 0.77+, so anything past 0.70 is flagged *weak* rather than silently dropped.
> That lets the agent distinguish "the library is silent" from "my phrasing missed,"
> and reformulate instead of giving up.

### 3.2 Agent #2: the Drive librarian — `exploration/loop.py` + `exploration/tools.py`

**[show: the /library page, "Ask the librarian", then `exploration/loop.py`]**

> The second agent solves the *"what should I read next"* problem: the library holds a
> hundred books, the Drive holds 57,000. This is the walk-into-a-bookstore agent — you
> say "early church history," it runs several differently-angled searches, and hands
> back a shortlist with reasons.

> It gets four tools, and the constraints are the design:
>
> - **`search_drive`** — hybrid semantic+lexical search over all 57k titles *and their
>   folder paths*, against a local Postgres mirror of Drive's metadata. Originally this
>   called Drive's `name contains`, which matches word *prefixes* — "parab" hit,
>   "arable" returned zero — and the agent's own traces showed it compensating by
>   guessing author surnames. Mirroring the metadata locally (57k title embeddings,
>   halfvec + HNSW) turned "the end times" into a query that actually reaches
>   Revelation and eschatology.
> - **`browse_folder`** — one level, never recursive, with explicit `truncated` and
>   `total_pdfs` fields, because *"a silently-cut list is indistinguishable from a
>   complete one, and an agent that cannot tell the difference will confidently say
>   'that folder only has 50 books'."*
> - **`estimate_pipeline`** — all cost/time arithmetic happens in code, never in the
>   model's head.
> - **`recommend`** — the agent's *output channel*. It must pass `{file_id, why}` picks,
>   and here's the hallucination guard: the loop remembers every file id any tool
>   actually returned this run, and `recommend` rejects any id not in that set. A
>   hallucinated pick becomes a flagged error instead of a dead link on screen.
>
> And the single most useful piece of context it gets: every search result is annotated
> `indexed: true` if the book is already in the library — because the most embarrassing
> failure mode for a librarian is recommending you a book you already own.

---

## 4. Close (30 s)

**[show: the chat page with a cited answer, panel open on a source]**

> So: a Postgres-native pipeline where every stage is idempotent, resumable, and carries
> the measurement that justified it — feeding two hand-built agents whose defining
> features are *restraint* (budgets, stop-checks, leashes) and *verifiability* (typed
> event streams, stable citations, hallucination guards, and a citation that ends at a
> real page of a real PDF).
>
> The through-line I'd want you to take away: almost every constant in `config.py` has
> a number next to it explaining *why* — eval tables, before/after counts, measured
> latencies. The system is built to be re-decided, not just built.

---

## Appendix — where everything lives (leave on screen for Q&A)

| Piece | Files |
|---|---|
| Queue / claims / heartbeat | `src/library_rag/db.py` (claim_next_book, touch_claim), `config.py` (CLAIM_STALE_MINUTES=5, MAX_ATTEMPTS=3) |
| Pipeline orchestrator | `src/library_rag/ingest.py` (process_book, process_queue, downloader_for) |
| Download + verify + mirror | `ingest.py:244-258`, `src/library_rag/storage.py` |
| Extraction (PyMuPDF / Mistral OCR) | `src/library_rag/pipeline/extract.py`; probe thresholds in `config.py:121-123` |
| Chunking (3 passes, heading trails) | `src/library_rag/pipeline/chunking.py`; sizes in `config.py:126-160` |
| Embedding (voyage-4-lite, 1024-dim) | `src/library_rag/pipeline/embed.py`; model in `config.py:167-173` |
| Vector + lexical + RRF search | `src/library_rag/db.py` (search, _search_fused); eval tables in `config.py:186-296` |
| Research agent | `src/library_rag/retrieval/loop.py` (prompt, tool schemas, event loop), `retrieval/tools.py` (Session, budget, 0.70 cutoff) |
| Drive librarian agent | `src/library_rag/exploration/loop.py` (4 tools, prompt), `exploration/tools.py` (mirror search, indexed_ids, recommend guard) |
| Drive metadata mirror | `src/library_rag/drive/mirror.py` (sync, path CTE, title embeddings), migrations 0003–0005 |
| Web app + SSE streaming | `src/library_rag/web/api.py`, `web/static/{index,library,queue}.html`, `app.js` (readEventStream) |
| Tests (182, no network) | `tests/` — scripted fake Anthropic/Voyage/Drive clients; local-Docker-only Postgres guard in `tests/conftest.py` |
