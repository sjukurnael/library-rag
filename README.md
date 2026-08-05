# library-rag

A RAG "library tutor" chatbot over ~130GB of theology PDFs in a shared
Google Drive. Built in phases:

- **Phase 0** (`explore.py`): a hand-written Anthropic tool-use agent that
  explores the Drive folder tree (metadata only, no downloads) and picks a
  pilot folder with an estimated time/dollar cost for ingesting it.
- **Phase 1** (`ingest.py`, `search.py`, `report.py`, this section): the real
  ingestion pipeline, run end-to-end on the pilot folder only — download →
  extract → chunk → embed → store in Postgres with vector search.
- **Phase 2+** (not built yet): the tutor chatbot itself, evals, S3 storage,
  the full 130GB backfill.

## Layout

```
library-rag/
├── explore.py            # Phase 0 CLI: pilot-folder exploration/estimation
├── agent/                # the exploration agent
│   ├── loop.py           #   system prompt + hand-written tool-use loop
│   ├── tools.py          #   list_folder (cached Drive listing) + estimate_pipeline
│   ├── assumptions.py    #   every estimation constant, each with a rationale comment
│   └── research.py       #   Phase 1: the retrieval agent's own search loop
├── drive/                # Drive access, shared by both phases
│   └── client.py         #   OAuth, paginated listing, download, retries
├── config.py              # Phase 1: every ingestion tunable (paths, models, chunk sizes)
├── db.py                  #   Postgres access: queue semantics (claim/reap) + CRUD
├── migrations/            #   *.sql schema, applied in filename order
│   ├── 0001_init.sql      #     books + chunks (halfvec embedding, tsv, doc_status enum)
│   ├── 0002_book_sources.sql        # drive_file_id -> source_id, + source column
│   └── 0002_rekey_local_uploads.py  # its filesystem half (one-off, not a migration)
├── migrate.py              #   tiny forward-only migration runner (schema_migrations)
├── docker-compose.yml      #   pgvector/pgvector:pg17, port 5434, healthcheck
├── ingest.py               #   the worker: Drive PDF -> markdown -> chunks -> embeddings
├── search.py               #   retrieval smoke test CLI + the deterministic control
├── evaluate.py             #   retrieval quality harness: hit-rate@k / MRR
├── eval/questions*.json    #   curated ground truth (book + page span, not chunk id)
├── api.py                  #   FastAPI: UI, /api/books, /api/books/upload, /api/research
├── static/index.html       #   the UI: question box, inline citations, PDF upload
├── report.py               #   calibration report: measured vs. assumptions.py
├── pipeline/               #   ingest.py's internals
│   ├── extract.py         #     PyMuPDF / Mistral OCR -> markdown + manifest
│   ├── chunking.py         #     markdown -> chunks with page ranges + ordinals
│   └── embed.py             #     the one place that calls the Voyage API
├── pyproject.toml          #   package + dev deps; `pip install -e ".[dev]"`
├── data/                    #   gitignored
│   ├── uploads/            #     uploaded originals, content-addressed (NOT disposable)
│   ├── markdown/           #     the permanent, rebuildable-from asset
│   └── pdfs/               #     per-book working cache, safe to delete
├── .github/workflows/ci.yml #   pgvector service -> ruff check -> pytest
└── tests/                   #   real Postgres, fake externals, zero-network
    ├── conftest.py         #     test-DB lifecycle + fake Voyage/Drive
    ├── test_estimate.py    #     Phase 0: estimate_pipeline arithmetic
    ├── test_migrations.py  #     migration runner idempotency
    ├── test_discover.py    #     enumerate/upsert
    ├── test_claim.py       #     queue: SKIP LOCKED, reaper, attempt cap
    ├── test_chunking.py    #     chunk boundaries, pages, ordinals, junk skip
    ├── test_embed_store.py #     atomic chunk+done, crash safety
    ├── test_extract.py     #     text-layer probe, corrupt-PDF recovery
    ├── test_research.py    #     the agent tool loop, scripted fake client
    ├── test_eval.py        #     the quality harness, incl. a negative control
    └── test_upload.py      #     upload validation, content-addressing, queueing
```

`drive/` is shared by both phases. `agent/` is Phase 0 only. Everything
under Phase 1 treats `data/markdown/*.md` (+ its `.manifest.json`) as the
permanent, rebuildable-from asset — Postgres is disposable
(`ingest.py --rechunk` rebuilds every chunk/embedding from local markdown
alone, no Drive or OCR calls).

## Setup

### 1. Python environment

```bash
cd ~/Desktop/library-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Anthropic API key

Create a `.env` file in the project root (or copy the example):

```bash
cp .env.example .env
```

Then edit `.env` and set your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`explore.py` loads this automatically via `python-dotenv`. Do not commit `.env`
— it is gitignored.

### 3. Google Cloud OAuth setup (first-time, ~10 minutes)

You need an OAuth "Desktop app" client so the script can read your Drive
account's "Shared with me" folder. Read-only access, nothing is ever written.

1. Go to https://console.cloud.google.com/ and create a new project (or pick
   an existing one) — the picker is in the top bar next to the Google Cloud logo.
2. In the left sidebar: **APIs & Services → Library**. Search for
   "Google Drive API" and click **Enable**.
3. **APIs & Services → OAuth consent screen**.
   - User type: **External** (unless you have a Workspace org, then Internal is fine).
   - Fill in an app name (e.g. "library-rag-explorer"), your email as support
     contact, and your email again as developer contact. Save and continue
     through the Scopes and Test users steps — you can leave scopes empty here.
   - On the **Test users** step, add your own Google account email (the one
     that has access to the shared Drive folder). While the app is in
     "Testing" mode, only test users you list can authorize it.
4. **APIs & Services → Credentials → + Create Credentials → OAuth client ID**.
   - Application type: **Desktop app**.
   - Name it anything (e.g. "library-rag-cli").
   - Click **Create**, then **Download JSON** on the resulting client.
5. Move that downloaded file into this repo and rename it exactly:
   ```bash
   mv ~/Downloads/client_secret_*.json ~/Desktop/library-rag/credentials.json
   ```
6. That's it — no need to touch scopes or verification. The script requests
   only `drive.readonly` at runtime, and the first run will open your browser
   for a one-time consent screen (click through the "unverified app" warning
   — that's expected for a personal test app in Testing mode — then Allow).
   It caches the resulting token to `token.json` so you won't be prompted again
   until the token expires.

`credentials.json` and `token.json` are both gitignored — never commit them.

## Running it

```bash
python explore.py
```

This prints a live `[agent] ...` trail of every Drive folder it lists and
every cost/time estimate it computes, then ends with a `RECOMMENDATION` block
naming a pilot folder, its URL, size, PDF count, a stage-by-stage time/cost
table, 1-2 runner-up options, and a full-130GB extrapolation for planning.

Flags:

- `--fresh` — bypass `cache.json` and re-crawl Drive from scratch (normal runs
  reuse the cache automatically, so a second run completes in seconds and
  makes zero Drive API calls).
- `--quiet` — suppress the `[agent]` tool-call trail, print only the final
  recommendation.

## Tuning the estimates

Every constant used by `estimate_pipeline` lives in `assumptions.py` with a
comment explaining where it came from. After running the actual pilot
pipeline (a later project), go back and replace these with real measured
numbers — that's the intended feedback loop. Nothing outside `assumptions.py`
needs to change when you retune a constant.

## Tests

```bash
python -m tests.test_estimate
```

## Troubleshooting

- **"Missing credentials.json"** — see step 5 above; the file must be named
  exactly `credentials.json` and live in this directory.
- **"token.json exists but refresh failed"** — delete `token.json` and re-run;
  you'll be sent through the browser consent flow again.
- **403 after retries** — the authenticated Google account doesn't have
  access to the shared folder, or the Drive API isn't enabled on your Cloud
  project (step 2).
- **404 on the root folder ID** — double check the folder ID and that it's
  actually shared with the account you authorized.

---

## Phase 1: Ingestion pipeline

Runs the pilot folder (`Books / Jensen Bible Self Study Guides`, 23 PDFs)
end to end: download → extract → chunk → embed → store in Postgres with
vector search.

### Setup

1. **Postgres.** Requires **Docker Desktop running** (human step).
   ```bash
   make db-up      # starts pgvector/pgvector:pg17 on localhost:5434
   python -m migrate   # apply migrations/*.sql (idempotent)
   ```
2. **API keys.** Add to `.env` (see `.env.example`):
   ```
   VOYAGE_API_KEY=pa-...       # required — embeddings (human step: obtain one)
   MISTRAL_API_KEY=            # optional — only for scanned/image-only PDFs
   ```
   The default `DATABASE_URL` is `postgresql://app:app@localhost:5434/library`
   (port 5434 matches docker-compose.yml; 5432 and 5433 are left free for any
   native Postgres install).
3. **Python deps** (if not already installed):
   ```bash
   source .venv/bin/activate
   pip install -e ".[dev]"     # runtime + dev (pytest, ruff)
   ```

### Run: empty DB → working search

```bash
python ingest.py --discover  # enumerate the pilot folder → 23 `discovered` rows
python ingest.py --limit 2   # process 2 books fully → CHECKPOINT (human step):
                             #   read data/markdown/*.md by hand before continuing
python ingest.py             # process the remaining books
python ingest.py --index     # build the HNSW index (run once, after ingestion)
python ingest.py --status    # counts-by-status table
python search.py "structure of the book of Romans"
python search.py "How should I study a Bible chapter?" -k 8
python report.py             # measured vs. assumptions.py, suggested corrections
```

Console output during `ingest.py` prints one line per book: title, pages,
scanned?, chunks, tokens, seconds. A book that fails is marked `failed` with
the error saved to `books.error` and the run continues — it does not stop the
whole batch. A scanned book with no `MISTRAL_API_KEY` set is marked
`needs_ocr` and skipped, not failed.

### Adding a book

Two sources, one pipeline. A book is a row in `books` with a `source`
(`drive` | `upload`) and a `source_id` unique within it; everything after the
download step is identical either way, because `process_book` takes the
downloader as an argument and `downloader_for` is the only place the two
diverge.

**From Google Drive** — enumerate the pilot folder, then drain the queue:

```bash
python ingest.py --discover
python ingest.py
```

**From a file you have** — either drag it into the web UI (see below), or:

```bash
python ingest.py --local some-book.pdf another.pdf
```

Both call the same `register_upload`, so the CLI cannot drift from what the
endpoint does.

Uploads are **content-addressed**: `source_id` is `upload:<md5 of the bytes>`
and the original is kept at `data/uploads/<md5>.pdf`. Three consequences:

- Re-uploading the identical file is recognised, not duplicated — and a book
  already indexed is not reprocessed.
- Two different books that happen to share a filename can never overwrite each
  other.
- A client-supplied filename never reaches the filesystem, so there is nothing
  to traverse out of. It is kept only as the title.

`data/uploads/` holds the **only** copy of an uploaded book's bytes — it is to
an upload what Drive is to a Drive book, and unlike `data/pdfs/` it is not
disposable.

### Removing a book

```bash
python ingest.py --delete 93 94        # or the × in the web UI
```

Hard delete: the row, its chunks (via `ON DELETE CASCADE`, so there is never a
moment where orphaned chunks still answer searches), the working PDF cache, the
extracted markdown and its manifest, and — for an upload — the original bytes.
Deleting the row alone would leave four orphans keyed by a `book_id` that will
never be reused.

Re-uploading afterwards creates a **new** book. Content-addressing decides
identity, not history, so a deletion is not permanent.

### Flags

- `--discover` — enumerate the pilot folder into `books`, then stop.
- `--limit N` — process at most N books this run.
- `--rechunk` — rebuild chunks + embeddings from local `data/markdown/*.md`
  only; makes **no Drive or OCR calls**. Use this after tuning chunk size, the
  junk-heading filter, or the embedding model. (Resets `done` → `extracted`
  and rebuilds — markdown is the permanent, rebuildable-from asset.)
- `--index` — build the HNSW index (`chunks_hnsw`); run once, after ingestion
  finishes (not before — building it against an empty/partial table is wasted
  work).
- `--status` — print a counts-by-status table.
- `--local a.pdf b.pdf` — ingest PDFs already on disk instead of from Drive.
  Reuses `process_book` unchanged by swapping the downloader for a file copy,
  so extraction, chunking, embedding and the atomic commit are the same code
  the Drive path runs. Books are keyed `local:<filename>`, so re-running is
  idempotent exactly like `--discover`.

### Recovery

`ingest.py` is safe to kill (`Ctrl-C`) and re-run at any point:

- Re-running on a fully `done` book is a no-op — enumeration is idempotent on
  `drive_file_id`, and `done`/`failed`/`needs_ocr` books are never re-claimed.
- A book killed mid-processing keeps its `claimed_at`; the next run's claim
  query treats a claim older than `CLAIM_STALE_MINUTES` (5) as stale and
  re-claims it (→ `failed` once it has exceeded 3 attempts). `claimed_at` is a
  **lease, not a lock** — no row is locked during the minutes a book takes, so
  `process_book` heartbeats (`db.touch_claim`) at each stage boundary. That is
  what lets the window be 5 minutes rather than longer than the slowest
  possible book.
- A book whose bytes change in Drive (new md5) is reset to `discovered` by the
  next `--discover` and reprocessed. A rename alone is metadata and does not
  reprocess.
- Re-processing a book always starts by wiping its old chunks and stale local
  PDF/markdown, and commits all of a book's chunks **and** its `done` status in
  a single transaction — so a restart never leaves a half-indexed book.

### Tests

```bash
pytest -v          # real Postgres, fake externals, zero network
ruff check
```

DB-backed tests create/drop a throwaway `library_test` database from
`DATABASE_URL`; if Postgres is unreachable they **skip** with a clear message
(the pure-function tests still run). CI (`.github/workflows/ci.yml`) runs the
full suite against a pgvector service container.

### Retrieval quality

The rest of the suite asserts that functions do what they say. None of it can
tell you whether retrieval is any *good* — chunk size, the embedding model, the
fusion and its constants all trade quality against each other silently, and the
only symptom of getting one wrong is that answers quietly get worse.

```bash
python -m evaluate                  # score the shipping config
python -m evaluate --compare        # every mode side by side
python -m evaluate --show-misses    # what came back instead
python -m evaluate --questions eval/questions_paraphrase.json
```

Ground truth lives in `eval/questions.json` as (book, page span) — never chunk
ids, which do not survive a re-chunk. Two sets of the same 22 questions: one
phrased in the books' own vocabulary, one phrased the way a user would ask.
Metrics are hit-rate@k (what matters — everything retrieved is handed to the
model at once) and MRR (a tiebreaker when hit-rate ties).

**Both a dense and a hybrid (dense + Postgres full-text, fused by Reciprocal
Rank Fusion) path are implemented. Dense is what ships, because that is what the
measurement supported:**

|              | book-voice hit@8 | MRR | user-voice hit@8 | MRR |
|--------------|---|---|---|---|
| **dense**    | **100.0%** | **0.865** | **77.3%** | **0.641** |
| hybrid/and   | 100.0% | 0.888 | 77.3% | 0.633 |
| hybrid/or    | 100.0% | 0.867 | 72.7% | 0.531 |
| lexical/and  | 63.6% | 0.614 | 9.1% | 0.091 |
| lexical/or   | 63.6% | 0.286 | 36.4% | 0.161 |

Hit-rate is *identical* between dense and hybrid on both sets — fusion only
reshuffles ranks inside a result set that already held the right passage, and
the MRR deltas point in opposite directions on the two sets (+0.023 / −0.008),
which is noise at n=22. At 1,423 chunks the dense leg is under no pressure, and
Postgres full-text has no IDF, so its ranking cannot tell that *Philemon* is
rarer than *ask*. `config.SEARCH_MODE` records the full reasoning and what would
change the answer. The lexical path stays because it makes re-running this at
100K chunks a one-flag experiment rather than a rewrite.

`tests/test_eval.py` runs the same scoring functions over a small synthetic
corpus with a deterministic bag-of-words embedder, so the harness is gated in CI
with no network and no API key — including a test that deliberately breaks
retrieval and asserts the harness goes red, because a gate that cannot fail is
decoration.

### Troubleshooting

- **`db-up` hangs / connection refused** — Docker Desktop isn't running.
  Launch it and wait for the whale icon before `make db-up`.
- **`relation "books" does not exist`** — run `python -m migrate`.
- **`VOYAGE_API_KEY is not set`** — copy it into `.env`; both `ingest.py` and
  `search.py` read it via `config.py`.
- **A book stuck in `needs_ocr`** — set `MISTRAL_API_KEY` in `.env`, then reset
  it (`needs_ocr` is terminal to the queue): `UPDATE books SET status='discovered'
  WHERE status='needs_ocr';` (via `make db-psql`) and re-run `ingest.py`.
- **`search.py` returns nothing** — confirm at least one `done` book
  (`python ingest.py --status`) and that `ingest.py --index` has run.

---

## Ask it questions (web UI)

The page also has an **Add a book (PDF)** control. An upload is validated
(magic bytes, not the extension or the browser-supplied content type), streamed
to disk under a size cap, registered, and queued; the response returns as soon
as it is queued and the page polls `/api/books` for progress. Ingestion takes
minutes, so holding the request open would tie the result to the browser
staying on the page.

The API's background worker claims **only uploads**. Drive ingestion stays a
deliberate `python ingest.py` — otherwise one uploaded PDF puts the whole Drive
backlog in flight behind it, and a Drive book claimed inside the API process
blocks on an OAuth flow that has no console to prompt.

```bash
make db-up                                   # Postgres must be running
./.venv/bin/uvicorn api:app --reload --port 8000
open http://localhost:8000
```

One question box. Every question goes through `agent/research.py`: the model
runs its own search loop and decides how many searches the question warrants —
one for a plain factual question, more for a comparative or multi-part one.

There is deliberately no separate "one search, top-k" mode in the UI. That is
just this with the decision hardcoded, and hardcoding it caps a comparative
question at whatever a single embedding of the user's phrasing happens to
reach. Asked *"how do these books differ on baptism in the Holy Spirit"*, the
one-shot path returned 8 passages from a single book and correctly reported it
could not answer; the agent noticed the same gap, called `list_books`, scoped
follow-up searches per book, and produced a real comparison.

`search.py` remains the deterministic control — same question, same vector,
same passages — which is what you need to tell whether a chunking or embedding
change actually helped. The agent varies run to run: better product, worse
measuring instrument.

Citations in the answer are clickable and open the exact passage with its book,
section, page range and position (`passage 173 of 330`).

Endpoints: `GET /`, `GET /api/books`, `POST /api/research` (server-sent events,
so the search trace streams rather than the page sitting silent).

**Cost, measured per question:** ~$0.10, of which Claude is essentially all of
it and Voyage is $0.0000002. Split across the two Claude calls, choosing the
query costs $0.0095 and writing the answer from ten retrieved passages costs
$0.0964 — so `k`, not the number of searches, is the real cost lever.

---

## Project status

- **Phase 0 — complete.** `explore.py` exploration agent; picked the pilot
  folder (Jensen Bible Self Study Guides).
- **Phase 1 — implemented; live pilot run pending.** Full pipeline (migrations,
  queue, download → extract → chunk → embed → Postgres, search, calibration
  report) plus a zero-network test suite and CI. The end-to-end 23-book run
  against real Drive + Voyage (Docker + `VOYAGE_API_KEY`) is the human's next
  step, following the command sequence above — including the `--limit 2`
  markdown-review checkpoint.
- **Phase 2+ — not started.** The tutor chatbot, evals, hybrid search/RRF (the
  `chunks.tsv` column already exists for it), S3 storage, and the full 130 GB
  backfill.
