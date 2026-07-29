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
│   └── assumptions.py    #   every estimation constant, each with a rationale comment
├── drive/                # Drive access, shared by both phases
│   └── client.py         #   OAuth, paginated listing, download, retries
├── config.py              # Phase 1: every ingestion tunable (paths, models, chunk sizes)
├── db.py                  #   Postgres access: queue semantics (claim/reap) + CRUD
├── migrations/            #   *.sql schema, applied in filename order
│   └── 0001_init.sql      #     books + chunks (halfvec embedding, tsv, doc_status enum)
├── migrate.py              #   tiny forward-only migration runner (schema_migrations)
├── docker-compose.yml      #   pgvector/pgvector:pg17, port 5434, healthcheck
├── ingest.py               #   the worker: Drive PDF -> markdown -> chunks -> embeddings
├── search.py               #   retrieval smoke test CLI
├── report.py               #   calibration report: measured vs. assumptions.py
├── pipeline/               #   ingest.py's internals
│   ├── extract.py         #     PyMuPDF / Mistral OCR -> markdown + manifest
│   ├── chunking.py         #     markdown -> chunks with page ranges + ordinals
│   └── embed.py             #     the one place that calls the Voyage API
├── pyproject.toml          #   package + dev deps; `pip install -e ".[dev]"`
├── data/                    #   gitignored: pdfs/, markdown/ (the permanent asset)
├── .github/workflows/ci.yml #   pgvector service -> ruff check -> pytest
└── tests/                   #   real Postgres, fake externals, zero-network
    ├── conftest.py         #     test-DB lifecycle + fake Voyage/Drive
    ├── test_estimate.py    #     Phase 0: estimate_pipeline arithmetic
    ├── test_migrations.py  #     migration runner idempotency
    ├── test_discover.py    #     enumerate/upsert
    ├── test_claim.py       #     queue: SKIP LOCKED, reaper, attempt cap
    ├── test_chunking.py    #     chunk boundaries, pages, ordinals, junk skip
    ├── test_embed_store.py #     atomic chunk+done, crash safety
    └── test_extract.py     #     text-layer probe, corrupt-PDF recovery
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

### Recovery

`ingest.py` is safe to kill (`Ctrl-C`) and re-run at any point:

- Re-running on a fully `done` book is a no-op — enumeration is idempotent on
  `drive_file_id`, and `done`/`failed`/`needs_ocr` books are never re-claimed.
- A book killed mid-processing keeps its `claimed_at`; the next run's claim
  query treats a claim older than 30 minutes as stale and re-claims it (→
  `failed` once it has exceeded 3 attempts).
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
