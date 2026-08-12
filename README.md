# library-rag

> **Just cloned this? Read [SETUP.md](SETUP.md)** — clone to a running app in
> about 15 minutes. This file is the engineering record: what each decision was,
> what was measured, and what would change the answer.

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
├── pyproject.toml          # deps, console scripts, packaging (src layout)
├── docker-compose.yml      # pgvector/pgvector:pg17 on port 5434
├── Makefile                # db-up / db-psql / migrate / test
├── migrations/             # *.sql, applied in filename order by migrate.py
├── scripts/                # one-off maintenance, not part of any pipeline
├── data/                   # gitignored, YOUR corpus (not packaged)
│   ├── uploads/            #   uploaded originals, content-addressed (NOT disposable)
│   ├── markdown/           #   extracted markdown -- permanent ONLY when storage is off
│   └── pdfs/               #   per-book working cache, safe to delete
├── tests/                  # real Postgres, fake externals, zero network
└── src/library_rag/
    ├── config.py           # every tunable, with the measurement behind it
    ├── db.py               # Postgres: the work queue (claim/lease) + CRUD + search
    ├── ingest.py           # the worker: discover, register_upload, process_book,
    │                       #   process_queue, purge_book
    ├── migrate.py          # forward-only migration runner
    ├── drive/client.py     # OAuth, paginated listing, download, retries
    ├── pipeline/           # how a PDF becomes searchable
    │   ├── extract.py      #   PyMuPDF / Mistral OCR -> markdown + manifest
    │   ├── chunking.py     #   markdown -> chunks with page ranges + ordinals
    │   └── embed.py        #   the one place that calls the Voyage API
    ├── bible.py            # the Bible page: download -> parse -> load, + 3 read
    │                       #   queries. One module, no Greek, no embeddings.
    ├── retrieval/
    │   └── research.py     # the agent that decides what to search for, and when to stop
    ├── evaluation/
    │   ├── harness.py      # hit-rate@k / MRR scoring -- no CLI, so tests reuse it
    │   └── questions/      # curated ground truth (book + page span, never chunk id)
    ├── web/
    │   ├── auth.py         # Google sign-in + the default-deny gate
    │   ├── api.py          # /api/books, /api/books/upload, DELETE, /api/research (SSE)
    │   └── static/         # the UI: question box, inline citations, PDF upload
    ├── exploration/        # the librarian (see below). Started as Phase 0 and
    │   │                   #   was repurposed -- web/api.py imports it live.
    │   ├── loop.py         #   hand-written tool-use loop, streamed as SSE
    │   ├── tools.py        #   search_drive + browse_folder + recommend
    │   └── assumptions.py  #   every estimation constant, each with a rationale
    └── cli/                # EVERY command. Library modules have no __main__.
        ├── ingest.py  search.py  evaluate.py
        └── migrate.py report.py  explore.py
```

One rule makes the tree navigable: **every command lives in `cli/`** — if a
module parses argv or prints, it goes there, and everything else is importable
library code the tests and the API can drive directly.

`exploration/` used to be the second rule ("finished work, nothing imports it").
That is no longer true and the directory name is now misleading: Phase 0's
tool-use loop was repurposed into **the librarian**, and `web/api.py` imports
`exploration.loop` and `exploration.tools` on the live path. Only
`assumptions.py` is still Phase 0 residue.

`src/` layout on purpose: the package is importable only once installed
(`pip install -e ".[dev]"`), so a test can never pass by accidentally picking up
the working directory instead of what would actually ship.

Everything under Phase 1 treats the extracted markdown (+ its `.manifest.json`)
as the permanent, rebuildable-from asset — Postgres is disposable
(`ingest.py --rechunk` rebuilds every chunk/embedding from markdown alone, no
Drive or OCR calls).

**Where that markdown lives depends on whether Supabase Storage is configured**,
and it is one place or the other, never both:

- **`SUPABASE_*` set** — `process_book` mirrors markdown and manifest to the
  bucket and then deletes the local pair, so the **bucket** is the durable copy
  and `data/markdown/` is working space. This is what makes `--rechunk` work on
  a machine that never extracted the book — before it, only the laptop that ran
  the ingest could rebuild its chunks. It also stops `data/pdfs/` growing
  without bound (measured: 488 MB against 18 MB of markdown at 197 books).
- **`SUPABASE_*` unset** — nothing is uploaded and nothing is deleted; the local
  files are the only copy, exactly as before.

The working PDF is deleted only once the bucket *confirms* it, so an outage or
an unconfigured install never loses the last copy. But note the trade: with
storage on there is no second copy of the markdown, and losing the bucket means
re-extracting — which for a scanned book means paying for OCR again.

## Setup

### 1. Python environment

```bash
cd ~/Desktop/library-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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
python -m library_rag.cli.explore
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
pytest tests/test_estimate.py
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
   python -m library_rag.cli.migrate   # apply migrations/*.sql (idempotent)
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
python -m library_rag.cli.ingest --discover  # enumerate the pilot folder → 23 `discovered` rows
python -m library_rag.cli.ingest --limit 2   # process 2 books fully → CHECKPOINT (human step):
                             #   read data/markdown/*.md by hand before continuing
python -m library_rag.cli.ingest             # process the remaining books
python -m library_rag.cli.ingest --index     # build the HNSW index (run once, after ingestion)
python -m library_rag.cli.ingest --status    # counts-by-status table
python -m library_rag.cli.search "structure of the book of Romans"
python -m library_rag.cli.search "How should I study a Bible chapter?" -k 8
python -m library_rag.cli.report             # measured vs. assumptions.py, suggested corrections
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
python -m library_rag.cli.ingest --discover
python -m library_rag.cli.ingest
```

**From a file you have** — either drag it into the web UI (see below), or:

```bash
python -m library_rag.cli.ingest --local some-book.pdf another.pdf
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
python -m library_rag.cli.ingest --delete 93 94        # or the × in the web UI
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
- `--rechunk` — rebuild chunks + embeddings from the extracted markdown only;
  makes **no Drive or OCR calls**. Use this after tuning chunk size, the
  junk-heading filter, or the embedding model. (Resets `done` → `extracted`
  and rebuilds — markdown is the permanent, rebuildable-from asset.) Markdown is
  read via `extract.load_markdown`, which falls back to Supabase Storage, so
  this works on a machine that never extracted the book.
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
python -m library_rag.cli.evaluate                  # score the shipping config
python -m library_rag.cli.evaluate --compare        # every mode side by side
python -m library_rag.cli.evaluate --show-misses    # what came back instead
python -m library_rag.cli.evaluate --questions src/library_rag/evaluation/questions/questions_paraphrase.json
```

Ground truth lives in `src/library_rag/evaluation/questions/questions.json` as (book, page span) — never chunk
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
- **`relation "books" does not exist`** — run `python -m library_rag.cli.migrate`.
- **`VOYAGE_API_KEY is not set`** — copy it into `.env`; both `ingest.py` and
  `search.py` read it via `config.py`.
- **A book stuck in `needs_ocr`** — set `MISTRAL_API_KEY` in `.env`, then reset
  it (`needs_ocr` is terminal to the queue): `UPDATE books SET status='discovered'
  WHERE status='needs_ocr';` (via `make db-psql`) and re-run `ingest.py`.
- **`search.py` returns nothing** — confirm at least one `done` book
  (`python -m library_rag.cli.ingest --status`) and that `ingest.py --index` has run.

---

## Ask it questions (web UI)

The page also has an **Add a book (PDF)** control. An upload is validated
(magic bytes, not the extension or the browser-supplied content type), streamed
to disk under a size cap, registered, and queued; the response returns as soon
as it is queued and the page polls `/api/books` for progress. Ingestion takes
minutes, so holding the request open would tie the result to the browser
staying on the page.

The API's background worker claims **only uploads**. Drive ingestion stays a
deliberate `python -m library_rag.cli.ingest` — otherwise one uploaded PDF puts the whole Drive
backlog in flight behind it, and a Drive book claimed inside the API process
blocks on an OAuth flow that has no console to prompt.

```bash
make db-up                                   # Postgres must be running
./.venv/bin/uvicorn library_rag.web.api:app --reload --port 8000
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

## The librarian (Drive page)

The Drive holds ~57,500 PDFs; the indexed library holds a few hundred. The
bottleneck is not searching what you have indexed — it is **deciding what to
index next**. The librarian is that triage step: describe what you want to
study, and an agent runs a few searches and returns 3–6 candidates with a
reason each and an Add button.

`exploration/loop.py` is a hand-written tool-use loop (~85 lines, no framework)
with four tools — `search_drive`, `browse_folder`, `estimate_pipeline`,
`recommend`. It is a **generator of events**, so `/api/browse` streams the trail
as server-sent events and a CLI could print the same events without a second
implementation. Watching which searches it ran is most of the value: a list of
titles with no visible provenance is indistinguishable from a hallucination.

It is **read-only**. The agent proposes; adding a book is a separate, explicit
POST that you trigger by pressing a button.

### What it can actually see — titles and folder paths, never contents

This is the limitation to understand before trusting it. **No book here has been
opened.** What gets embedded is `drive_files.search_text`: the full breadcrumb
plus the filename, cleaned (see `migrations/0004_drive_search_text.sql`) —

```
Exclusive / Books / Bockmuehl Revelation and Mystery in Ancient Judaism...
Exclusive / Dissertations / Jassen Mediating the divine- Prophecy and revelation...
```

The folder path is doing real work here: a book filed under
`Books / Bible Study Guides / Revelation /` inherits that topic even when its own
filename is opaque. That, plus academic filenames tending to describe their
contents, is why this works better on this corpus than it would on a fiction
library — `Snodgrass_Stories With Intent Parables.pdf` tells you what is inside;
`Harry Potter and the Philosopher's Stone` does not.

**Where it fails, measured against the live mirror:**

| query | result |
|---|---|
| `"why do bad things happen to good people"` | finds `The problem of evil`, `Nature and origin of evil` — but **not Job**, whose filename carries no topical signal. `word_and_meaning = 0` correctly reports low confidence. |
| `"a boy who discovers he has magical powers"` | returns NT scholarship on magic in antiquity, and scores a confident-looking `word_and_meaning = 5`. It matched the *words* "magic" and "power". |

So it cannot bridge from a description of contents to a title that does not name
the topic. It reads spines, not books. Treat it as triage, not comprehension —
the thing that reads the books is the RAG pipeline, and only after you index one.

### On "the end times"

An earlier version of this file, and several comments still in the source,
claimed the mirror search turns `"the end times"` into a query that reaches
Revelation. Re-measured, that is **half true and worth stating precisely**:

```
dense only   Segal_Calculating the End_ Inner-Danielic Chronological...
             Pfandl_The latter days and the time of the end in Daniel     <- real eschatology
hybrid (shipped, w=0.5)
             Times as Task, Not Timing- Reconsidering Qoheleth's...
             Christ and Time—Part Three- "Telling Time" in the Fourth...  <- matched on "time"
```

The dense leg does reach genuine eschatology. The lexical leg then drags in
titles that merely contain "time" or "end", and on this query the fusion is
**worse than dense alone**. `config.DRIVE_RRF_LEXICAL_WEIGHT` already documents
this failure mode at weight 1.0; it is milder at the shipped 0.5, not absent.

Note this does not overturn the weight choice: on the 8-query set recorded in
`config.py`, hybrid at 0.5 scored 8/8 against dense's 7/8. It does mean that set
is too small to have caught this, exactly as the comment there warns. A larger
ground-truth set is the outstanding work.

---

## The Bible page

A plain English Bible at `/bible` — pick a book and chapter, read it, arrow
forward and back, search for a phrase. Deliberately unrelated to the RAG
pipeline: no embeddings, no agent, no PDFs. It shares the database and the
sidebar and nothing else.

```bash
python -m library_rag.cli.migrate
python -m library_rag.cli.bible --load     # ~2 s; expects exactly 31,102 verses
python -m library_rag.cli.bible --status   # books and verse counts
```

The text is the [Berean Standard Bible](https://bereanbible.com), placed in the
public domain in April 2023. `--load` downloads `bsb.txt` (4.3 MB) into
gitignored `data/` on first run and reads from disk after that; the running web
app never touches the network.

### One table

[migrations/0006_bible_verses.sql](migrations/0006_bible_verses.sql) — 31,102
rows, **6 MB**. For scale, one `chunks` row carries 2 KB of embedding, so the
whole Bible costs less disk than ~3,500 chunks.

```sql
CREATE TABLE bible_verses (
    book       SMALLINT NOT NULL,   -- 1..66, canonical order
    book_name  TEXT     NOT NULL,   -- 'Genesis'
    chapter    SMALLINT NOT NULL,
    verse      SMALLINT NOT NULL,
    text       TEXT     NOT NULL,
    PRIMARY KEY (book, chapter, verse)
);
```

No surrogate id — `(book, chapter, verse)` already names a verse, and its index
is also the chapter lookup. No book table: 66 names repeated costs a few hundred
KB and removes a join from every query. No search index — `text ILIKE '%…%'`
over 31k rows is a sequential scan measured at ~115 ms against Supabase, and an
index nobody needs is a thing to have to explain.

### Three stages, deliberately separate

[src/library_rag/bible.py](src/library_rag/bible.py) is one module, ~200 lines
including comments, no classes:

```
download()   network -> data/bsb.txt     runs once, then the file is there
parse()      data/bsb.txt -> tuples      no database
load()       tuples -> Postgres          no filesystem, one COPY, one transaction
```

`parse()` being pure is what lets most of the tests run with no database at all.

### Two things the data forced

**Book names contain spaces and digits** — `1 Samuel`, `Song of Solomon`,
`3 John`. The regex takes the name greedily up to the last ` <digits>:<digits>`;
splitting on spaces breaks on all eighteen of them. Note the source spells it
**`Psalm`**, singular — kept as-is, because "correcting" it stops lookups
matching.

**16 verses carry no text** — Matthew 17:21, Mark 9:44, John 5:4, Acts 8:37 and
the rest: present in the KJV, absent from the earliest manuscripts. The BSB
keeps the *number* as a placeholder, and so do we. Dropping them would make
Matthew 17 jump from verse 20 to 22 and read as a loader bug. The page renders
them as *"— not found in the earliest manuscripts"*, because a blank line just
looks broken.

### Dig deeper — the librarian, on a verse

Hovering a verse reveals a **✦** button. It runs the **same agent** the Drive
page runs, given the verse instead of a typed interest:

```
I am studying this Bible verse and want to go deeper:

John 3:16 — "For God so loved the world that He gave His one and only Son, …"

Find books in the library that would help me understand it.
```

There is **no new endpoint and no new agent**. `/api/browse` already takes free
text, and a verse is just a well-specified way of saying what you want to study
— so this is one composed string and a place to render the result. What was
shared is the *frontend*: `toolLine`, `resultLine`, `recCard`, `runLibrarian`
and `wireIndexButtons` moved from `library.html` into `app.js` (−98 lines from
that page), for the same reason `esc()` lives there — it was about to have a
second copy.

Measured, unchanged prompt. John 3:16 produced four searches (John commentaries,
the love of God, eternal life in Johannine theology, the Nicodemus context) and
six picks including a paper on how John 3:16 itself has been translated.
Philemon 6 surfaced a journal article devoted entirely to interpreting Philemon
6. The agent's honest closing on the first: the collection is "strong on
Johannine theology and thin on classic verse-by-verse commentaries".

The 16 empty verses get no button — there is no text to hand the agent, and an
action that cannot work is worse than none. One panel is open at a time, since
the Index buttons index into the open verse's picks.

### Search

Literal substring matching, case-insensitive, in Bible order — typing "love"
finds the letters. `%` and `_` are escaped before they reach `LIKE`, or a search
for `100%` would become the pattern `%100%%` and return the entire Bible. Capped
at 200 results, and the page says when it hit the cap rather than presenting a
truncated list as complete.

---

## Deploying

The app runs on **Google Cloud Run**, gated by **Google sign-in** against an
allowlist. At two users this costs **$0** — measured, 1,800 requests and 13,800
vCPU-seconds a month against free allowances of 2,000,000 and 180,000, so 7.7%
of the tightest one.

### Sign-in, and why it is not Supabase Auth

Two questions, answered separately. **Who are you?** — Google, via a signed ID
token the browser hands us and `google-auth` verifies. **May you in?** — the
`allowed_users` table. Anyone on earth has a Google account, so identity is not
authorisation.

Not Supabase Auth, for one architectural reason: this app serves **server-rendered
pages**. A plain navigation to `/bible` arrives before any JavaScript has run, so
there is no header to attach a browser-held token to and nothing for a gate to
inspect. A server-set cookie can gate that; a localStorage token cannot. Supabase
Auth is the right tool when you want per-user data with RLS.

Nothing here is Supabase-specific either: `allowed_users` is a plain Postgres
table read with `psycopg`, and the whole design moves to any other Postgres
unchanged. Supabase's dashboard just happens to be a convenient editor for it.

```bash
python -m library_rag.cli.users --add you@gmail.com
python -m library_rag.cli.users --add friend@gmail.com --note "borrowed Philemon"
python -m library_rag.cli.users --list
```

**The gate is off when `GOOGLE_CLIENT_ID` is unset**, so local development and
the test suite are unaffected — and the server prints a loud warning when it
starts unauthenticated, because the failure mode is silent.

The middleware is **default-deny** ([web/auth.py](src/library_rag/web/auth.py)):
open paths are an explicit short list and everything else needs a session, so a
route added next month is protected without anyone remembering to protect it.
A browser gets `302 /login`; a `fetch` gets `401 JSON`, because a login page
delivered into `await r.json()` is a parse error rather than a message.

### Two Google Cloud projects, not one

This is the non-obvious part. The existing OAuth consent screen is **External +
Testing**, and Google revokes refresh tokens issued under that status after
**7 days**. Publishing to production would fix that, but `drive.readonly` is a
**restricted scope** and publishing with it requires a paid CASA security
assessment. `openid`/`email`/`profile` are not restricted and need no review.

| | Project A — existing | Project B — new |
|---|---|---|
| purpose | Drive access | sign-in + hosting |
| scope | `drive.readonly` (restricted) | `openid email profile` |
| consent screen | stays **Testing** | **published**, no assessment |
| OAuth client | Desktop (`credentials.json`) | **Web application** |
| used from | your laptop only | the deployed app |

Splitting them is what lets anyone sign in, indefinitely, for free — while the
restricted scope stays quarantined on the machine that actually indexes.

In **Project B**: enable `run.googleapis.com`, `cloudbuild.googleapis.com` and
`artifactregistry.googleapis.com`; publish the consent screen; create a **Web
application** OAuth client with *Authorized JavaScript origins*
`http://localhost:8000` and your Cloud Run URL. Origins, not redirect URIs —
this flow never redirects to Google.

### Deploy

```bash
gcloud run deploy library-rag \
  --source . --region us-west1 \
  --max-instances=2 \        # default is 100; this bounds worst-case spend
  --memory=1Gi --cpu=1 \
  --timeout=300 \            # agent runs take ~45s and stream
  --env-vars-file=env.yaml   # gitignored; keeps DATABASE_URL out of shell history
```

The first deploy uses `--no-allow-unauthenticated`: the Cloud Run URL is unknown
until the service exists, but the OAuth client needs it as an origin, and
`GOOGLE_CLIENT_ID` cannot be set until then. Locking the service to IAM for that
window avoids publishing an ungated app. Get the URL → add the origin → set the
env vars → redeploy with `--allow-unauthenticated`.

Then set a **spend cap budget** on Cloud Run. Google's spend caps genuinely pause
the service at 100% rather than only emailing — $5/month is ~13× expected usage.

Image is ~740 MB (PyMuPDF and googleapiclient are 200 MB between them), which
puts cold start around 5–15s. `LIBRARY_RAG_DATA_DIR=/tmp/data` matters more than
it looks: [config.py](src/library_rag/config.py) calls `PDF_DIR.mkdir(...)` at
**import** time, so an unwritable data directory is an import error, not a
runtime one.

### What the deployed app deliberately cannot do

**No Drive credentials ship in the image** — `.dockerignore` excludes
`credentials.json` and `token.json`, and a layer is not undone by a later `rm`.
With Project A in Testing, a mirrored token would be revoked weekly anyway; a
feature that breaks on a timer is worse than one that is honestly absent.
**Indexing runs on your laptop**, where the consent flow already exists and where
extracting a 60 MB scan will not OOM a 1 GB instance.

| | deployed | |
|---|---|---|
| Bible reader + search | ✅ | Postgres only |
| Chat over indexed books | ✅ | Postgres + Anthropic + Voyage |
| Librarian `search_drive` | ✅ | reads the **mirror**, not Drive |
| Drive browse + search | ✅ | mirror-backed |
| PDF viewer | ✅ mostly | signed Supabase Storage URL |
| Librarian `browse_folder` | ❌ | dropped from the agent's tools automatically |
| Drive sync, indexing, upload | ❌ | live Drive / heavy CPU |

`exploration.loop.available_tools()` removes `browse_folder` when
`credentials_status()` reports no Drive: offering a tool that can only error
wastes the model's turns and fills the trail with failures that look like a bug.

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
