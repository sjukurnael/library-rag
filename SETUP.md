# Setup — from `git clone` to a running app

This is the getting-started guide. `README.md` is the engineering record: why
each piece works the way it does, what was measured, what the flags do. Start
here, go there when something surprises you.

You will have a working app with your own books in about 15 minutes. Google
Drive is **optional** and adds ~10 minutes — skip it on the first pass and
upload a PDF instead.

---

## What is NOT in this repo

Cloning gets you the code and nothing else. Four things are deliberately
gitignored, and you have to supply your own:

| Missing | What it is | How to get it |
|---|---|---|
| `.env` | Every API key and the database URL | `cp .env.example .env`, then fill it in — see below |
| `credentials.json` | Google OAuth desktop client | Only for Drive. See `README.md` → "Google Cloud OAuth setup" |
| `token.json` | Your signed-in Google token | Written automatically on first sign-in |
| `data/` | The PDFs, extracted markdown, uploads | Created as you add books |

The database is not in the repo either — you point at your own, and the
migrations build the schema.

---

## 1. Prerequisites

- **Python 3.11+**
- **Docker Desktop** — for local Postgres. It must be *running*, not just
  installed. (Skip only if you have a Postgres 16+ with the `pgvector`
  extension available some other way.)
- **An Anthropic API key** — <https://console.anthropic.com/>
- **A Voyage AI API key** — <https://voyageai.com/> — embeddings, for both
  ingestion and search. `voyage-4-lite` has a free token tier.
- *(Optional)* A Mistral API key, only for scanned/image-only PDFs.

## 2. Clone and install

```bash
git clone https://github.com/sjukurnael/library-rag.git
cd library-rag
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Given a **zip** instead of a repo, unzip it and run the last two lines — nothing
else differs. You lose the commit history, which costs you nothing to run it.

> **If you are the one sending the zip: do not compress the folder in Finder.**
> That sweeps up `.env` with every API key in it, plus `credentials.json`,
> `token.json`, the whole `data/` corpus and a `.venv` full of absolute paths
> that are wrong on any other machine. Use
> `git archive --format=zip --prefix=library-rag/ HEAD -o library-rag.zip`
> instead: it exports exactly the tracked files, so everything gitignored is
> excluded by construction rather than by remembering to exclude it. Note it
> archives the last **commit** — uncommitted work in your tree will not be in
> the zip.

The `src/` layout means the package is importable **only once installed** —
that is deliberate, so a test can never pass by accidentally picking up the
working directory instead of what would actually ship.

## 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`. The defaults for
everything else point at local Docker and work as-is.

## 4. Start the database and build the schema

```bash
make db-up                              # pgvector/pgvector:pg17 on localhost:5434
.venv/bin/python -m library_rag.cli.migrate   # applies migrations/*.sql, idempotent
```

`db-up` waits until Postgres actually accepts connections, so when it returns
you are ready. Port **5434**, not 5432, so it cannot collide with a native
Postgres install.

## 5. Run it

```bash
.venv/bin/python -m uvicorn library_rag.web.api:app --port 8000
```

Open <http://localhost:8000>. Three pages:

- **/** — chat. Ask a question; the research agent runs its own searches and
  answers with citations that open the source passage.
- **/library** — browse and search Google Drive (needs Drive connected).
- **/queue** — processing dashboard, and the **Upload** button.

Add `--reload` while editing code. Be aware it restarts the server on every
`.py` save, which kills any in-flight research run or ingest — fine while
developing, annoying while demoing.

## 6. Put a book in it

**The fast way, no Google account needed:** go to **/queue**, click **Upload**,
pick a PDF. It is validated, queued, and ingested in the background — download
→ extract → chunk → embed. Watch it move through the stages on that page. When
it lands in `done`, ask it something on **/**.

**Or from the CLI:**

```bash
.venv/bin/python -m library_rag.cli.ingest --local some-book.pdf
.venv/bin/python -m library_rag.cli.ingest --status   # counts by status
```

**Or from Google Drive** — this is the part that needs `credentials.json`.
Follow "Google Cloud OAuth setup" in `README.md`, drop the file in the project
root, then click **Sign in to Google Drive** in the sidebar and **Sync from
Drive** on /library. Set `DRIVE_ROOT_FOLDER_ID` in `.env` to your own folder
first — the default is the shared drive this project was built against, which
you will not have access to.

## 7. Check it works

```bash
.venv/bin/python -m pytest -q      # ~180 tests, no network, no API keys needed
.venv/bin/ruff check
```

Tests need local Postgres running. If it is unreachable the DB-backed tests
**skip** with a clear message rather than fail, and the pure-function tests
still run — so a green-but-mostly-skipped run means Docker is not up.

---

## Joining someone else's library

If the point is to see a library that already has books in it, you skip most of
the above: no Docker, no migrations, no ingestion. You point at their database
and the corpus is simply there.

Ask them for three values and put them in your `.env`:

```
DATABASE_URL=            # their Supabase SESSION pooler URI
SUPABASE_URL=            # their project URL
SUPABASE_SERVICE_KEY=    # their service_role key
```

Then keep these **your own** — they are not part of what is shared:

```
ANTHROPIC_API_KEY=       # your key, your bill
VOYAGE_API_KEY=          # your key
TEST_DATABASE_URL=postgresql://app:app@localhost:5434/library
```

Your own Voyage key is fine and does not corrupt anything: the model is pinned
in `config.EMBED_MODEL`, so vectors you produce land in the same space as
theirs. It is the *model* that has to match, not the account.

Then step 5 — start the server — and you are done. Steps 4 and 6 do not apply.

**What works immediately:** the chat, every indexed book, and browsing/searching
all 57k Drive titles — the metadata mirror lives in the shared database, so no
Google account is involved. **What still needs your own Google setup:** indexing
a *new* book from Drive, because that downloads the actual file. Uploading a PDF
from your machine never needs Drive.

Two people can use one database at once. The `books` table is the work queue and
claims are taken with `FOR UPDATE SKIP LOCKED`, so two servers draining it will
not collide or double-process — that is the design, not a happy accident.

### What you are actually being given

The `service_role` key is **unscoped read and write on their database and file
storage**. There is no read-only mode and no per-user permission. Deleting a
book on your screen deletes it from theirs, permanently, along with its chunks
and its stored PDF. Treat the whole thing as shared-owner access to someone
else's work.

For the person sharing: send these through a password manager or another
end-to-end encrypted channel, not chat or email, and never commit them. If you
later want the access back, rotate the `service_role` key in the Supabase
dashboard and change the database password — revoking is per-project, not per
person, so everyone reconnects.

`TEST_DATABASE_URL` is the one line you must not skip. The suite CREATEs and
DROPs databases; `tests/conftest.py` refuses any non-local host outright, so a
missing value is a hard stop rather than a disaster — but point it at local
Docker and the tests just work.

---

## When something goes wrong

| Symptom | Cause |
|---|---|
| `db-up` hangs, connection refused | Docker Desktop is not running |
| `relation "books" does not exist` | Run the migrate step |
| `VOYAGE_API_KEY is not set` | Not in `.env`, or you started the server from another directory |
| `ModuleNotFoundError: library_rag` | `pip install -e ".[dev]"` was skipped, or the venv is not the one running |
| Page says "server may be down or restarting" | `--reload` restarted it mid-request; retry |
| A book stuck in `needs_ocr` | It is scanned and `MISTRAL_API_KEY` is unset. Terminal by design, not a failure |
| Search returns nothing | No `done` books yet, or `ingest --index` has not been run |

`.venv/bin/python -m ...` is used throughout rather than activating the venv,
because it works regardless of shell state — and if the repo directory is ever
moved, the console scripts in `.venv/bin/` break on their absolute-path
shebangs while `python -m` keeps working.
