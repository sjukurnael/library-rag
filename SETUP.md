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

## Two ways to get books, and why it matters

**Your own library (recommended).** Everything above. Empty database, your
keys, your books. Fully independent.

**Sharing an existing library.** If someone hands you their `DATABASE_URL` and
`SUPABASE_*` values, you get their books immediately — same corpus, same
citations. Understand what that is: the Supabase `service_role` key is
**full read and write on their database and file storage**, with no scoping.
Deleting a book from your copy deletes it from theirs. Only do this with
someone who means to give you that, and never commit the values.

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
