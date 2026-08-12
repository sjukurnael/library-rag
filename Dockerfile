# The serving image. Reading, searching and the agents -- NOT indexing.
#
# Indexing (ingest, drive_sync, bible --load) stays on a laptop: it needs Drive
# credentials that expire every 7 days under a Testing consent screen, and it
# peaks memory extracting large scanned PDFs. See README, "Deploying".

FROM python:3.12-slim

# PyMuPDF, psycopg[binary] and numpy all ship manylinux wheels, so no compiler
# and no apt packages are needed. If that ever stops being true the build fails
# loudly at pip install rather than at runtime.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing application code does not re-resolve and
# re-download ~600 MB of wheels on every build. pyproject.toml is the whole
# dependency declaration; README.md is only here because it is referenced by it.
COPY pyproject.toml README.md ./
COPY src/ ./src/
# No [dev] extra: pytest and ruff have no business in a serving image.
RUN pip install .

# config.py calls PDF_DIR.mkdir(...) at IMPORT time, so an unwritable data
# directory is an import error rather than a runtime one -- the container would
# fail to start with a traceback about mkdir. Cloud Run's filesystem is
# writable but in-memory, so this also means anything written here counts
# against the instance's memory. Nothing durable lives here: markdown and
# originals go to Supabase Storage.
ENV LIBRARY_RAG_DATA_DIR=/tmp/data

# Cloud Run injects PORT and can change it; binding a hardcoded 8080 works
# today and breaks silently the day it does not. `exec` so uvicorn is PID 1 and
# receives SIGTERM directly -- without it the shell swallows the signal and
# every deploy waits out the 10s kill timeout.
ENV PORT=8080
# JSON form (so Docker does not wrap this in an implicit shell of its own)
# around an explicit `sh -c`, which is what expands ${PORT}. `exec` then
# replaces that shell so uvicorn is PID 1.
CMD ["sh", "-c", "exec uvicorn library_rag.web.api:app --host 0.0.0.0 --port ${PORT}"]
