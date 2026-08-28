.PHONY: db-up db-down db-psql db-logs migrate test lint serve ui clean

db-up:
	docker compose up -d
	@echo "Waiting for Postgres to accept connections..."
	@until docker compose exec -T db pg_isready -U app -d library > /dev/null 2>&1; do sleep 1; done
	@echo "Postgres is up on localhost:5434"

migrate:
	python -m library_rag.cli.migrate

db-down:
	docker compose down

db-psql:
	docker compose exec db psql -U app -d library

db-logs:
	docker compose logs -f db

test:
	pytest -q

lint:
	ruff check

serve:
	uvicorn library_rag.web.api:app --reload --port 8000

# UI work. Same app, same Supabase data, sign-in switched off.
#
# Blanking GOOGLE_CLIENT_ID makes config.auth_enabled() false, which drops the
# deny-by-default middleware -- so no Google popup, no origin to register, no
# session cookie. That matters for a browser on http://localhost:8000, which is
# a different origin from the deployed app and would need its own OAuth
# registration to sign in at all.
#
# Reads and writes the REAL library. list_classrooms() takes no owner filter, so
# every classroom you made while signed in is here, with all 8,679 books behind
# it. Convenient for UI work and worth remembering before you click Delete.
#
# HTML/CSS/JS need no restart: _page() and static_file() both hand back a
# FileResponse, which re-reads from disk per request, and Cache-Control:
# no-cache makes the browser revalidate. Edit, refresh, done. --reload is only
# for the Python.
ui:
	GOOGLE_CLIENT_ID= uvicorn library_rag.web.api:app --reload --port 8000

# Bytecode outlives the source it came from. After a rename, the old package
# directory survives as nothing but __pycache__ -- and an empty directory is an
# importable namespace package, so `import agent` succeeds and then fails
# confusingly at `from agent import research` instead of saying "no such module".
clean:
	find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
