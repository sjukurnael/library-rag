.PHONY: db-up db-down db-psql db-logs migrate test lint serve clean

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

# Bytecode outlives the source it came from. After a rename, the old package
# directory survives as nothing but __pycache__ -- and an empty directory is an
# importable namespace package, so `import agent` succeeds and then fails
# confusingly at `from agent import research` instead of saying "no such module".
clean:
	find . -path ./.venv -prune -o -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
