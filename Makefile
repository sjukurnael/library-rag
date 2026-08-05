.PHONY: db-up db-down db-psql db-logs migrate test lint serve

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
