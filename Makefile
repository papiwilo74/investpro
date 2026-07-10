.PHONY: lint test migrate docker-build docker-up dev help

lint:
	ruff check .
	black --check --diff .

lint-fix:
	ruff check --fix .
	black .

test:
	python -m pytest tests/ -v --tb=short -x

migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-logs:
	docker compose logs -f

dev:
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

help:
	@echo "lint        — Ruff + Black check"
	@echo "lint-fix    — Ruff + Black auto-fix"
	@echo "test        — Run pytest"
	@echo "migrate     — Run alembic migrations"
	@echo "migrate-new — Create new migration"
	@echo "docker-build— Build Docker images"
	@echo "docker-up   — Start services (app + db)"
	@echo "docker-logs — Tail logs"
	@echo "dev         — Run uvicorn with hot reload"
