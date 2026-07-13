.PHONY: lint test migrate docker-build docker-up dev frontend frontend-dev help

lint:
	ruff check .
	ruff format --check .

lint-fix:
	ruff check --fix .
	ruff format .

test:
	python -m pytest tests/ -v --tb=short -x

migrate:
	alembic upgrade head

migrate-new:
	@read -p "Migration name: " name; alembic revision --autogenerate -m "$$name"

frontend:
	cd frontend && npm install && npm run build

frontend-dev:
	cd frontend && npm run dev

frontend-lint:
	cd frontend && npm run lint

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-logs:
	docker compose logs -f

dev:
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

help:
	@echo "lint          — Ruff check + format"
	@echo "lint-fix      — Ruff auto-fix + format"
	@echo "test          — Run pytest"
	@echo "migrate       — Run alembic migrations"
	@echo "migrate-new   — Create new migration"
	@echo "frontend      — Build React (npm install + npm run build)"
	@echo "frontend-dev  — Vite dev server (hot reload, port 3000)"
	@echo "frontend-lint — ESLint check"
	@echo "docker-build  — Build Docker images"
	@echo "docker-up     — Start services (app + db)"
	@echo "docker-logs   — Tail logs"
	@echo "dev           — Run uvicorn with hot reload"
