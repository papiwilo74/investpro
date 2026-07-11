FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv

COPY pyproject.toml .
RUN uv sync --frozen --no-cache

COPY . .
RUN uv sync --frozen --no-cache

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "--web"]
