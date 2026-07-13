FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv

# Instalar Node para build del frontend
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev

COPY . .
RUN uv sync --frozen --no-cache --no-dev

# Build del frontend -> api/static/
WORKDIR /app/frontend
RUN npm install && npm run build
WORKDIR /app

EXPOSE 8000

CMD ["uv", "run", "python", "main.py", "--web"]
