# syntax=docker/dockerfile:1
# Multi-stage build: frontend (Node) + backend (Python).
# El frontend se construye desde source; la imagen final no necesita Node.js.

# ═══════════════════════════════════════════════════════════
# Stage 1: Frontend build
# ═══════════════════════════════════════════════════════════
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Cache layer de dependencias (incluye devDeps necesarias para build)
COPY frontend/package*.json ./
RUN npm ci

# Copiar source y build
COPY frontend/ ./
RUN npm run build

# Verificar que el build se generó
RUN test -f dist/index.html || (echo "ERROR: frontend/dist/index.html no encontrado" && exit 1)


# ═══════════════════════════════════════════════════════════
# Stage 2: Python runtime
# ═══════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

WORKDIR /app

# Variables de entorno recomendadas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema necesarias para compilar algunos paquetes de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copiar el código fuente
COPY . .

# Traer el build del frontend desde el stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Sanity check final
RUN test -f frontend/dist/index.html || (echo "ERROR: frontend/dist/index.html no encontrado" && exit 1)

EXPOSE 8000

# Render/Fly proveen $PORT; si no, usar 8000 por defecto
CMD ["sh", "-c", "python main.py --web --port ${PORT:-8000}"]
