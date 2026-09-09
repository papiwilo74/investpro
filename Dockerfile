# syntax=docker/dockerfile:1
# Python runtime optimizado para Render 512MB RAM
# El frontend se sirve independientemente en Vercel.

FROM python:3.12-slim AS runtime

WORKDIR /app

# Variables de entorno para máxima eficiencia de memoria en Linux containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=65536 \
    PYTHONMALLOC=malloc

# Dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python (cloud-optimized, sin torch/transformers)
COPY requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copiar el código fuente
COPY . .

EXPOSE 8000

# Render provee $PORT; si no, usar 8000 por defecto
CMD ["sh", "-c", "python main.py --web --port ${PORT:-8000}"]
