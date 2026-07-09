FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar (versión ligera sin torch)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copiar el código
COPY . .

# Crear directorios necesarios
RUN mkdir -p data/logs data/cache

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Arrancar la web app
CMD ["python", "main.py", "--web", "--port", "8000"]
