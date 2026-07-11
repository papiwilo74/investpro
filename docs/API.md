# API Reference

## Endpoints

### Market Data

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/watchlist` | Lista de tickers monitoreados |
| GET | `/api/market/scan?universe=nasdaq100` | Escanear oportunidades |
| GET | `/api/market/analysis/{ticker}` | Análisis completo de un ticker |
| GET | `/api/data/info` | Estado del data layer y caché |

### Análisis

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/analysis/indicators/{ticker}` | Indicadores técnicos actuales |
| GET | `/api/analysis/signals/{ticker}` | Señales activas |
| GET | `/api/analysis/overview/{ticker}` | Resumen completo |

### Machine Learning

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/ml/predict/{ticker}` | Predicción ensemble con todos los modelos |
| GET | `/api/ml/status` | Estado de todos los modelos |
| GET | `/api/ml/gate/status` | Estado del Model Gate |
| POST | `/api/ml/train/{ticker}` | Entrenar modelo para un ticker |
| GET | `/api/ensemble/status` | Pesos y accuracy del ensemble |

### Backtest

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/backtest/run` | Ejecutar backtest |
| POST | `/api/backtest/optimize` | Optimizar parámetros |

### Broker

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/broker/account` | Resumen de cuenta |
| GET | `/api/broker/positions` | Posiciones abiertas |
| POST | `/api/broker/bot/toggle` | Iniciar/detener bot |
| GET | `/api/broker/bot/status` | Estado del bot |

### Portafolio

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/portfolio/optimize` | Optimizar portafolio (Max Sharpe / Min Vol) |

### Configuración

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/config/flags` | Feature flags activos |

### Monitoreo

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Health check completo |
| GET | `/metrics` | Métricas Prometheus |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |

## Formato de Respuesta

### Predicción ML

```json
{
  "ticker": "AAPL",
  "direction": "ALCISTA",
  "probability": 0.72,
  "confidence": 0.65,
  "regime": "BULL",
  "model_signals": {
    "xgboost": {"direction": "ALCISTA", "probability": 0.68, "score": 0.36},
    "neural_brain": {"direction": "ALCISTA", "probability": 0.74, "score": 0.48},
    "panel": {"direction": "ALCISTA", "probability": 0.71, "score": 0.42}
  },
  "model_weights": {
    "xgboost": 0.22, "neural_brain": 0.18, "panel": 0.11
  },
  "blended_score": 0.42
}
```

### Health Check

```json
{
  "status": "ok",
  "checks": {
    "api": "ok",
    "data": {
      "provider": {"name": "yfinance", "status": "ok", "latency_ms": 450},
      "cache": {"total_entries": 42, "fresh_entries": 38, "expired_entries": 4},
      "quality_check": {"status": "ok", "latency_ms": 520, "rows": 252, "null_pct": 0.0}
    },
    "broker": "ok",
    "bot": "running"
  }
}
```

## Códigos de Error

| Código | Significado |
|--------|-------------|
| 200 | OK |
| 400 | Bad Request — parámetros inválidos |
| 401 | Unauthorized — JWT faltante o inválido |
| 404 | Ticker no encontrado o sin datos |
| 429 | Rate limit excedido |
| 503 | Servicio degradado (broker desconectado, datos lentos) |
