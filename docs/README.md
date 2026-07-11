# Inversion Helper — Documentación Técnica

Sistema de trading automatizado con análisis técnico, machine learning y gestión de riesgo.
Arquitectura modular con FastAPI backend, React frontend, y pipeline ML escalable.

## Índice

1. [Arquitectura del Sistema](ARCHITECTURE.md)
2. [Guía de Setup](SETUP.md)
3. [Referencia de API](API.md)
4. [Pipeline de Machine Learning](ML_PIPELINE.md)
5. [Flujo de Datos](DATA_FLOW.md)
6. [Configuración](CONFIG.md)

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| ML | XGBoost, LightGBM, PyTorch, scikit-learn |
| Datos | yfinance, Alpaca API, Polygon.io |
| Cache | SQLite (índice) + Parquet (datos) |
| Frontend | React, Vite, TypeScript, Lightweight Charts |
| Broker | Alpaca Markets (paper + live) |
| CI/CD | GitHub Actions (ruff, mypy, pytest, bandit) |
| Monitoreo | Prometheus + Grafana, structlog JSON |
| Alertas | Telegram, Discord |
| Deploy | Docker, Render |

## Componentes Principales

```
inversion-helper/
├── api/              # FastAPI REST endpoints
├── backtesting/      # Motores de backtest + validación
├── bot/              # Bot de trading, risk manager, alerts
├── broker/           # Conexión Alpaca
├── config/           # Settings unificados + feature flags
├── data/             # Data providers, caché, split adjuster
├── indicators/       # Indicadores técnicos + señales
├── ml/               # Modelos ML (XGBoost, NN, RL, Panel, Ensemble)
├── portfolio/        # Optimización de portafolio
├── tests/            # Tests unitarios + de integración
├── frontend/         # React SPA
└── docs/             # Esta documentación
```
