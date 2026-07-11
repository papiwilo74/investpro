# Guía de Setup

## Requisitos

- Python 3.12+
- uv (gestor de paquetes)
- Node.js 20+ (para frontend React)
- Git

## Instalación

```bash
# Clonar
git clone https://github.com/papiwilo74/investpro.git
cd investpro

# Backend
uv sync --dev
cp .env.example .env
# Editar .env con tus API keys

# Frontend (opcional)
cd frontend
npm install
cd ..

# Ejecutar
uv run python main.py --web
```

## Configuración Mínima (.env)

```env
ENV=development
ALPACA_API_KEY=tu_key
ALPACA_SECRET_KEY=tu_secret
ALPACA_PAPER=true
DATA_PROVIDER=yfinance
```

## Comandos Útiles

```bash
# Web App
uv run python main.py --web              # FastAPI en :8000
uv run python main.py --app              # Streamlit dashboard

# CLI
uv run python main.py --ticker AAPL      # Pipeline completo
uv run python main.py --scan-market      # Escanear oportunidades
uv run python main.py --bot-backtest     # Backtest del bot

# Machine Learning
uv run python main.py --train-ml AAPL    # Entrenar XGBoost
uv run python main.py --train-panel      # Entrenar modelo panel
uv run python main.py --panel-predict AAPL  # Predecir con panel

# Validación
uv run python main.py --full-validation --ticker AAPL --period 2y

# Tests
uv run pytest                             # Todos los tests
uv run pytest tests/test_panel_model.py   # Tests específicos
uv run pytest -k "ensemble"               # Filtrar por nombre

# Linting
uv run ruff check .
uv run ruff format --check .
uv run mypy . --ignore-missing-imports
uv run bandit -r .

# Docker
docker build -t inversion-helper .
docker run -p 8000:8000 inversion-helper
```

## Estructura del Proyecto

```
inversion-helper/
├── api/                # FastAPI REST API
│   ├── routes/         #   Endpoints por módulo
│   ├── metrics.py      #   Prometheus metrics
│   ├── middleware.py   #   CORS, rate limit, security
│   ├── auth.py         #   JWT authentication
│   └── server.py       #   App FastAPI principal
├── backtesting/        # Motores de simulación
│   ├── engine.py       #   Backtest básico
│   ├── bot_engine.py   #   Backtest con bot completo
│   ├── validation.py   #   Walk-Forward + Monte Carlo
│   └── full_validation.py  # Pipeline unificado
├── bot/                # Bot de trading
│   ├── engine.py       #   Bot principal
│   ├── risk.py         #   Risk Manager
│   ├── safety.py       #   Signal Journal + Safety Gate
│   ├── alerts.py       #   Telegram/Discord alerts
│   └── scanner.py      #   Market scanner
├── broker/             # Conexión a brokers
│   ├── alpaca_client.py  # Alpaca API
│   └── smart_router.py   # Smart order routing
├── config/             # Configuración
│   ├── settings.py     #   Pydantic Settings
│   └── __init__.py     #   Backward-compat exports
├── data/               # Capa de datos
│   ├── fetcher.py      #   DataFetcher (legacy)
│   ├── provider.py     #   DataProvider abstracto + impls
│   ├── cache_manager.py #  SQLite + Parquet cache
│   ├── data_manager.py #   Orquestador
│   └── split_adjuster.py # Splits/dividendos
├── indicators/         # Indicadores técnicos
│   ├── technical.py    #   Cálculo de indicadores
│   └── signals.py      #   Generación de señales
├── ml/                 # Machine Learning
│   ├── train.py        #   XGBoost trainer
│   ├── panel_model.py  #   LightGBM panel model
│   ├── neural_brain.py #   PyTorch Neural Network
│   ├── rl_train.py     #   RL Agent
│   ├── lstm_model.py   #   LSTM
│   ├── ensemble.py     #   Adaptive Ensemble
│   ├── model_gate.py   #   Fail-closed gate
│   ├── champion_challenger.py  # Promoción
│   └── features.py     #   Feature engineering
├── portfolio/          # Optimización
│   ├── optimizer.py    #   Markowitz
│   └── genetic_optimizer.py  # Genético
├── frontend/           # React SPA
├── tests/              # Tests
├── docs/               # Documentación
└── main.py             # Punto de entrada CLI
```
