# InvestPro — Inversion Helper

<div align="center">

**Bot de trading automatizado con inteligencia artificial, backtesting, optimización genética y risk management institucional.**

[![CI](https://github.com/papiwilo74/investpro/actions/workflows/ci.yml/badge.svg)](https://github.com/papiwilo74/investpro/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Personal](https://img.shields.io/badge/License-Personal-red.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-78%25-green.svg)](tests/)
[![Security: Bandit](https://img.shields.io/badge/Security-Bandit-success.svg)](pyproject.toml)

</div>


## Arquitectura

```
investpro/
├── api/            # FastAPI REST + autenticación JWT
│   ├── routes/     # broker, market, analysis, backtest, portfolio, ml, advisor
│   ├── auth.py     # JWT login
│   ├── schemas.py  # Pydantic v2
│   ├── server.py   # App principal + serve del build React
│   └── static/     # Build de Vite (generado, gitignored)
├── backtesting/    # WFO, Monte Carlo, validación estadística
│   ├── validation.py
│   ├── bot_engine.py
│   └── engine.py
├── bot/            # Core del bot
│   ├── engine.py        # Orquestador principal
│   ├── strategy.py      # TradingBrain + Kelly + Decision engine
│   ├── risk.py          # RiskManager (correlación, circuit breaker, Kelly)
│   ├── online_advisor.py# Q-learning para filtro de entradas
│   ├── notifications.py # Telegram/Discord/log
│   ├── performance_tracker.py  # Equity curve, rolling metrics
│   ├── state_manager.py # Estado + daily orders vía SQLite
│   └── market_regime.py # Filtro de régimen (SPY SMA + VIX)
├── broker/         # Cliente Alpaca
├── db/             # SQLAlchemy models + repositories
├── ml/             # Machine Learning
│   ├── ensemble.py         # Adaptive Ensemble (NUEVO)
│   ├── train.py            # XGBoost per-ticker
│   ├── neural_brain.py     # PyTorch NN (HOLD/BUY/SELL/SHORT/COVER)
│   ├── rl.py               # RL exit agent (Q-learning)
│   ├── lstm_model.py       # LSTM price forecaster
│   ├── sentiment.py        # VADER news sentiment
│   └── features.py         # Feature engineering
├── indicators/     # Technical indicators + SignalGenerator
├── portfolio/      # Markowitz optimization
├── data/           # SQLite DBs, JSON state, model weights
└── frontend/       # React + TypeScript + Vite + Zustand
    ├── src/
    │   ├── api/           # API client (fetch + JWT)
    │   ├── components/    # panels, layout, ui, charts
    │   ├── hooks/         # useApi, useTheme
    │   ├── store/         # Zustand appStore
    │   └── types/         # TypeScript types
    └── vite.config.ts     # Build -> ../api/static
```

## Stack

| Componente | Tecnología |
|------------|-----------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Base de datos | SQLite (x4), SQLAlchemy 2.0 |
| ML | XGBoost, PyTorch, scikit-learn, hmmlearn |
| Broker | Alpaca Markets API (Paper Trading) |
| Frontend | React 18, TypeScript, Vite, Zustand, Tailwind CSS, Lightweight Charts |
| CI/CD | GitHub Actions, Render |
| Notificaciones | Telegram Bot API, Discord Webhook |

## Requisitos

- Python 3.12+
- Node.js 18+ (para el frontend)
- Cuenta Alpaca (paper) — [alpaca.markets](https://alpaca.markets)
- (Opcional) Bot de Telegram + Chat ID para notificaciones

## Instalación

```bash
git clone https://github.com/papiwilo74/investpro.git
cd investpro

# Backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

## Configuración

Crear archivo `.env` en la raíz:

```env
ALPACA_API_KEY="tu-api-key"
ALPACA_SECRET_KEY="tu-secret-key"
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true

TELEGRAM_BOT_TOKEN="tu-token"    # opcional
TELEGRAM_CHAT_ID="tu-chat-id"    # opcional
```

## Uso

### Desarrollo (frontend + backend separados)

```bash
# Terminal 1: Backend
python -m uvicorn api.server:app --reload --port 8000

# Terminal 2: Frontend (dev server con hot reload + proxy a /api)
cd frontend
npm run dev    # http://localhost:3000
```

### Producción (build + backend único)

```bash
# Build del frontend -> genera api/static/
cd frontend && npm run build && cd ..

# Backend sirve el build en /
python main.py --web --port 8000
```

Abrir `http://localhost:8000` en el navegador.

### Bot autónomo (CLI)

```bash
python -m bot.engine --daemon       # Loop cada 10 min
python -m bot.engine --intraday     # Loop cada 5 min (día de mercado)
python -m bot.engine --ticker AAPL  # Modo single-ticker
```

### Backtesting

```bash
python -m backtesting.run --ticker AAPL --period 2y
python -m backtesting.genetic       # Optimización genética
```

### Entrenar ML

```bash
python -m ml.train --ticker AAPL              # XGBoost
python -m ml.train --ticker AAPL --optimize   # XGBoost + Grid Search
python -m ml.neural_brain                     # Neural Trading Brain
```

## Endpoints API

| Ruta | Descripción |
|------|-------------|
| `GET /` | Frontend SPA |
| `GET /api/watchlist` | Tickers de la watchlist |
| `GET /api/market/{ticker}` | Datos de mercado + indicadores |
| `GET /api/analysis/{ticker}/signals` | Señales técnicas + score compuesto |
| `GET /api/advisor/{ticker}` | Asesor de inversiones IA |
| `GET /api/ml/{ticker}` | Estado del modelo ML + predicción |
| `POST /api/ml/{ticker}/train` | Entrenar XGBoost |
| `GET /api/backtest/{ticker}` | Backtest completo |
| `GET /api/backtest/{ticker}/validate` | Validación WFO + Monte Carlo |
| `POST /api/backtest/genetic` | Optimización genética (async) |
| `GET /api/backtest/genetic/{job_id}` | Estado del job genético |
| `GET /api/broker/dashboard` | Dashboard completo del bot |
| `POST /api/broker/bot/toggle` | Activar/detener bot |
| `GET /api/ensemble/status` | Pesos y precisión del ensemble |
| `GET /health` | Health check |

## ML Ensemble

El Adaptive Ensemble combina 5 fuentes de señal con pesos dinámicos que se ajustan según precisión reciente:

| Modelo | Framework | Estado |
|--------|-----------|--------|
| XGBoost | sklearn/xgboost | ✅ Entrenado (7 tickers) |
| Neural Trading Brain | PyTorch | ✅ Entrenado |
| RL Exit Agent | Q-learning | ✅ Activo |
| Online Advisor | Q-learning | ✅ Activo |
| TA Clásico | Indicadores técnicos | ✅ Siempre activo |
| LSTM Forecaster | PyTorch LSTM | 🔄 Pendiente |

Los pesos se actualizan automáticamente cada 10 predicciones basado en accuracy por régimen (BULL/BEAR/LATERAL/HIGH_VOL).

## Tests

```bash
pytest tests/ -v                    # Todos
pytest tests/test_ensemble.py -v    # Ensemble
pytest tests/test_auth.py -v        # JWT auth
pytest tests/test_db.py -v          # Database
pytest tests/test_kelly.py -v       # Kelly Calculator
pytest tests/test_risk_manager.py -v # Risk Manager
```

## Despliegue

El repositorio incluye `render.yaml` para despliegue automático en [Render](https://render.com).

El `buildCommand` instala dependencias de Python y compila el frontend (React → `api/static/`). El `startCommand` lanza FastAPI que sirve la API + el build de React.

Variables de entorno requeridas en Render:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`
- `ALPACA_PAPER=true`
- `TELEGRAM_BOT_TOKEN` (opcional)
- `TELEGRAM_CHAT_ID` (opcional)

## Estructura de Datos

El bot mantiene 4 bases SQLite en `data/`:

| Base | Tablas principales |
|------|--------------------|
| `performance.sqlite3` | equity_snapshots, trade_log_telemetry, rolling_metrics |
| `bot_state.sqlite3` | bot_state, open_positions, daily_orders |
| `paper_journal.sqlite3` | paper_signals |
| `inversion_helper.db` | kelly_trades, risk_state, advisor_state (SQLAlchemy) |

## Licencia

Uso personal. No apto para trading real sin validación exhaustiva.
