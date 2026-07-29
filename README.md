# 🚀 InvestPro — Sistema de Trading Algorítmico Institucional

<div align="center">

**Bot de trading automatizado con IA, backtesting profesional, arbitraje estadístico y gestión de riesgo de nivel Hedge Fund.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Deploy: Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com)
[![Tests: 30+](https://img.shields.io/badge/Tests-30+-green.svg)](tests/)
[![License: Personal](https://img.shields.io/badge/License-Personal-red.svg)](LICENSE)

</div>

---

## 📋 Tabla de Contenidos

- [Resumen](#-resumen)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Stack Tecnológico](#-stack-tecnológico)
- [Módulos del Bot (Core Intelligence)](#-módulos-del-bot-core-intelligence)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Gestión de Riesgo](#-gestión-de-riesgo)
- [Frontend & API](#-frontend--api)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Tests](#-tests)
- [Despliegue en Render](#-despliegue-en-render)
- [Estructura de Datos](#-estructura-de-datos)

---

## 🎯 Resumen

InvestPro es un sistema de trading algorítmico completo que opera de forma autónoma 24/7:

- **Acciones** (NYSE/NASDAQ) durante horarios bursátiles vía Alpaca Markets.
- **Criptomonedas** (BTC, ETH, SOL) durante noches y fines de semana.
- **Arbitraje Estadístico** (Pairs Trading) en pares cointegrados como KO/PEP, JPM/BAC, XOM/CVX.

El sistema combina **Machine Learning (XGBoost)**, **Análisis Técnico avanzado**, **Reinforcement Learning (Q-Learning)**, **Detección de Smart Money** y **Análisis de Sentimiento de Noticias** para generar señales de alta probabilidad. Una capa de gestión de riesgo institucional protege el capital con trailing stops adaptativos, guardián de correlación, filtro de earnings y calendario macro (FOMC).

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INVESTPRO ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   Frontend    │    │  Telegram     │    │   Render Cloud       │   │
│  │  (React SPA)  │◄──►│  Bot Control  │    │  (Deploy 24/7)       │   │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘   │
│         │                   │                       │               │
│  ┌──────▼───────────────────▼───────────────────────▼───────────┐   │
│  │                    FastAPI REST Server                        │   │
│  │              (api/server.py + api/routes/)                    │   │
│  └──────┬───────────────────┬───────────────────────┬───────────┘   │
│         │                   │                       │               │
│  ┌──────▼───────┐  ┌───────▼────────┐  ┌───────────▼───────────┐   │
│  │  TradingBot   │  │  Backtesting   │  │   Portfolio           │   │
│  │  (bot/engine) │  │  Engine        │  │   Optimizer           │   │
│  └──────┬───────┘  │  + WFO         │  │   (Genetic)           │   │
│         │          │  + Monte Carlo  │  └───────────────────────┘   │
│  ┌──────▼──────────────────────────────────────────────────────┐    │
│  │              DECISION LAYER (bot/)                          │    │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐│    │
│  │  │TradingBrain│ │Pairs       │ │MultiStrategy             ││    │
│  │  │(strategy)  │ │Trading     │ │Allocator                 ││    │
│  │  │            │ │Engine      │ │(Rotación de estrategias) ││    │
│  │  └─────┬──────┘ └─────┬──────┘ └──────────┬───────────────┘│    │
│  │        └──────────┬───┘                   │                │    │
│  │  ┌────────────────▼───────────────────────▼────────────┐   │    │
│  │  │            RISK & PROTECTION LAYER                  │   │    │
│  │  │  • RiskManager (correlación, circuit breaker)       │   │    │
│  │  │  • CorrelationRiskGuard (>85% → bloqueo)            │   │    │
│  │  │  • AdaptiveThresholdManager (VIX/SPY)               │   │    │
│  │  │  • MacroTracker (FOMC, Earnings filter)             │   │    │
│  │  │  • ATR Trailing Stop Progresivo                     │   │    │
│  │  │  • News Sentinel (sentimiento de noticias)          │   │    │
│  │  │  • Smart Money Tracker (RVOL, OBV, Put/Call)        │   │    │
│  │  └─────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                   │                       │               │
│  ┌──────▼───────┐  ┌───────▼────────┐  ┌───────────▼───────────┐   │
│  │  ML Ensemble  │  │  Data Layer    │  │   Broker Layer        │   │
│  │  (XGBoost,    │  │  (yfinance,    │  │   (Alpaca API,        │   │
│  │   NN, RL,     │  │   SQLite,      │  │    CryptoBroker)      │   │
│  │   Sentiment)  │  │   DataFetcher) │  │                       │   │
│  └──────────────┘  └────────────────┘  └───────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Stack Tecnológico

| Capa | Tecnología |
|------|------------|
| **Backend** | Python 3.12, FastAPI, Uvicorn, SQLAlchemy 2.0 |
| **Machine Learning** | XGBoost, scikit-learn, NumPy, Pandas |
| **Neural Networks** | PyTorch (desactivado en cloud por límite de RAM) |
| **NLP / Sentimiento** | Léxico financiero curado + FinBERT (fallback ligero en cloud) |
| **Broker** | Alpaca Markets API (Paper + Live Trading) |
| **Cripto** | CryptoBrokerClient (Alpaca Crypto 24/7) |
| **Frontend** | HTML/CSS/JS (Diseño Premium Quant, Glassmorphism Teal/Cyan) |
| **Base de Datos** | SQLite ×4 (estado, journal, performance, advisor) |
| **Control Remoto** | Telegram Bot API (`/start`, `/stop`, `/status`) |
| **CI/CD & Deploy** | GitHub → Render (Auto-deploy on push) |
| **Testing** | pytest (30+ archivos de prueba) |

---

## 🧠 Módulos del Bot (Core Intelligence)

```
bot/
├── engine.py                   # Orquestador principal del ciclo de trading
├── strategy.py                 # TradingBrain: genera decisiones BUY/SELL/SHORT/COVER/HOLD
├── statistical_arbitrage.py    # PairsTradingEngine: Z-Score en pares cointegrados (KO/PEP, JPM/BAC)
├── multi_strategy_allocator.py # Rotación dinámica de capital entre Momentum, Mean-Rev y Pairs
├── adaptive_thresholds.py      # Ajuste automático de SL/TP/RSI según VIX y SPY
├── smart_money.py              # Detección de acumulación institucional (RVOL, OBV, Put/Call)
├── macro_calendar.py           # Filtro FOMC, Earnings, y VIX panic mode
├── risk.py                     # RiskManager: correlación, circuit breaker, Kelly, position limits
├── portfolio_allocator.py      # Kelly Criterion, Volatility Parity, CorrelationRiskGuard
├── position_state.py           # Trailing Stop ATR Progresivo (+5%→1.8x, +10%→1.2x, +15%→0.9x)
├── online_advisor.py           # Q-Learning RL online: BLOCK / REDUCE / ALLOW por trade
├── market_regime.py            # Clasificación de mercado: BULL / BEAR / LATERAL / HIGH_VOL
├── market_breadth.py           # Amplitud de mercado (% acciones sobre SMA200)
├── scanner.py                  # Escaneo automático de oportunidades en la watchlist
├── signal_executor.py          # Ejecución final de órdenes con pre-trade checklist
├── notifications.py            # Alertas vía Telegram / Discord / log
├── telegram_listener.py        # Control remoto del bot vía comandos de Telegram
├── performance_tracker.py      # Equity curve, Sharpe ratio, rolling metrics
└── state_manager.py            # Persistencia de estado del bot en SQLite
```

---

## 📊 Machine Learning Pipeline

```
ml/
├── ensemble.py              # Adaptive Ensemble: combina 5+ modelos con pesos dinámicos
├── train.py                 # Entrenamiento XGBoost per-ticker + Champion/Challenger
├── champion_challenger.py   # Sistema de comparación de modelos (re-entrena cada 7 días)
├── features.py              # Feature engineering (40+ features técnicas)
├── sentiment.py             # Léxico financiero curado + News Sentinel por ticker
├── neural_brain.py          # Red neuronal PyTorch (TCN/FFN) para BUY/SELL/SHORT/COVER
├── lstm_model.py            # LSTM price forecaster
├── rl.py                    # RL Exit Agent (Q-learning para optimizar salidas)
├── reddit_sentiment.py      # Sentimiento de Reddit/WallStreetBets
└── stocktwits_sentiment.py  # Sentimiento de StockTwits
```

| Modelo | Framework | Función |
|--------|-----------|---------|
| XGBoost | scikit-learn / xgboost | Predicción principal (clasificación BUY/SELL) |
| Neural Trading Brain | PyTorch | Decisión multi-clase (5 acciones posibles) |
| RL Exit Agent | Q-learning tabular | Optimiza el momento de venta |
| Online Advisor | Q-learning online | Filtra entradas de baja calidad en tiempo real |
| News Sentinel | Léxico financiero | Bloquea compras ante noticias negativas (< -0.4) |
| TA Clásico | Indicadores técnicos | Señales RSI, MACD, EMA, Bollinger, ADX |

---

## 🛡️ Gestión de Riesgo

| Capa de Protección | Módulo | Descripción |
|---------------------|--------|-------------|
| **Umbrales Adaptativos** | `adaptive_thresholds.py` | Ajusta SL, TP y RSI según régimen de mercado (VIX/SPY) |
| **Guardián de Correlación** | `portfolio_allocator.py` | Bloquea compras si el activo está correlacionado >85% con el portafolio |
| **Trailing Stop ATR** | `position_state.py` | Aprieta progresivamente: +5%→1.8x, +10%→1.2x, +15%→0.9x ATR |
| **Smart Money** | `smart_money.py` | Detecta acumulación/distribución institucional (RVOL > 2.0x) |
| **News Sentinel** | `sentiment.py` | Escanea titulares y bloquea compras si sentimiento < -0.4 |
| **Filtro de Earnings** | `macro_calendar.py` | Bloquea compras 3 días antes de reportes de ganancias |
| **Calendario FOMC** | `macro_calendar.py` | Alerta y reduce exposición en días de reunión de la Fed |
| **Rotación de Estrategias** | `multi_strategy_allocator.py` | Asigna 1.4x capital a la estrategia con mejor Win Rate, 0.5x a la peor |
| **Kelly Criterion** | `portfolio_allocator.py` | Quarter-Kelly para sizing óptimo basado en historial |

---

## 🖥️ Frontend & API

### Endpoints Principales

| Ruta | Descripción |
|------|-------------|
| `GET /` | Frontend SPA (Dashboard Premium Quant) |
| `GET /api/watchlist` | Tickers de la watchlist activa |
| `GET /api/market/{ticker}` | Datos de mercado + indicadores técnicos |
| `GET /api/analysis/{ticker}/signals` | Señales técnicas + score compuesto |
| `GET /api/advisor/{ticker}` | Asesor de inversiones IA |
| `GET /api/ml/{ticker}` | Estado del modelo ML + predicción |
| `POST /api/ml/{ticker}/train` | Entrenar modelo XGBoost |
| `GET /api/backtest/{ticker}` | Backtest completo con métricas |
| `GET /api/backtest/{ticker}/validate` | Validación Walk-Forward + Monte Carlo |
| `POST /api/backtest/genetic` | Optimización genética (async con job_id) |
| `GET /api/broker/dashboard` | Dashboard completo del bot (posiciones, PnL, estado) |
| `POST /api/broker/bot/toggle` | Activar/detener bot |
| `GET /api/ensemble/status` | Pesos y precisión del ensemble ML |
| `GET /health` | Health check ligero (cacheado 30s) |

---

## 🔧 Instalación

```bash
# Clonar repositorio
git clone https://github.com/papiwilo74/investpro.git
cd investpro

# Crear entorno virtual e instalar dependencias
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## ⚙️ Configuración

Crear archivo `.env` en la raíz del proyecto:

```env
# Broker (Requerido)
ALPACA_API_KEY="tu-api-key"
ALPACA_SECRET_KEY="tu-secret-key"
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true

# Notificaciones (Opcional)
TELEGRAM_BOT_TOKEN="tu-token"
TELEGRAM_CHAT_ID="tu-chat-id"
```

---

## 🚀 Uso

### Modo Web (Dashboard + Bot)
```bash
python main.py --web --port 8000
# Abrir http://localhost:8000
```

### Bot Autónomo (CLI con Telegram)
```bash
python main.py --daemon          # Loop cada 10 min + control por Telegram
python main.py --intraday        # Loop cada 5 min (día de mercado)
```

### Comandos de Telegram
| Comando | Acción |
|---------|--------|
| `/start` | Activa el bot |
| `/stop` | Detiene el bot |
| `/status` | Muestra posiciones, PnL y estado |

### Backtesting
```bash
python -m backtesting.run --ticker AAPL --period 2y
python -m backtesting.genetic     # Optimización genética
```

### Entrenar Modelos ML
```bash
python -m ml.train --ticker AAPL              # XGBoost
python -m ml.train --ticker AAPL --optimize   # XGBoost + Grid Search
```

---

## 🧪 Tests

```bash
# Suite completa rápida (7 módulos principales)
python run_tests.py

# Suite completa con pytest (30+ archivos)
pytest tests/ -v

# Tests específicos
pytest tests/test_strategy.py -v          # Estrategia / TradingBrain
pytest tests/test_engine.py -v            # Motor del bot
pytest tests/test_ensemble.py -v          # ML Ensemble
pytest tests/test_backtest.py -v          # Backtesting
pytest tests/test_risk_manager.py -v      # Risk Manager
pytest tests/test_statistical_arbitrage.py -v  # Pairs Trading
pytest tests/test_advanced_triad.py -v    # Smart Money + Trailing + Sentinel
```

---

## ☁️ Despliegue en Render

El repositorio incluye `render.yaml` para despliegue automático.

**Optimizaciones para el plan gratuito (512 MB RAM):**
- Lazy Init de todos los módulos pesados (XGBoost, SciPy, SQLAlchemy).
- PyTorch/Transformers desactivados en cloud (usa léxico financiero como fallback).
- Watchdog de auto-reinicio con backoff exponencial.
- Health check cacheado (30s) que no carga módulos pesados.
- Monte Carlo vectorizado con NumPy (~50x más rápido).
- Garbage Collector agresivo post-ciclo.

**Variables de entorno requeridas en Render:**
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`
- `ALPACA_PAPER=true`
- `TELEGRAM_BOT_TOKEN` (opcional)
- `TELEGRAM_CHAT_ID` (opcional)

---

## 💾 Estructura de Datos

| Base de Datos | Tablas Principales |
|---------------|-------------------|
| `performance.sqlite3` | equity_snapshots, trade_log_telemetry, rolling_metrics |
| `bot_state.sqlite3` | bot_state, open_positions, daily_orders |
| `paper_journal.sqlite3` | paper_signals |
| `inversion_helper.db` | kelly_trades, risk_state, advisor_state (SQLAlchemy) |

---

## 📜 Licencia

Uso personal. No apto para trading real sin validación exhaustiva.
