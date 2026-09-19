# AXIOM — Enciclopedia y Cuaderno de Estudio (Notebox)

Bienvenido a la documentación integral de **Axiom** (anteriormente *InvestPro*). Esta serie de guías está diseñada para que comprendas en profundidad cada componente, algoritmo, modelo matemático y salvaguarda de riesgo del sistema.

---

## 🗺️ Mapa de Contenidos

| Cuaderno | Título | Descripción |
| :--- | :--- | :--- |
| [01_ARQUITECTURA_Y_SISTEMAS.md](01_ARQUITECTURA_Y_SISTEMAS.md) | **Arquitectura y Sistemas** | Estructura modular, FastAPI, React Frontend, Broker Alpaca, flujo de datos y despliegue en Render (512MB Docker). |
| [02_ESTRATEGIAS_Y_ANALISIS_TECNICO.md](02_ESTRATEGIAS_Y_ANALISIS_TECNICO.md) | **Estrategias Cuantitativas** | Indicadores técnicos, cálculo del Composite Score, MTF Sniper, StatArb, y asignación de portafolio 60/40. |
| [03_MACHINE_LEARNING_E_IA.md](03_MACHINE_LEARNING_E_IA.md) | **Inteligencia Artificial y ML** | Ensemble (XGBoost + Random Forest), PyTorch Neural Brain, Q-Learning RL, Shadow Trader y Model Gate. |
| [04_GESTION_DE_RIESGO_Y_ESCUDOS.md](04_GESTION_DE_RIESGO_Y_ESCUDOS.md) | **Gestión de Riesgo y Escudos** | Circuit Breakers (2% diario), regla del 5% max sizing, Escudo Macro BTC, Cooldown anti-churn y modo HOLD manual. |
| [05_GUIA_OPERATIVA_Y_COMANDOS.md](05_GUIA_OPERATIVA_Y_COMANDOS.md) | **Guía Operativa y Comandos** | Comandos CLI, ejecución local, tests automatizados con pytest, variables de entorno y despliegue a la nube. |

---

## ⚡ ¿Qué es Axiom en 30 Segundos?

**Axiom** es un motor de inversión autónomo cuantitativo (*Autonomous Quantitative Trading Engine*) diseñado para gestionar un portafolio bivalente de **Criptomonedas (60%)** y **Acciones de Empresa (40%)** en tiempo real.

El sistema combina:
1. **Análisis Técnico Multi-Timeframe**: Velas diarias (1D) para macro-tendencia y velas de 1 hora (1H) para sincronizar entradas sin sobrecompra.
2. **Inteligencia Artificial Híbrida**: Árboles de decisión gradient boosted, redes neuronales profundas y aprendizaje por refuerzo.
3. **Escudos Institucionales de Riesgo**: Límites estrictos de pérdida diaria, paridad de volatilidad y filtros de correlación macro para proteger el capital.
4. **Operación 24/7/365**: Escaneo continuo de mercados tradicionales en horario bursátil y criptomonedas 24 horas al día.

---

## 📂 Directorios Clave del Código

```text
inversion-helper/
├── api/             # Backend FastAPI (REST APIs, websockets, autenticación, rutas)
├── bot/             # Motor algorítmico central (TradingBot, TradingBrain, Scanner, Risk)
├── broker/          # Conectores con brokers (AlpacaClient, CryptoBrokerClient, Paper)
├── config/          # Parámetros globales, configuración de riesgo y broker
├── data/            # Fetchers (Yahoo Finance, Alpaca), cache managers, bases de datos SQLite
├── frontend/        # Interfaz de usuario interactiva construida en Vite + React + Tailwind
├── indicators/      # Indicadores matemáticos (ATR, RSI, MACD, VWAP) y generador de señales
├── ml/              # Modelos de Machine Learning (XGBoost, Neural Brain PyTorch, RL Q-table)
├── notebox/         # Esta colección de documentación educativa
├── tests/           # Suite completa de tests unitarios y de integración (pytest)
├── main.py          # Punto de entrada CLI y servidor web unificado
└── render.yaml      # Manifiesto de infraestructura cloud para Render
```
