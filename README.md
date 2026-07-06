# Inversion Helper

Dashboard y bot de apoyo para analisis tecnico, backtesting, machine learning, optimizacion de portafolio y paper trading con Alpaca.

> Uso educativo. No es recomendacion financiera.

## Configuracion

1. Instala dependencias:

```bash
pip install -r requirements.txt
```

2. Copia `.env.example` a `.env` y configura tus credenciales de Alpaca paper:

```bash
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_PAPER=true
```

## Comandos utiles

### Instalacion y arranque

Instala dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Arranca la web app FastAPI + frontend:

```bash
python main.py --web
```

Abre la web app en otro puerto si el `8000` esta ocupado:

```bash
python main.py --web --port 8080
```

### Modo Hedge Fund (Trading 100% Automático)

Ejecutar el Bot Multi-Ticker en vivo (Escanea NVDA, AAPL, etc., y opera con Alpaca):
```bash
python bot/multi_daemon.py
```

Activar "Cazador Autónomo" (Escanea todo NASDAQ en busca de oportunidades desconocidas):
```bash
python bot/multi_daemon.py --auto-scan
```

Modo Alto Riesgo (Opera Opciones Financieras Calls/Puts):
```bash
python bot/multi_daemon.py --auto-scan --options
```

Forzar evolución de IA manual (Re-entrena la IA con datos frescos de hoy):
```bash
python main.py --train
```

Arranca el dashboard alternativo en Streamlit:

```bash
python main.py --app
```

### Analisis de mercado

Analiza un ticker con indicadores, senales y backtest basico:

```bash
python main.py --ticker AAPL --period 1y
```

Analiza otro intervalo:

```bash
python main.py --ticker MSFT --period 6mo --interval 1d
```

Optimiza un portafolio con varios tickers:

```bash
python main.py --portfolio AAPL,MSFT,NVDA,GOOGL --period 2y
```

### Watchlist inteligente

Escanea oportunidades en Nasdaq 100:

```bash
python main.py --scan-market --universe nasdaq100 --scan-limit 15
```

Escanea oportunidades en acciones liquidas del S&P 500:

```bash
python main.py --scan-market --universe sp500 --scan-limit 15
```
```

Escanea la watchlist pequena definida en `config.py`:

```bash
python main.py --scan-market --universe watchlist --scan-limit 10
```

Escanea todos los universos disponibles:

```bash
python main.py --scan-market --universe all --scan-limit 20
```

Escanea y guarda las oportunidades como senales paper auditables:

```bash
python main.py --scan-market --universe nasdaq100 --scan-limit 15 --record-paper-signals
```

El scanner filtra por liquidez, tendencia, volatilidad, precio y score tecnico. Tambien muestra por que acepta o rechaza cada ticker.

### Modo seguro antes de dinero real

Consulta si el paper trading tiene suficiente consistencia:

```bash
python main.py --paper-safety
```

Actualiza resultados de senales guardadas y luego muestra el filtro de seguridad:

```bash
python main.py --paper-safety --update-paper-outcomes
```

El modo seguro revisa:

- Dias observados.
- Senales cerradas.
- Win rate.
- Retorno promedio.
- Si aun falta historial antes de pensar en dinero real.

### Machine Learning

Entrena un modelo ML para un ticker:

```bash
python main.py --train-ml AAPL --period 2y
```

Entrena y optimiza hiperparametros del modelo ML:

```bash
python main.py --train-ml AAPL --period 2y --optimize-ml
```

Entrena un agente de reinforcement learning:

```bash
python main.py --train-rl AAPL --period 2y
```

### Backtesting y optimizacion del bot

Ejecuta backtest completo del bot:

```bash
python main.py --bot-backtest --ticker AAPL --period 2y
```

Ejecuta backtest con otro intervalo:

```bash
python main.py --bot-backtest --ticker NVDA --period 1y --interval 1d
```

Optimiza parametros del bot por grid search:

```bash
python main.py --optimize-bot --ticker AAPL --period 2y
```

### Broker y paper trading

Valida conexion con Alpaca paper:

```bash
python main.py --paper-check
```

Arranca el bot en modo daemon 24/7:

```bash
python main.py --daemon
```

Arranca el bot daemon para un ticker especifico:

```bash
python main.py --daemon --ticker AAPL --interval 1d
```

Importante: el bot esta pensado para paper trading. Antes de operar dinero real, usa el modo seguro durante al menos 1-2 meses.

## Endpoints utiles

Watchlist fija:

```text
GET /api/watchlist
```

Datos de mercado:

```text
GET /api/market/AAPL?period=1y&interval=1d
```

Scanner inteligente:

```text
GET /api/market/scanner/opportunities?universe=nasdaq100&limit=15
```

Senales tecnicas:

```text
GET /api/analysis/AAPL/signals?period=1y&interval=1d
```

Estado del bot:

```text
GET /api/broker/bot/status
```

Activar o detener bot:

```text
POST /api/broker/bot/toggle
```

Senales paper guardadas:

```text
GET /api/broker/paper/signals
```

Resumen del paper trading:

```text
GET /api/broker/paper/summary
```

Actualizar resultados paper:

```text
POST /api/broker/paper/update-outcomes
```

Filtro de seguridad antes de dinero real:

```text
GET /api/broker/paper/safety-gate
```

## Seguridad del bot

El bot esta bloqueado para live trading por defecto. Para arrancar desde la API o UI de Broker:

- `ALPACA_PAPER` debe ser `true`.
- Las credenciales deben conectar correctamente.
- Se aplican limites de riesgo configurados en `config.py`.

Controles actuales:

- Tamano maximo por posicion.
- Maximo de ordenes por dia.
- Stop-loss.
- Take-profit.
- Trailing stop por ATR.
- Tamano de posicion por volatilidad.
- Filtro de tendencia por SMA200.
- Filtro de RSI alto.
- Confirmacion opcional por ML si existe modelo entrenado.
- Scanner inteligente para evitar activos con baja liquidez, volatilidad mala o tendencia debil.
- Bitacora paper en `data/paper_journal.sqlite3`.
- Filtro de seguridad antes de considerar live trading.
