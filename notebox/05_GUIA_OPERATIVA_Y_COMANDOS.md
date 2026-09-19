# 05 — Guía Operativa, Comandos y Despliegue

Este cuaderno es tu manual de instrucciones para operar, probar, monitorear y desplegar Axiom en el día a día.

---

## 1. Variables de Entorno y Configuración (`.env`)

Para que el bot se conecte a tu cuenta de Alpaca y envíe notificaciones, requiere un archivo `.env` en la raíz del proyecto:

```bash
# ── Broker: Alpaca (Paper Trading por defecto) ───────────────────────
ALPACA_API_KEY="tu_api_key_de_alpaca_aqui"
ALPACA_SECRET_KEY="tu_secret_key_de_alpaca_aqui"
ALPACA_BASE_URL="https://paper-api.alpaca.markets"
ALPACA_PAPER="true"

# ── Notificaciones de Telegram (Opcional) ────────────────────────────
TELEGRAM_BOT_TOKEN="tu_token_de_bot_aqui"
TELEGRAM_CHAT_ID="tu_chat_id_aqui"

# ── Servidor Web ─────────────────────────────────────────────────────
PORT=8000
HOST="0.0.0.0"
```

---

## 2. Comandos para Ejecutar Axiom Localmente

Todos los comandos se ejecutan desde la raíz del proyecto con el entorno virtual activado:

```powershell
# Activar entorno virtual en Windows
.venv\Scripts\activate
```

### A. Iniciar Servidor Web + Dashboard + Bot Automático
```powershell
python main.py --web --port 8000
```
* Abre la API de FastAPI en `http://localhost:8000`.
* Activa el watchdog que arranca el bot automáticamente a los 10 segundos.

### B. Iniciar en Modo Daemon (Solo Proceso en Segundo Plano)
```powershell
python main.py --daemon
```

### C. Ejecutar un Backtest Histórico sobre un Activo
```powershell
python main.py --backtest AAPL --period 2y
```

### D. Optimizar Portafolio (Máximo Sharpe y Mínima Varianza)
```powershell
python main.py --portfolio "AAPL,MSFT,NVDA,GOOGL" --period 1y
```

---

## 3. Pruebas Automatizadas y Calidad de Código

Antes de enviar cualquier cambio a producción, Axiom cuenta con una batería de tests automáticos:

```powershell
# 1. Ejecutar tests de riesgo y estrategias crypto
.venv\Scripts\python.exe -m pytest tests/test_crypto_risk_guard.py tests/test_crypto_focus_and_protection.py tests/test_strategy.py

# 2. Ejecutar toda la suite de tests del motor
.venv\Scripts\python.exe -m pytest tests/test_engine.py

# 3. Revisar calidad y estilo de código con Ruff (linter ultrarrápido)
.venv\Scripts\python.exe -m ruff check bot/ tests/

# 4. Formatear código automáticamente con Black
.venv\Scripts\python.exe -m black bot/ tests/
```

---

## 4. Consultas Rápidas del Estado de la Cuenta por Consola

Puedes consultar tu balance y posiciones de Alpaca en 3 segundos con este comando:

```powershell
.venv\Scripts\python.exe -c "from broker.alpaca_client import AlpacaClient; a = AlpacaClient(); print('BALANCE:', a.get_account_summary()); [print(p.get('symbol'), 'Valor:$' + str(p.get('market_value')), 'PnL:' + str(p.get('unrealized_pl')), f'{float(p.get(\"unrealized_plpc\", 0))*100:.2f}%') for p in a.get_positions()]"
```

Para ver las últimas 10 órdenes ejecutadas:
```powershell
.venv\Scripts\python.exe -c "from broker.alpaca_client import AlpacaClient; from alpaca.trading.requests import GetOrdersRequest; from alpaca.trading.enums import QueryOrderStatus; a = AlpacaClient(); [print(o.submitted_at, o.symbol, o.side, o.qty, o.filled_avg_price, o.status) for o in a.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))]"
```

---

## 5. Despliegue en la Nube (Render)

Axiom está configurado con **despliegue continuo (CI/CD)**: cada vez que haces `git push origin master`, Render detecta los cambios, compila la imagen Docker y despliega la nueva versión automáticamente.

### Especificaciones del Despliegue (`Dockerfile` y `render.yaml`)
* **Imagen base**: `python:3.12-slim`.
* **Health Check**: Render consulta `/health`. Si el contenedor responde código 200, marca el despliegue como exitoso.
* **Comando de inicio del contenedor**:
  ```bash
  python main.py --web --port ${PORT:-8000}
  ```

### ¿Por qué pasar al Plan Starter (\$7/mes) el próximo mes?
1. **Always-On**: El plan gratuito duerme el contenedor a los 15 minutos sin visitas. El plan de \$7 mantiene a Axiom despierto **24 horas al día, los 7 días de la semana**, garantizando que el escáner de criptomonedas no se detenga jamás de noche ni fines de semana.
2. **5x más CPU**: El cálculo de medias móviles y señales para 50 acciones y 25 criptomonedas se realiza en segundos.
3. **Cero límites de horas**: Evita que el servicio se apague a final de mes por exceder las 750 horas gratuitas.

---

## 6. Procedimientos de Emergencia

### ¿Qué hacer si el bot se congeló por un Circuit Breaker y quieres resetearlo?
Ejecuta el script dedicado:
```powershell
python reset_risk.py
```
Este script limpia las rachas de pérdidas acumuladas, resetea el temporizador del freno de emergencia y limpia la base de datos de telemetría de Kelly.

### ¿Cómo agregar un activo a HOLD manual para que el bot no lo venda?
En [`bot/strategy_params.py`](file:///c:/Users/villa/.gemini/antigravity/scratch/inversion-helper/bot/strategy_params.py), busca la tupla `manual_hold_symbols` y agrega el símbolo:
```python
manual_hold_symbols: tuple[str, ...] = ("DOT/USD", "DOTUSD", "NUEVO/USD")
```
O simplemente define la variable en tu archivo `.env` o en Render:
```bash
MANUAL_HOLD_SYMBOLS="DOT/USD,DOTUSD,NUEVO/USD"
```
Axiom respetará la posición inmediatamente y omitirá cualquier venta automática hasta que tú decidas lo contrario.
