# Configuración

## Variables de Entorno

Ver `.env.example` para todas las variables.

### Ambiente

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ENV` | `development` | `development`, `staging`, `production` |

### Broker

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ALPACA_API_KEY` | `""` | API Key de Alpaca |
| `ALPACA_SECRET_KEY` | `""` | Secret Key de Alpaca |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | URL de la API |
| `ALPACA_PAPER` | `true` | Modo paper trading |

### Data Provider

| Variable | Default | Descripción |
|----------|---------|-------------|
| `DATA_PROVIDER` | `yfinance` | Fuente: `yfinance`, `alpaca`, `polygon` |
| `DATA_CACHE_TTL_HOURS` | `4` | TTL del caché en horas |
| `POLYGON_API_KEY` | `""` | API Key de Polygon.io |

### Riesgo

| Variable | Default | Descripción |
|----------|---------|-------------|
| `INITIAL_CAPITAL` | `100000.0` | Capital inicial |
| `MAX_DAILY_LOSS_PCT` | `-0.02` | Pérdida máxima diaria (-2%) |
| `MAX_WEEKLY_DRAWDOWN_PCT` | `-0.05` | Drawdown máximo semanal |
| `MAX_SECTOR_EXPOSURE_PCT` | `0.25` | Exposición máxima por sector |
| `MAX_TOTAL_EXPOSURE_PCT` | `0.65` | Exposición total máxima |
| `CONSECUTIVE_LOSS_LIMIT` | `3` | Pérdidas consecutivas antes de pausa |
| `CIRCUIT_BREAKER_MINUTES` | `60` | Minutos de pausa tras pérdidas |
| `LEVERAGE_ENABLED` | `true` | Apalancamiento activado |
| `MIN_LEVERAGE` | `2.0` | Apalancamiento mínimo |
| `MAX_LEVERAGE` | `3.0` | Apalancamiento máximo |

### Model Gate

| Variable | Default | Descripción |
|----------|---------|-------------|
| `GATE_MIN_ACCURACY` | `0.55` | Accuracy mínima para aprobar |
| `GATE_MIN_PRECISION` | `0.50` | Precisión mínima |
| `GATE_MIN_TEST_SIZE` | `30` | Mínimo de muestras OOS |
| `GATE_MIN_EDGE` | `0.03` | Ventaja mínima vs baseline (3pp) |

### Champion/Challenger

| Variable | Default | Descripción |
|----------|---------|-------------|
| `CC_PROMO_MARGIN` | `0.02` | Margen para promover (2pp) |
| `CC_MIN_CHALLENGER_ACCURACY` | `0.52` | Piso de accuracy del challenger |
| `CC_MAX_AGE_DAYS` | `14` | Re-entrenar si > 14 días |
| `CC_DRIFT_FLOOR` | `0.45` | Re-entrenar si accuracy en vivo < 0.45 |

## Feature Flags

Los feature flags cambian según `ENV`:

| Flag | Development | Staging | Production |
|------|-------------|---------|------------|
| `live_trading` | ❌ | ❌ | ✅ |
| `paper_trading` | ✅ | ✅ | ✅ |
| `train_ml` | ✅ | ✅ | ✅ |
| `train_nn` | ✅ | ✅ | ✅ |
| `web_app` | ✅ | ✅ | ✅ |
| `metrics` | ✅ | ✅ | ✅ |
| `alerts` | ✅ | ✅ | ✅ |
| `genetic_optimizer` | ✅ | ✅ | ❌ |
| `intraday` | ✅ | ❌ | ❌ |
| `global_backtest` | ✅ | ✅ | ✅ |
| `full_validation` | ✅ | ❌ | ❌ |
| `panel_model` | ✅ | ✅ | ✅ |

## Uso en Código

```python
# Moderno (recomendado)
from config import settings, feature_flags

settings.ENV                          # "development"
settings.ALPACA_API_KEY               # del .env
settings.MAX_DAILY_LOSS_PCT           # -0.02
settings.GATE_MIN_ACCURACY            # 0.55
feature_flags.live_trading            # False en dev
feature_flags.to_dict()               # {train_ml: True, ...}

# Legacy (sigue funcionando)
from config import WATCHLIST, BROKER_CONFIG, RISK_CONFIG
```
