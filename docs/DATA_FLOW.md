# Flujo de Datos

## Data Providers

| Provider | Fuente | API Key | Velocidad | Costo | Estado |
|----------|--------|---------|-----------|-------|--------|
| YFinanceProvider | Yahoo Finance | No | ~500ms | Gratis | ✅ Producción |
| AlpacaDataProvider | Alpaca Markets | Sí | ~200ms | Gratis (paper) | ✅ Producción |
| PolygonProvider | Polygon.io | Sí | ~100ms | Pago | 🔧 Stub |

### Failover Chain

```
DataManager.get_data()
  → 1. CacheManager.get()           # SQLite + Parquet
  → 2. YFinanceProvider.fetch()     # Gratuito, rápido
  → 3. AlpacaProvider.fetch()       # Fallback si yfinance falla
  → 4. PolygonProvider.fetch()      # Fallback si Alpaca falla
  → 5. CacheManager.set()           # Cachear resultado
```

## Caché

```
cache/
├── cache_index.sqlite3     # Índice SQLite (ticker, period, interval, TTL, rows)
├── AAPL_1y_1d.parquet      # Datos OHLCV en Parquet
├── MSFT_1y_1d.parquet
└── ...
```

- TTL default: 4 horas
- Limpieza automática al arrancar (`cache_manager.clear_expired()`)
- Thread-safe con `threading.Lock`

## Quality Check

Cada descarga pasa por `DataQuality`:

| Campo | Descripción | Umbral |
|-------|-------------|--------|
| rows | Número de filas | > 20 |
| null_pct | % de valores nulos | < 5% |
| duplicate_dates | Fechas duplicadas | 0 |
| gap_days_max | Máximo gap entre días | < 5 días |
| latency_seconds | Latencia de descarga | < 10s |

## Split/Dividend Adjustment

```python
report = SplitAdjuster.check_adjustment("AAPL", df)
report.splits_found      # Lista de splits detectados
report.dividends_found   # Lista de dividendos
report.verified          # True si todos los splits están ajustados
report.price_discrepancy_pct  # Discrepancia máxima de precio
```
