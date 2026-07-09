"""
Data fetching — descarga OHLCV de yfinance con caché local en parquet.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import CACHE_CONFIG, CacheConfig


class DataFetcher:
    """Descarga y cachea datos de mercado OHLCV."""

    def __init__(self, cache_config: CacheConfig | None = None) -> None:
        self.config = cache_config or CACHE_CONFIG
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers privados ──────────────────────────────────────────────

    def _cache_path(self, ticker: str, period: str, interval: str) -> Path:
        return self.cache_dir / f"{ticker}_{period}_{interval}.parquet"

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < self.config.ttl_hours

    # ── API pública ───────────────────────────────────────────────────

    def get_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Obtiene datos OHLCV para *ticker*.  Retorna copia cacheada si
        el archivo tiene menos de ``ttl_hours`` horas de antigüedad.
        """
        cache_path = self._cache_path(ticker, period, interval)

        if self._is_cache_fresh(cache_path):
            return pd.read_parquet(cache_path)

        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No se encontraron datos para «{ticker}»")

        # Normalizar nombres de columnas
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "date"
        df.index = df.index.tz_localize(None)

        # Conservar solo OHLCV
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep]

        # Persistir en caché
        df.to_parquet(cache_path)
        return df

    def fetch_batch(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 14,
    ) -> dict[str, pd.DataFrame]:
        """Descarga datos para múltiples tickers en paralelo con ThreadPoolExecutor.

        Con 14 workers en una i7-13650HX (20 hilos), descargar 100 tickers
        pasa de ~50s (secuencial) a ~4-5s (paralelo).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.get_data, ticker, period, interval): ticker
                for ticker in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as exc:
                    errors[ticker] = str(exc)

        if errors:
            from loguru import logger
            logger.warning("fetch_batch: {} errores en {} tickers", len(errors), len(tickers))
            for t, e in list(errors.items())[:3]:
                logger.warning("  {} → {}", t, e)

        return results

    def clear_cache(self, ticker: str | None = None) -> int:
        """Elimina archivos de caché.  Retorna cantidad de archivos borrados."""
        pattern = f"{ticker}_*" if ticker else "*.parquet"
        files = list(self.cache_dir.glob(pattern))
        for f in files:
            f.unlink()
        return len(files)
