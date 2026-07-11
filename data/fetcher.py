"""
Data fetching — descarga OHLCV con caché local en parquet.

Ahora soporta dos modos:
  - DataFetcher (original, retrocompatible)
  - data_manager (nuevo, con DataProvider + CacheManager + SplitAdjuster)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CACHE_CONFIG, CacheConfig
from data.data_manager import DataManager
from data.provider import DataProvider


class DataFetcher:
    """Descarga y cachea datos de mercado OHLCV.

    Mantiene compatibilidad hacia atrás. Internamente delega en DataManager.
    """

    def __init__(self, cache_config: CacheConfig | None = None, provider: DataProvider | None = None) -> None:
        self.config = cache_config or CACHE_CONFIG
        self._dm = DataManager(
            provider=provider,
            default_ttl_hours=float(self.config.ttl_hours),
        )

    def _cache_path(self, ticker: str, period: str, interval: str) -> Path:
        return Path(self.config.cache_dir) / f"{ticker}_{period}_{interval}.parquet"

    def _is_cache_fresh(self, path: Path, ttl_hours: float | None = None) -> bool:
        """Método legacy: checkea archivo directo. Usar CacheManager en su lugar."""
        import time
        if not path.exists():
            return False
        ttl = ttl_hours if ttl_hours is not None else self.config.ttl_hours
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < ttl

    def get_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        return self._dm.get_data(ticker, period, interval, force_refresh=force_refresh)

    def fetch_batch(
        self,
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 14,
    ) -> dict[str, pd.DataFrame]:
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
        from data.cache_manager import cache_manager
        if ticker:
            return cache_manager.invalidate(ticker)
        # No hay API para limpiar todo aún
        return 0

    @property
    def data_manager(self) -> DataManager:
        return self._dm
