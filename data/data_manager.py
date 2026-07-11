from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from data.cache_manager import CacheManager, cache_manager
from data.provider import (
    AlpacaDataProvider,
    DataProvider,
    DataQuality,
    PolygonProvider,
    YFinanceProvider,
    create_provider,
)
from data.split_adjuster import SplitAdjuster

logger = logging.getLogger(__name__)


class DataManager:
    """Manejador unificado de datos con failover chain: YFinance -> Alpaca -> Polygon."""

    def __init__(
        self,
        provider: DataProvider | None = None,
        cache: CacheManager | None = None,
        default_ttl_hours: float = 4.0,
        fallback_providers: list[DataProvider] | None = None,
    ):
        self.provider = provider or YFinanceProvider()
        self.cache = cache or cache_manager
        self.default_ttl_hours = default_ttl_hours
        self.fallback_providers = fallback_providers or []

    def _all_providers(self) -> list[DataProvider]:
        """Retorna [primary, *fallbacks]."""
        return [self.provider, *self.fallback_providers]

    def get_data(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
        check_quality: bool = True,
    ) -> pd.DataFrame:
        """Obtiene datos OHLCV con failover chain."""
        if not force_refresh:
            cached = self.cache.get(ticker, period, interval)
            if cached is not None and not cached.empty:
                return cached

        errors: list[str] = []
        for prov in self._all_providers():
            try:
                df = prov.fetch(ticker, period=period, interval=interval)
                if df.empty:
                    raise ValueError(f"empty data from {prov.name()}")

                self.cache.set(
                    ticker,
                    period,
                    interval,
                    df,
                    ttl_hours=self.default_ttl_hours,
                    provider=prov.name(),
                )

                if prov is not self.provider:
                    logger.info(
                        "Failover: %s served data for %s (primary %s failed)", prov.name(), ticker, self.provider.name()
                    )

                return df
            except Exception as e:
                msg = f"{prov.name()}: {e}"
                errors.append(msg)
                logger.warning("DataManager failover: %s", msg)

        raise RuntimeError(
            f"DataManager: all providers failed for {ticker} ({period}/{interval}). " f"Errors: {'; '.join(errors)}"
        )

    def get_data_with_quality(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, DataQuality]:
        """Como get_data pero retorna (df, quality_report)."""
        t0 = time.time()
        df = self.get_data(ticker, period, interval, force_refresh=force_refresh)
        latency = time.time() - t0

        quality = self.provider.quality_check(df, ticker)
        quality.latency_seconds = round(latency, 3)
        return df, quality

    def get_data_with_adjustment_check(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = self.get_data(ticker, period, interval)
        report = SplitAdjuster.check_adjustment(ticker, df)
        return df, SplitAdjuster.to_dict(report)

    def health(self) -> dict[str, Any]:
        """Health check completo del data layer."""
        providers_health: list[dict[str, Any]] = []
        for prov in self._all_providers():
            try:
                h = prov.health()
            except Exception as e:
                h = {"name": prov.name(), "status": "error", "error": str(e)}
            providers_health.append(h)

        cache_stats = self.cache.stats()

        quality_result: dict[str, Any] = {"status": "unknown", "latency_ms": 0}
        try:
            df, quality = self.get_data_with_quality("SPY", period="1mo", interval="1d", force_refresh=True)
            quality_result = {
                "status": "ok" if not df.empty else "degraded",
                "latency_ms": round(quality.latency_seconds * 1000),
                "rows": quality.rows,
                "null_pct": quality.null_pct,
            }
        except Exception as e:
            quality_result = {"status": "error", "error": str(e)}

        return {
            "providers": providers_health,
            "cache": cache_stats,
            "quality_check": quality_result,
        }

    def set_provider(self, source: str, **kwargs: Any) -> None:
        """Cambia el provider activo."""
        self.provider = create_provider(source, **kwargs)


data_manager = DataManager(
    fallback_providers=[
        AlpacaDataProvider(),
        PolygonProvider(),
    ],
)
