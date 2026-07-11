"""Abstract Data Provider — interfaz común para fuentes de datos OHLCV.

Implementaciones:
  - YFinanceProvider  (gratuito, sin API key)
  - AlpacaDataProvider  (requiere API key Alpaca)
  - PolygonProvider  (stub, requiere API key Polygon)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataQuality:
    ticker: str
    rows: int
    date_from: str
    date_to: str
    null_pct: float
    duplicate_dates: int
    gap_days_max: int
    latency_seconds: float


class DataProvider(ABC):
    """Interfaz abstracta para descarga de OHLCV."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame: ...

    def health(self) -> dict[str, Any]:
        """Health check del provider. Default: ok."""
        return {"name": self.name(), "status": "ok", "latency_ms": 0}

    def quality_check(self, df: pd.DataFrame, ticker: str) -> DataQuality:
        date_from = str(df.index.min().date()) if not df.empty else ""
        date_to = str(df.index.max().date()) if not df.empty else ""
        total = len(df)
        nulls = df.isnull().sum().sum()
        null_pct = nulls / max(total, 1)
        dupes = df.index.duplicated().sum()
        max_gap = 0
        if not df.empty:
            try:
                gaps = df.index.to_series().diff().dt.days.dropna()
                max_gap = int(gaps.max()) if not gaps.empty else 0
            except AttributeError:
                max_gap = 0
        return DataQuality(
            ticker=ticker,
            rows=total,
            date_from=date_from,
            date_to=date_to,
            null_pct=round(null_pct, 4),
            duplicate_dates=int(dupes),
            gap_days_max=max_gap,
            latency_seconds=0.0,
        )


class YFinanceProvider(DataProvider):
    """Provider basado en yfinance (gratuito, sin autenticación)."""

    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        t0 = time.time()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        latency = time.time() - t0

        if df.empty:
            raise ValueError(f"YFinance: no data for {ticker} ({period}/{interval})")

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "date"

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep]

        df.attrs["_provider"] = self.name()
        df.attrs["_latency"] = round(latency, 3)
        return df

    def health(self) -> dict[str, Any]:
        t0 = time.time()
        try:
            import yfinance as yf

            stock = yf.Ticker("SPY")
            df = stock.history(period="5d", interval="1d")
            ok = not df.empty
            latency_ms = round((time.time() - t0) * 1000)
            return {"name": "yfinance", "status": "ok" if ok else "degraded", "latency_ms": latency_ms}
        except Exception as e:
            return {"name": "yfinance", "status": "error", "error": str(e)}


class AlpacaDataProvider(DataProvider):
    """Provider basado en Alpaca Markets API (requiere API key)."""

    def __init__(self, api_key: str = "", secret_key: str = "", base_url: str = "https://paper-api.alpaca.markets"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self._client = None

    def name(self) -> str:
        return "alpaca"

    def _get_client(self):
        if self._client is None:
            from alpaca.data import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def fetch(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("AlpacaDataProvider: ALPACA_API_KEY not configured")

        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        t0 = time.time()
        client = self._get_client()

        # Map period/interval to Alpaca params
        tf_map = {
            "1d": TimeFrame.Day,
            "1h": TimeFrame.Hour,
            "15m": TimeFrame.Minute,
            "5m": TimeFrame.Minute,
            "1m": TimeFrame.Minute,
        }
        tf = tf_map.get(interval, TimeFrame.Day)
        multiplier = 1
        if interval == "15m":
            multiplier = 15
        elif interval == "5m":
            multiplier = 5

        days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = days_map.get(period, 365)

        from datetime import datetime, timedelta

        import pytz

        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(multiplier, tf),
            start=start,
            end=end,
            adjustment=Adjustment.SPLIT,
            limit=10000,
        )
        bars = client.get_stock_bars(request)

        if ticker not in bars.data or not bars.data[ticker]:
            raise ValueError(f"Alpaca: no data for {ticker}")

        records = []
        for bar in bars.data[ticker]:
            records.append(
                {
                    "date": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.index.name = "date"
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.attrs["_provider"] = self.name()
        df.attrs["_latency"] = round(time.time() - t0, 3)
        return df

    def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {"name": "alpaca", "status": "disabled", "error": "no API key"}
        t0 = time.time()
        try:
            df = self.fetch("SPY", period="5d", interval="1d")
            latency_ms = round((time.time() - t0) * 1000)
            return {"name": "alpaca", "status": "ok" if not df.empty else "degraded", "latency_ms": latency_ms}
        except Exception as e:
            return {"name": "alpaca", "status": "error", "error": str(e)}


class PolygonProvider(DataProvider):
    """Provider basado en Polygon.io REST API v2 (requiere API key)."""

    _BASE = "https://api.polygon.io"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def name(self) -> str:
        return "polygon"

    def fetch(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("PolygonProvider: POLYGON_API_KEY not configured")

        multiplier, timespan = self._parse_interval(interval)
        start, end = self._parse_period(period)

        url = f"{self._BASE}/v2/aggs/ticker/{ticker.upper()}/range/" f"{multiplier}/{timespan}/{start}/{end}"
        params = {"apiKey": self.api_key, "limit": 50000, "adjusted": "true"}

        t0 = time.time()
        import json
        import urllib.parse
        import urllib.request

        full_url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={"User-Agent": "InversionHelper/1.0"})

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        latency = time.time() - t0

        if data.get("status") != "OK" or "results" not in data or not data["results"]:
            raise ValueError(f"Polygon: no data for {ticker} ({period}/{interval}): {data.get('error', 'unknown')}")

        records = []
        for bar in data["results"]:
            records.append(
                {
                    "date": pd.Timestamp(bar["t"], unit="ms"),
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar.get("v", 0)) if bar.get("v") else 0,
                }
            )

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        df.index.name = "date"
        df.sort_index(inplace=True)

        df.attrs["_provider"] = self.name()
        df.attrs["_latency"] = round(latency, 3)
        return df

    @staticmethod
    def _parse_interval(interval: str) -> tuple[int, str]:
        """Convierte '1d' -> (1, 'day'), '1h' -> (1, 'hour'), etc."""
        mapping = {
            "1d": (1, "day"),
            "1h": (1, "hour"),
            "15m": (15, "minute"),
            "5m": (5, "minute"),
            "1m": (1, "minute"),
        }
        if interval not in mapping:
            return 1, "day"
        return mapping[interval]

    @staticmethod
    def _parse_period(period: str) -> tuple[str, str]:
        """Convierte '1y' -> ('2025-07-11', '2026-07-11') approx."""
        from datetime import datetime, timedelta

        end = datetime.utcnow()
        days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = days_map.get(period, 365)
        start = end - timedelta(days=days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def health(self) -> dict[str, Any]:
        if not self.api_key:
            return {"name": "polygon", "status": "disabled", "error": "no API key"}
        t0 = time.time()
        try:
            df = self.fetch("SPY", period="5d", interval="1d")
            latency_ms = round((time.time() - t0) * 1000)
            return {"name": "polygon", "status": "ok" if not df.empty else "degraded", "latency_ms": latency_ms}
        except Exception as e:
            return {"name": "polygon", "status": "error", "error": str(e)}


def create_provider(source: str = "yfinance", **kwargs: Any) -> DataProvider:
    """Factory: crea un DataProvider según el nombre."""
    providers = {
        "yfinance": YFinanceProvider,
        "alpaca": AlpacaDataProvider,
        "polygon": PolygonProvider,
    }
    cls = providers.get(source)
    if cls is None:
        raise ValueError(f"Unknown provider: {source}. Options: {list(providers.keys())}")
    return cls(**kwargs)
