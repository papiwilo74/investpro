from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from data.cache_manager import CacheManager
from data.data_manager import DataManager
from data.provider import (
    AlpacaDataProvider,
    DataProvider,
    DataQuality,
    PolygonProvider,
    YFinanceProvider,
    create_provider,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(10) * 0.5)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, size=10),
        },
        index=dates,
    )


@pytest.fixture
def mock_provider():
    p = MagicMock(spec=DataProvider)
    p.name.return_value = "mock"
    return p


@pytest.fixture
def mock_cache(tmp_path):
    return CacheManager(cache_dir=str(tmp_path / "cache"))


# ── DataManager ────────────────────────────────────────────────────────


class TestDataManager:
    def test_get_data_cache_hit(self, mock_provider, mock_cache, sample_df):
        mock_cache.set("AAPL", "1y", "1d", sample_df, ttl_hours=24)
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        df = dm.get_data("AAPL", "1y", "1d")
        assert len(df) == 10
        mock_provider.fetch.assert_not_called()

    def test_get_data_cache_miss(self, mock_provider, mock_cache, sample_df):
        mock_provider.fetch.return_value = sample_df
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        df = dm.get_data("AAPL", "1y", "1d")
        assert len(df) == 10
        mock_provider.fetch.assert_called_once_with("AAPL", period="1y", interval="1d")

    def test_get_data_force_refresh(self, mock_provider, mock_cache, sample_df):
        mock_cache.set("AAPL", "1y", "1d", sample_df, ttl_hours=24)
        mock_provider.fetch.return_value = sample_df
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        df = dm.get_data("AAPL", "1y", "1d", force_refresh=True)
        assert len(df) == 10
        mock_provider.fetch.assert_called_once()

    def test_get_data_provider_returns_empty(self, mock_provider, mock_cache):
        mock_provider.fetch.return_value = pd.DataFrame()
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        with pytest.raises(RuntimeError, match="all providers failed"):
            dm.get_data("AAPL", "1y", "1d")

    def test_get_data_with_quality(self, mock_provider, mock_cache, sample_df):
        mock_provider.fetch.return_value = sample_df
        mock_provider.quality_check.return_value = DataQuality(
            ticker="AAPL",
            rows=10,
            date_from="2024-01-01",
            date_to="2024-01-10",
            null_pct=0.0,
            duplicate_dates=0,
            gap_days_max=1,
            latency_seconds=0.1,
        )
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        _df, quality = dm.get_data_with_quality("AAPL", "1y", "1d")
        assert isinstance(quality, DataQuality)
        assert quality.rows == 10
        assert quality.ticker == "AAPL"

    def test_get_data_with_adjustment_check(self, mock_provider, mock_cache, sample_df):
        mock_provider.fetch.return_value = sample_df
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        _df, report = dm.get_data_with_adjustment_check("AAPL", "1y", "1d")
        assert report["ticker"] == "AAPL"
        assert "splits_found" in report

    def test_set_provider(self, mock_cache):
        dm = DataManager(cache=mock_cache)
        dm.set_provider("yfinance")
        assert dm.provider.name() == "yfinance"
        dm.set_provider("polygon", api_key="test")
        assert dm.provider.name() == "polygon"

    def test_health(self, mock_provider, mock_cache, sample_df):
        mock_provider.fetch.return_value = sample_df
        mock_provider.health.return_value = {"name": "mock", "status": "ok", "latency_ms": 5}
        dm = DataManager(provider=mock_provider, cache=mock_cache)
        h = dm.health()
        assert "providers" in h
        assert h["providers"][0]["status"] == "ok"
        assert "cache" in h
        assert "quality_check" in h


# ── Failover Chain ─────────────────────────────────────────────────────


class TestFailoverChain:
    def test_primary_succeeds(self, mock_cache, sample_df):
        primary = MagicMock(spec=DataProvider)
        primary.name.return_value = "primary"
        primary.fetch.return_value = sample_df
        fallback = MagicMock(spec=DataProvider)
        dm = DataManager(provider=primary, cache=mock_cache, fallback_providers=[fallback])
        df = dm.get_data("AAPL", "1y", "1d")
        assert len(df) == 10
        primary.fetch.assert_called_once()
        fallback.fetch.assert_not_called()

    def test_failover_to_fallback(self, mock_cache, sample_df):
        primary = MagicMock(spec=DataProvider)
        primary.name.return_value = "primary"
        primary.fetch.side_effect = ValueError("primary down")
        fallback = MagicMock(spec=DataProvider)
        fallback.name.return_value = "fallback"
        fallback.fetch.return_value = sample_df
        dm = DataManager(provider=primary, cache=mock_cache, fallback_providers=[fallback])
        df = dm.get_data("AAPL", "1y", "1d")
        assert len(df) == 10
        primary.fetch.assert_called_once()
        fallback.fetch.assert_called_once()

    def test_all_providers_fail(self, mock_cache):
        primary = MagicMock(spec=DataProvider)
        primary.name.return_value = "primary"
        primary.fetch.side_effect = ValueError("primary down")
        fallback = MagicMock(spec=DataProvider)
        fallback.name.return_value = "fallback"
        fallback.fetch.side_effect = ConnectionError("fallback down")
        dm = DataManager(provider=primary, cache=mock_cache, fallback_providers=[fallback])
        with pytest.raises(RuntimeError, match="all providers failed"):
            dm.get_data("AAPL", "1y", "1d")

    def test_fallback_saves_to_cache(self, mock_cache, sample_df):
        primary = MagicMock(spec=DataProvider)
        primary.name.return_value = "primary"
        primary.fetch.side_effect = ValueError("primary down")
        fallback = MagicMock(spec=DataProvider)
        fallback.name.return_value = "fallback"
        fallback.fetch.return_value = sample_df
        dm = DataManager(provider=primary, cache=mock_cache, fallback_providers=[fallback])
        dm.get_data("AAPL", "1y", "1d")
        cached = mock_cache.get("AAPL", "1y", "1d")
        assert cached is not None
        assert len(cached) == 10

    def test_fallback_does_not_override_primary_cache(self, mock_cache, sample_df):
        primary = MagicMock(spec=DataProvider)
        primary.name.return_value = "primary"
        primary.fetch.side_effect = ValueError("primary down")
        fallback = MagicMock(spec=DataProvider)
        fallback.name.return_value = "fallback"
        fallback.fetch.return_value = sample_df
        dm = DataManager(provider=primary, cache=mock_cache, fallback_providers=[fallback])
        dm.get_data("AAPL", "1y", "1d")
        entry = mock_cache.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert entry.provider == "fallback"


# ── Providers ──────────────────────────────────────────────────────────


class TestYFinanceProvider:
    def test_name(self):
        p = YFinanceProvider()
        assert p.name() == "yfinance"

    def test_normalize_symbol_crypto(self):
        p = YFinanceProvider()
        assert p._normalize_symbol("BTCUSD") == "BTC-USD"
        assert p._normalize_symbol("ETHUSD") == "ETH-USD"
        assert p._normalize_symbol("SOLUSD") == "SOL-USD"

    def test_normalize_symbol_stock_untouched(self):
        p = YFinanceProvider()
        assert p._normalize_symbol("AAPL") == "AAPL"
        assert p._normalize_symbol("msft") == "msft"
        assert p._normalize_symbol("BTC/USD") == "BTC/USD"
        assert p._normalize_symbol("BRK-B") == "BRK-B"

    def test_quality_check(self, sample_df):
        p = YFinanceProvider()
        q = p.quality_check(sample_df, "AAPL")
        assert q.rows == 10
        assert q.duplicate_dates == 0

    def test_quality_check_empty(self):
        p = YFinanceProvider()
        q = p.quality_check(pd.DataFrame({"close": []}), "EMPTY")
        assert q.rows == 0
        assert q.null_pct == 0.0
        assert q.gap_days_max == 0

    def test_quality_check_empty_no_columns(self):
        p = YFinanceProvider()
        q = p.quality_check(pd.DataFrame(), "EMPTY")
        assert q.rows == 0


class TestAlpacaDataProvider:
    def test_no_api_key_raises(self):
        p = AlpacaDataProvider()
        with pytest.raises(ValueError, match="ALPACA_API_KEY not configured"):
            p.fetch("AAPL")

    def test_health_no_key(self):
        p = AlpacaDataProvider()
        h = p.health()
        assert h["status"] == "disabled"

    def test_name(self):
        p = AlpacaDataProvider()
        assert p.name() == "alpaca"


class TestPolygonProvider:
    def test_no_api_key_raises(self):
        p = PolygonProvider()
        with pytest.raises(ValueError, match="POLYGON_API_KEY not configured"):
            p.fetch("AAPL")

    def test_health_no_key(self):
        p = PolygonProvider()
        h = p.health()
        assert h["status"] == "disabled"

    def test_name(self):
        p = PolygonProvider()
        assert p.name() == "polygon"

    def test_parse_interval(self):
        assert PolygonProvider._parse_interval("1d") == (1, "day")
        assert PolygonProvider._parse_interval("1h") == (1, "hour")
        assert PolygonProvider._parse_interval("15m") == (15, "minute")
        assert PolygonProvider._parse_interval("5m") == (5, "minute")
        assert PolygonProvider._parse_interval("1m") == (1, "minute")

    def test_parse_period(self):
        start, end = PolygonProvider._parse_period("1y")
        assert start < end
        assert start.count("-") == 2
        assert end.count("-") == 2

    def test_quality_check(self, sample_df):
        p = PolygonProvider(api_key="test")
        q = p.quality_check(sample_df, "AAPL")
        assert q.rows == 10

    def test_create_provider_factory(self):
        p = create_provider("yfinance")
        assert isinstance(p, YFinanceProvider)
        p = create_provider("polygon", api_key="test")
        assert isinstance(p, PolygonProvider)
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")
