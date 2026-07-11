from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.cache_manager import CacheEntry, CacheManager


@pytest.fixture(autouse=True)
def _patch_global_db(monkeypatch, tmp_path: Path) -> None:
    """Aísla el SQLite global de la base de datos de caché para cada prueba."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / "cache_index.sqlite3"
    monkeypatch.setattr("data.cache_manager._DB_PATH", db_path)
    monkeypatch.setattr("data.cache_manager._CACHE_DIR", cache_dir)
    from data import cache_manager as cm

    cm._init_db()


@pytest.fixture
def sample_df() -> pd.DataFrame:
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
def cache_manager(tmp_path: Path) -> CacheManager:
    return CacheManager(cache_dir=str(tmp_path / "cache"), default_ttl_hours=4.0)


class TestCacheManager:
    def test_store_and_retrieve(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        result = cache_manager.get("AAPL", "1y", "1d")
        assert result is not None
        assert len(result) == 10
        assert list(result.columns) == list(sample_df.columns)
        assert list(result["close"]) == list(sample_df["close"])

    def test_get_nonexistent_returns_none(self, cache_manager: CacheManager) -> None:
        result = cache_manager.get("NONEXISTENT", "1y", "1d")
        assert result is None

    def test_get_entry_metadata(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, provider="yfinance")
        entry = cache_manager.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert isinstance(entry, CacheEntry)
        assert entry.ticker == "AAPL"
        assert entry.period == "1y"
        assert entry.interval == "1d"
        assert entry.rows == 10
        assert entry.provider == "yfinance"
        assert entry.ttl_hours == 4.0
        assert entry.cached_at > 0

    def test_get_entry_nonexistent(self, cache_manager: CacheManager) -> None:
        entry = cache_manager.get_entry("NONEXISTENT", "1y", "1d")
        assert entry is None

    def test_ttl_expiration(self, cache_manager: CacheManager, monkeypatch: Any, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=0.0)
        time.sleep(0.01)
        result = cache_manager.get("AAPL", "1y", "1d")
        assert result is None

    def test_fresh_within_ttl(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=24)
        result = cache_manager.get("AAPL", "1y", "1d")
        assert result is not None

    def test_custom_ttl(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=48.0)
        entry = cache_manager.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert entry.ttl_hours == 48.0

    def test_invalidate_all_for_ticker(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        cache_manager.set("AAPL", "6mo", "1h", sample_df)
        cache_manager.set("MSFT", "1y", "1d", sample_df)
        count = cache_manager.invalidate("AAPL")
        assert count == 2
        assert cache_manager.get("AAPL", "1y", "1d") is None
        assert cache_manager.get("AAPL", "6mo", "1h") is None
        assert cache_manager.get("MSFT", "1y", "1d") is not None

    def test_invalidate_with_period_filter(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        cache_manager.set("AAPL", "6mo", "1h", sample_df)
        count = cache_manager.invalidate("AAPL", period="1y")
        assert count == 1
        assert cache_manager.get("AAPL", "1y", "1d") is None
        assert cache_manager.get("AAPL", "6mo", "1h") is not None

    def test_invalidate_with_period_and_interval_filter(
        self, cache_manager: CacheManager, sample_df: pd.DataFrame
    ) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        cache_manager.set("AAPL", "1y", "1h", sample_df)
        cache_manager.set("AAPL", "6mo", "1d", sample_df)
        count = cache_manager.invalidate("AAPL", period="1y", interval="1d")
        assert count == 1
        assert cache_manager.get("AAPL", "1y", "1d") is None
        assert cache_manager.get("AAPL", "1y", "1h") is not None
        assert cache_manager.get("AAPL", "6mo", "1d") is not None

    def test_invalidate_nonexistent_ticker(self, cache_manager: CacheManager) -> None:
        count = cache_manager.invalidate("NONEXISTENT")
        assert count == 0

    def test_invalidate_removes_parquet_file(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        parquet_path = cache_manager._parquet_path("AAPL", "1y", "1d")
        assert parquet_path.exists()
        cache_manager.invalidate("AAPL")
        assert not parquet_path.exists()

    def test_clear_expired(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=0.0)
        cache_manager.set("MSFT", "1y", "1d", sample_df, ttl_hours=24)
        time.sleep(0.01)
        count = cache_manager.clear_expired()
        assert count == 1
        assert cache_manager.get("AAPL", "1y", "1d") is None
        assert cache_manager.get("MSFT", "1y", "1d") is not None

    def test_clear_expired_no_expired(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=24)
        count = cache_manager.clear_expired()
        assert count == 0

    def test_stats_empty(self, cache_manager: CacheManager) -> None:
        stats = cache_manager.stats()
        assert stats["total_entries"] == 0
        assert stats["fresh_entries"] == 0
        assert stats["expired_entries"] == 0
        assert stats["total_rows"] == 0
        assert "cache_dir" in stats

    def test_stats_after_set(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df)
        stats = cache_manager.stats()
        assert stats["total_entries"] == 1
        assert stats["fresh_entries"] == 1
        assert stats["expired_entries"] == 0
        assert stats["total_rows"] == 10

    def test_stats_with_expired(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, ttl_hours=0.0)
        cache_manager.set("MSFT", "1y", "1d", sample_df, ttl_hours=24)
        time.sleep(0.01)
        stats = cache_manager.stats()
        assert stats["total_entries"] == 2
        assert stats["fresh_entries"] == 1
        assert stats["expired_entries"] == 1

    def test_to_dict(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, provider="yfinance")
        d = cache_manager.to_dict()
        assert "AAPL_1y_1d" in d
        assert d["AAPL_1y_1d"]["ticker"] == "AAPL"
        assert d["AAPL_1y_1d"]["provider"] == "yfinance"

    def test_to_dict_empty(self, cache_manager: CacheManager) -> None:
        d = cache_manager.to_dict()
        assert d == {}

    def test_uppercase_ticker_normalization(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("aapl", "1y", "1d", sample_df)
        assert cache_manager.get("AAPL", "1y", "1d") is not None
        entry = cache_manager.get_entry("aapl", "1y", "1d")
        assert entry is not None
        assert entry.ticker == "AAPL"

    def test_provider_in_entry(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        cache_manager.set("AAPL", "1y", "1d", sample_df, provider="test_provider")
        entry = cache_manager.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert entry.provider == "test_provider"

    def test_provider_from_df_attrs(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df.attrs["_provider"] = "df_provider"
        cache_manager.set("AAPL", "1y", "1d", df)
        entry = cache_manager.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert entry.provider == "df_provider"

    def test_latency_from_df_attrs(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        df = sample_df.copy()
        df.attrs["_latency"] = 0.15  # seconds -> will be converted to ms
        cache_manager.set("AAPL", "1y", "1d", df)
        entry = cache_manager.get_entry("AAPL", "1y", "1d")
        assert entry is not None
        assert entry.latency_ms == 150.0

    def test_thread_safety(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        n = 20

        def set_and_get(i: int) -> pd.DataFrame | None:
            cache_manager.set(f"TICKER{i}", "1y", "1d", sample_df)
            return cache_manager.get(f"TICKER{i}", "1y", "1d")

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(set_and_get, i) for i in range(n)]
            results = [f.result() for f in as_completed(futures)]

        stats = cache_manager.stats()
        assert stats["total_entries"] == n
        assert all(r is not None for r in results)

    def test_thread_safety_concurrent_invalidate(self, cache_manager: CacheManager, sample_df: pd.DataFrame) -> None:
        n = 30
        for i in range(n):
            cache_manager.set(f"TICKER{i}", "1y", "1d", sample_df)

        def invalidate_one(i: int) -> int:
            return cache_manager.invalidate(f"TICKER{i}")

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(invalidate_one, i) for i in range(n)]
            counts = [f.result() for f in as_completed(futures)]

        stats = cache_manager.stats()
        assert stats["total_entries"] == 0
        assert sum(counts) == n
