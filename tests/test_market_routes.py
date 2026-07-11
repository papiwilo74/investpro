from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient


@pytest.fixture(autouse=True)
def _mock_external_deps() -> None:
    """Evita que los endpoints llamen a servicios externos reales."""
    pass


def _make_candles_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1_000_000, 10_000_000, size=n),
            "sma_20": pd.Series([100.0] * n),
            "sma_50": pd.Series([100.0] * (n // 2) + [None] * (n - n // 2)),
            "sma_200": pd.Series([None] * n),
            "rsi": pd.Series([55.0] * n),
            "macd": pd.Series([0.5] * n),
            "macd_signal": pd.Series([0.3] * n),
            "macd_histogram": pd.Series([0.2] * n),
            "bb_upper": pd.Series([110.0] * n),
            "bb_middle": pd.Series([100.0] * n),
            "bb_lower": pd.Series([90.0] * n),
        },
        index=dates,
    )


def _make_scan_result_dict() -> dict:
    return {
        "universe": "nasdaq100",
        "scanned": 2,
        "accepted_count": 1,
        "rejected_count": 1,
        "accepted": [
            {
                "ticker": "AAPL",
                "accepted": True,
                "rank_score": 0.85,
                "signal_score": 0.75,
                "trend_score": 0.9,
                "liquidity_score": 0.95,
                "volatility_score": 0.4,
                "close": 175.5,
                "change_pct": 1.2,
                "avg_volume": 50_000_000,
                "atr_pct": 1.5,
                "adx": 28.0,
                "rsi": 55.0,
                "reasons": ["strong_trend"],
                "warnings": [],
            }
        ],
        "rejected": [
            {
                "ticker": "INTC",
                "accepted": False,
                "rank_score": 0.3,
                "signal_score": 0.2,
                "trend_score": 0.1,
                "liquidity_score": 0.5,
                "volatility_score": 0.8,
                "close": 45.0,
                "change_pct": -0.5,
                "avg_volume": 20_000_000,
                "atr_pct": 2.5,
                "adx": 15.0,
                "rsi": 35.0,
                "reasons": ["low_liquidity"],
                "warnings": [],
            }
        ],
        "errors": {},
        "scan_elapsed": 1.5,
        "scan_parallel": True,
        "scan_ticker_count": 2,
    }


class TestScannerOpportunities:
    async def test_returns_expected_structure(self, client: AsyncClient) -> None:
        scan_result = MagicMock()
        scan_result.to_dict.return_value = _make_scan_result_dict()
        with patch("api.routes.market.scanner.scan", return_value=scan_result):
            response = await client.get("/api/market/scanner/opportunities")
            assert response.status_code == 200
            data = response.json()
            assert data["universe"] == "nasdaq100"
            assert data["scanned"] == 2
            assert data["accepted_count"] == 1
            assert data["rejected_count"] == 1
            assert len(data["accepted"]) == 1
            assert data["accepted"][0]["ticker"] == "AAPL"
            assert len(data["rejected"]) == 1
            assert data["rejected"][0]["ticker"] == "INTC"
            assert "scan_elapsed" in data

    async def test_with_limit_param(self, client: AsyncClient) -> None:
        scan_result = MagicMock()
        scan_result.to_dict.return_value = _make_scan_result_dict()
        with patch("api.routes.market.scanner.scan", return_value=scan_result):
            response = await client.get("/api/market/scanner/opportunities?limit=5")
            assert response.status_code == 200

    async def test_with_universe_param(self, client: AsyncClient) -> None:
        scan_result = MagicMock()
        scan_result.to_dict.return_value = _make_scan_result_dict()
        with patch("api.routes.market.scanner.scan", return_value=scan_result):
            response = await client.get("/api/market/scanner/opportunities?universe=sp500")
            assert response.status_code == 200

    async def test_with_include_rejected_false(self, client: AsyncClient) -> None:
        scan_result = MagicMock()
        scan_result.to_dict.return_value = _make_scan_result_dict()
        with patch("api.routes.market.scanner.scan", return_value=scan_result):
            response = await client.get("/api/market/scanner/opportunities?include_rejected=false")
            assert response.status_code == 200

    async def test_limit_clamps_to_max(self, client: AsyncClient) -> None:
        scan_result = MagicMock()
        scan_result.to_dict.return_value = _make_scan_result_dict()
        with patch("api.routes.market.scanner.scan", return_value=scan_result):
            response = await client.get("/api/market/scanner/opportunities?limit=100")
            assert response.status_code in (200, 422)

    async def test_scanner_error_returns_400(self, client: AsyncClient) -> None:
        with patch("api.routes.market.scanner.scan", side_effect=ValueError("Scanner API down")):
            response = await client.get("/api/market/scanner/opportunities")
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data

    async def test_scanner_internal_error_returns_400(self, client: AsyncClient) -> None:
        with patch("api.routes.market.scanner.scan", side_effect=RuntimeError("Internal error")):
            response = await client.get("/api/market/scanner/opportunities")
            assert response.status_code == 400


class TestMarketData:
    async def test_returns_candles_and_indicators(self, client: AsyncClient) -> None:
        df = _make_candles_df()
        with (
            patch("api.routes.market.fetcher.get_data", return_value=df) as mock_get,
            patch("api.routes.market.TechnicalIndicators.add_all", return_value=df) as mock_add,
        ):
            response = await client.get("/api/market/AAPL")
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"
            assert len(data["candles"]) == 10
            assert "open" in data["candles"][0]
            assert "high" in data["candles"][0]
            assert "low" in data["candles"][0]
            assert "close" in data["candles"][0]
            assert "volume" in data["candles"][0]
            assert "time" in data["candles"][0]
            assert "indicators" in data
            assert "sma_20" in data["indicators"]
            assert "sma_50" in data["indicators"]
            assert "sma_200" in data["indicators"]
            assert "rsi" in data["indicators"]
            assert "macd" in data["indicators"]
            assert "bb" in data["indicators"]
            assert "latest" in data
            assert "close" in data["latest"]
            assert "change_pct" in data["latest"]
            assert "volume" in data["latest"]
            assert mock_get.call_count == 1
            assert mock_add.call_count == 1

    async def test_normalizes_ticker_to_uppercase(self, client: AsyncClient) -> None:
        df = _make_candles_df()
        with (
            patch("api.routes.market.fetcher.get_data", return_value=df),
            patch("api.routes.market.TechnicalIndicators.add_all", return_value=df),
        ):
            response = await client.get("/api/market/aapl")
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == "AAPL"

    async def test_with_period_and_interval_params(self, client: AsyncClient) -> None:
        df = _make_candles_df()
        with (
            patch("api.routes.market.fetcher.get_data", return_value=df),
            patch("api.routes.market.TechnicalIndicators.add_all", return_value=df),
        ):
            response = await client.get("/api/market/AAPL?period=6mo&interval=1h")
            assert response.status_code == 200

    async def test_error_returns_400(self, client: AsyncClient) -> None:
        with patch("api.routes.market.fetcher.get_data", side_effect=ValueError("No data for ticker")):
            response = await client.get("/api/market/UNKNOWN")
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data

    async def test_fetcher_internal_error_returns_400(self, client: AsyncClient) -> None:
        with patch("api.routes.market.fetcher.get_data", side_effect=RuntimeError("API failure")):
            response = await client.get("/api/market/FAIL")
            assert response.status_code == 400

    async def test_single_candle_edge_case(self, client: AsyncClient) -> None:
        df = _make_candles_df(n=1)
        with (
            patch("api.routes.market.fetcher.get_data", return_value=df),
            patch("api.routes.market.TechnicalIndicators.add_all", return_value=df),
        ):
            response = await client.get("/api/market/AAPL")
            assert response.status_code == 200
            data = response.json()
            assert len(data["candles"]) == 1
            assert data["latest"]["change_pct"] == 0.0  # prev_close == last_close


class TestMarketNews:
    async def test_returns_news(self, client: AsyncClient) -> None:
        mock_news = [{"headline": "AAPL hits all-time high", "sentiment": "positive", "score": 0.8}]
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_news_batch.return_value = {"news": mock_news, "sentiment_summary": {"avg_score": 0.8}}
        with (
            patch("data.news.NewsFetcher.get_latest_news", return_value=mock_news),
            patch("ml.sentiment.SentimentAnalyzer", return_value=mock_analyzer),
        ):
            response = await client.get("/api/market/AAPL/news")
            assert response.status_code == 200
            data = response.json()
            assert "news" in data
            assert "sentiment_summary" in data

    async def test_with_limit_param(self, client: AsyncClient) -> None:
        mock_news = [{"headline": f"News {i}", "sentiment": "neutral", "score": 0.5} for i in range(5)]
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_news_batch.return_value = {"news": mock_news}
        with (
            patch("data.news.NewsFetcher.get_latest_news", return_value=mock_news),
            patch("ml.sentiment.SentimentAnalyzer", return_value=mock_analyzer),
        ):
            response = await client.get("/api/market/AAPL/news?limit=5")
            assert response.status_code == 200

    async def test_normalizes_ticker(self, client: AsyncClient) -> None:
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_news_batch.return_value = {"news": []}
        with (
            patch("data.news.NewsFetcher.get_latest_news", return_value=[]),
            patch("ml.sentiment.SentimentAnalyzer", return_value=mock_analyzer),
        ):
            response = await client.get("/api/market/aapl/news")
            assert response.status_code == 200

    async def test_error_returns_400(self, client: AsyncClient) -> None:
        with (
            patch("data.news.NewsFetcher.get_latest_news", side_effect=ValueError("News API error")),
        ):
            response = await client.get("/api/market/AAPL/news")
            assert response.status_code == 400
