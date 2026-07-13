import numpy as np
import pandas as pd

from bot.scanner import MarketScanner
from config import ScannerConfig


class FakeFetcher:
    def __init__(self, frames):
        self.frames = frames

    def get_data(self, ticker, period="1y", interval="1d"):
        return self.frames[ticker].copy()


def make_frame(close_start=100, close_end=130, volume=2_000_000):
    index = pd.date_range("2025-01-01", periods=260, freq="B")
    close = np.linspace(close_start, close_end, len(index))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(index), volume),
        },
        index=index,
    )


def test_scanner_accepts_liquid_trending_candidate():
    scanner = MarketScanner(
        fetcher=FakeFetcher({"GOOD": make_frame()}),
        config=ScannerConfig(min_score=-1.0, min_adx=0.0),
    )

    result = scanner.scan(universe=["GOOD"], limit=5)

    assert result.scanned == 1
    assert len(result.accepted) == 1
    candidate = result.accepted[0]
    assert candidate.ticker == "GOOD"
    assert candidate.accepted is True
    assert candidate.reasons
    assert candidate.rank_score > 0


def test_scanner_rejects_low_volume_candidate_with_reason():
    scanner = MarketScanner(
        fetcher=FakeFetcher({"THIN": make_frame(volume=50_000)}),
        config=ScannerConfig(min_score=-1.0, min_adx=0.0),
    )

    result = scanner.scan(universe=["THIN"], limit=5)

    assert len(result.accepted) == 0
    assert len(result.rejected) == 1
    assert "volumen bajo" in "; ".join(result.rejected[0].warnings)


def test_scanner_orders_candidates_by_rank_score():
    scanner = MarketScanner(
        fetcher=FakeFetcher(
            {
                "STRONG": make_frame(close_start=80, close_end=140, volume=6_000_000),
                "WEAK": make_frame(close_start=100, close_end=105, volume=1_500_000),
            }
        ),
        config=ScannerConfig(min_score=-1.0, min_adx=0.0),
    )

    result = scanner.scan(universe=["WEAK", "STRONG"], limit=5)

    assert [c.ticker for c in result.accepted][:2] == ["STRONG", "WEAK"]
