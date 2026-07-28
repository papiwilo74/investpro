"""Pruebas para la Tríada de Mejoras Avanzadas: Smart Money, Trailing ATR y News Sentinel."""

import numpy as np
import pandas as pd

from bot.position_state import PositionState
from bot.smart_money import SmartMoneyTracker
from bot.strategy_params import StrategyParams
from ml.sentiment import SentimentAnalyzer


def test_smart_money_rvol_and_obv():
    tracker = SmartMoneyTracker(rvol_threshold=2.0)
    dates = pd.date_range("2026-01-01", periods=30)
    volumes = [1000] * 29 + [5000]  # Pico de volumen 5x
    prices = np.linspace(100, 110, 30)

    df = pd.DataFrame({"close": prices, "volume": volumes}, index=dates)

    flow = tracker.analyze_institutional_flow(df)

    assert flow["rvol"] >= 4.0, "RVOL debe detectar el pico de volumen de 5x"
    assert flow["smart_money_score"] > 0, "Precio subiendo con pico de volumen debe ser acumulación"
    assert flow["is_accumulation"] is True


def test_progressive_atr_trailing_stop():
    params = StrategyParams(
        use_dynamic_trailing=True, trailing_stop_atr_mult=2.5, trail_atr_base=2.5, trail_atr_tight=1.0
    )
    pos = PositionState(entry_price=100.0, entry_atr=2.0, params=params)

    # 1. Sin ganancias (PnL 0%) -> Mult base (2.5x)
    assert pos._effective_trail_mult() == 2.5

    # 2. Con ganancia de 7% -> Mult medio (1.8x)
    pos.update_extremes(107.0)
    assert pos._effective_trail_mult() == 1.8

    # 3. Con ganancia de 16% -> Mult apretado (0.9x)
    pos.update_extremes(116.0)
    assert pos._effective_trail_mult() == 0.9


def test_news_sentinel_batch():
    analyzer = SentimentAnalyzer()

    mock_bad_news = [
        {
            "headline": "Company faces massive fraud investigation and bankruptcy risk",
            "summary": "Shares plummet as SEC opens probe",
        },
        {"headline": "Earnings miss estimates significantly, guidance cut", "summary": "Revenue down 20%"},
    ]

    res = analyzer.analyze_news_batch(mock_bad_news)
    assert res["average_sentiment"] < -0.4
    assert res["global_label"] == "BAJISTA"
