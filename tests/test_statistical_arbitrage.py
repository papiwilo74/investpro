"""Tests para el motor de Arbitraje Estadístico (Pairs Trading)."""

import numpy as np
import pandas as pd

from bot.statistical_arbitrage import PairsTradingEngine


def test_pairs_trading_zscore_calculation():
    engine = PairsTradingEngine(lookback_window=20)

    # Crear datos donde A y B están perfectamente cointegrados, pero en el punto final A sube abruptamente
    dates = pd.date_range("2026-01-01", periods=30)
    b_prices = np.linspace(100, 110, 30)
    a_prices = b_prices * 1.5

    # En el último punto, A se dispara creando un Z-score alto positivo
    a_prices[-1] += 20.0

    df_a = pd.DataFrame({"close": a_prices}, index=dates)
    df_b = pd.DataFrame({"close": b_prices}, index=dates)

    spread, zscore = engine.calculate_spread_and_zscore(df_a["close"], df_b["close"])

    assert len(spread) == 30
    assert zscore > 2.0


def test_pairs_trading_signals():
    engine = PairsTradingEngine(entry_zscore=2.0, exit_zscore=0.5)

    dates = pd.date_range("2026-01-01", periods=30)
    b_prices = np.linspace(50, 55, 30)
    a_prices = b_prices * 2.0
    a_prices[-1] += 15.0  # Divergencia positiva

    df_a = pd.DataFrame({"close": a_prices}, index=dates)
    df_b = pd.DataFrame({"close": b_prices}, index=dates)

    signal = engine.analyze_pair("KO", df_a, "PEP", df_b)

    assert signal is not None
    assert signal.pair == ("KO", "PEP")
    assert signal.decision_a.action == "SHORT"
    assert signal.decision_b.action == "BUY"
    assert "Short KO, Buy PEP" in signal.reason
