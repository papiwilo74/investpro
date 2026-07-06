import pytest
import pandas as pd
import numpy as np

from indicators.signals import SignalGenerator, Action


class TestSignalGenerator:

    def test_add_signal_columns_rsi(self):
        df = pd.DataFrame({
            "close": [100.0] * 20,
            "rsi": [75.0] * 10 + [25.0] * 10,
        }, index=pd.date_range("2023-01-01", periods=20, freq="D"))

        df = SignalGenerator.add_signal_columns(df)
        assert "sig_rsi" in df.columns
        # RSI > 70 debe ser señal de venta (-1)
        assert df["sig_rsi"].iloc[5] == -1
        # RSI < 30 debe ser señal de compra (1)
        assert df["sig_rsi"].iloc[15] == 1

    def test_add_signal_columns_macd_crossover(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "close": [100.0] * 10,
            "macd": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=dates)

        df = SignalGenerator.add_signal_columns(df)
        assert "sig_macd" in df.columns
        # Cruce alcista en índice 6
        assert df["sig_macd"].iloc[6] == 1
        # Cruce bajista en índice 3
        assert df["sig_macd"].iloc[3] == -1

    def test_add_signal_columns_bollinger(self):
        df = pd.DataFrame({
            "close": [90.0, 110.0, 100.0],
            "bb_upper": [105.0, 105.0, 105.0],
            "bb_lower": [95.0, 95.0, 95.0],
        }, index=pd.date_range("2023-01-01", periods=3, freq="D"))

        df = SignalGenerator.add_signal_columns(df)
        assert df["sig_bb"].iloc[0] == 1   # close <= bb_lower
        assert df["sig_bb"].iloc[1] == -1  # close >= bb_upper
        assert df["sig_bb"].iloc[2] == 0   # dentro de bandas

    def test_add_signal_columns_sma_cross(self):
        df = pd.DataFrame({
            "sma_50": [100.0, 100.0, 100.0, 110.0, 110.0],
            "sma_200": [100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=pd.date_range("2023-01-01", periods=5, freq="D"))

        df = SignalGenerator.add_signal_columns(df)
        # Golden cross en índice 3
        assert df["sig_sma"].iloc[3] == 1

    def test_composite_score_range(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        score = SignalGenerator.composite_score(df)
        assert -1.0 <= score <= 1.0

    def test_get_latest_signals_structure(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        signals = SignalGenerator.get_latest_signals(df, "AAPL")
        assert isinstance(signals, list)
        for s in signals:
            assert s.ticker == "AAPL"
            assert s.action in (Action.BUY, Action.SELL, Action.HOLD)
            assert 0.0 <= s.strength <= 1.0
            assert isinstance(s.reason, str)
