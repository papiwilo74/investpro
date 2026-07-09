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
        # RSI > 70 debe ser negativo (señal de venta)
        assert df["sig_rsi"].iloc[5] < 0
        # RSI < 30 debe ser positivo (señal de compra)
        assert df["sig_rsi"].iloc[15] > 0

    def test_add_signal_columns_macd_crossover(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "close": [100.0] * 10,
            "macd": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            "macd_signal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=dates)

        df = SignalGenerator.add_signal_columns(df)
        assert "sig_macd" in df.columns
        # MACD > signal en filas con std > 0 = señal positiva
        assert pd.notna(df["sig_macd"].iloc[1]) and df["sig_macd"].iloc[1] > 0
        # MACD < signal = señal negativa (bajista)
        assert df["sig_macd"].iloc[3] < 0
        # Cruce alcista binario
        assert "sig_macd_cross" in df.columns
        assert df["sig_macd_cross"].iloc[6] == 1
        assert df["sig_macd_cross"].iloc[3] == -1

    def test_add_signal_columns_bollinger(self):
        df = pd.DataFrame({
            "close": [90.0, 110.0, 100.0],
            "bb_upper": [105.0, 105.0, 105.0],
            "bb_lower": [95.0, 95.0, 95.0],
        }, index=pd.date_range("2023-01-01", periods=3, freq="D"))

        df = SignalGenerator.add_signal_columns(df)
        # close <= bb_lower debe ser señal positiva (compra)
        assert df["sig_bb"].iloc[0] > 0
        # close >= bb_upper debe ser señal negativa (venta)
        assert df["sig_bb"].iloc[1] < 0
        # dentro de bandas debe ser cercano a 0
        assert abs(df["sig_bb"].iloc[2]) < 0.5

    def test_add_signal_columns_sma_cross(self):
        df = pd.DataFrame({
            "sma_50": [100.0, 100.0, 100.0, 110.0, 110.0],
            "sma_200": [100.0, 100.0, 100.0, 100.0, 100.0],
        }, index=pd.date_range("2023-01-01", periods=5, freq="D"))

        df = SignalGenerator.add_signal_columns(df)
        # SMA50 > SMA200 = señal positiva
        assert df["sig_sma"].iloc[3] > 0
        # Golden cross binario
        assert "sig_sma_cross" in df.columns
        assert df["sig_sma_cross"].iloc[3] == 1

    def test_composite_score_range(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        score = SignalGenerator.composite_score(df)
        assert not np.isnan(score)
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

    def test_composite_score_bounds(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        score = SignalGenerator.composite_score(df)
        assert -1.0 <= score <= 1.0

    def test_composite_score_float(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        score = SignalGenerator.composite_score(df)
        assert isinstance(score, float)

    def test_empty_df_composite_score(self):
        df = pd.DataFrame()
        score = SignalGenerator.composite_score(df)
        assert score == 0.0

    def test_signal_generator_add_all_columns(self, dummy_with_indicators):
        df = SignalGenerator.add_signal_columns(dummy_with_indicators)
        expected_cols = {"sig_composite", "sig_momentum", "sig_volume", "sig_rsi", "sig_macd"}
        assert expected_cols.issubset(set(df.columns))
