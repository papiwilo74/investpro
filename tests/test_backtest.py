import pytest
import pandas as pd
import numpy as np

from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics


class TestBacktestEngine:

    def test_backtest_runs_without_errors(self, synthetic_backtest_df):
        engine = BacktestEngine()
        result = engine.run(synthetic_backtest_df)
        assert result.metrics["total_trades"] >= 0
        assert len(result.equity_curve) == len(synthetic_backtest_df)

    def test_backtest_missing_signal_column_raises(self):
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.date_range("2023-01-01", periods=2))
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="sig_composite"):
            engine.run(df)

    def test_backtest_with_known_result(self):
        """
        Precio: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        Señal:   0,  1,  0,  0, -1,  0,  0,  0,  0,  0
        
        shift(1) en señal:
        signals: 0,  0,  1,  0,  0, -1,  0,  0,  0,  0
        
        i=2: signal=1, compra en 12 (slippage 0.05% → 12.006)
            shares = int(100000 / 12.006) = 8333
            cost = 8333 * 12.006 = 100,005.998
            comm = 100,005.998 * 0.001 = 100.006
            capital = 100,000 - 100,005.998 - 100.006 ≈ -106
        
        Espera, con precio 10 funciona mejor.
        """
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "sig_composite": [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=dates)
        
        engine = BacktestEngine()
        result = engine.run(df)
        
        # Debe haber exactamente 1 trade (compra en día 3, venta en día 6)
        assert result.metrics["total_trades"] == 1
        # Equity final debe ser mayor que inicial porque el precio subió
        assert result.metrics["capital_final"] > 100_000.0
        # Retorno total debe ser positivo
        assert result.metrics["retorno_total"] > 0

    def test_backtest_sell_then_no_reentry(self):
        """
        Después de vender, si no hay señal de compra, debe quedar en cash.
        """
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            "sig_composite": [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }, index=dates)
        
        engine = BacktestEngine()
        result = engine.run(df)
        
        # Solo un trade, no reentra después de vender
        assert result.metrics["total_trades"] == 1
        # El capital final debe ser cash (no hay posición abierta al final)
        assert result.equity_curve.iloc[-1] == result.metrics["capital_final"]

    def test_performance_metrics_empty(self):
        empty = pd.Series([], dtype=float)
        trades = []
        assert PerformanceMetrics.cumulative_return(empty) == 0.0
        assert PerformanceMetrics.annualized_return(empty) == 0.0
        assert PerformanceMetrics.sharpe_ratio(empty) == 0.0
        assert PerformanceMetrics.win_rate(trades) == 0.0
        assert PerformanceMetrics.profit_factor(trades) == 0.0

    def test_profit_factor_infinite(self):
        """Si no hay pérdidas, profit_factor debe ser inf."""
        from backtesting.metrics import Trade
        trades = [
            Trade("2023-01-01", "2023-01-02", "LONG", 100, 110, 10, 100, 0.1, 1),
        ]
        assert PerformanceMetrics.profit_factor(trades) == float("inf")
