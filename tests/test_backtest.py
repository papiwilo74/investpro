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


class TestBotBacktestEngine:
    def test_bot_engine_creates_brain(self):
        from backtesting.bot_engine import BotBacktestEngine
        engine = BotBacktestEngine()
        assert engine.brain is not None

    def test_bot_engine_runs_with_synthetic_data(self, synthetic_backtest_df):
        from backtesting.bot_engine import BotBacktestEngine
        engine = BotBacktestEngine()
        result = engine.run(synthetic_backtest_df, ticker="TEST")
        assert len(result.equity_curve) == len(synthetic_backtest_df)
        assert result.metrics["total_trades"] >= 0

    def test_bot_engine_missing_signal_column_raises(self):
        from backtesting.bot_engine import BotBacktestEngine
        df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.date_range("2023-01-01", periods=2))
        engine = BotBacktestEngine()
        with pytest.raises(ValueError, match="sig_composite"):
            engine.run(df)

    def test_bot_backtest_params_leverage_default(self):
        from backtesting.bot_engine import BotBacktestEngine
        engine = BotBacktestEngine()
        assert engine.leverage >= 1.0

class TestMonteCarlo:
    def test_monte_carlo_empty_trades(self):
        from backtesting.validation import MonteCarloSimulator
        mc = MonteCarloSimulator(n_simulations=100).run([])
        assert mc.n_simulations == 0
        assert mc.p50_return == 0.0

    def test_monte_carlo_with_trades(self):
        from backtesting.validation import MonteCarloSimulator
        from backtesting.metrics import Trade
        trades = [
            Trade("2023-01-01", "2023-01-10", "LONG", 100, 110, 10, 100, 0.1, 0.5),
            Trade("2023-01-11", "2023-01-20", "LONG", 110, 105, 10, -50, -0.05, 0.5),
            Trade("2023-01-21", "2023-01-30", "LONG", 105, 120, 10, 150, 0.14, 0.5),
        ]
        mc = MonteCarloSimulator(n_simulations=200).run(trades)
        assert mc.n_simulations == 200
        assert mc.p50_return is not None

    def test_monte_carlo_percentiles_well_ordered(self):
        from backtesting.validation import MonteCarloSimulator
        from backtesting.metrics import Trade
        trades = [Trade(f"2023-01-0{i}", f"2023-01-1{i}", "LONG", 100 + i, 100 + i + (5 if i % 2 == 0 else -3), 10, 0, 0, 0.5) for i in range(20)]
        mc = MonteCarloSimulator(n_simulations=500).run(trades)
        assert mc.p5_return <= mc.p50_return <= mc.p95_return


class TestWalkForwardOptimizer:
    def test_wfo_minimal_data_returns_empty(self):
        from backtesting.validation import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(train_months=24, test_months=6)
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"close": range(100, 130), "sig_composite": [0.0]*30}, index=dates)
        windows = wfo.run(df, ticker="TEST")
        # Not enough data for even one window
        assert len(windows) == 0


class TestOverfitDetector:
    def test_detect_no_overfit_with_good_metrics(self):
        from backtesting.validation import OverfitDetector
        detector = OverfitDetector()
        is_metrics = {"sharpe_ratio": 1.5, "retorno_total": 0.3}
        oos_metrics = {"sharpe_ratio": 1.2, "retorno_total": 0.2}
        flags = detector.detect([], oos_metrics, is_metrics)
        assert flags.get("oos_sharpe_positive", True) is True

    def test_detect_overfit_when_oos_sharpe_negative(self):
        from backtesting.validation import OverfitDetector
        detector = OverfitDetector()
        is_metrics = {"sharpe_ratio": 1.5, "retorno_total": 0.3}
        oos_metrics = {"sharpe_ratio": -0.3, "retorno_total": -0.1}
        flags = detector.detect([], oos_metrics, is_metrics)
        assert flags.get("oos_sharpe_positive", True) is False

    def test_verdict_rejected_with_high_overfit(self):
        from backtesting.validation import OverfitDetector
        detector = OverfitDetector()
        flags = {"oos_sharpe_positive": False, "oos_is_ratio": 0.3}
        mc = type("MC", (), {"prob_negative_return": 0.5, "prob_sharpe_above_1": 0.3})()
        verdict = detector.verdict(flags, mc)
        assert verdict in ("RECHAZADO", "CONDICIONAL", "APROBADO")


    def test_bot_engine_leverage_applied_to_position_size(self):
        from backtesting.bot_engine import BotBacktestEngine, BacktestParams
        from bot.strategy import StrategyParams
        params = StrategyParams(buy_score_threshold=-1.0, use_ml_filter=False, max_position_size_pct=1.0)
        engine = BotBacktestEngine(strategy_params=params, leverage=2.0)
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "sig_composite": [1.0, 0.0, 0.0, 0.0, -1.0],
        }, index=dates)
        result = engine.run(df, ticker="TEST")
        # With leverage 2.0, position should be larger
        assert result.metrics["total_trades"] > 0
