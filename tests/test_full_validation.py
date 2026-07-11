"""Tests for the Full Validation Pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from backtesting.full_validation import (
    FullValidationPipeline,
    FullValidationResult,
    ValidationConfig,
    run_full_validation,
)
from bot.strategy import StrategyParams


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generates a synthetic price DataFrame with technical indicators."""
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + np.random.rand(n) * 2.0,
            "low": close - np.random.rand(n) * 2.0,
            "open": close - np.random.randn(n) * 0.3,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )

    # Add minimal indicators needed by BotBacktestEngine
    df["sig_composite"] = np.where(np.random.rand(n) > 0.55, 0.3, -0.2)
    # Add SMA columns that TradingBrain might use
    df["sma_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(50, min_periods=1).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=1).mean()
    # Add RSI-like column
    df["rsi"] = 50.0 + np.random.rand(n) * 30
    df.loc[df["rsi"] > 80, "rsi"] = 75.0
    df.loc[df["rsi"] < 20, "rsi"] = 25.0
    # Add ADX-like column
    df["adx"] = 20.0 + np.random.rand(n) * 15
    # Add ATR
    df["atr"] = df["close"] * 0.02 + np.random.rand(n) * 0.5
    # Add Bollinger Bands
    df["bb_upper"] = df["sma_20"] + df["close"] * 0.02
    df["bb_lower"] = df["sma_20"] - df["close"] * 0.02
    # Add MACD
    df["macd"] = np.random.randn(n) * 0.5
    df["macd_signal"] = df["macd"] * 0.8
    # Add Donchian
    df["donchian_upper_20"] = df["high"].rolling(20, min_periods=1).max()
    df["donchian_lower_20"] = df["low"].rolling(20, min_periods=1).min()
    # Add VWAP
    df["vwap"] = df["close"] * 0.99 + np.random.rand(n) * 0.5
    # Add momentum/volume signals
    df["sig_momentum"] = np.random.randn(n) * 0.3
    df["sig_volume"] = np.random.rand(n) * 0.5

    # Drop NaN rows
    df = df.dropna().reset_index(drop=False)
    # Ensure close column is float64
    df["close"] = df["close"].astype(float)
    return df


class TestValidationConfig:
    def test_defaults(self):
        config = ValidationConfig()
        assert config.train_months == 18
        assert config.test_months == 6
        assert config.n_mc_simulations == 1000
        assert config.oos_split_pct == 0.15
        assert config.min_oos_sharpe == 0.3
        assert config.run_champion_challenger is True
        assert config.evaluate_model_gate is True

    def test_custom_values(self):
        config = ValidationConfig(
            train_months=12,
            test_months=3,
            n_mc_simulations=500,
            oos_split_pct=0.2,
            min_oos_sharpe=0.5,
            run_champion_challenger=False,
            evaluate_model_gate=False,
        )
        assert config.train_months == 12
        assert config.test_months == 3
        assert config.n_mc_simulations == 500
        assert config.oos_split_pct == 0.2
        assert config.min_oos_sharpe == 0.5
        assert config.run_champion_challenger is False
        assert config.evaluate_model_gate is False


class TestFullValidationResult:
    def test_to_dict_serializes(self):
        result = FullValidationResult(
            ticker="TEST",
            period="2y",
            interval="1d",
            timestamp=time.time(),
            config=ValidationConfig(),
            walk_forward=[],
            monte_carlo=None,
            oos_metrics={},
            is_metrics={},
            overfit_flags=[],
            verdict="PENDING",
        )
        d = result.to_dict()
        assert d["ticker"] == "TEST"
        assert d["verdict"] == "PENDING"
        assert "walk_forward" in d
        assert "monte_carlo" in d

    def test_save_creates_file(self, tmp_path):
        result = FullValidationResult(
            ticker="SAVETEST",
            period="1y",
            interval="1d",
            timestamp=time.time(),
            config=ValidationConfig(save_report=False),
            walk_forward=[],
            monte_carlo=None,
            oos_metrics={"sharpe_ratio": 1.5, "retorno_total": 0.25},
            is_metrics={"sharpe_ratio": 2.0},
            overfit_flags=["No overfitting detected"],
            verdict="APPROVED",
            html_report="<html><body>OK</body></html>",
        )
        saved = result.save(tmp_path / "test_report.json")
        assert saved.exists()
        data = saved.read_text(encoding="utf-8")
        assert "SAVETEST" in data
        assert "APPROVED" in data

        # HTML should also be saved
        html_path = saved.with_suffix(".html")
        assert html_path.exists()
        assert "OK" in html_path.read_text(encoding="utf-8")


class TestFullValidationPipeline:
    def test_run_returns_result(self, sample_df):
        config = ValidationConfig(
            train_months=12,
            test_months=3,
            n_mc_simulations=100,
            min_oos_bars=10,
            run_champion_challenger=False,
            evaluate_model_gate=False,
            save_report=False,
        )
        pipeline = FullValidationPipeline(config=config)

        progress_log = []

        def progress(msg, pct):
            progress_log.append((msg, pct))

        result = pipeline.run(
            df=sample_df,
            ticker="TEST",
            period="2y",
            interval="1d",
        )

        assert isinstance(result, FullValidationResult)
        assert result.ticker == "TEST"
        assert result.verdict in ("APPROVED", "CONDITIONAL", "REJECTED")
        assert len(progress_log) > 0

    def test_run_with_bot_params(self, sample_df):
        config = ValidationConfig(
            train_months=6,
            test_months=2,
            n_mc_simulations=50,
            min_oos_bars=5,
            run_champion_challenger=False,
            evaluate_model_gate=False,
            save_report=False,
        )
        params = StrategyParams(
            buy_score_threshold=0.10,
            stop_loss_pct=-0.05,
            take_profit_pct=0.15,
            max_position_size_pct=0.25,
            use_regime_filter=False,
            use_multi_timeframe=False,
        )
        pipeline = FullValidationPipeline(config=config)
        result = pipeline.run(
            df=sample_df,
            ticker="TEST",
            period="1y",
            interval="1d",
            strategy_params=params,
        )
        assert isinstance(result, FullValidationResult)
        assert result.verdict in ("APPROVED", "CONDITIONAL", "REJECTED")

    def test_median_params_from_empty_windows(self):
        pipeline = FullValidationPipeline()
        base = StrategyParams(buy_score_threshold=0.20, stop_loss_pct=-0.05)
        result = pipeline._median_params_from_wfo([], base)
        assert result.buy_score_threshold == 0.20
        assert result.stop_loss_pct == -0.05

    def test_aggregated_metrics(self):
        from backtesting.validation import WindowResult

        windows = [
            WindowResult(
                window_idx=0,
                train_start="2020-01",
                train_end="2021-06",
                test_start="2021-07",
                test_end="2021-12",
                train_metrics={"sharpe_ratio": 2.0},
                test_metrics={"sharpe_ratio": 1.2},
                best_params={"buy_score_threshold": 0.1},
                sharpe_oos=1.2,
                sharpe_is=2.0,
                overfit_ratio=0.6,
            ),
            WindowResult(
                window_idx=1,
                train_start="2020-07",
                train_end="2021-12",
                test_start="2022-01",
                test_end="2022-06",
                train_metrics={"sharpe_ratio": 1.8},
                test_metrics={"sharpe_ratio": 0.8},
                best_params={"buy_score_threshold": 0.15},
                sharpe_oos=0.8,
                sharpe_is=1.8,
                overfit_ratio=0.44,
            ),
        ]
        pipeline = FullValidationPipeline()
        aggregated = pipeline._calculate_aggregated_metrics(
            windows,
            {"sharpe_ratio": 1.0, "retorno_total": 0.15, "max_drawdown": -0.08},
            None,
        )
        assert "oos_sharpe" in aggregated
        assert "oos_return" in aggregated
        assert "oos_max_dd" in aggregated
        assert "consistency" in aggregated
        # 1 of 2 windows has positive OOS sharpe → consistency = 0.5
        assert aggregated["consistency"] <= 0.5


def test_run_full_validation_convenience(sample_df):
    config = ValidationConfig(
        train_months=6,
        test_months=2,
        n_mc_simulations=50,
        min_oos_bars=5,
        run_champion_challenger=False,
        evaluate_model_gate=False,
        save_report=False,
    )
    result = run_full_validation(
        df=sample_df,
        ticker="CONV",
        period="1y",
        interval="1d",
        config=config,
    )
    assert isinstance(result, FullValidationResult)
    assert result.ticker == "CONV"
    assert result.verdict in ("APPROVED", "CONDITIONAL", "REJECTED")


class TestIntegrationWithBotEngine:
    def test_backtest_and_validate(self, sample_df):
        """Smoke test: run a regular BotBacktestEngine backtest, then validate."""
        from backtesting.bot_engine import BotBacktestEngine

        engine = BotBacktestEngine(
            StrategyParams(
                buy_score_threshold=0.10,
                stop_loss_pct=-0.05,
                take_profit_pct=0.15,
            )
        )
        bt_result = engine.run(sample_df, ticker="INTEGRATION")
        assert bt_result.metrics["total_trades"] >= 0

        # Now validate
        config = ValidationConfig(
            train_months=6,
            test_months=2,
            n_mc_simulations=50,
            min_oos_bars=5,
            run_champion_challenger=False,
            evaluate_model_gate=False,
            save_report=False,
        )
        pipeline = FullValidationPipeline(config=config)
        val_result = pipeline.run(sample_df, ticker="INTEGRATION", period="2y")
        assert val_result.ticker == "INTEGRATION"
        assert val_result.verdict in ("APPROVED", "CONDITIONAL", "REJECTED")
