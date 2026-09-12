from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.engine import TradingBot
from bot.strategy import Decision, StrategyParams, TradingBrain


def test_adaptive_sltp_negative_clamp_not_stuck():
    """Verifica que el clamp de _adaptive_sltp respete los límites negativos y no quede trabado en -2.0%."""
    params = StrategyParams(
        use_adaptive_sltp=True,
        adaptive_sltp_min_stop_pct=-0.04,
        adaptive_sltp_max_stop_pct=-0.08,
    )
    brain = TradingBrain(params)

    # DataFrame con baja volatilidad (debería dar el stop más ajustado: -0.04)
    df_low = pd.DataFrame(
        {
            "close": [100.0] * 30,
            "atr": [0.5] * 30,
        }
    )
    sl_low, _ = brain._adaptive_sltp(df_low, 29)
    assert np.isclose(sl_low, -0.04), f"Expected -0.04 but got {sl_low}"

    # DataFrame con alta volatilidad (debería dar el stop más holgado: -0.08)
    df_high = pd.DataFrame(
        {
            "close": [100.0] * 30,
            "atr": [10.0] * 30,
        }
    )
    sl_high, _ = brain._adaptive_sltp(df_high, 29)
    assert np.isclose(sl_high, -0.08), f"Expected -0.08 but got {sl_high}"


@pytest.mark.asyncio
async def test_crypto_daily_loss_circuit_breaker_blocks_buys():
    """Verifica que si la pérdida diaria alcanza o supera el 2%, se congelen las compras en crypto."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        crypto_daily_max_loss_pct=0.02,
        crypto_symbols=["ETH/USD"],
        use_stop_loss_cooldown=False,
    )
    bot.is_running = True
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    # Pérdida diaria de -2.5% hoy
    bot.crypto_client.get_account_summary.return_value = {
        "equity": 97500.0,
        "last_equity": 100000.0,
        "pnl_pct_today": -0.025,
        "buying_power": 50000.0,
    }
    bot.crypto_client.get_positions.return_value = []
    bot._execute_crypto_buy = AsyncMock(return_value=1000.0)
    bot.fetcher = MagicMock()

    await bot._scan_and_trade_crypto(interval="1d")

    # Ninguna compra debe haber sido ejecutada
    bot._execute_crypto_buy.assert_not_called()
    log_messages = [call[0][0] for call in bot._log.call_args_list]
    assert any("CRYPTO RISK GATE" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_crypto_btc_macro_shield_blocks_altcoins_when_btc_dumping():
    """Verifica que si BTC tiene score fuertemente bajista, se bloqueen compras de altcoins."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        use_crypto_btc_macro_filter=True,
        crypto_btc_min_score=-0.10,
        crypto_symbols=["AVAX/USD"],
        use_stop_loss_cooldown=False,
    )
    bot.is_running = True
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.get_account_summary.return_value = {
        "equity": 100000.0,
        "pnl_pct_today": 0.0,
        "buying_power": 50000.0,
    }
    bot.crypto_client.get_positions.return_value = []
    bot._execute_crypto_buy = AsyncMock(return_value=1000.0)

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df_sample = pd.DataFrame(
        {
            "open": [10.0] * 30,
            "high": [11.0] * 30,
            "low": [9.0] * 30,
            "close": [10.5] * 30,
            "volume": [1000.0] * 30,
            "atr": [0.5] * 30,
        },
        index=dates,
    )
    bot.fetcher = MagicMock()
    bot.fetcher.get_data.return_value = df_sample.copy()
    bot.brain = MagicMock()
    bot.brain.decide.return_value = Decision("BUY", "Setup", confidence=0.8)
    bot._get_ml_prediction = MagicMock(return_value=("BUY", 0.8))

    with (
        patch("indicators.technical.TechnicalIndicators.add_all", side_effect=lambda df, **k: df),
        patch("indicators.signals.SignalGenerator.add_signal_columns", side_effect=lambda df: df),
        patch("indicators.signals.SignalGenerator.composite_score", return_value=-0.35),
    ):
        await bot._scan_and_trade_crypto(interval="1d")

    bot._execute_crypto_buy.assert_not_called()
    log_messages = [call[0][0] for call in bot._log.call_args_list]
    assert any("CRYPTO BTC SHIELD" in msg for msg in log_messages)


@pytest.mark.asyncio
async def test_crypto_loss_exit_triggers_4h_cooldown_and_risk_manager():
    """Verifica que cualquier venta con PnL negativo active cooldown de 4h y reporte al RiskManager."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(stop_loss_cooldown_seconds=14400)
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.place_market_order.return_value = {"status": "success"}

    bot.risk_manager = MagicMock()
    bot._stop_loss_exit_timestamps = {}

    position = {
        "symbol": "AVAX/USD",
        "qty": 100.0,
        "current_price": 7.33,
        "cost_basis": 760.0,
        "market_value": 733.0,
    }
    decision = Decision("SELL", "score bearish (-0.40)")

    await bot._execute_crypto_sell("AVAX/USD", decision, position, equity=100000.0, pnl_pct=-0.0355)

    assert "AVAXUSD" in bot._stop_loss_exit_timestamps
    assert time.time() - bot._stop_loss_exit_timestamps["AVAXUSD"] < 2.0
    bot.risk_manager.record_trade.assert_called_once_with("AVAX/USD", "SELL", -0.0355, -27.0)
