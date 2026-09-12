"""Pruebas unitarias para las 4 mejoras cuantitativas de Cripto:

1. Universo ampliado 27+ pares y mapeo Yahoo Finance (CRYPTO_YFINANCE_MAP).
2. Multi-Timeframe (MTF) Sniper (1D Macro + 1H Timing con RSI <= 70).
3. Dynamic Volatility Parity Sizing (dimensionamiento inversamente proporcional a ATR%).
5. Crypto Statistical Arbitrage / Pairs Trading (Z-score contra benchmark cointegrado).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.engine import TradingBot
from bot.statistical_arbitrage import PairsTradingEngine
from bot.strategy import Decision
from bot.strategy_params import StrategyParams
from broker.crypto_client import CRYPTO_YFINANCE_MAP, DEFAULT_CRYPTO_WATCHLIST


def test_crypto_watchlist_expansion_and_mapping():
    """Verifica que el universo crypto tenga al menos 25 pares y los contratos especiales estén mapeados."""
    assert len(DEFAULT_CRYPTO_WATCHLIST) >= 25
    assert "BTC/USD" in DEFAULT_CRYPTO_WATCHLIST
    assert "ETH/USD" in DEFAULT_CRYPTO_WATCHLIST
    assert "SOL/USD" in DEFAULT_CRYPTO_WATCHLIST
    assert "UNI/USD" in DEFAULT_CRYPTO_WATCHLIST
    assert "GRT/USD" in DEFAULT_CRYPTO_WATCHLIST
    assert "SUI/USD" in DEFAULT_CRYPTO_WATCHLIST

    # Verificación de mapeo a Yahoo Finance
    assert CRYPTO_YFINANCE_MAP.get("UNI/USD") == "UNI7083-USD"
    assert CRYPTO_YFINANCE_MAP.get("GRT/USD") == "GRT6719-USD"
    assert CRYPTO_YFINANCE_MAP.get("SUI/USD") == "SUI20947-USD"


def test_crypto_pairs_trading_zscore():
    """Verifica que get_crypto_pair_zscore calcule el Z-score en series sintéticas cointegradas."""
    engine = PairsTradingEngine(lookback_window=20)
    np.random.seed(42)
    n = 50
    btc_prices = pd.Series(50000 + np.cumsum(np.random.randn(n) * 200))
    # ETH fuertemente correlacionada pero con un rezago / caída artificial al final
    eth_prices = pd.Series(btc_prices * 0.06)
    eth_prices.iloc[-5:] = eth_prices.iloc[-5:] * 0.90  # ETH se rezaga

    zscore = engine.get_crypto_pair_zscore("ETH/USD", eth_prices, btc_prices)
    assert isinstance(zscore, float)
    # ETH se rezagó, por ende el spread cayó -> Z-score fuertemente negativo
    assert zscore < -1.0


@pytest.mark.asyncio
async def test_crypto_volatility_parity_sizing():
    """Verifica que la paridad de volatilidad asigne más tamaño a activos calmos (BTC) que a volátiles (SHIB)."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        use_crypto_volatility_parity=True,
        crypto_target_risk_per_trade_pct=0.02,  # 2% riesgo
        crypto_min_position_size_pct=0.05,
        crypto_max_position_size_pct=0.25,
    )
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.place_market_order.return_value = {
        "status": "success",
        "filled_avg_price": 60000.0,
    }

    equity = 10000.0
    buying_power = 10000.0
    decision = Decision("BUY", "Strong trend", position_size_pct=0.15)

    # 1. Activo de baja volatilidad (BTC): ATR% = 1200 / 60000 = 2.0%
    # Sizing = 0.02 / 0.02 = 1.0 -> capeado a max_size = 25% ($2,500)
    invested_btc = await bot._execute_crypto_buy(
        "BTC/USD", decision, last_close=60000.0, equity=equity, buying_power=buying_power, atr=1200.0
    )
    assert invested_btc > 0
    # Cantidad comprada con $2500 a $60000 = 0.041667 BTC
    assert np.isclose(invested_btc, 2500.0, atol=10.0)

    # 2. Activo de alta volatilidad (Meme coin): ATR% = 0.20 / 1.0 = 20.0%
    # Sizing = 0.02 / 0.20 = 0.10 (10%) ($1,000)
    bot.crypto_client.place_market_order.return_value = {
        "status": "success",
        "filled_avg_price": 1.0,
    }
    invested_alt = await bot._execute_crypto_buy(
        "DOGE/USD", decision, last_close=1.0, equity=equity, buying_power=buying_power, atr=0.20
    )
    assert invested_alt > 0
    assert np.isclose(invested_alt, 1000.0, atol=10.0)

    # 3. Si se desactiva volatility parity, usa el sizing base fijo
    bot._strategy_params = StrategyParams(
        use_crypto_volatility_parity=False,
        crypto_position_size_mult=1.75,
        crypto_max_position_size_pct=0.30,
    )
    # base_pct 0.15 * mult 1.75 = 26.25% de equity ($2,625)
    invested_legacy = await bot._execute_crypto_buy(
        "DOGE/USD", decision, last_close=1.0, equity=equity, buying_power=buying_power, atr=0.20
    )
    assert np.isclose(invested_legacy, 2625.0, atol=10.0)


@pytest.mark.asyncio
async def test_crypto_mtf_sniper_blocks_overbought_1h():
    """Verifica que el MTF Sniper posponga compras cuando 1D es BUY pero 1H RSI > 70."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        use_crypto_mtf_sniper=True,
        crypto_sniper_max_rsi_1h=70.0,
        crypto_symbols=["BTC/USD"],
        use_stop_loss_cooldown=False,
    )
    bot.is_running = True
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.get_account_summary.return_value = {
        "equity": 10000.0,
        "buying_power": 10000.0,
    }
    bot.crypto_client.get_positions.return_value = []
    bot._execute_crypto_buy = AsyncMock(return_value=1000.0)

    # Mock DataFetcher
    bot.fetcher = MagicMock()
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df_1d = pd.DataFrame(
        {
            "open": np.linspace(50000, 60000, 30),
            "high": np.linspace(50500, 60500, 30),
            "low": np.linspace(49500, 59500, 30),
            "close": np.linspace(50200, 60200, 30),
            "volume": [1000.0] * 30,
        },
        index=dates,
    )

    # 1H con RSI sobrecomprado (>70)
    dates_1h = pd.date_range("2026-01-20", periods=20, freq="h")
    df_1h_overbought = pd.DataFrame(
        {
            "open": np.linspace(59000, 61000, 20),
            "high": np.linspace(59500, 61500, 20),
            "low": np.linspace(58500, 60500, 20),
            "close": np.linspace(59200, 61200, 20),
            "volume": [100.0] * 20,
            "rsi": [80.0] * 20,  # Sobrecompra en 1H
        },
        index=dates_1h,
    )

    bot.fetcher.get_data.side_effect = lambda ticker, period, interval: (
        df_1h_overbought.copy() if interval == "1h" else df_1d.copy()
    )

    bot._get_ml_prediction = MagicMock(return_value=("BUY", 0.85))
    bot.brain = MagicMock()
    bot.brain.decide.return_value = Decision("BUY", "1D Trend Bullish", confidence=0.85)

    with (
        patch("indicators.technical.TechnicalIndicators.add_all", side_effect=lambda df, **k: df),
        patch("indicators.signals.SignalGenerator.add_signal_columns", side_effect=lambda df: df),
        patch("indicators.signals.SignalGenerator.composite_score", return_value=0.85),
    ):
        await bot._scan_and_trade_crypto(interval="1d")

    # La compra debe ser bloqueada / pospuesta por el sniper
    bot._execute_crypto_buy.assert_not_called()

    # Ahora cambiamos el 1H RSI a 55.0 (sin sobrecompra)
    df_1h_healthy = df_1h_overbought.copy()
    df_1h_healthy["rsi"] = [55.0] * 20
    bot.fetcher.get_data.side_effect = lambda ticker, period, interval: (
        df_1h_healthy.copy() if interval == "1h" else df_1d.copy()
    )

    with (
        patch("indicators.technical.TechnicalIndicators.add_all", side_effect=lambda df, **k: df),
        patch("indicators.signals.SignalGenerator.add_signal_columns", side_effect=lambda df: df),
        patch("indicators.signals.SignalGenerator.composite_score", return_value=0.85),
    ):
        await bot._scan_and_trade_crypto(interval="1d")

    # Ahora sí debe haber llamado a _execute_crypto_buy
    bot._execute_crypto_buy.assert_called_once()


@pytest.mark.asyncio
async def test_crypto_pairs_arbitrage_boost_in_scanner():
    """Verifica que el scanner aplique el boost (+0.08) cuando el Z-score indica infravaloración."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        use_crypto_pairs_arbitrage=True,
        crypto_pairs_zscore_entry=-1.75,
        crypto_pairs_score_boost=0.08,
        crypto_symbols=["ETH/USD"],
        use_crypto_mtf_sniper=False,  # Desactivado para aislar el test de pairs boost
        use_stop_loss_cooldown=False,
    )
    bot.is_running = True
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.get_account_summary.return_value = {
        "equity": 10000.0,
        "buying_power": 10000.0,
    }
    bot.crypto_client.get_positions.return_value = []
    bot._execute_crypto_buy = AsyncMock(return_value=1000.0)

    # Pairs engine mock
    bot.pairs_engine = MagicMock()
    bot.pairs_engine.DEFAULT_CRYPTO_PAIRS = [("ETH/USD", "BTC/USD")]
    bot.pairs_engine.crypto_pairs = [("ETH/USD", "BTC/USD")]
    # Retorna Z-score de -2.20 (<= -1.75 -> infravalorado)
    bot.pairs_engine.get_crypto_pair_zscore.return_value = -2.20

    # Mock DataFetcher
    bot.fetcher = MagicMock()
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    df_eth = pd.DataFrame(
        {
            "open": np.linspace(2500, 2700, 30),
            "high": np.linspace(2550, 2750, 30),
            "low": np.linspace(2450, 2650, 30),
            "close": np.linspace(2520, 2720, 30),
            "volume": [5000.0] * 30,
            "atr": [50.0] * 30,
        },
        index=dates,
    )
    df_btc = pd.DataFrame(
        {
            "open": np.linspace(50000, 60000, 30),
            "high": np.linspace(50500, 60500, 30),
            "low": np.linspace(49500, 59500, 30),
            "close": np.linspace(50200, 60200, 30),
            "volume": [1000.0] * 30,
            "atr": [1000.0] * 30,
        },
        index=dates,
    )

    bot.fetcher.get_data.side_effect = lambda ticker, period, interval: (
        df_btc.copy() if "BTC" in ticker else df_eth.copy()
    )

    bot._get_ml_prediction = MagicMock(return_value=("BUY", 0.85))
    bot.brain = MagicMock()
    bot.brain.decide.return_value = Decision("BUY", "Good score", confidence=0.85)

    base_score = 0.50
    with (
        patch("indicators.technical.TechnicalIndicators.add_all", side_effect=lambda df, **k: df),
        patch("indicators.signals.SignalGenerator.add_signal_columns", side_effect=lambda df: df),
        patch("indicators.signals.SignalGenerator.composite_score", return_value=base_score),
    ):
        await bot._scan_and_trade_crypto(interval="1d")

    # Verificar que TradingBrain.decide recibió el score con el boost de +0.08 (0.50 + 0.08 = 0.58)
    decide_call_args = bot.brain.decide.call_args
    passed_score = decide_call_args.kwargs.get("score")
    assert np.isclose(passed_score, 0.58, atol=0.001)
