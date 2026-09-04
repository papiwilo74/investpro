"""Tests unitarios para la Opción A: Enfoque Crypto 85/15 y Protección de Posiciones."""

import pandas as pd
import pytest

from bot.multi_strategy_allocator import MultiStrategyAllocator
from bot.strategy import TradingBrain
from bot.strategy_params import StrategyParams


@pytest.fixture
def sample_crypto_df():
    """Genera un DataFrame sintético representativo de un activo cripto alcista."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    prices = [50000.0 * (1.005**i) for i in range(60)]  # Tendencia alcista
    df = pd.DataFrame(
        {
            "close": prices,
            "open": [p * 0.998 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.995 for p in prices],
            "volume": [1000.0] * 60,
            "atr": [1200.0] * 60,
            "rsi": [65.0] * 60,
            "adx": [28.0] * 60,
            "sma_200": [40000.0] * 60,
            "sig_composite": [0.65] * 60,
        },
        index=dates,
    )
    return df


def test_strategy_params_allocation():
    params = StrategyParams()
    assert params.crypto_portfolio_target_pct == 0.85
    assert params.stock_portfolio_target_pct == 0.15
    assert params.crypto_position_size_mult == 1.75
    assert "BTC/USD" in params.crypto_symbols
    assert "ETH/USD" in params.crypto_symbols
    assert "SOL/USD" in params.crypto_symbols


def test_multi_strategy_allocator_crypto_weighting():
    allocator = MultiStrategyAllocator()
    # Crypto con WR neutral o bueno recibe factor de impulso
    scale_crypto = allocator.get_allocation_scale("MOMENTUM", asset_type="CRYPTO")
    # Stock recibe ponderación acotada al 15%
    scale_stock = allocator.get_allocation_scale("MOMENTUM", asset_type="STOCK")

    assert scale_crypto > scale_stock
    assert scale_crypto >= 1.75  # Boost activo para crypto
    assert scale_stock <= 0.50  # Stock ponderado hacia 15%


def test_trading_brain_canonical_crypto_symbol_matching(sample_crypto_df):
    brain = TradingBrain()
    # Registrar posición con formato de Alpaca: BTCUSD
    brain.on_position_opened("BTCUSD", 64000.0, sample_crypto_df, side="LONG")

    # Consultar decisión usando ticker con formato yfinance: BTC-USD
    decision = brain.decide(
        df=sample_crypto_df,
        score=0.40,
        has_position=True,
        position_pnl_pct=0.25,
        ticker="BTC-USD",
        market_regime="BULL",
        weekly_trend="BULLISH",
    )

    # La posición debe haber sido encontrada y procesada (no 'position valid (no state)')
    assert decision.reason != "position valid (no state)"


def test_trading_brain_hot_reconstruction_and_failsafe_stoploss(sample_crypto_df):
    brain = TradingBrain()
    # Sin posición previa en _positions, pero has_position es True con pérdida severa (ej. PANW -11.9%)
    decision = brain.decide(
        df=sample_crypto_df,
        score=-0.10,
        has_position=True,
        position_pnl_pct=-0.119,  # -11.9%
        ticker="PANW",
        market_regime="NEUTRAL",
        weekly_trend="NEUTRAL",
    )

    # El fail-safe o PositionState en caliente debe ordenar la venta por Stop Loss
    assert decision.action == "SELL"
    assert "stop" in decision.reason.lower()


def test_trading_brain_hot_reconstruction_trailing_profit(sample_crypto_df):
    brain = TradingBrain()
    # Sin posición previa en _positions, pero has_position es True con ganancia de BTC (+25%)
    decision = brain.decide(
        df=sample_crypto_df,
        score=0.50,
        has_position=True,
        position_pnl_pct=0.25,
        ticker="BTC/USD",
        market_regime="BULL",
        weekly_trend="BULLISH",
    )
    # Ahora la posición ya existe en _positions con max_price actualizado
    assert decision is not None
    pos = brain._positions.get("BTC/USD")
    assert pos is not None
    assert pos.entry_price > 0
