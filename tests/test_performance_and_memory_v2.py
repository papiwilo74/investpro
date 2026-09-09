"""Pruebas unitarias para las optimizaciones de RAM y Rendimiento al 110%:

1. Downcast a float32 en TechnicalIndicators.add_all.
2. Chandelier Trailing Mode (dejar correr ganancias tras superar el Take-Profit).
3. Modulación Half-Kelly en el dimensionamiento crypto.
4. Verificación de Dockerfile sin Node.js y con MALLOC_ARENA_MAX=2.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from bot.engine import TradingBot
from bot.position_state import PositionState
from bot.strategy import Decision
from bot.strategy_params import StrategyParams
from indicators.technical import TechnicalIndicators


def test_float32_downcasting_in_technical_indicators():
    """Verifica que TechnicalIndicators.add_all devuelva float32 reduciendo 50% de RAM."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 150, 50, dtype=np.float64),
            "high": np.linspace(105, 155, 50, dtype=np.float64),
            "low": np.linspace(95, 145, 50, dtype=np.float64),
            "close": np.linspace(102, 152, 50, dtype=np.float64),
            "volume": np.linspace(1000, 2000, 50, dtype=np.float64),
        },
        index=dates,
    )

    df_res = TechnicalIndicators.add_all(df)

    # Columnas clave deben ser float32
    assert df_res["close"].dtype == np.float32
    assert df_res["rsi"].dtype == np.float32
    assert df_res["atr"].dtype == np.float32
    assert df_res["macd"].dtype == np.float32


def test_chandelier_trailing_stop_lets_winners_run():
    """Verifica que una posición que supera el Take-Profit inicial (+6%) continúe abierta

    si sigue subiendo (hasta +20%) y solo cierre cuando retroceda por debajo del trailing stop.
    """
    params = StrategyParams(
        stop_loss_pct=-0.03,
        take_profit_pct=0.06,  # 6% TP base
        use_trailing_stop=True,
        trailing_stop_atr_mult=2.0,
    )
    entry_price = 100.0
    entry_atr = 2.0  # ATR de $2

    pos = PositionState(entry_price=entry_price, entry_atr=entry_atr, params=params, side="LONG")

    # 1. El precio sube a $108 (+8%, superó el 6% de TP).
    # Con Chandelier Trailing Mode, el stop es max_price - 2*ATR = 108 - 4 = $104.
    # $108 > $104 -> la posición NO debe cerrarse inmediatamente, debe dejar correr la ganancia.
    pos.update_extremes(108.0, current_atr=2.0)
    should_exit, reason = pos.should_exit(108.0)
    assert not should_exit

    # 2. El precio continúa el rally explosivo hasta $120 (+20% de ganancia).
    # Nuevo trailing stop = 120 - 4 = $116.
    pos.update_extremes(120.0, current_atr=2.0)
    should_exit, reason = pos.should_exit(120.0)
    assert not should_exit

    # 3. El precio se gira y cae a $115 (por debajo de $116).
    # Ahora sí debe salir, asegurando una ganancia de +15% (muy superior al +6% rígido).
    pos.update_extremes(115.0, current_atr=2.0)
    should_exit, reason = pos.should_exit(115.0)
    assert should_exit
    assert "chandelier trailing-stop" in reason


@pytest.mark.asyncio
async def test_crypto_kelly_position_sizing():
    """Verifica que el Half-Kelly module el tamaño hacia arriba con buen win rate y hacia abajo con racha perdedora."""
    bot = TradingBot.__new__(TradingBot)
    bot._strategy_params = StrategyParams(
        use_crypto_volatility_parity=True,
        crypto_target_risk_per_trade_pct=0.02,
        crypto_min_position_size_pct=0.05,
        crypto_max_position_size_pct=0.25,
    )
    bot._log = MagicMock()
    bot.crypto_client = MagicMock()
    bot.crypto_client.place_market_order.return_value = {
        "status": "success",
        "filled_avg_price": 100.0,
    }

    equity = 10000.0
    buying_power = 10000.0
    decision = Decision("BUY", "Setup", position_size_pct=0.10)

    # 1. Brain con Kelly alto (racha ganadora: 8 de 10 aciertos)
    bot.brain = MagicMock()
    bot.brain._kelly = MagicMock()
    bot.brain._kelly.trades = [0.05] * 8 + [-0.02] * 2
    bot.brain._kelly.kelly_pct = 0.25  # Kelly 25% -> multiplier > 1.0

    # Activo con ATR 10% -> base sizing = 0.02 / 0.10 = 0.20 (20%)
    # Con Kelly boost -> sizing aumenta
    invested_high_kelly = await bot._execute_crypto_buy(
        "SOL/USD", decision, last_close=100.0, equity=equity, buying_power=buying_power, atr=10.0
    )

    # 2. Brain con Kelly bajo (racha perdedora: 2 aciertos de 10)
    bot.brain._kelly.trades = [0.02] * 2 + [-0.03] * 8
    bot.brain._kelly.kelly_pct = 0.0  # Kelly 0% -> multiplier < 1.0 (0.85x)
    invested_low_kelly = await bot._execute_crypto_buy(
        "SOL/USD", decision, last_close=100.0, equity=equity, buying_power=buying_power, atr=10.0
    )

    # Con racha ganadora se debe asignar más capital que con racha perdedora
    assert invested_high_kelly > invested_low_kelly


def test_dockerfile_optimizations():
    """Verifica que el Dockerfile no incluya Node.js y configure MALLOC_ARENA_MAX=2."""
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists()
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "MALLOC_ARENA_MAX=2" in content
    assert "FROM node:" not in content
    assert "npm ci" not in content
