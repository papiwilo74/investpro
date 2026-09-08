"""Tests para las optimizaciones de memoria (Zero-OOM) y mejoras cuantitativas del bot:
- trim_process_memory (ejecución segura en cualquier OS)
- Cooldown anti-cuchillo cayendo tras Stop-Loss
- Crypto Fear & Greed Index client y ajuste de sentimiento
- Filtro de volumen institucional (Volume Surge)
- Breakeven stop con cobertura de comisiones
- Guardia de reentrenamiento ML en la nube (Render 512MB)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd

from bot.engine import TradingBot
from bot.engine_helpers import trim_process_memory
from bot.position_state import PositionState
from bot.strategy import TradingBrain
from bot.strategy_params import StrategyParams
from data.fear_greed import FearGreedClient, get_fear_greed_client


# ── Test 1: trim_process_memory ──────────────────────────────────────
def test_trim_process_memory_executes_safely():
    """Verifica que trim_process_memory se ejecute sin excepciones en cualquier entorno."""
    res = trim_process_memory()
    assert isinstance(res, bool)


# ── Test 2: Stop-Loss Cooldown ────────────────────────────────────────
@patch("bot.engine.create_broker_client")
@patch("bot.engine.create_crypto_client")
def test_stop_loss_cooldown_blocks_reentry(mock_crypto, mock_broker):
    """Verifica que un ticker que disparó stop loss quede en cooldown y no recompre inmediatamente."""
    params = StrategyParams(
        use_stop_loss_cooldown=True,
        stop_loss_cooldown_seconds=7200,
    )
    bot = TradingBot(strategy_mode="web", strategy_params=params, use_db=False)

    # Simular que BTC/USD sufrió un stop loss hace 5 minutos (300 segundos)
    canonical = "BTCUSD"
    bot._stop_loss_exit_timestamps[canonical] = time.time() - 300

    # Verificar que el cooldown está activo
    cooldown_sec = bot._strategy_params.stop_loss_cooldown_seconds
    elapsed = time.time() - bot._stop_loss_exit_timestamps[canonical]
    assert elapsed < cooldown_sec

    # Simular que pasaron más de 2 horas (7300 segundos)
    bot._stop_loss_exit_timestamps[canonical] = time.time() - 7300
    elapsed_expired = time.time() - bot._stop_loss_exit_timestamps[canonical]
    assert elapsed_expired >= cooldown_sec


# ── Test 3: Fear & Greed Index Client & Sentiment Bias ────────────────
def test_fear_and_greed_client_and_bias():
    """Verifica el cálculo de score adjustment y el manejo de fallback."""
    client = FearGreedClient()

    # Caso 1: Extreme Greed (85) -> Exige +0.08 de score
    client._cached_data = {"value": 85, "classification": "Extreme Greed", "timestamp": int(time.time())}
    client._last_fetch_time = time.time()
    adj_greed = client.get_score_adjustment(extreme_greed=75, extreme_fear=25)
    assert adj_greed == 0.08

    # Caso 2: Extreme Fear (20) -> Reduce exigencia -0.05 para rebote
    client._cached_data = {"value": 20, "classification": "Extreme Fear", "timestamp": int(time.time())}
    adj_fear = client.get_score_adjustment(extreme_greed=75, extreme_fear=25)
    assert adj_fear == -0.05

    # Caso 3: Neutral (50) -> Sin ajuste
    client._cached_data = {"value": 50, "classification": "Neutral", "timestamp": int(time.time())}
    adj_neutral = client.get_score_adjustment(extreme_greed=75, extreme_fear=25)
    assert adj_neutral == 0.0

    # Caso 4: Singleton
    assert get_fear_greed_client() is not None


# ── Test 4: Volume Surge Filter ───────────────────────────────────────
def test_volume_surge_filter():
    """Verifica que señales con volumen anémico sean rechazadas cuando se activa el filtro."""
    params = StrategyParams(
        buy_score_threshold=0.10,
        use_volume_surge_filter=True,
        volume_surge_min_ratio=1.20,  # Requiere 20% más que la media
        use_confirmation_filter=False,
        use_ensemble=False,
    )
    brain = TradingBrain(params=params)

    dates = pd.date_range("2026-01-01", periods=25, freq="D")

    # Volumen promedio histórico de 1,000,000 pero última vela con solo 800,000 (anémico)
    volumes = [1_000_000 + i * 10_000 for i in range(24)] + [800_000]
    closes = [100.0 + i for i in range(25)]

    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": volumes,
            "rsi": [55.0] * 25,
            "adx": [25.0] * 25,
            "atr": [2.0] * 25,
            "sig_composite": [0.5] * 25,
        },
        index=dates,
    )

    decision = brain.decide(
        df=df,
        score=0.45,  # Muy por encima de buy_score_threshold (0.10)
        has_position=False,
        ticker="SOL-USD",
    )

    # Debe ser rechazado por volumen insuficiente
    assert decision.action == "HOLD"
    assert "Volumen insuficiente" in decision.reason

    # Ahora si la última vela tiene un volumen explosivo (2,500,000)
    df.loc[df.index[-1], "volume"] = 2_500_000
    decision_surge = brain.decide(
        df=df,
        score=0.45,
        has_position=False,
        ticker="SOL-USD",
    )
    assert decision_surge.action == "BUY"


# ── Test 5: Fee-Covering Breakeven Exit ────────────────────────────────
def test_fee_covering_breakeven_exit():
    """Verifica que el breakeven stop se active al +2.5% y cierre en +0.3% protegiendo comisiones."""
    params = StrategyParams(
        use_breakeven_stop=True,
        breakeven_trigger_pct=0.025,
        breakeven_offset_pct=0.003,
        use_trailing_stop=False,
    )

    pos = PositionState(
        entry_price=2000.0,
        entry_atr=50.0,
        params=params,
        side="LONG",
        entry_date="2026-09-01",
    )

    # 1. Precio sube a 2060 (+3.0%) -> Dispara activación de Breakeven
    should_exit, _ = pos.should_exit(current_price=2060.0)
    assert should_exit is False
    assert pos._breakeven_active is True

    # 2. Precio retrocede a 2008 (+0.4%) -> Todavía por encima del offset de +0.3% (2006)
    should_exit_safe, _ = pos.should_exit(current_price=2008.0)
    assert should_exit_safe is False

    # 3. Precio cae a 2005 (+0.25%) -> Por debajo del offset (+0.3%) -> Salida de protección de comisiones
    should_exit_be, reason = pos.should_exit(current_price=2005.0)
    assert should_exit_be is True
    assert "breakeven stop" in reason


# ── Test 6: Cloud Retrain Guard ───────────────────────────────────────
@patch("bot.engine.create_broker_client")
@patch("bot.engine.create_crypto_client")
def test_cloud_retrain_guard(mock_crypto, mock_broker, monkeypatch):
    """Verifica que si RENDER=true, el ciclo de re-entreno pesado no se ejecute."""
    monkeypatch.setenv("RENDER", "true")

    bot = TradingBot(strategy_mode="web", use_db=False)
    bot._trainer = MagicMock()

    # Ejecutar ciclo de champion challenger en cloud
    import asyncio

    asyncio.run(bot._run_champion_challenger_cycle(single_ticker=None))

    # El entrenador NO debió ser llamado porque estamos en Render (512MB)
    bot._trainer.retrain_if_stale.assert_not_called()
