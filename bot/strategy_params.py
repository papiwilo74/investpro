"""Strategy parameters dataclass.

Centraliza la definición de parámetros de estrategia para evitar
importar todo el módulo de estrategia desde componentes pequeños.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    """Parámetros inmutables que controlan el comportamiento del motor de decisiones."""

    # ── Distribución de Portafolio (70% Crypto / 30% Acciones) ───────
    crypto_portfolio_target_pct: float = 0.70
    stock_portfolio_target_pct: float = 0.30
    crypto_position_size_mult: float = 1.50

    # ── Optimización para Render (512 MB RAM) & Neon DB ────────────
    render_low_memory_mode: bool = True
    sequential_ticker_processing: bool = True
    max_memory_history_days: int = 90  # Acota el historial cargado en RAM a 90 días

    # Lista de las 20 principales criptomonedas para análisis secuencial
    crypto_symbols: tuple[str, ...] = (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "AVAX/USD",
        "NEAR/USD",
        "ADA/USD",
        "LINK/USD",
        "DOT/USD",
        "DOGE/USD",
        "XRP/USD",
        "BNB/USD",
        "LTC/USD",
        "SUI/USD",
        "FET/USD",
        "INJ/USD",
        "MATIC/USD",
        "ATOM/USD",
        "APT/USD",
        "ARB/USD",
        "OP/USD",
    )

    # ── LONG (compra en tendencia alcista) ─────────────────────────
    buy_score_threshold: float = 0.10
    sell_score_threshold: float = -0.50
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.15
    trailing_stop_atr_mult: float = 2.5
    use_trailing_stop: bool = True
    max_position_size_pct: float = 0.25
    min_position_size_pct: float = 0.15
    atr_position_sizing: bool = True
    atr_risk_pct: float = 0.02
    min_ml_buy_probability: float = 0.52
    require_price_above_sma200: bool = False
    max_buy_rsi: float = 75.0
    use_ml_filter: bool = False
    use_donchian_breakout: bool = True

    # ── Momentum Scalping ──────────────────────────────────────────
    use_momentum_scalp: bool = True
    scalp_momentum_min: float = 0.40
    scalp_volume_min: float = 1.3
    scalp_stop_loss_pct: float = -0.03
    scalp_take_profit_pct: float = 0.04
    scalp_position_size_pct: float = 0.08

    # ── Mean Reversion ─────────────────────────────────────────────
    use_mean_reversion: bool = True
    mean_rev_rsi_max: float = 28.0
    mean_rev_drop_pct: float = -0.02
    mean_rev_stop_loss_pct: float = -0.02
    mean_rev_take_profit_pct: float = 0.03
    mean_rev_position_size_pct: float = 0.05

    # ── Buy the Dip ────────────────────────────────────────────────
    use_contrarian_dip: bool = True
    dip_drop_pct: float = -0.04
    dip_drop_days: int = 3
    dip_rsi_max: float = 35.0
    dip_position_size_pct: float = 0.12

    # ── Short Selling ──────────────────────────────────────────────
    use_short_selling: bool = True
    short_score_threshold: float = -0.25
    short_min_rsi: float = 55.0
    short_stop_loss_pct: float = 0.020
    short_take_profit_pct: float = -0.030
    short_position_size_pct: float = 0.10
    short_momentum_threshold: float = -0.30
    short_min_adx: float = 18.0

    # ── Filtros y mejoras de win rate ──────────────────────────────
    confirm_candle: bool = False
    confirm_candle_days: int = 2
    use_multi_timeframe: bool = True
    use_regime_filter: bool = True
    use_earnings_blackout: bool = True
    earnings_blackout_days: int = 5
    auto_retrain_days: int = 30
    use_rl_exits: bool = True

    # ── Partial Take Profit ────────────────────────────────────────
    use_partial_take_profit: bool = True
    partial_tp1_pct: float = 0.05
    partial_tp1_fraction: float = 0.33
    partial_tp2_pct: float = 0.10
    partial_tp2_fraction: float = 0.33

    # ── Dynamic trailing stop ──────────────────────────────────────
    use_dynamic_trailing: bool = True
    trail_atr_base: float = 3.0
    trail_atr_tight: float = 1.5

    # ── Confirmación de velas ──────────────────────────────────────
    use_confirmation_filter: bool = True
    confirmation_bars: int = 10
    confirmation_min_ratio: float = 0.6

    # Suavizado de señal
    signal_smoothing_periods: int = 3

    # ADX mínimo para operar
    min_adx_to_trade: float = 15.0

    # ── Intraday / Scalping ────────────────────────────────────────
    use_intraday_scalp: bool = False
    intraday_scalp_momentum_min: float = 0.60
    intraday_scalp_volume_min: float = 1.5
    intraday_scalp_stop_loss_pct: float = -0.015
    intraday_scalp_take_profit_pct: float = 0.025
    intraday_scalp_position_size_pct: float = 0.12
    intraday_max_hold_minutes: int = 90

    # Filtro de sesión
    use_session_filter: bool = True
    session_start_hour: int = 9
    session_start_minute: int = 30
    session_end_hour: int = 16
    session_end_minute: int = 0

    # VWAP
    use_vwap_filter: bool = True
    vwap_deviation_pct: float = 0.005

    # Régimen
    disable_scalp_in_bear: bool = True
    disable_meanrev_in_trend: bool = True

    # ── Neural Brain ───────────────────────────────────────────────
    use_neural_brain: bool = False
    neural_brain_min_confidence: float = 0.35

    # ── Adaptive SL/TP (Ajustado para volatilidad Crypto) ─────────
    use_adaptive_sltp: bool = True
    adaptive_sltp_atr_mult_stop: float = 2.0
    adaptive_sltp_atr_mult_stop_bear: float = 1.5
    adaptive_sltp_atr_mult_tp: float = 3.5
    adaptive_sltp_vol_lookback: int = 20
    adaptive_sltp_min_stop_pct: float = -0.02
    adaptive_sltp_max_stop_pct: float = -0.12
    adaptive_sltp_min_tp_pct: float = 0.03
    adaptive_sltp_max_tp_pct: float = 0.45

    # ── Time-based exit ────────────────────────────────────────────
    use_time_based_exit: bool = True
    max_hold_days: int = 20

    # Breakeven stop
    use_breakeven_stop: bool = True
    breakeven_trigger_pct: float = 0.03

    # Volatility targeting
    use_volatility_targeting: bool = True
    target_annual_volatility: float = 0.15

    # Ensemble
    use_ensemble: bool = True

    # Cautious regime boost
    cautious_regime_score_boost: float = 0.15
