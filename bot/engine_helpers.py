"""Standalone helpers for the trading engine.

These functions do not depend on TradingBot state and can be tested/imported
without instantiating the full engine.
"""

from __future__ import annotations

from bot.strategy_params import StrategyParams


def sanitize_web_params(params: StrategyParams) -> StrategyParams:
    """Garantiza que el modo web no ejecute estrategias de alto riesgo."""
    return StrategyParams(
        **{
            **params.__dict__,
            "use_neural_brain": False,
            "use_rl_exits": False,
            "use_momentum_scalp": False,
            "use_mean_reversion": False,
            "use_contrarian_dip": False,
            "use_intraday_scalp": False,
            "use_session_filter": False,
            "use_vwap_filter": False,
            "use_donchian_breakout": False,
            "use_ml_filter": False,
        }
    )


def fmt_value(value, suffix: str = "", digits: int = 2) -> str:
    """Format a numeric value for logging."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"
