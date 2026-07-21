"""Defaults unificados para la estrategia web.

Centraliza los parámetros por defecto del modo web para evitar dispersión
entre bot/strategy.py, api/schemas.py y otros consumidores.
"""

from __future__ import annotations

# Defaults del modo web. Mantener sincronizado con create_web_bot_strategy_params().
WEB_STRATEGY_DEFAULTS: dict[str, object] = {
    "buy_score_threshold": 0.15,
    "sell_score_threshold": -0.30,
    "stop_loss_pct": -0.06,
    "take_profit_pct": 0.12,
    "trailing_stop_atr_mult": 2.0,
    "max_position_size_pct": 0.08,
    "min_position_size_pct": 0.05,
    "require_price_above_sma200": False,
    "max_buy_rsi": 70.0,
    "use_short_selling": True,
}
