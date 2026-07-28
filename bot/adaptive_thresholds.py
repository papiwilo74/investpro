"""Adaptive Threshold Manager — ajuste dinámico de umbrales según el régimen de mercado (VIX y SPY)."""

from __future__ import annotations

import logging

from bot.market_regime import MarketRegime, MarketRegimeFilter
from bot.strategy import StrategyParams

logger = logging.getLogger("inversion_helper.adaptive_thresholds")


class AdaptiveThresholdManager:
    """Ajusta dinámicamente los umbrales de compra, stop loss y take profit en tiempo real."""

    def __init__(self, regime_filter: MarketRegimeFilter | None = None):
        self.regime_filter = regime_filter or MarketRegimeFilter()

    def get_adapted_params(self, base_params: StrategyParams) -> StrategyParams:
        """Adapta StrategyParams según el régimen de mercado actual (SPY + VIX)."""
        try:
            regime_status: MarketRegime = self.regime_filter.get_regime()
        except Exception as e:
            logger.warning("Error al obtener régimen de mercado en AdaptiveThresholdManager: %s", e)
            return base_params

        vix = regime_status.vix_value or 20.0
        vix_level = regime_status.vix_level
        spy_trend = regime_status.spy_trend

        # Crear dict con los valores base
        p_dict = {
            "buy_score_threshold": base_params.buy_score_threshold,
            "stop_loss_pct": base_params.stop_loss_pct,
            "take_profit_pct": base_params.take_profit_pct,
            "trailing_stop_atr_mult": base_params.trailing_stop_atr_mult,
            "max_buy_rsi": base_params.max_buy_rsi,
            "min_ml_buy_probability": base_params.min_ml_buy_probability,
            "use_trailing_stop": base_params.use_trailing_stop,
            "max_position_size_pct": base_params.max_position_size_pct,
            "min_position_size_pct": base_params.min_position_size_pct,
            "atr_position_sizing": base_params.atr_position_sizing,
            "atr_risk_pct": base_params.atr_risk_pct,
            "require_price_above_sma200": base_params.require_price_above_sma200,
            "use_ml_filter": base_params.use_ml_filter,
            "use_donchian_breakout": base_params.use_donchian_breakout,
            "use_momentum_scalp": base_params.use_momentum_scalp,
            "use_mean_reversion": base_params.use_mean_reversion,
            "use_contrarian_dip": base_params.use_contrarian_dip,
            "use_short_selling": base_params.use_short_selling,
            "use_multi_timeframe": base_params.use_multi_timeframe,
            "use_regime_filter": base_params.use_regime_filter,
            "use_earnings_blackout": base_params.use_earnings_blackout,
            "use_rl_exits": base_params.use_rl_exits,
        }

        # 1. Régimen de Alta Volatilidad / Pánico (VIX > 25 o EXTREME)
        if vix_level in ("HIGH", "EXTREME") or vix > 25.0:
            logger.info("Adaptando parámetros para ALTA VOLATILIDAD (VIX: %.1f)", vix)
            p_dict["buy_score_threshold"] = max(0.35, base_params.buy_score_threshold * 1.4)
            p_dict["take_profit_pct"] = min(0.06, base_params.take_profit_pct * 0.5)  # Tomar ganancias rápido
            p_dict["stop_loss_pct"] = max(-0.03, base_params.stop_loss_pct * 0.7)  # Stop loss más estrecho
            p_dict["max_buy_rsi"] = 62.0  # Evitar entradas cerca de techos
            p_dict["min_ml_buy_probability"] = max(0.60, base_params.min_ml_buy_probability)

        # 2. Mercado Alcista Estable (BULL trend + Low VIX < 16)
        elif spy_trend == "BULL" and (vix_level == "LOW" or vix < 16.0):
            logger.info("Adaptando parámetros para TENDENCIA ALCISTA FUERTE (VIX: %.1f)", vix)
            p_dict["buy_score_threshold"] = min(0.15, base_params.buy_score_threshold * 0.8)  # Entrar temprano
            p_dict["take_profit_pct"] = max(0.20, base_params.take_profit_pct * 1.3)  # Dejar correr ganadores
            p_dict["trailing_stop_atr_mult"] = max(3.0, base_params.trailing_stop_atr_mult * 1.2)
            p_dict["max_buy_rsi"] = 78.0  # Permitir impulsos fuertes de momentum

        # 3. Mercado Bajista (BEAR trend)
        elif spy_trend == "BEAR":
            logger.info("Adaptando parámetros para MERCADO BAJISTA (SPY BEAR)")
            p_dict["buy_score_threshold"] = max(0.40, base_params.buy_score_threshold * 1.5)
            p_dict["short_score_threshold"] = -0.20  # Facilitar entradas en Short

        return StrategyParams(**p_dict)
