"""Decision engine shared by live trading, bot backtests, and optimization.

Estrategias implementadas:
- LONG:  Compra cuando señales técnicas/ML son alcistas.
- DIP:   Compra cuando hay una caída fuerte + RSI sobrevendido (Buy the Dip).
- SHORT: Vende en corto cuando señales son fuertemente bajistas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from ml.rl import RLExitAgent
from ml.vision import VisualAnalyzer

# Global RL Agent para no recargarlo en cada paso
rl_agent = RLExitAgent()
@dataclass(frozen=True)
class StrategyParams:
    # ── LONG (compra en tendencia alcista) ─────────────────────────
    buy_score_threshold: float = 0.20        # Óptimo encontrado por grid search
    sell_score_threshold: float = -0.50      # Vender LONG si score cae aquí
    stop_loss_pct: float = -0.05             # Stop loss general: -5.0% (dejar respirar)
    take_profit_pct: float = 0.15            # Take profit: +15.0% (capturar grandes subidas)
    trailing_stop_atr_mult: float = 2.5      # Trailing stop basado en ATR
    use_trailing_stop: bool = True
    max_position_size_pct: float = 0.25      # Apuestas más fuertes
    min_position_size_pct: float = 0.20      # Asignación mínima alta
    atr_position_sizing: bool = True
    atr_risk_pct: float = 0.02
    min_ml_buy_probability: float = 0.52     # Restaurado al óptimo
    require_price_above_sma200: bool = False
    max_buy_rsi: float = 72.0                # Restaurado para permitir momentum
    use_ml_filter: bool = True
    use_donchian_breakout: bool = True       # Solo comprar si rompe máximos de 20 días

    # ── BUY THE DIP (compra caídas fuertes en sobrevendido) ─────────
    use_contrarian_dip: bool = True          # Activar estrategia Buy the Dip
    dip_drop_pct: float = -0.04             # Caída mínima en N días para activar (reducida a 4% para scalping)
    dip_drop_days: int = 3                  # Ventana de días (más rápida)
    dip_rsi_max: float = 35.0               # RSI máximo para considerar sobreventa
    dip_position_size_pct: float = 0.12     # Tamaño posición en Dip (12% del capital)

    # ── SHORT SELLING (vende en corto para ganar cuando cae) ─────────
    use_short_selling: bool = True           # Activar short selling
    short_score_threshold: float = -0.45    # Score mínimo para abrir un short
    short_min_rsi: float = 68.0             # RSI mínimo (sobrecompra) para short
    short_stop_loss_pct: float = 0.015      # Stop loss short (sube 1.5% → salir)
    short_take_profit_pct: float = -0.025   # Take profit short (baja 2.5% → salir)
    short_position_size_pct: float = 0.10   # Tamaño posición short (10% del capital)

    # ── MEJORAS DE WIN RATE (Ajustado para Scalping) ─────────────────
    # 1. Confirmación por vela: APAGADO para entrar el mismo día
    confirm_candle: bool = False
    confirm_candle_days: int = 2
    # 2. Multi-temporalidad: APAGADO porque en scalping no importa la tendencia semanal
    use_multi_timeframe: bool = False
    # 3. Filtro de régimen: PRENDIDO para no ir contra pánico general
    use_regime_filter: bool = True
    # 4. Blackout de earnings: PRENDIDO (muy peligroso holdear cerca de earnings)
    use_earnings_blackout: bool = True
    earnings_blackout_days: int = 5
    # 5. Re-entrenamiento automático (manejado en daemon, no aquí)
    auto_retrain_days: int = 30
    # 6. Salidas dinámicas por Reinforcement Learning
    use_rl_exits: bool = True


@dataclass(frozen=True)
class Decision:
    action: str                              # BUY / SELL / HOLD / SHORT / COVER
    reason: str
    confidence: float = 0.0
    position_size_pct: float = 0.0
    side: str = "LONG"                       # LONG, DIP, SHORT


class PositionState:
    """Guarda estado de una posición abierta para trailing stop."""

    def __init__(self, entry_price: float, entry_atr: float, params: StrategyParams, side: str = "LONG"):
        self.entry_price = entry_price
        self.entry_atr = entry_atr
        self.max_price = entry_price
        self.min_price = entry_price   # Para shorts: seguimos el precio mínimo
        self.params = params
        self.side = side  # LONG, DIP, SHORT

    def update_extremes(self, current_price: float, current_atr: float | None = None) -> None:
        if current_price > self.max_price:
            self.max_price = current_price
        if current_price < self.min_price:
            self.min_price = current_price
        if current_atr is not None and current_atr > 0:
            self.entry_atr = current_atr

    def should_exit(self, current_price: float, rsi: float = 50.0, regime: str = "BULL") -> tuple[bool, str]:
        """Retorna (should_exit, reason)."""
        p = self.params

        if self.side in ("LONG", "DIP"):
            pnl_pct = (current_price / self.entry_price) - 1.0

            if p.use_rl_exits:
                action = rl_agent.get_action(pnl_pct, rsi, regime, is_training=False)
                if action == 1:
                    return True, f"RL agent chose CLOSE (PnL: {pnl_pct:.2%})"
                # Si el agente dice HOLD, ignoramos TP/SL excepto stop-loss extremo como red de seguridad
                if pnl_pct <= (p.stop_loss_pct * 1.5):
                    return True, f"stop-loss extremo de seguridad ({pnl_pct:.2%})"
                
                # Aún evaluamos Trailing Stop como seguridad
                if p.use_trailing_stop and self.entry_atr > 0:
                    trailing_dist = p.trailing_stop_atr_mult * self.entry_atr
                    trailing_stop = self.max_price - trailing_dist
                    if current_price <= trailing_stop:
                        return True, f"trailing-stop de seguridad ({current_price:.2f} <= {trailing_stop:.2f})"
                
                return False, ""

            # Logica Clasica sin RL
            if pnl_pct <= p.stop_loss_pct:
                return True, f"stop-loss ({pnl_pct:.2%})"

            if pnl_pct >= p.take_profit_pct:
                return True, f"take-profit ({pnl_pct:.2%})"

            # Trailing stop dinámico
            if p.use_trailing_stop and self.entry_atr > 0:
                trailing_dist = p.trailing_stop_atr_mult * self.entry_atr
                trailing_stop = self.max_price - trailing_dist
                if current_price <= trailing_stop:
                    return True, f"trailing-stop ({current_price:.2f} <= {trailing_stop:.2f}, ATR={self.entry_atr:.2f})"

        elif self.side == "SHORT":
            # Para short: ganamos si el precio baja
            pnl_pct = (self.entry_price / current_price) - 1.0

            if p.use_rl_exits:
                action = rl_agent.get_action(pnl_pct, rsi, regime, is_training=False)
                if action == 1:
                    return True, f"RL agent chose CLOSE SHORT (PnL: {pnl_pct:.2%})"
                if pnl_pct <= -(p.short_stop_loss_pct * 1.5): # Stop safety
                    return True, f"short stop-loss extremo ({pnl_pct:.2%})"
                return False, ""

            # Stop loss del short: el precio sube más del umbral
            price_change = (current_price / self.entry_price) - 1.0
            if price_change >= p.short_stop_loss_pct:
                return True, f"short stop-loss (subió {price_change:.2%})"

            # Take profit del short: el precio bajó suficiente
            if price_change <= p.short_take_profit_pct:
                return True, f"short take-profit (bajó {abs(price_change):.2%})"

            # Trailing stop del short: si rebota desde el mínimo más de 2x ATR, salir
            if p.use_trailing_stop and self.entry_atr > 0:
                trailing_dist = p.trailing_stop_atr_mult * self.entry_atr
                trailing_stop = self.min_price + trailing_dist
                if current_price >= trailing_stop:
                    return True, f"short trailing-stop ({current_price:.2f} >= {trailing_stop:.2f})"

        return False, ""

    def current_pnl_pct(self, current_price: float) -> float:
        if self.side == "SHORT":
            return (self.entry_price / current_price) - 1.0
        return (current_price / self.entry_price) - 1.0


class TradingBrain:
    """Turns market state into BUY, SHORT, SELL, COVER, or HOLD decisions."""

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()
        self._positions: dict[str, PositionState] = {}
        self.vision = VisualAnalyzer()

    def _position_size(self, df: pd.DataFrame) -> float:
        """Calcula tamaño de posición en % del capital usando ATR."""
        p = self.params
        if not p.atr_position_sizing:
            return p.max_position_size_pct

        last = df.iloc[-1]
        atr = last.get("atr")
        close = float(last["close"])

        if pd.notna(atr) and atr > 0 and close > 0:
            stop_distance = p.trailing_stop_atr_mult * float(atr)
            if stop_distance > 0:
                size_pct = (p.atr_risk_pct * close) / stop_distance
                return max(p.min_position_size_pct, min(p.max_position_size_pct, size_pct))

        return p.max_position_size_pct

    def _check_dip(self, df: pd.DataFrame) -> bool:
        """Detecta Buy the Dip: caída >= dip_drop_pct en N días + RSI sobrevendido."""
        p = self.params
        if not p.use_contrarian_dip or len(df) < p.dip_drop_days + 1:
            return False

        last = df.iloc[-1]
        rsi = last.get("rsi")
        if pd.isna(rsi) or float(rsi) > p.dip_rsi_max:
            return False

        # Caída del precio en los últimos N días
        recent_high = float(df["close"].iloc[-(p.dip_drop_days + 1)])
        current = float(last["close"])
        drop = (current / recent_high) - 1.0

        return drop <= p.dip_drop_pct

    def _check_short_entry(self, df: pd.DataFrame, score: float) -> bool:
        """Detecta oportunidad de Short: score muy bajista + RSI sobrecomprado."""
        p = self.params
        if not p.use_short_selling:
            return False
        if score > p.short_score_threshold:
            return False

        last = df.iloc[-1]
        rsi = last.get("rsi")
        if pd.isna(rsi) or float(rsi) < p.short_min_rsi:
            return False

        return True

    def decide(
        self,
        df: pd.DataFrame,
        score: float,
        has_position: bool,
        position_pnl_pct: float = 0.0,
        ml_direction: str | None = None,
        ml_probability: float | None = None,
        sentiment_label: str | None = None,
        ticker: str = "",
        position_side: str = "LONG",
        # ── Mejoras de Win Rate ──────────────────────────
        prev_score: float = 0.0,           # Score del día anterior (confirmación)
        weekly_trend: str = "NEUTRAL",     # BULLISH / BEARISH / NEUTRAL (multi-TF)
        market_regime: str = "BULL",       # BULL / BEAR / LATERAL (HMM)
        earnings_blackout: bool = False,   # True si estamos cerca de earnings
    ) -> Decision:
        if df.empty:
            return Decision("HOLD", "no market data")

        last = df.iloc[-1]
        close = float(last["close"])
        atr = last.get("atr")
        rsi = float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0
        ticker_key = ticker or "default"
        pos = self._positions.get(ticker_key)

        # ── Gestionar posición abierta ──────────────────────────────
        if has_position and pos is not None:
            current_atr = float(atr) if pd.notna(atr) else None
            pos.update_extremes(close, current_atr=current_atr)
            should_exit, reason = pos.should_exit(close, rsi=rsi, regime=market_regime)
            if should_exit:
                del self._positions[ticker_key]
                if pos.side == "SHORT":
                    return Decision("COVER", reason, confidence=1.0, side="SHORT")
                return Decision("SELL", reason, confidence=1.0, side=pos.side)

            # Mantener short si score sigue bajista
            if pos.side == "SHORT":
                if score <= self.params.short_score_threshold:
                    return Decision("HOLD", f"short holding (score={score:.2f})", side="SHORT")
                else:
                    # Score mejoró → cubrir short
                    del self._positions[ticker_key]
                    return Decision("COVER", f"short cubierto por mejora de score ({score:.2f})", confidence=0.7, side="SHORT")

            # Mantener LONG/DIP en uptrend
            adx = float(last.get("adx", 0))
            sma_200 = last.get("sma_200")
            in_uptrend = adx > 20 and pd.notna(sma_200) and close > float(sma_200)
            if in_uptrend and score > -0.30:
                return Decision("HOLD", f"uptrend holding (score={score:.2f})", side=pos.side)
            if score < -0.30:
                del self._positions[ticker_key]
                return Decision("SELL", f"score bearish ({score:.2f})", confidence=abs(score), side=pos.side)
            return Decision("HOLD", "position still valid", side=pos.side)

        if has_position:
            return Decision("HOLD", "position valid (no state)", side=position_side)

        # ── Sin posición: buscar entrada ────────────────────────────

        p = self.params
        if p.use_earnings_blackout and earnings_blackout:
            return Decision("HOLD", "Blackout por Earnings")

        # 1. Buy the Dip
        if self._check_dip(df):
            if p.use_regime_filter and market_regime == "BEAR":
                return Decision("HOLD", "DIP cancelado: Régimen BEAR")
            if p.use_multi_timeframe and weekly_trend == "BEARISH":
                return Decision("HOLD", "DIP cancelado: Tendencia Semanal BEARISH")
                
            return Decision(
                "BUY",
                f"DIP detectado: RSI={float(last.get('rsi', 0)):.1f}, caída fuerte",
                confidence=0.75,
                position_size_pct=p.dip_position_size_pct,
                side="DIP",
            )

        # 2. Short Selling
        if self._check_short_entry(df, score):
            if p.use_regime_filter and market_regime == "BULL":
                return Decision("HOLD", "SHORT cancelado: Régimen BULL")
            
            donchian_lower = last.get("donchian_lower_20")
            if pd.notna(donchian_lower):
                close_val = float(last["close"])
                # For short breakout, we want close to be hitting the lowest of 20 days
                if close_val > float(donchian_lower) * 1.01:
                    return Decision("HOLD", "SHORT cancelado: No hay ruptura bajista (Donchian)")
            if p.use_multi_timeframe and weekly_trend == "BULLISH":
                return Decision("HOLD", "SHORT cancelado: Tendencia Semanal BULLISH")
            if p.confirm_candle and prev_score > p.short_score_threshold:
                return Decision("HOLD", "SHORT esperando confirmación 2da vela")

            return Decision(
                "SHORT",
                f"Short: score={score:.2f}, RSI={float(last.get('rsi', 0)):.1f} sobrecomprado",
                confidence=min(1.0, abs(score)),
                position_size_pct=p.short_position_size_pct,
                side="SHORT",
            )

        # 3. Long tradicional
        return self._decide_entry(
            df=df,
            score=score,
            ml_direction=ml_direction,
            ml_probability=ml_probability,
            sentiment_label=sentiment_label,
            prev_score=prev_score,
            weekly_trend=weekly_trend,
            market_regime=market_regime
        )

    def _decide_entry(
        self,
        df: pd.DataFrame,
        score: float,
        ml_direction: str | None,
        ml_probability: float | None,
        sentiment_label: str | None,
        prev_score: float = 0.0,
        weekly_trend: str = "NEUTRAL",
        market_regime: str = "BULL"
    ) -> Decision:
        p = self.params
        if score < p.buy_score_threshold:
            return Decision("HOLD", f"score below buy threshold ({score:.2f})")

        # ── Nuevos Filtros ──────────────────────────────────────────
        if p.use_regime_filter and market_regime == "BEAR":
            return Decision("HOLD", "LONG cancelado: Régimen BEAR")
            
        if p.use_multi_timeframe and weekly_trend == "BEARISH":
            return Decision("HOLD", "LONG cancelado: Tendencia Semanal BEARISH")
            
        if p.confirm_candle and prev_score < p.buy_score_threshold:
            return Decision("HOLD", "LONG esperando confirmación 2da vela")
        # ─────────────────────────────────────────────────────────────

        last = df.iloc[-1]
        close = float(last["close"])
        sma_200 = last.get("sma_200")
        rsi = last.get("rsi")
        adx = last.get("adx", 50.0)

        if p.require_price_above_sma200 and pd.notna(sma_200) and close < float(sma_200):
            return Decision("HOLD", "price below SMA200")

        if pd.notna(rsi) and float(rsi) > p.max_buy_rsi:
            return Decision("HOLD", f"RSI too high ({float(rsi):.1f})")

        if pd.notna(adx) and adx < 15.0:
            return Decision("HOLD", f"ADX too low ({float(adx):.1f}), no directional trend")

        if p.use_ml_filter:
            if ml_direction is None or ml_probability is None:
                return Decision("HOLD", "ML confirmation missing")
            if ml_direction != "ALCISTA" or ml_probability < p.min_ml_buy_probability:
                return Decision("HOLD", f"ML rejected buy ({ml_direction}, {ml_probability:.1%})")

        if sentiment_label == "BAJISTA":
        
        if p.use_donchian_breakout:
            donchian_upper = last.get("donchian_upper_20")
            # We want to know if the close is breaking or at least very close to the upper band
            # Since rolling max includes the current close, if close == donchian_upper, it is a breakout
            if pd.notna(donchian_upper) and close < float(donchian_upper) * 0.99:
                return Decision("HOLD", "No breakout (price below Donchian Upper)")
            return Decision("HOLD", "News sentiment is BAJISTA")

        # Forzar confianza a 50% según solicitud del usuario
        confidence = 0.50

        size = self._position_size(df)
        return Decision(
            "BUY",
            f"LONG: score {score:.2f} passed all filters",
            confidence=confidence,
            position_size_pct=size,
            side="LONG",
        )

    def on_position_opened(self, ticker: str, entry_price: float, df: pd.DataFrame, side: str = "LONG") -> None:
        """Registra una posición abierta para trailing stop."""
        last = df.iloc[-1]
        atr = last.get("atr", 0.0)
        atr_val = float(atr) if pd.notna(atr) else 0.0
        self._positions[ticker] = PositionState(entry_price, atr_val, self.params, side=side)
