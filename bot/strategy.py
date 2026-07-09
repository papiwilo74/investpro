"""Decision engine shared by live trading, bot backtests, and optimization.

Estrategias implementadas:
- LONG:  Compra cuando señales técnicas/ML son alcistas.
- DIP:   Compra cuando hay una caída fuerte + RSI sobrevendido (Buy the Dip).
- SHORT: Vende en corto cuando señales son fuertemente bajistas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.repositories import KellyRepository
from ml.rl import RLExitAgent
from ml.ensemble import ensemble, ModelSignal

# Instancias globales para compatibilidad con código existente.
# Los nuevos componentes deben inyectar rl_agent y kelly_tracker vía constructor.
rl_agent = RLExitAgent()


def get_rl_agent() -> RLExitAgent:
    """Factory para obtener el agente RL singleton."""
    return rl_agent


class KellyCalculator:
    """Calcula el tamaño óptimo de posición vía Kelly Criterion.

    f* = p - (1-p) / b
    p = win rate, b = avg_win / avg_loss
    Se usa Half-Kelly (50%) para reducir riesgo.

    Soporta dos backends de persistencia:
    - SQLAlchemy (recomendado): pasar session= al constructor
    - JSON file (legacy): usa data/kelly_trades.json
    """

    def __init__(self, fractional: float = 0.25, file_path: str = "",
                 session: Session | None = None):
        self.fractional = fractional
        self.trades: list[float] = []
        self._file_path = file_path or str(Path(__file__).resolve().parent.parent / "data" / "kelly_trades.json")
        self._repo: KellyRepository | None = None
        self._use_db = session is not None
        if session is not None:
            self._repo = KellyRepository(session)
            self._load_from_db()
        else:
            self._load_from_json()

    def _load_from_db(self) -> None:
        if self._repo is None:
            return
        try:
            self.trades = self._repo.get_all_trades()
        except Exception as e:
            import logging
            logging.getLogger("inversion_helper.kelly").warning("Error cargando Kelly trades desde DB: %s", e)
            self.trades = []

    def _load_from_json(self) -> None:
        try:
            if Path(self._file_path).exists():
                raw = Path(self._file_path).read_text(encoding="utf-8")
                data = json.loads(raw)
                self.trades = data.get("trades", [])
                self.fractional = data.get("fractional", self.fractional)
        except Exception as e:
            import logging
            logging.getLogger("inversion_helper.kelly").warning("Error cargando Kelly trades desde JSON: %s", e)
            self.trades = []

    def load(self) -> None:
        if self._use_db:
            self._load_from_db()
        else:
            self._load_from_json()

    def save(self) -> None:
        if self._use_db:
            return
        try:
            Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)
            data = {"trades": self.trades, "fractional": self.fractional}
            Path(self._file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            import logging
            logging.getLogger("inversion_helper.kelly").warning("Error guardando Kelly trades: %s", e)

    def record(self, pnl_pct: float) -> None:
        self.trades.append(pnl_pct)
        if self._use_db and self._repo is not None:
            try:
                self._repo.add_trade(pnl_pct, self.fractional)
            except Exception as e:
                import logging
                logging.getLogger("inversion_helper.kelly").warning("Error guardando Kelly trade en DB: %s", e)
        else:
            self.save()

    def reset(self) -> None:
        self.trades.clear()
        if self._use_db and self._repo is not None:
            try:
                self._repo.clear()
            except Exception as e:
                import logging
                logging.getLogger("inversion_helper.kelly").warning("Error limpiando Kelly trades en DB: %s", e)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0  # No asumir edge sin evidencia
        wins = sum(1 for t in self.trades if t > 0)
        return wins / len(self.trades)

    @property
    def avg_win(self) -> float:
        wins = [t for t in self.trades if t > 0]
        if not wins:
            return 0.01
        return sum(wins) / len(wins)

    @property
    def avg_loss(self) -> float:
        losses = [t for t in self.trades if t < 0]
        if not losses:
            return 0.01
        return abs(sum(losses) / len(losses))

    @property
    def odds_ratio(self) -> float:
        return self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0

    @property
    def kelly_pct(self) -> float:
        p = self.win_rate
        b = self.odds_ratio
        if b <= 0:
            return 0.05
        k = p - (1 - p) / b
        return max(0.01, min(0.30, k * self.fractional))

    def to_dict(self) -> dict:
        return {
            "win_rate": round(self.win_rate, 3),
            "avg_win_pct": round(self.avg_win, 4),
            "avg_loss_pct": round(self.avg_loss, 4),
            "odds_ratio": round(self.odds_ratio, 2),
            "kelly_pct": round(self.kelly_pct, 4),
            "total_trades": len(self.trades),
        }


# Global Kelly tracker
kelly_tracker = KellyCalculator()
@dataclass(frozen=True)
class StrategyParams:
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

    # ── NUEVAS: Momentum Scalping (operaciones rápidas) ────────────
    use_momentum_scalp: bool = True
    scalp_momentum_min: float = 0.40
    scalp_volume_min: float = 1.3
    scalp_stop_loss_pct: float = -0.03
    scalp_take_profit_pct: float = 0.04
    scalp_position_size_pct: float = 0.08

    # ── NUEVAS: Mean Reversion (1-2 día, compra en sobreventa) ────
    use_mean_reversion: bool = True
    mean_rev_rsi_max: float = 28.0
    mean_rev_drop_pct: float = -0.02
    mean_rev_stop_loss_pct: float = -0.02
    mean_rev_take_profit_pct: float = 0.03
    mean_rev_position_size_pct: float = 0.05

    # ── BUY THE DIP (compra caídas fuertes en sobrevendido) ─────────
    use_contrarian_dip: bool = True
    dip_drop_pct: float = -0.04
    dip_drop_days: int = 3
    dip_rsi_max: float = 35.0
    dip_position_size_pct: float = 0.12

    # ── SHORT SELLING ───────────────────────────────────────────────
    use_short_selling: bool = True
    short_score_threshold: float = -0.25
    short_min_rsi: float = 55.0
    short_stop_loss_pct: float = 0.020
    short_take_profit_pct: float = -0.030
    short_position_size_pct: float = 0.10
    short_momentum_threshold: float = -0.30
    short_min_adx: float = 18.0

    # ── MEJORAS DE WIN RATE ─────────────────────────────────────────
    confirm_candle: bool = False
    confirm_candle_days: int = 2
    use_multi_timeframe: bool = True
    use_regime_filter: bool = True
    use_earnings_blackout: bool = True
    earnings_blackout_days: int = 5
    auto_retrain_days: int = 30
    use_rl_exits: bool = True

    # ── NUEVAS: Partial Take Profit ─────────────────────────────────
    use_partial_take_profit: bool = True
    partial_tp1_pct: float = 0.05
    partial_tp1_fraction: float = 0.33
    partial_tp2_pct: float = 0.10
    partial_tp2_fraction: float = 0.33

    # ── NUEVAS: Dynamic trailing stop ───────────────────────────────
    use_dynamic_trailing: bool = True
    trail_atr_base: float = 3.0
    trail_atr_tight: float = 1.5

    # ── MEJORAS DE WIN RATE ─────────────────────────────────────────
    # Filtro de confirmación: requiere que N/M velas recientes sean alcistas
    use_confirmation_filter: bool = True
    confirmation_bars: int = 10
    confirmation_min_ratio: float = 0.6   # 60% de velas alcistas

    # Suavizado de señal con EMA para reducir ruido
    signal_smoothing_periods: int = 3

    # ADX mínimo para operar (evita mercado lateral)
    min_adx_to_trade: float = 15.0

    # ── NUEVAS: Intraday / Scalping agresivo ──────────────────────────
    use_intraday_scalp: bool = False
    intraday_scalp_momentum_min: float = 0.60
    intraday_scalp_volume_min: float = 1.5
    intraday_scalp_stop_loss_pct: float = -0.015
    intraday_scalp_take_profit_pct: float = 0.025
    intraday_scalp_position_size_pct: float = 0.12
    intraday_max_hold_minutes: int = 90
    
    # Filtro de sesión: solo operar en horas de mercado líquido (9:30-16:00 ET)
    use_session_filter: bool = True
    session_start_hour: int = 9
    session_start_minute: int = 30
    session_end_hour: int = 16
    session_end_minute: int = 0
    
    # VWAP como referencia intradía
    use_vwap_filter: bool = True
    vwap_deviation_pct: float = 0.005  # 0.5% desviación de VWAP

    # Régimen: qué estrategias activar según mercado
    disable_scalp_in_bear: bool = True
    disable_meanrev_in_trend: bool = True

    # ── Neural Brain ─────────────────────────────────────────────────
    use_neural_brain: bool = False
    neural_brain_min_confidence: float = 0.35

    # ── MEJORAS DE SALIDA PARA WIN RATE ──────────────────────────────
    # Time-based exit: cerrar si no pasa nada en N días
    use_time_based_exit: bool = True
    max_hold_days: int = 20

    # Breakeven stop: mover stop a breakeven tras ganancia inicial
    use_breakeven_stop: bool = True
    breakeven_trigger_pct: float = 0.03

    # Volatility targeting: reducir tamaño en alta volatilidad
    use_volatility_targeting: bool = True
    target_annual_volatility: float = 0.15

    # Ensemble adaptativo: blend de modelos con pesos dinámicos
    use_ensemble: bool = True

    # Score mínimo más alto en régimen cauteloso (VIX alto / SPY lateral)
    cautious_regime_score_boost: float = 0.15


@dataclass(frozen=True)
class Decision:
    action: str                              # BUY / SELL / HOLD / SHORT / COVER
    reason: str
    confidence: float = 0.0
    position_size_pct: float = 0.0
    side: str = "LONG"                       # LONG, DIP, SHORT
    partial_exit_fraction: float = 0.0       # >0 = vender fracción, no toda la posición


class PositionState:
    """Guarda estado de una posición abierta para trailing stop, partial TP, time exit y breakeven."""

    def __init__(self, entry_price: float, entry_atr: float, params: StrategyParams, side: str = "LONG", entry_date=None):
        self.entry_price = entry_price
        self.entry_atr = entry_atr
        self.max_price = entry_price
        self.min_price = entry_price
        self.params = params
        self.side = side
        self.entry_date = entry_date
        self._tp1_hit = False
        self._tp2_hit = False
        self._breakeven_active = False

    def to_dict(self) -> dict:
        return {
            "entry_price": self.entry_price,
            "entry_atr": self.entry_atr,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "side": self.side,
            "entry_date": str(self.entry_date) if self.entry_date is not None else None,
            "tp1_hit": self._tp1_hit,
            "tp2_hit": self._tp2_hit,
            "breakeven_active": self._breakeven_active,
        }

    @classmethod
    def from_dict(cls, data: dict, params: StrategyParams) -> "PositionState":
        state = cls(
            entry_price=data["entry_price"],
            entry_atr=data.get("entry_atr", data["entry_price"] * 0.02),
            params=params,
            side=data.get("side", "LONG"),
            entry_date=data.get("entry_date"),
        )
        state.max_price = data.get("max_price", data["entry_price"])
        state.min_price = data.get("min_price", data["entry_price"])
        state._tp1_hit = data.get("tp1_hit", False)
        state._tp2_hit = data.get("tp2_hit", False)
        state._breakeven_active = data.get("breakeven_active", False)
        return state

    def update_extremes(self, current_price: float, current_atr: float | None = None) -> None:
        if current_price > self.max_price:
            self.max_price = current_price
        if current_price < self.min_price:
            self.min_price = current_price
        if current_atr is not None and current_atr > 0:
            self.entry_atr = current_atr

    def _effective_trail_mult(self) -> float:
        p = self.params
        if not p.use_dynamic_trailing:
            return p.trailing_stop_atr_mult
        # Si ya vendimos parcialmente, el trailing se pone mucho más agresivo
        if self._tp2_hit:
            return 1.0
        if self._tp1_hit:
            return p.trail_atr_tight
        pnl_pct = self.current_pnl_pct(self.max_price)
        if pnl_pct >= 0.10:
            return p.trail_atr_tight
        if pnl_pct >= 0.05:
            return (p.trail_atr_base + p.trail_atr_tight) / 2
        return p.trail_atr_base

    def check_partial_tp(self, current_price: float) -> float:
        """Retorna fracción (0.0-1.0) de la posición a vender en take profit parcial."""
        p = self.params
        if not p.use_partial_take_profit:
            return 0.0
        pnl_pct = self.current_pnl_pct(current_price)
        fraction = 0.0
        if not self._tp1_hit and pnl_pct >= p.partial_tp1_pct:
            self._tp1_hit = True
            fraction += p.partial_tp1_fraction
        if not self._tp2_hit and pnl_pct >= p.partial_tp2_pct:
            self._tp2_hit = True
            fraction += p.partial_tp2_fraction
        return fraction

    def should_exit(self, current_price: float, rsi: float = 50.0, regime: str = "BULL", current_date=None) -> tuple[bool, str]:
        """Retorna (should_exit, reason)."""
        p = self.params

        # ── Time-based exit ───────────────────────────────────────────
        if p.use_time_based_exit and current_date is not None and self.entry_date is not None:
            try:
                days_held = (pd.to_datetime(current_date) - pd.to_datetime(self.entry_date)).days
                if days_held >= p.max_hold_days:
                    return True, f"time-based exit ({days_held}d >= {p.max_hold_days}d)"
            except Exception:
                pass

        # ── Breakeven stop ────────────────────────────────────────────
        if p.use_breakeven_stop and self.side in ("LONG", "DIP"):
            pnl_pct = (current_price / self.entry_price) - 1.0
            if not self._breakeven_active and pnl_pct >= p.breakeven_trigger_pct:
                self._breakeven_active = True
            if self._breakeven_active and pnl_pct <= 0:
                return True, f"breakeven stop (pnl={pnl_pct:.2%})"

        if self.side == "MEANREV":
            pnl_pct = (current_price / self.entry_price) - 1.0
            if pnl_pct <= p.mean_rev_stop_loss_pct:
                return True, f"meanrev stop-loss ({pnl_pct:.2%})"
            if pnl_pct >= p.mean_rev_take_profit_pct:
                return True, f"meanrev take-profit ({pnl_pct:.2%})"
            if p.use_trailing_stop and self.entry_atr > 0:
                trailing_stop = self.max_price - 1.0 * self.entry_atr
                if current_price <= trailing_stop:
                    return True, f"meanrev trailing-stop ({current_price:.2f} <= {trailing_stop:.2f})"
            return False, ""

        if self.side in ("SCALP", "SCALP_INTRADAY"):
            pnl_pct = (current_price / self.entry_price) - 1.0
            is_intraday = self.side == "SCALP_INTRADAY"
            sl = p.intraday_scalp_stop_loss_pct if is_intraday else p.scalp_stop_loss_pct
            tp = p.intraday_scalp_take_profit_pct if is_intraday else p.scalp_take_profit_pct
            if pnl_pct <= sl:
                return True, f"{self.side} stop-loss ({pnl_pct:.2%})"
            if pnl_pct >= tp:
                return True, f"{self.side} take-profit ({pnl_pct:.2%})"
            if p.use_trailing_stop and self.entry_atr > 0:
                mult = 1.0 if is_intraday else 1.5
                trailing_stop = self.max_price - mult * self.entry_atr
                if current_price <= trailing_stop:
                    return True, f"{self.side} trailing-stop ({current_price:.2f} <= {trailing_stop:.2f})"
            return False, ""

        if self.side in ("LONG", "DIP"):
            pnl_pct = (current_price / self.entry_price) - 1.0

            if p.use_rl_exits:
                action = self._rl_agent.get_action(pnl_pct, rsi, regime, is_training=False)
                if action == 1:
                    return True, f"RL agent chose CLOSE (PnL: {pnl_pct:.2%})"
                if pnl_pct <= (p.stop_loss_pct * 1.5):
                    return True, f"stop-loss extremo ({pnl_pct:.2%})"
                if p.use_trailing_stop and self.entry_atr > 0:
                    mult = self._effective_trail_mult()
                    trailing_stop = self.max_price - mult * self.entry_atr
                    if current_price <= trailing_stop:
                        return True, f"trailing-stop RL ({current_price:.2f} <= {trailing_stop:.2f})"
                return False, ""

            if pnl_pct <= p.stop_loss_pct:
                return True, f"stop-loss ({pnl_pct:.2%})"

            if pnl_pct >= p.take_profit_pct:
                return True, f"take-profit ({pnl_pct:.2%})"

            if p.use_trailing_stop and self.entry_atr > 0:
                mult = self._effective_trail_mult()
                trailing_stop = self.max_price - mult * self.entry_atr
                if current_price <= trailing_stop:
                    return True, f"trailing-stop ({current_price:.2f} <= {trailing_stop:.2f}, ATR={self.entry_atr:.2f}, mult={mult:.1f})"

        elif self.side == "SHORT":
            pnl_pct = (self.entry_price / current_price) - 1.0

            if p.use_rl_exits:
                action = self._rl_agent.get_action(pnl_pct, rsi, regime, is_training=False)
                if action == 1:
                    return True, f"RL agent CLOSE SHORT (PnL: {pnl_pct:.2%})"
                if pnl_pct <= -(p.short_stop_loss_pct * 1.5):
                    return True, f"short stop-loss extremo ({pnl_pct:.2%})"
                return False, ""

            price_change = (current_price / self.entry_price) - 1.0
            if price_change >= p.short_stop_loss_pct:
                return True, f"short stop-loss (subió {price_change:.2%})"
            if price_change <= p.short_take_profit_pct:
                return True, f"short take-profit (bajó {abs(price_change):.2%})"
            if p.use_trailing_stop and self.entry_atr > 0:
                mult = self._effective_trail_mult()
                trailing_stop = self.min_price + mult * self.entry_atr
                if current_price >= trailing_stop:
                    return True, f"short trailing-stop ({current_price:.2f} >= {trailing_stop:.2f})"

        return False, ""

    def current_pnl_pct(self, current_price: float) -> float:
        if self.side == "SHORT":
            return (self.entry_price / current_price) - 1.0
        return (current_price / self.entry_price) - 1.0


class TradingBrain:
    """Turns market state into BUY, SHORT, SELL, COVER, or HOLD decisions."""

    # Lazy-loaded Neural Brain compartido
    _neural_brain: object | None = None

    def __init__(self, params: StrategyParams | None = None,
                 rl_agent_instance: RLExitAgent | None = None,
                 kelly_instance: "KellyCalculator | None" = None) -> None:
        self.params = params or StrategyParams()
        self._positions: dict[str, PositionState] = {}
        # Inyección de dependencias: permite testear sin singletons globales
        self._rl_agent = rl_agent_instance or rl_agent
        self._kelly = kelly_instance or kelly_tracker
        self._ensemble = ensemble
        self._load_neural_if_needed()

    @classmethod
    def _load_neural_if_needed(cls) -> None:
        if cls._neural_brain is not None:
            return
        try:
            from ml.neural_brain import NeuralTradingBrain, extract_features
            model = NeuralTradingBrain()
            model_path = Path(__file__).resolve().parent.parent / "data" / "neural_brain.pth"
            if model_path.exists():
                ckpt = __import__("torch").load(str(model_path), map_location="cpu", weights_only=True)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
            cls._neural_brain = model
        except Exception:
            cls._neural_brain = False  # no disponible

    @classmethod
    def _neural_predict(cls, df, idx, score, has_position, position_pnl_pct, weekly_trend, market_regime, position_side, prev_score):
        """Decisión vía Neural Brain."""
        from ml.neural_brain import extract_features
        if cls._neural_brain is None or cls._neural_brain is False:
            return None
        try:
            feats = extract_features(df, idx, score, has_position, position_pnl_pct,
                                     weekly_trend, market_regime, position_side, prev_score)
            result = cls._neural_brain.predict(feats)
            return result
        except Exception:
            return None

    def _position_size(self, df: pd.DataFrame, current_index: int, score: float = 0.0) -> float:
        """Tamaño de posición combinando ATR risk + Volatility Targeting + Quarter-Kelly + convicción."""
        p = self.params
        last = df.iloc[current_index]
        close = float(last["close"])
        atr = last.get("atr")

        # 1. Base: ATR risk sizing
        atr_size = p.max_position_size_pct
        if pd.notna(atr) and atr > 0 and close > 0:
            stop_distance = p.trail_atr_base * float(atr)
            if stop_distance > 0:
                atr_size = (p.atr_risk_pct * close) / stop_distance

        # 2. Volatility Targeting: reducir tamaño cuando la volatilidad reciente es alta
        vol_multiplier = 1.0
        if p.use_volatility_targeting and current_index >= 20:
            try:
                recent_returns = df["close"].iloc[max(0, current_index - 20):current_index + 1].pct_change().dropna()
                if len(recent_returns) >= 10:
                    annual_vol = float(recent_returns.std() * np.sqrt(252))
                    if annual_vol > 0:
                        vol_multiplier = min(1.5, max(0.25, p.target_annual_volatility / annual_vol))
            except Exception:
                vol_multiplier = 1.0

        # 3. Quarter-Kelly: si hay historial suficiente, ajustar sizing
        kelly_multiplier = 1.0
        if self._kelly.trades and len(self._kelly.trades) >= 10:
            kelly_pct = self._kelly.kelly_pct
            # Usar quarter-kelly como multiplicador suave (0.5x - 1.2x)
            kelly_multiplier = min(1.2, max(0.5, 1.0 + (kelly_pct - 0.10) * 2.0))

        # 4. Convicción: score más alto = más capital
        conviction_boost = 1.0
        if score >= 0.50:
            conviction_boost = 1.3
        elif score >= 0.35:
            conviction_boost = 1.1
        elif score >= 0.20:
            conviction_boost = 1.0
        else:
            conviction_boost = 0.7

        size = atr_size * vol_multiplier * kelly_multiplier * conviction_boost
        return max(p.min_position_size_pct, min(p.max_position_size_pct, size))

    def _check_dip(self, df: pd.DataFrame, current_index: int) -> bool:
        """Detecta Buy the Dip: caída >= dip_drop_pct en N días + RSI sobrevendido."""
        p = self.params
        if not p.use_contrarian_dip or current_index + 1 < p.dip_drop_days + 1:
            return False

        last = df.iloc[current_index]
        rsi = last.get("rsi")
        if pd.isna(rsi) or float(rsi) > p.dip_rsi_max:
            return False

        # Caída del precio en los últimos N días
        recent_high = float(df["close"].iloc[current_index - p.dip_drop_days])
        current = float(last["close"])
        drop = (current / recent_high) - 1.0

        return drop <= p.dip_drop_pct

    def _check_short_entry(self, df: pd.DataFrame, current_index: int, score: float) -> bool:
        """Short: score negativo + breakdown Donchian o momentum negativo fuerte."""
        p = self.params
        if not p.use_short_selling:
            return False
        if score > p.short_score_threshold:
            return False

        last = df.iloc[current_index]
        close = float(last["close"])
        adx = last.get("adx", 0)
        if pd.isna(adx) or adx < p.short_min_adx:
            return False

        rsi = last.get("rsi")
        rsi_ok = pd.isna(rsi) or float(rsi) >= p.short_min_rsi
        if not rsi_ok:
            return False

        donchian_lower = last.get("donchian_lower_20")
        momentum = last.get("sig_momentum", 0)
        volume = last.get("sig_volume", 0)

        # Breakdown Donchian
        if pd.notna(donchian_lower) and close <= float(donchian_lower) * 1.01:
            return True

        # Momentum negativo + volumen (breakdown suave)
        if (pd.notna(momentum) and float(momentum) <= p.short_momentum_threshold
            and pd.notna(volume) and float(volume) >= 0.15):
            return True

        return False

    def decide(
        self,
        df: pd.DataFrame,
        score: float,
        has_position: bool,
        current_index: int | None = None,
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

        # current_index=None → modo legacy (df es un slice que termina en el día actual)
        idx = len(df) - 1 if current_index is None else current_index
        if idx < 0:
            return Decision("HOLD", "no market data")
        last = df.iloc[idx]
        close = float(last["close"])
        atr = last.get("atr")
        rsi = float(last.get("rsi", 50.0)) if pd.notna(last.get("rsi")) else 50.0
        adx_val = float(last.get("adx", 0)) if pd.notna(last.get("adx")) else None
        ticker_key = ticker or "default"
        pos = self._positions.get(ticker_key)
        p = self.params

        # ── FILTRO OBLIGATORIO DE RÉGIMEN Y TENDENCIA SEMANAL ──────────
        # Evita operar en chop/bear donde la mayoría de estrategias fallan
        if not has_position:
            if p.use_regime_filter and market_regime != "BULL" and market_regime != "NEUTRAL":
                return Decision("HOLD", f"Régimen {market_regime} no favorable para entradas")
            if p.use_multi_timeframe and weekly_trend != "BULLISH" and weekly_trend != "NEUTRAL":
                return Decision("HOLD", f"Tendencia semanal {weekly_trend} no favorable")
            if p.min_adx_to_trade > 0 and adx_val is not None and adx_val < p.min_adx_to_trade:
                return Decision("HOLD", f"ADX bajo ({adx_val:.1f}) — mercado lateral")

        # ── Suavizar señal con EMA para reducir ruido ──────────────
        smooth = p.signal_smoothing_periods
        n = idx + 1
        if smooth > 1 and "sig_composite" in df.columns and n >= smooth:
            raw = df["sig_composite"].values[:n]
            alpha = 2.0 / (smooth + 1)
            smoothed = np.empty(n)
            smoothed[0] = raw[0]
            for j in range(1, n):
                if np.isnan(raw[j]):
                    smoothed[j] = smoothed[j - 1]
                else:
                    smoothed[j] = raw[j] * alpha + smoothed[j - 1] * (1.0 - alpha)
            smooth_score = float(smoothed[-1])
        else:
            smooth_score = score
        self._last_smooth_score = smooth_score

        # ── ADX mínimo para operar (evita mercado lateral) ─────────
        adx_raw = last.get("adx")
        adx_val = float(adx_raw) if pd.notna(adx_raw) else None
        regime_block = False
        if not has_position and p.min_adx_to_trade > 0 and adx_val is not None and adx_val < p.min_adx_to_trade:
            regime_block = True

        # ── Gestionar posición abierta ──────────────────────────────
        if has_position and pos is not None:
            current_atr = float(atr) if pd.notna(atr) else None
            pos.update_extremes(close, current_atr=current_atr)

            # Partial take profit: vender fracción si se alcanzó un nivel
            partial_frac = pos.check_partial_tp(close)
            if partial_frac > 0:
                return Decision("SELL", f"partial TP: vendiendo {partial_frac:.0%}",
                                confidence=0.8, side=pos.side, partial_exit_fraction=partial_frac)

            current_date = df.index[idx] if idx is not None and idx < len(df) else None
            should_exit, reason = pos.should_exit(close, rsi=rsi, regime=market_regime, current_date=current_date)
            if should_exit:
                pnl = pos.current_pnl_pct(close)
                self._kelly.record(pnl)
                self._rl_agent.update(pnl, rsi, market_regime, action=1, reward=pnl, next_pnl_pct=0.0, next_rsi=rsi, next_regime=market_regime)
                self._rl_agent.save_model()
                del self._positions[ticker_key]
                if pos.side == "SHORT":
                    return Decision("COVER", reason, confidence=1.0, side="SHORT")
                return Decision("SELL", reason, confidence=1.0, side=pos.side)

            # Mantener short: solo cubrir si score mejora significativamente
            if pos.side == "SHORT":
                short_pnl = pos.current_pnl_pct(close)
                # Cubrir si: score >= 0 (se puso alcista) O stop-loss O take-profit
                if score >= 0.0:
                    self._kelly.record(short_pnl)
                    del self._positions[ticker_key]
                    return Decision("COVER", f"short: score turned bullish ({score:.2f})", confidence=0.7, side="SHORT")
                if short_pnl <= self.params.short_take_profit_pct:
                    self._kelly.record(short_pnl)
                    del self._positions[ticker_key]
                    return Decision("COVER", f"short take-profit ({short_pnl:.2%})", confidence=0.9, side="SHORT")
                return Decision("HOLD", f"short holding (score={score:.2f}, pnl={short_pnl:.2%})", side="SHORT")

            # Mantener LONG/DIP en uptrend
            adx = float(last.get("adx", 0))
            sma_200 = last.get("sma_200")
            in_uptrend = adx > 20 and pd.notna(sma_200) and close > float(sma_200)
            if in_uptrend and score > -0.30:
                return Decision("HOLD", f"uptrend holding (score={score:.2f})", side=pos.side)
            if score < -0.30:
                pnl = pos.current_pnl_pct(close)
                self._kelly.record(pnl)
                del self._positions[ticker_key]
                return Decision("SELL", f"score bearish ({score:.2f})", confidence=abs(score), side=pos.side)
            return Decision("HOLD", "position still valid", side=pos.side)

        if has_position:
            return Decision("HOLD", "position valid (no state)", side=position_side)

        # ── Sin posición: buscar entrada ────────────────────────────

        if p.use_earnings_blackout and earnings_blackout:
            return Decision("HOLD", "Blackout por Earnings")

        # ── ADX mínimo: si no hay tendencia, no operar ─────────────
        if regime_block:
            return Decision("HOLD", f"ADX bajo ({adx_val:.1f}) — mercado lateral")

        # ── Neural Brain: reemplaza TODAS las reglas si está activo ──────
        if p.use_neural_brain:
            nn_result = self._neural_predict(
                df, idx, smooth_score, has_position, position_pnl_pct,
                weekly_trend, market_regime, position_side, prev_score,
            )
            if nn_result and nn_result["confidence"] >= p.neural_brain_min_confidence:
                action = nn_result["action"]
                conf = nn_result["confidence"]
                size = nn_result["position_size_pct"]
                if action == "BUY":
                    return Decision("BUY", f"NN: buy (conf={conf:.2%})", confidence=conf,
                                    position_size_pct=size, side="LONG")
                elif action == "SHORT":
                    return Decision("SHORT", f"NN: short (conf={conf:.2%})", confidence=conf,
                                    position_size_pct=size, side="SHORT")
                elif action in ("SELL", "COVER"):
                    return Decision(action, f"NN: exit (conf={conf:.2%})", confidence=conf, side=position_side)
                else:  # HOLD
                    return Decision("HOLD", f"NN: hold (conf={conf:.2%})", confidence=conf)
            fallback_conf = nn_result["confidence"] if nn_result else 0.0
            return Decision("HOLD", f"NN: baja confianza ({fallback_conf:.2%})" if nn_result else "NN: no disponible")

        # Estrategias permitidas (el régimen BULL/NEUTRAL ya pasó filtro arriba)
        can_dip = p.use_contrarian_dip
        can_short = p.use_short_selling
        can_long = True

        # 0. Momentum Scalping (rápido, antes que otras estrategias)
        if p.use_momentum_scalp:
            sig_momentum = float(last.get("sig_momentum", 0.0))
            sig_volume = float(last.get("sig_volume", 0.0))
            volume_ok = sig_volume >= (p.scalp_volume_min - 1.0)
            if (sig_momentum >= p.scalp_momentum_min
                and volume_ok):
                return Decision(
                    "BUY",
                    f"Scalp: momentum={sig_momentum:.2f} vol={sig_volume:.2f}",
                    confidence=min(1.0, sig_momentum),
                    position_size_pct=p.scalp_position_size_pct,
                    side="SCALP",
                )

        # 0.2 Intraday Scalping (ultra-rápido, 1-5 min, alta frecuencia)
        if p.use_intraday_scalp:
            if self._check_intraday_session(last):
                sig_momentum = float(last.get("sig_momentum", 0.0))
                sig_volume = float(last.get("sig_volume", 0.0))
                volume_ok = sig_volume >= (p.intraday_scalp_volume_min - 1.0)
                vwap_ok = self._check_vwap(last, p) if p.use_vwap_filter else True
                if (sig_momentum >= p.intraday_scalp_momentum_min
                    and volume_ok and vwap_ok):
                    return Decision(
                        "BUY",
                        f"Intraday Scalp: mom={sig_momentum:.2f} vol={sig_volume:.2f}",
                        confidence=min(1.0, sig_momentum),
                        position_size_pct=p.intraday_scalp_position_size_pct,
                        side="SCALP_INTRADAY",
                    )

        # 0.5 Mean Reversion (compra en sobreventa, 1-2 días)
        can_meanrev = p.use_mean_reversion
        if p.disable_meanrev_in_trend:
            can_meanrev = False
        if can_meanrev:
            rsi_val = last.get("rsi")
            prev_close = float(df["close"].iloc[idx - 1]) if idx > 0 else close
            one_day_drop = (close / prev_close) - 1.0
            if (pd.notna(rsi_val) and float(rsi_val) <= p.mean_rev_rsi_max
                and one_day_drop <= p.mean_rev_drop_pct):
                return Decision(
                    "BUY",
                    f"MeanRev: RSI={float(rsi_val):.1f} drop={one_day_drop:.2%}",
                    confidence=0.6,
                    position_size_pct=p.mean_rev_position_size_pct,
                    side="MEANREV",
                )

        # 1. Buy the Dip
        if can_dip and self._check_dip(df, idx):
            return Decision(
                "BUY",
                f"DIP detectado: RSI={float(last.get('rsi', 0)):.1f}, caída fuerte",
                confidence=0.75,
                position_size_pct=p.dip_position_size_pct,
                side="DIP",
            )

        # 2. Short Selling (permitido en cualquier régimen si setup fuerte)
        if p.use_short_selling and self._check_short_entry(df, idx, smooth_score):
            sig_momentum = float(last.get("sig_momentum", 0.0))
            confidence = min(1.0, max(0.3, abs(smooth_score)))
            return Decision(
                "SHORT",
                f"Short: score={smooth_score:.2f}, momentum={sig_momentum:.2f}",
                confidence=confidence,
                position_size_pct=p.short_position_size_pct,
                side="SHORT",
            )

        # 3. Long tradicional
        return self._decide_entry(
            df=df,
            current_index=idx,
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
        current_index: int,
        score: float,
        ml_direction: str | None,
        ml_probability: float | None,
        sentiment_label: str | None,
        prev_score: float = 0.0,
        weekly_trend: str = "NEUTRAL",
        market_regime: str = "BULL"
    ) -> Decision:
        p = self.params
        # Usar score suavizado si está disponible
        smooth_score = getattr(self, '_last_smooth_score', score)
        entry_score = smooth_score if smooth_score != score else score

        # ── Ensemble adaptativo ────────────────────────────────────────
        if p.use_ensemble:
            xgb_signal = None
            if ml_direction is not None and ml_probability is not None:
                xgb_dir = "BULLISH" if ml_direction == "ALCISTA" else "BEARISH"
                xgb_signal = ModelSignal(direction=xgb_dir, probability=ml_probability, score=(ml_probability * 2 - 1))
            nn_signal = None
            if p.use_neural_brain and self._neural_brain and self._neural_brain is not False:
                nn_result = self._neural_predict(
                    df, current_index, entry_score, has_position=False,
                    position_pnl_pct=0.0, weekly_trend=weekly_trend,
                    market_regime=market_regime, position_side="LONG",
                    prev_score=prev_score,
                )
                if nn_result and nn_result["confidence"] >= 0.3:
                    nn_dir = {"BUY": "BULLISH", "HOLD": "NEUTRAL", "SELL": "BEARISH"}.get(nn_result["action"], "NEUTRAL")
                    nn_signal = ModelSignal(direction=nn_dir, probability=nn_result["confidence"], score=(nn_result["confidence"] * 2 - 1))

            ensemble_regime = market_regime
            if ensemble_regime not in ("BULL", "BEAR", "LATERAL"):
                ensemble_regime = "BULL" if ensemble_regime == "NEUTRAL" else "HIGH_VOL"

            ens_result = self._ensemble.predict(
                regime=ensemble_regime,
                xgboost_signal=xgb_signal,
                neural_brain_signal=nn_signal,
                ta_score=entry_score,
            )

            if ens_result.consensus_direction == "BULLISH" and ens_result.confidence >= 0.3:
                entry_score = max(entry_score, ens_result.blended_score)
                ml_direction = "ALCISTA"
                ml_probability = ens_result.confidence
            elif ens_result.consensus_direction == "BEARISH" and ens_result.confidence >= 0.4:
                return Decision("HOLD", f"Ensemble bearish ({ens_result.blended_score:.2f}, conf={ens_result.confidence:.2f})")

        if entry_score < p.buy_score_threshold:
            return Decision("HOLD", f"score below buy threshold ({entry_score:.2f})")

        # ── Filtro de confirmación: N/M velas recientes alcistas ────
        n = current_index + 1
        if p.use_confirmation_filter and "sig_composite" in df.columns and n >= p.confirmation_bars:
            recent = df["sig_composite"].iloc[current_index - p.confirmation_bars + 1 : current_index + 1].fillna(0)
            alcistas = (recent > 0).sum()
            required = int(p.confirmation_bars * p.confirmation_min_ratio)
            if alcistas < required:
                return Decision(
                    "HOLD",
                    f"Confirmación insuficiente: {alcistas}/{p.confirmation_bars} velas alcistas (req. {required})"
                )

        # ── Nuevos Filtros ──────────────────────────────────────────
        if p.confirm_candle and prev_score < p.buy_score_threshold:
            return Decision("HOLD", "LONG esperando confirmación 2da vela")
        # ─────────────────────────────────────────────────────────────

        last = df.iloc[current_index]
        close = float(last["close"])
        sma_200 = last.get("sma_200")
        rsi = last.get("rsi")
        adx = last.get("adx", 50.0)

        if p.require_price_above_sma200 and pd.notna(sma_200) and close < float(sma_200):
            return Decision("HOLD", "price below SMA200")

        if pd.notna(rsi) and float(rsi) > p.max_buy_rsi:
            return Decision("HOLD", f"RSI too high ({float(rsi):.1f})")

        if pd.notna(adx) and adx < 12.0:
            return Decision("HOLD", f"ADX too low ({float(adx):.1f}), no directional trend")

        if p.use_ml_filter:
            if ml_direction is None or ml_probability is None:
                return Decision("HOLD", "ML confirmation missing")
            if ml_direction != "ALCISTA" or ml_probability < p.min_ml_buy_probability:
                return Decision("HOLD", f"ML rejected buy ({ml_direction}, {ml_probability:.1%})")

        if p.use_donchian_breakout:
            donchian_upper = last.get("donchian_upper_20")
            if pd.notna(donchian_upper) and close < float(donchian_upper) * 0.97:
                return Decision("HOLD", "No breakout (close below Donchian Upper 20)")

        if sentiment_label == "BAJISTA":
            return Decision("HOLD", "News sentiment is BAJISTA")

        confidence = min(1.0, max(0.3, abs(entry_score) * 1.5))
        size = self._position_size(df, current_index, score=entry_score)
        return Decision(
            "BUY",
            f"LONG: score {entry_score:.2f} passed all filters",
            confidence=confidence,
            position_size_pct=size,
            side="LONG",
        )

    @staticmethod
    def _infer_weekly_trend(df: pd.DataFrame) -> str:
        if df is None or len(df) < 5:
            return "NEUTRAL"
        sma50 = df["close"].rolling(50).mean()
        if pd.isna(sma50.iloc[-1]):
            return "NEUTRAL"
        return "BULLISH" if df["close"].iloc[-1] > sma50.iloc[-1] else "BEARISH"

    @staticmethod
    def _infer_market_regime(df: pd.DataFrame) -> str:
        if df is None or len(df) < 200:
            return "BULL"
        sma200 = df["close"].rolling(200).mean()
        if pd.isna(sma200.iloc[-1]):
            return "BULL"
        close = df["close"].iloc[-1]
        if close > sma200.iloc[-1] * 1.05:
            return "BULL"
        elif close < sma200.iloc[-1] * 0.95:
            return "BEAR"
        return "LATERAL"

    def on_position_opened(self, ticker: str, entry_price: float, df: pd.DataFrame, current_index: int | None = None, side: str = "LONG") -> None:
        """Registra una posición abierta para trailing stop."""
        idx = len(df) - 1 if current_index is None else current_index
        last = df.iloc[idx]
        atr = last.get("atr", 0.0)
        atr_val = float(atr) if pd.notna(atr) else 0.0
        entry_date = df.index[idx] if idx < len(df) else None
        self._positions[ticker] = PositionState(entry_price, atr_val, self.params, side=side, entry_date=entry_date)

    def restore_positions(self, state_manager, alpaca_positions: list[dict]) -> int:
        """Recupera PositionState desde SQLite + Alpaca al iniciar el bot.
        
        Si hay posición en Alpaca pero no en SQLite, la crea desde cero.
        Si hay en SQLite, restaura el estado completo (trailing, breakeven, etc.).
        Retorna cantidad de posiciones restauradas.
        """
        if state_manager is None:
            return 0
        
        saved = {p["ticker"]: p for p in state_manager.get_positions()}
        count = 0
        
        for pos in alpaca_positions:
            ticker = pos.get("symbol", "")
            if not ticker:
                continue
            entry_price = float(pos.get("avg_entry_price", pos.get("current_price", 0)))
            current_price = float(pos.get("current_price", entry_price))
            
            if ticker in saved:
                s = saved[ticker]
                ps = PositionState.from_dict(s, self.params)
                ps.update_extremes(current_price)
                self._positions[ticker] = ps
            else:
                atr_est = entry_price * 0.02
                ps = PositionState(entry_price, atr_est, self.params, side="LONG")
                ps.update_extremes(current_price)
                self._positions[ticker] = ps
                
                state_manager.save_position(
                    ticker=ticker, side="LONG", entry_price=entry_price,
                    entry_atr=atr_est, qty=float(pos.get("qty", 0)),
                    max_price=ps.max_price, min_price=ps.min_price,
                    breakeven_active=ps._breakeven_active,
                    tp1_hit=ps._tp1_hit, tp2_hit=ps._tp2_hit,
                )
            count += 1
        
        return count

    def save_position_state(self, state_manager, ticker: str, qty: float = 0) -> None:
        """Persiste el estado actual de una posición a SQLite."""
        if state_manager is None:
            return
        ps = self._positions.get(ticker)
        if ps is None:
            return
        state_manager.save_position(
            ticker=ticker, side=ps.side, entry_price=ps.entry_price,
            entry_atr=ps.entry_atr, qty=qty,
            max_price=ps.max_price, min_price=ps.min_price,
            breakeven_active=ps._breakeven_active,
            tp1_hit=ps._tp1_hit, tp2_hit=ps._tp2_hit,
        )

    @staticmethod
    def _check_intraday_session(last: pd.Series) -> bool:
        """Verifica que la vela esté dentro del horario de mercado líquido."""
        p = StrategyParams()  # valores por defecto para session
        ts = last.name
        if not hasattr(ts, 'hour'):
            return True  # datos diarios, no filtrar
        h, m = ts.hour, ts.minute
        sess_start = p.session_start_hour * 60 + p.session_start_minute
        sess_end = p.session_end_hour * 60 + p.session_end_minute
        now = h * 60 + m
        return sess_start <= now <= sess_end

    @staticmethod
    def _check_vwap(last: pd.Series, p: StrategyParams) -> bool:
        """Verifica que el precio no esté muy lejos de VWAP."""
        vwap = last.get("vwap")
        close_val = float(last["close"])
        if pd.isna(vwap) or vwap == 0:
            return True
        deviation = abs(close_val / float(vwap) - 1.0)
        return deviation <= p.vwap_deviation_pct


def create_web_bot_strategy_params() -> StrategyParams:
    """
    Parámetros conservadores para el bot web:
    - Estrategia LONG con trend-following + SHORT en mercados bajistas.
    - Desactiva NN, RL, scalping, mean-reversion y dip-buying.
    - Filtros de régimen, tendencia semanal y confirmación activos.
    - Tamaño de posición reducido (max 10% LONG, 7% SHORT).
    - Shorts más tight: TP +5%, SL -3%.
    """
    return StrategyParams(
        buy_score_threshold=0.15,
        sell_score_threshold=-0.40,
        stop_loss_pct=-0.06,
        take_profit_pct=0.18,
        trailing_stop_atr_mult=2.5,
        use_trailing_stop=True,
        max_position_size_pct=0.10,
        min_position_size_pct=0.05,
        atr_position_sizing=True,
        atr_risk_pct=0.015,
        min_ml_buy_probability=0.60,
        require_price_above_sma200=True,
        max_buy_rsi=70.0,
        use_ml_filter=False,
        use_donchian_breakout=False,
        use_momentum_scalp=False,
        use_mean_reversion=False,
        use_contrarian_dip=False,
        # ── SHORT enabled in web mode (conservador) ───────────────
        use_short_selling=True,
        short_score_threshold=-0.25,
        short_min_rsi=55.0,
        short_stop_loss_pct=0.030,      # +3% price rise = cover
        short_take_profit_pct=-0.050,    # -5% price drop = cover
        short_position_size_pct=0.07,    # 7% max per short
        short_momentum_threshold=-0.30,
        short_min_adx=22.0,              # Solo shorts si hay momentum bajista real
        use_partial_take_profit=False,
        use_dynamic_trailing=True,
        use_confirmation_filter=True,
        confirmation_bars=10,
        confirmation_min_ratio=0.6,
        use_multi_timeframe=True,
        use_regime_filter=True,
        use_earnings_blackout=False,
        use_rl_exits=False,
        use_intraday_scalp=False,
        use_session_filter=False,
        use_vwap_filter=False,
        use_neural_brain=False,
        signal_smoothing_periods=3,
        min_adx_to_trade=15.0,
        confirm_candle=False,
        use_time_based_exit=True,
        max_hold_days=20,
        use_breakeven_stop=True,
        breakeven_trigger_pct=0.03,
        use_volatility_targeting=True,
        target_annual_volatility=0.15,
        cautious_regime_score_boost=0.15,
    )
