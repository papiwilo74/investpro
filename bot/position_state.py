"""Position state management for exits and trailing stops."""

from __future__ import annotations

import pandas as pd

from bot.rl_agent import get_rl_agent
from bot.strategy_params import StrategyParams


class PositionState:
    """Guarda estado de una posición abierta para trailing stop, partial TP, time exit y breakeven."""

    def __init__(
        self,
        entry_price: float,
        entry_atr: float,
        params: StrategyParams,
        side: str = "LONG",
        entry_date=None,
    ):
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
    def from_dict(cls, data: dict, params: StrategyParams) -> PositionState:
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

    def should_exit(
        self,
        current_price: float,
        rsi: float = 50.0,
        regime: str = "BULL",
        current_date=None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
    ) -> tuple[bool, str]:
        """Retorna (should_exit, reason)."""
        p = self.params

        # Time-based exit
        if p.use_time_based_exit and current_date is not None and self.entry_date is not None:
            try:
                days_held = (pd.to_datetime(current_date) - pd.to_datetime(self.entry_date)).days
                if days_held >= p.max_hold_days:
                    return True, f"time-based exit ({days_held}d >= {p.max_hold_days}d)"
            except Exception:
                pass

        # Breakeven stop
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
                try:
                    rl = get_rl_agent()
                    action = rl.get_action(pnl_pct, rsi, regime, is_training=False)
                    if action == 1:
                        return True, f"RL agent chose CLOSE (PnL: {pnl_pct:.2%})"
                except Exception:
                    pass
                if pnl_pct <= (p.stop_loss_pct * 1.5):
                    return True, f"stop-loss extremo ({pnl_pct:.2%})"
                if p.use_trailing_stop and self.entry_atr > 0:
                    mult = self._effective_trail_mult()
                    trailing_stop = self.max_price - mult * self.entry_atr
                    if current_price <= trailing_stop:
                        return True, f"trailing-stop RL ({current_price:.2f} <= {trailing_stop:.2f})"
                return False, ""

            effective_sl = stop_loss_pct if stop_loss_pct is not None else p.stop_loss_pct
            effective_tp = take_profit_pct if take_profit_pct is not None else p.take_profit_pct
            if pnl_pct <= effective_sl:
                return True, f"stop-loss ({pnl_pct:.2%})"

            if pnl_pct >= effective_tp:
                return True, f"take-profit ({pnl_pct:.2%})"

            if p.use_trailing_stop and self.entry_atr > 0:
                mult = self._effective_trail_mult()
                trailing_stop = self.max_price - mult * self.entry_atr
                if current_price <= trailing_stop:
                    return (
                        True,
                        f"trailing-stop ({current_price:.2f} <= {trailing_stop:.2f}, ATR={self.entry_atr:.2f}, mult={mult:.1f})",
                    )

        elif self.side == "SHORT":
            pnl_pct = (self.entry_price / current_price) - 1.0

            if p.use_rl_exits:
                try:
                    rl = get_rl_agent()
                    action = rl.get_action(pnl_pct, rsi, regime, is_training=False)
                    if action == 1:
                        return True, f"RL agent CLOSE SHORT (PnL: {pnl_pct:.2%})"
                except Exception:
                    pass
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
