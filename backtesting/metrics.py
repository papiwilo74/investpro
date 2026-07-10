"""
Métricas de rendimiento para resultados de backtesting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import BACKTEST_PARAMS

# ── Tipo de trade ─────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date: Any
    exit_date: Any
    side: str               # "LONG"
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    commission: float
    reason: str = ""        # Razón de cierre (stop-loss, take-profit, trailing-stop, score)

# ── Calculadora de métricas ───────────────────────────────────────────

class PerformanceMetrics:
    """Calcula métricas estándar a partir de equity curve y lista de trades."""

    @staticmethod
    def cumulative_return(equity: pd.Series) -> float:
        """Retorno acumulado total."""
        if equity.empty:
            return 0.0
        return (equity.iloc[-1] / equity.iloc[0]) - 1.0

    @staticmethod
    def annualized_return(equity: pd.Series) -> float:
        """Retorno anualizado."""
        if len(equity) < 2:
            return 0.0
        total = PerformanceMetrics.cumulative_return(equity)
        n_days = (equity.index[-1] - equity.index[0]).days
        if n_days <= 0:
            return 0.0
        years = n_days / 365.25
        if years <= 0:
            return 0.0
        return (1 + total) ** (1 / years) - 1

    @staticmethod
    def sharpe_ratio(
        equity: pd.Series,
        risk_free: float | None = None,
        trading_days: int = 252,
    ) -> float:
        """Ratio de Sharpe anualizado."""
        if len(equity) < 2:
            return 0.0
        rf = risk_free if risk_free is not None else BACKTEST_PARAMS.risk_free_rate
        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        excess = returns.mean() - rf / trading_days
        return float(excess / returns.std() * np.sqrt(trading_days))

    @staticmethod
    def max_drawdown(equity: pd.Series) -> dict[str, Any]:
        """Máximo drawdown con fechas de pico y valle."""
        if equity.empty:
            return {"max_drawdown": 0.0, "peak_date": None, "valley_date": None}
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        valley_idx = drawdown.idxmin()
        peak_idx = equity.loc[:valley_idx].idxmax()
        return {
            "max_drawdown": float(drawdown.min()),
            "peak_date": peak_idx,
            "valley_date": valley_idx,
        }

    @staticmethod
    def win_rate(trades: list[Trade]) -> float:
        """Porcentaje de trades ganadores."""
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def profit_factor(trades: list[Trade]) -> float:
        """Ratio ganancia bruta / pérdida bruta."""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def summary(equity: pd.Series, trades: list[Trade]) -> dict[str, Any]:
        """Diccionario con todas las métricas."""
        dd = PerformanceMetrics.max_drawdown(equity)
        return {
            "retorno_total": PerformanceMetrics.cumulative_return(equity),
            "retorno_anualizado": PerformanceMetrics.annualized_return(equity),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(equity),
            "max_drawdown": dd["max_drawdown"],
            "dd_peak": dd["peak_date"],
            "dd_valley": dd["valley_date"],
            "total_trades": len(trades),
            "win_rate": PerformanceMetrics.win_rate(trades),
            "profit_factor": PerformanceMetrics.profit_factor(trades),
            "capital_final": float(equity.iloc[-1]) if not equity.empty else 0.0,
        }
