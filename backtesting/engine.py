"""
Motor de backtesting — simula una estrategia basada en señales sobre
datos históricos.

Usa señales del ``sig_composite`` column desplazadas 1 periodo para
evitar look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from config import BACKTEST_PARAMS, BacktestParams
from backtesting.metrics import Trade, PerformanceMetrics


@dataclass
class BacktestResult:
    """Resultado de una simulación de backtest."""
    equity_curve: pd.Series
    trades: List[Trade]
    metrics: dict


class BacktestEngine:
    """
    Backtester basado en eventos.

    Lee la columna ``sig_composite`` del DataFrame.  Un valor positivo
    dispara compra, negativo dispara venta (cierre de posición), y cero
    es espera.  Las señales se aplican con un desplazamiento de 1 periodo
    para prevenir look-ahead bias.
    """

    def __init__(self, params: BacktestParams | None = None) -> None:
        self.params = params or BACKTEST_PARAMS

    # ── Helpers ───────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        factor = 1 + self.params.slippage_pct if is_buy else 1 - self.params.slippage_pct
        return price * factor

    def _commission(self, price: float, shares: float) -> float:
        return price * shares * self.params.commission_pct

    # ── Ejecución ─────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, signal_col: str = "sig_composite") -> BacktestResult:
        """
        Ejecuta el backtest.  *df* debe contener ``close`` y
        ``sig_composite`` (generado por ``SignalGenerator.add_signal_columns``) o la columna especificada en ``signal_col``.
        """
        if signal_col not in df.columns:
            raise ValueError(
                f"El DataFrame necesita la columna '{signal_col}'. "
                "Asegúrate de que la columna de señales exista."
            )

        capital = self.params.initial_capital
        position_shares: float = 0.0
        entry_price: float = 0.0
        entry_date = None

        equity_values: List[float] = []
        equity_dates: list = []
        trades: List[Trade] = []

        # Desplazar señales +1 para evitar look-ahead bias
        signals = df[signal_col].shift(1).fillna(0)

        for i in range(len(df)):
            date = df.index[i]
            price = df["close"].iloc[i]
            signal = signals.iloc[i]

            # ── COMPRA ────────────────────────────────────────────────
            if position_shares == 0 and signal > 0:
                exec_price = self._apply_slippage(price, is_buy=True)
                shares = int(capital / exec_price)          # acciones enteras
                if shares > 0:
                    cost = exec_price * shares
                    comm = self._commission(exec_price, shares)
                    capital -= cost + comm
                    position_shares = shares
                    entry_price = exec_price
                    entry_date = date

            # ── VENTA (cierre de posición) ────────────────────────────
            elif position_shares > 0 and signal < 0:
                exec_price = self._apply_slippage(price, is_buy=False)
                revenue = exec_price * position_shares
                comm = self._commission(exec_price, position_shares)
                pnl = revenue - (entry_price * position_shares) - comm
                pnl_pct = pnl / (entry_price * position_shares)

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    side="LONG",
                    entry_price=entry_price,
                    exit_price=exec_price,
                    shares=position_shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    commission=comm,
                ))

                capital += revenue - comm
                position_shares = 0
                entry_price = 0.0

            # Mark-to-market
            mtm = capital + position_shares * price
            equity_values.append(mtm)
            equity_dates.append(date)

        equity = pd.Series(equity_values, index=equity_dates, name="equity")
        metrics = PerformanceMetrics.summary(equity, trades)

        return BacktestResult(
            equity_curve=equity,
            trades=trades,
            metrics=metrics,
        )
