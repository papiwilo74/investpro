"""Multi-Strategy Allocator — Asignación dinámica de capital según rendimiento de estrategia.

Evalúa periódicamente el Win Rate y Profit Factor de:
1. MOMENTUM (Tendencia con RSI/MACD)
2. MEAN_REVERSION (Compras en sobreventa extremas)
3. PAIRS_TRADING (Arbitraje estadístico market-neutral)

Asigna un multiplicador de capital (0.5x a 1.5x) para favorecer las estrategias ganadoras en el entorno actual.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("inversion_helper.multi_strategy_allocator")

StrategyType = Literal["MOMENTUM", "MEAN_REVERSION", "PAIRS_TRADING"]


@dataclass
class StrategyStats:
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5  # Neutral por defecto
        return self.winning_trades / self.total_trades


class MultiStrategyAllocator:
    """Asignador de peso de capital por estrategia."""

    def __init__(self, min_trades_to_adjust: int = 5) -> None:
        self.min_trades_to_adjust = min_trades_to_adjust
        self.stats: dict[str, StrategyStats] = {
            "MOMENTUM": StrategyStats(),
            "MEAN_REVERSION": StrategyStats(),
            "PAIRS_TRADING": StrategyStats(),
        }

    def record_trade(self, strategy_type: str, pnl_pct: float) -> None:
        """Registra el resultado de una operación para una estrategia específica."""
        st = strategy_type.upper()
        if st not in self.stats:
            self.stats[st] = StrategyStats()

        stat = self.stats[st]
        stat.total_trades += 1
        if pnl_pct > 0:
            stat.winning_trades += 1
        stat.total_pnl += pnl_pct

    def get_allocation_scale(self, strategy_type: str) -> float:
        """Retorna un multiplicador de tamaño de posición (0.5x - 1.5x)."""
        st = strategy_type.upper()
        stat = self.stats.get(st)

        if not stat or stat.total_trades < self.min_trades_to_adjust:
            return 1.0  # Sin historial suficiente -> Sizing base (1.0x)

        wr = stat.win_rate

        # Escalar según Win Rate
        if wr >= 0.70:
            return 1.4  # Excelente rendimiento -> Aumenta capital 40%
        elif wr >= 0.55:
            return 1.2  # Buen rendimiento -> Aumenta 20%
        elif wr >= 0.40:
            return 1.0  # Rendimiento aceptable -> Base
        elif wr >= 0.25:
            return 0.7  # Bajo rendimiento -> Reduce 30%
        else:
            return 0.5  # Pésimo rendimiento -> Reduce al mínimo 50%
