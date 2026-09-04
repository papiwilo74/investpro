"""Multi-Strategy Allocator — Asignación dinámica de capital según rendimiento de estrategia y tipo de activo.

Evalúa periódicamente el Win Rate y Profit Factor de:
1. MOMENTUM (Tendencia con RSI/MACD)
2. MEAN_REVERSION (Compras en sobreventa extremas)
3. PAIRS_TRADING (Arbitraje estadístico market-neutral)

Asigna un multiplicador de capital (0.5x a 2.0x) para favorecer las estrategias ganadoras en el entorno actual,
aplicando una distribución objetivo del 85% para Crypto y 15% para Acciones.
"""

from __future__ import annotations

import gc
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
    """Asignador de peso de capital por estrategia y tipo de activo (85% Crypto / 15% Stock)."""

    def __init__(
        self,
        min_trades_to_adjust: int = 5,
        crypto_target_allocation: float = 0.85,
        stock_target_allocation: float = 0.15,
        crypto_boost_factor: float = 1.75,
    ) -> None:
        self.min_trades_to_adjust = min_trades_to_adjust
        self.crypto_target_allocation = crypto_target_allocation
        self.stock_target_allocation = stock_target_allocation
        self.crypto_boost_factor = crypto_boost_factor
        self.stats: dict[str, StrategyStats] = {
            "MOMENTUM": StrategyStats(),
            "MEAN_REVERSION": StrategyStats(),
            "PAIRS_TRADING": StrategyStats(),
        }
        self.asset_stats: dict[str, StrategyStats] = {
            "CRYPTO": StrategyStats(),
            "STOCK": StrategyStats(),
        }

    def record_trade(
        self,
        strategy_type: str,
        pnl_pct: float,
        asset_type: str = "STOCK",
    ) -> None:
        """Registra el resultado de una operación para una estrategia y tipo de activo."""
        st = strategy_type.upper()
        if st not in self.stats:
            self.stats[st] = StrategyStats()

        stat = self.stats[st]
        stat.total_trades += 1
        if pnl_pct > 0:
            stat.winning_trades += 1
        stat.total_pnl += pnl_pct

        # Registrar estadísticas por tipo de activo (CRYPTO vs STOCK)
        at = asset_type.upper()
        if at not in self.asset_stats:
            self.asset_stats[at] = StrategyStats()
        astat = self.asset_stats[at]
        astat.total_trades += 1
        if pnl_pct > 0:
            astat.winning_trades += 1
        astat.total_pnl += pnl_pct

        # Liberación explícita de memoria para Render (512MB RAM limit)
        gc.collect()

    def get_allocation_scale(
        self,
        strategy_type: str,
        asset_type: str | None = None,
    ) -> float:
        """Retorna un multiplicador de tamaño de posición (0.5x - 2.0x) respetando el peso 85/15."""
        st = strategy_type.upper()
        stat = self.stats.get(st)

        scale = 1.0
        if stat and stat.total_trades >= self.min_trades_to_adjust:
            wr = stat.win_rate
            # Escalar según Win Rate
            if wr >= 0.70:
                scale = 1.4  # Excelente rendimiento -> Aumenta capital 40%
            elif wr >= 0.55:
                scale = 1.2  # Buen rendimiento -> Aumenta 20%
            elif wr >= 0.40:
                scale = 1.0  # Rendimiento aceptable -> Base
            elif wr >= 0.25:
                scale = 0.7  # Bajo rendimiento -> Reduce 30%
            else:
                scale = 0.5  # Pésimo rendimiento -> Reduce al mínimo 50%

        # Si se especifica tipo de activo, escalar ponderación según 85% Crypto vs 15% Stocks
        if asset_type is not None:
            is_crypto = asset_type.upper() == "CRYPTO"
            if is_crypto:
                crypto_stat = self.asset_stats.get("CRYPTO")
                if not crypto_stat or crypto_stat.win_rate >= 0.40:
                    # Factor de impulso para Crypto alineado a la meta del 85%
                    scale *= self.crypto_boost_factor * (self.crypto_target_allocation / 0.50)
            else:
                # Ponderación moderada para Acciones (objetivo 15%)
                scale *= self.stock_target_allocation / 0.50

        return min(round(scale, 2), 2.0)

    def clear_memory_cache() -> None:
        """Fuerza la recolección de basura en el ciclo de escaneo."""
        gc.collect()
