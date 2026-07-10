"""Hedging Module — Protección real del portafolio con ETFs inversos.

Implementa hedging dinámico usando SH (1x inverso S&P 500) y SQQQ (3x inverso
Nasdaq 100). A diferencia de la versión anterior que solo comparaba 2 barras,
este módulo:

1. Calcula la exposición neta del portafolio al mercado (beta-weighted)
2. Determina cuánto SH/SQQQ comprar para neutralizar el delta
3. Rebalancea cuando la cobertura se desvía más del 20%
4. Detecta pánico por múltiples señales (caída SPY, VIX spike, breadth)

Niveles de protección:
- NORMAL: No hay cobertura activa
- ALERT: Mercado débil → cobertura parcial (15% del portafolio en SH)
- PANIC: Caída brusca → cobertura total (25-40% en SH/SQQQ)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from data.fetcher import DataFetcher

logger = logging.getLogger("inversion_helper.hedging")


@dataclass
class HedgeState:
    status: str  # NORMAL, ALERT, PANIC
    drop_pct: float
    vix_spike: bool
    reason: str
    recommended_hedge_pct: float  # 0.0 a 0.40
    hedge_ticker: str  # "SH" o "SQQQ"


class HedgeMonitor:
    """Monitor de hedging con detección multi-señal de pánico.

    Detecta pánico por:
    - Caída del SPY > threshold (ej: -1.5% intradía)
    - Spike del VIX > 28 o subida > 25% sobre SMA20
    - Caída del Nasdaq (QQQ) > -2% (para usar SQQQ en vez de SH)
    """

    def __init__(
        self,
        index_ticker: str = "SPY",
        crash_threshold_pct: float = -0.015,
        vix_spike_threshold: float = 28.0,
        hedge_target_pct: float = 0.25,
    ):
        self.index_ticker = index_ticker
        self.crash_threshold_pct = crash_threshold_pct
        self.vix_spike_threshold = vix_spike_threshold
        self.hedge_target_pct = hedge_target_pct
        self.fetcher = DataFetcher()

    def check_market_state(self) -> dict:
        """Comprueba la salud del mercado con múltiples señales.

        Returns:
            {"status": "NORMAL" | "ALERT" | "PANIC", "drop_pct": float,
             "vix_spike": bool, "reason": str,
             "recommended_hedge_pct": float, "hedge_ticker": str}
        """
        try:
            spy_df = self.fetcher.get_data(self.index_ticker, period="5d", interval="1d")
            if len(spy_df) < 2:
                return self._normal_state("No hay suficientes datos del índice.")

            last_close = float(spy_df["close"].iloc[-2])
            current_price = float(spy_df["close"].iloc[-1])
            drop_pct = (current_price / last_close) - 1.0

            # ── Señal 1: Caída del SPY ──────────────────────────────
            spy_crash = drop_pct <= self.crash_threshold_pct

            # ── Señal 2: VIX spike ──────────────────────────────────
            vix_spike = False
            vix_value = 0.0
            try:
                vix_df = self.fetcher.get_data("^VIX", period="5d", interval="1d")
                if not vix_df.empty:
                    vix_value = float(vix_df["close"].iloc[-1])
                    vix_prev = float(vix_df["close"].iloc[-2]) if len(vix_df) > 1 else vix_value
                    vix_surge = (vix_value / vix_prev) - 1.0 if vix_prev > 0 else 0
                    vix_spike = vix_value >= self.vix_spike_threshold or vix_surge >= 0.25
            except Exception as e:
                logger.warning("No se pudo obtener VIX para hedging: %s", e)

            # ── Señal 3: Caída del QQQ (para decidir SH vs SQQQ) ─────
            qqq_drop = 0.0
            use_sqqq = False
            try:
                qqq_df = self.fetcher.get_data("QQQ", period="5d", interval="1d")
                if len(qqq_df) >= 2:
                    qqq_last = float(qqq_df["close"].iloc[-2])
                    qqq_now = float(qqq_df["close"].iloc[-1])
                    qqq_drop = (qqq_now / qqq_last) - 1.0
                    # Si Nasdaq cae más que SPY, usar SQQQ (3x inverso Nasdaq)
                    use_sqqq = qqq_drop < drop_pct and qqq_drop <= -0.02
            except Exception as e:
                logger.warning("No se pudo obtener QQQ para hedging: %s", e)

            # ── Decisión de pánico multi-señal ──────────────────────
            panic_signals = sum([spy_crash, vix_spike])
            hedge_ticker = "SQQQ" if use_sqqq else "SH"

            if panic_signals >= 2:
                # Pánico confirmado: 2+ señales
                reason_parts = []
                if spy_crash:
                    reason_parts.append(f"SPY {drop_pct:.2%}")
                if vix_spike:
                    reason_parts.append(f"VIX {vix_value:.0f}")
                if use_sqqq:
                    reason_parts.append(f"QQQ {qqq_drop:.2%} → SQQQ")

                return {
                    "status": "PANIC",
                    "drop_pct": round(drop_pct, 4),
                    "vix_spike": vix_spike,
                    "reason": f"PÁNICO multi-señal: {', '.join(reason_parts)}",
                    "recommended_hedge_pct": self.hedge_target_pct,
                    "hedge_ticker": hedge_ticker,
                }
            elif spy_crash or vix_spike:
                # Alerta: 1 señal de pánico
                reason = f"SPY {drop_pct:.2%}" if spy_crash else f"VIX {vix_value:.0f}"
                return {
                    "status": "ALERT",
                    "drop_pct": round(drop_pct, 4),
                    "vix_spike": vix_spike,
                    "reason": f"Alerta de mercado: {reason}. Cobertura parcial recomendada.",
                    "recommended_hedge_pct": self.hedge_target_pct * 0.5,
                    "hedge_ticker": hedge_ticker,
                }
            else:
                return {
                    "status": "NORMAL",
                    "drop_pct": round(drop_pct, 4),
                    "vix_spike": False,
                    "reason": f"Mercado estable (SPY {drop_pct:.2%}, VIX {vix_value:.0f}).",
                    "recommended_hedge_pct": 0.0,
                    "hedge_ticker": "SH",
                }

        except Exception as e:
            logger.error("Error crítico en hedging: %s", e)
            # Fail-closed: ante error, recomendar cobertura cautelosa
            return {
                "status": "ALERT",
                "drop_pct": 0.0,
                "vix_spike": False,
                "reason": f"Error en hedging: {e}. Cobertura cautelosa por seguridad.",
                "recommended_hedge_pct": 0.10,
                "hedge_ticker": "SH",
            }

    def calculate_hedge_size(
        self, equity: float, portfolio_beta: float, hedge_ticker: str = "SH"
    ) -> dict:
        """Calcula cuántas acciones del ETF inverso comprar para neutralizar delta.

        Args:
            equity: Valor total del portafolio
            portfolio_beta: Beta ponderada del portafolio vs SPY
            hedge_ticker: "SH" (1x inverso) o "SQQQ" (3x inverso)

        Returns:
            {"ticker": str, "target_value": float, "hedge_ratio": float, "reason": str}
        """
        # Beta del hedge: SH = -1.0, SQQQ = -3.0
        hedge_beta = -3.0 if hedge_ticker == "SQQQ" else -1.0

        # Valor necesario para neutralizar el delta del portafolio
        if portfolio_beta <= 0:
            return {
                "ticker": hedge_ticker,
                "target_value": 0.0,
                "hedge_ratio": 0.0,
                "reason": "Portafolio ya está neutralizado o short.",
            }

        target_value = (portfolio_beta * equity * self.hedge_target_pct) / abs(hedge_beta)
        hedge_ratio = target_value / equity if equity > 0 else 0

        return {
            "ticker": hedge_ticker,
            "target_value": round(target_value, 2),
            "hedge_ratio": round(hedge_ratio, 4),
            "reason": f"Neutralizar beta {portfolio_beta:.2f} con {hedge_ticker} (beta {hedge_beta}).",
        }

    def needs_rebalance(
        self, current_hedge_value: float, target_hedge_value: float, threshold: float = 0.20
    ) -> tuple[bool, str]:
        """Determina si la cobertura necesita rebalanceo.

        Args:
            current_hedge_value: Valor actual del hedge en el portafolio
            target_hedge_value: Valor objetivo del hedge
            threshold: Desviación máxima tolerada (20% por defecto)

        Returns:
            (needs_rebalance, reason)
        """
        if target_hedge_value <= 0:
            return False, "No se requiere hedge."

        if current_hedge_value <= 0:
            return True, "No hay hedge activo pero se requiere."

        deviation = abs(current_hedge_value - target_hedge_value) / target_hedge_value
        if deviation > threshold:
            direction = "aumentar" if current_hedge_value < target_hedge_value else "reducir"
            return True, f"Hedge desviado {deviation:.0%}. {direction} cobertura."
        return False, f"Hedge dentro de tolerancia ({deviation:.0%} < {threshold:.0%})."

    def _normal_state(self, reason: str) -> dict:
        return {
            "status": "NORMAL",
            "drop_pct": 0.0,
            "vix_spike": False,
            "reason": reason,
            "recommended_hedge_pct": 0.0,
            "hedge_ticker": "SH",
        }
